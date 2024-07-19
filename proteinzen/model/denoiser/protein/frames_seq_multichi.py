""" Denoising model """

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import knn_graph

from proteinzen.model.modules.common import GaussianRandomFourierBasis
from proteinzen.model.modules.layers.edge.sitewise import EdgeTransition
from proteinzen.model.modules.layers.node.attention import GraphInvariantPointAttention
from proteinzen.model.modules.layers.node.mpnn import IPMP

from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.utils.framediff.all_atom import compute_backbone, compute_atom14, compute_all_atom14, adjust_oxygen_pos
from proteinzen.utils.openfold.feats import atom14_to_atom37

from proteinzen.data.datasets.featurize.common import _node_positional_embeddings
from proteinzen.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from proteinzen.data.datasets.featurize.common import _edge_positional_embeddings, _rbf

from proteinzen.model.modules.openfold.layers import BackboneUpdate, StructureModuleTransition
from proteinzen.model.modules.layers.edge.embed import MLMPairwiseAtomicEmbedding, SelfConditionPairwiseAtomicEmbedding

from proteinzen.model.utils.graph import sample_inv_cubic_edges, sequence_local_graph, gen_spatial_graph_features, batchwise_to_nodewise

from proteinzen.model.denoiser.bb.framediff import TorsionAngles
from proteinzen.model.modules.openfold.layers import Linear

from proteinzen.runtime.loss.common import _collect_from_seq
from proteinzen.data.openfold.data_transforms import make_atom14_masks

from torch_geometric.utils import coalesce


class ChiUpdate(nn.Module):
    def __init__(self, c_s, num_aa, num_chis):
        super().__init__()
        self.num_aa = num_aa
        self.num_chis = num_chis
        self.chi_update = FeedForward(c_s, num_aa * num_chis)

    def forward(self, node_features, chis, chi_noising_mask):
        angle_update = torch.tanh(self.chi_update(node_features)) * torch.pi
        chi_update = torch.stack([
            torch.cos(angle_update),
            torch.sin(angle_update)
        ], dim=-1)
        chi_update = chi_update.view(-1, self.num_aa, self.num_chis, 2)
        rot_col_2 = torch.stack(
            [
                -chi_update[..., 1],
                chi_update[..., 0]
            ],
            dim=-1
        )
        rot_matrix = torch.stack(
            [chi_update, rot_col_2],
            dim=-2
        )
        new_chis = torch.einsum("...ij,...j->...i", rot_matrix, chis)
        new_chis = new_chis * chi_noising_mask[..., None, None, None] + chis * (~chi_noising_mask[..., None, None, None])
        return new_chis


class SeqProbsUpdate(nn.Module):
    def __init__(self, c_s, num_aa, seq_matrix_updates=False):
        super().__init__()
        self.seq_matrix_updates = seq_matrix_updates
        self.num_aa = num_aa
        if seq_matrix_updates:
            self.transition_matrix_layer = nn.Sequential(
                Linear(c_s, c_s, init='relu'),
                nn.ReLU(),
                Linear(c_s, c_s, init='relu'),
                nn.ReLU(),
                Linear(c_s, num_aa ** 2, init='final')
            )
        else:
            self.transition_trunk = nn.Sequential(
                Linear(c_s, c_s, init='relu'),
                nn.ReLU(),
                Linear(c_s, c_s, init='relu'),
                nn.ReLU(),
            )
            self.transition_out = Linear(c_s, num_aa, init='final')
            self.transition_mix = Linear(c_s, 1, init='gating')


    def forward(self, node_features, seq_probs, seq_noising_mask):
        if self.seq_matrix_updates:
            base = torch.eye(self.num_aa, device=node_features.device)[None]
            transition_matrix = base + self.transition_matrix_layer(node_features).view(-1, self.num_aa, self.num_aa)
            transition_matrix = torch.softmax(transition_matrix, dim=-1)
            transition_matrix = transition_matrix * seq_noising_mask[..., None, None] + base * (~seq_noising_mask[..., None, None])
            # transition_matrix[~seq_noising_mask] = base
            seq_probs = torch.einsum("...i,...ij->...j", seq_probs, transition_matrix)
            return seq_probs
        else:
            x = self.transition_trunk(node_features)
            seq_update = torch.softmax(self.transition_out(x), dim=-1)
            seq_mix = torch.sigmoid(self.transition_mix(x))
            seq_new = seq_probs * seq_mix + seq_update * (1 - seq_mix)
            assert (seq_new >= 0).all()
            return seq_new * seq_noising_mask[..., None] + seq_probs * (~seq_noising_mask[..., None])


class FeedForward(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.):
        super().__init__()

        self.linear_1 = Linear(c_in, c_out, init="relu")
        self.linear_2 = Linear(c_out, c_out, init="relu")
        self.linear_3 = Linear(c_out, c_out, init="final")
        self.relu = nn.ReLU()
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, s):
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        if self.dropout is not None:
            s = self.dropout(s)

        return s


class GraphIpaFrameDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 self_conditioning=False,
                 use_seq_edge=True,
                 num_rbf=64,
                 knn_k=20,
                 lrange_k=40,
                 num_aa=20,
                 seq_matrix_updates=True,
                 ):
        """
        Args
        ----
        """
        super().__init__()
        self.num_aa = num_aa

        self.seq_chi_to_node_update = FeedForward(num_aa + c_s + num_aa * 8, c_s)
        self.init_ln = nn.LayerNorm(c_s)

        self.edge_embed = nn.Sequential(
            nn.Linear(c_s*2 + c_z + 4 + self_conditioning * (c_z//2 + 4), c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.LayerNorm(c_z)
        )
        self.seq_edge_update = nn.Sequential(
            nn.Linear(c_s*2 + c_z + 4 + self_conditioning * (c_z//2 + 4), c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
        )
        self.seq_edge_ln = nn.LayerNorm(c_z)

        self.attn_spatial = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )
        self.ln_s1 = nn.LayerNorm(c_s)

        self.seq_edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )
        self.attn_seq = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )
        self.ln_s2 = nn.LayerNorm(c_s)

        self.bb_update = BackboneUpdate(
            c_s
        )
        self.chi_update = ChiUpdate(c_s, num_aa, 4)
        self.node_transition = StructureModuleTransition(
            c=c_s
        )
        self.seq_probs_update = SeqProbsUpdate(
            c_s=c_s, num_aa=num_aa, seq_matrix_updates=seq_matrix_updates
        )

    def forward(
            self,
            node_features,
            rigids,
            seq_probs,
            chis,
            edge_features,
            edge_index,
            new_seq_edge_inputs,
            seq_edge_features,
            seq_edge_index,
            res_data,
    ):
        res_mask = res_data['res_mask'].bool()
        seq_mask = res_data['seq_mask'].bool()
        res_noising_mask = res_data['res_noising_mask']
        seq_noising_mask = res_data['seq_noising_mask']
        chi_noising_mask = res_data['chi_noising_mask']
        node_update = self.seq_chi_to_node_update(
            torch.cat([
                node_features,
                chis.view(-1, self.num_aa * 8),
                seq_probs],
            dim=-1)
        )
        node_features = self.init_ln(node_features + node_update * (~res_mask)[..., None])

        edge_inputs = torch.cat([
            edge_features,
            node_features[edge_index[0]],
            node_features[edge_index[1]]
        ], dim=-1)
        edge_features = self.edge_embed(edge_inputs)

        seq_edge_inputs = torch.cat([
            new_seq_edge_inputs,
            node_features[seq_edge_index[0]],
            node_features[seq_edge_index[1]]
        ], dim=-1)
        seq_edge_features = self.seq_edge_ln(
            seq_edge_features +
            self.seq_edge_update(seq_edge_inputs)
        )

        node_s_update = self.attn_spatial(
            s=node_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=(~res_mask).float()
        )
        node_s_update = node_s_update * res_mask[..., None]
        node_features = self.ln_s1(node_features + node_s_update)

        node_s_update = self.attn_seq(
            s=node_features,
            z=seq_edge_features,
            edge_index=seq_edge_index,
            r=rigids,
            mask=res_mask.float()
        )
        node_s_update = node_s_update * res_mask[..., None]
        node_features = self.ln_s2(node_features + node_s_update)

        node_features = self.node_transition(node_features)
        node_features = node_features * res_mask[..., None]
        rigids_update = self.bb_update(
            node_features * res_noising_mask[..., None])

        rigids = rigids.compose_q_update_vec(
            rigids_update * res_noising_mask[..., None]
        )
        seq_edge_features = self.seq_edge_transition(node_features, seq_edge_features, seq_edge_index)

        # TODO: i should probably have a dedicated "do chis exist" mask rather than using seq_mask
        chis = self.chi_update(node_features, chis, chi_noising_mask) * seq_mask[..., None, None, None]
        seq_probs = self.seq_probs_update(
            node_features,
            seq_probs,
            seq_noising_mask
        ) * seq_mask[..., None]

        return node_features, rigids, seq_probs, chis, edge_features, seq_edge_features

class IPMPUpdateLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_aa=20
                 ):
        """
        Args
        ----
        """
        super().__init__()
        self.num_aa = num_aa

        self.chis_to_node = Linear(c_s + num_aa*8, c_s, init='final')
        self.init_ln = nn.LayerNorm(c_s)
        self.ipmp = IPMP(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
        )

        self.chi_update = ChiUpdate(c_s, num_aa, 4)

    def forward(
            self,
            node_features,
            rigids,
            edge_features,
            edge_index,
            chis,
            res_data,
    ):
        res_mask = res_data['res_mask'].bool()
        node_update = self.chis_to_node(
            torch.cat([node_features, chis.view(-1, self.num_aa * 8)], dim=-1)
        )
        node_features = self.init_ln(node_features + node_update * res_mask[..., None])
        node_features, edge_features = self.ipmp(
            s=node_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=res_mask
        )
        chis = self.chi_update(node_features, chis)

        return node_features, edge_features, chis


class DynamicGraphIpaFrameSeqMultiChiDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_hidden=16,
                 num_heads=16,
                 num_qk_pts=8,
                 num_v_pts=12,
                 h_time=64,
                 n_layers=8,
                 knn_k=20,
                 lrange_k=40,
                 self_conditioning=False,
                 use_seq_edge=True,
                 impute_oxy=False,
                 use_ipmp_trunk=False,
                 seq_matrix_updates=False,
                 num_aa=20):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.num_aa = num_aa
        self.num_rbf = c_z//2
        self.self_conditioning = self_conditioning
        self.n_layers = n_layers
        self.impute_oxy = impute_oxy

        self.h_time = h_time
        self.time_rbf = GaussianRandomFourierBasis(n_basis=h_time//2)

        # node_embedding + time_embedding + fixed_mask + self_conditioning
        self.node_in = c_s + h_time + 1 + num_aa + 8 + self_conditioning * (7 + num_aa + 8 * num_aa)

        self.embed_node = nn.Sequential(
            nn.Linear(
                self.node_in,
                c_s
            ),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.LayerNorm(c_s)
        )
        self.seq_edge_embed = nn.Sequential(
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.LayerNorm(c_z)
        )

        self.denoiser = nn.ModuleList([
            GraphIpaFrameDenoisingLayer(
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 self_conditioning=self_conditioning,
                 use_seq_edge=use_seq_edge,
                 knn_k=knn_k,
                 lrange_k=lrange_k,
                 num_rbf=self.num_rbf,
                 seq_matrix_updates=seq_matrix_updates,
                 num_aa=num_aa
            )
            for i in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k

        self.torsion_angles = nn.Linear(c_s, 2)

        self.use_ipmp_trunk = use_ipmp_trunk
        if use_ipmp_trunk:
            self.ipmp_num_rbf = 16
            self.ipmp_num_pos_embed = 16
            atoms_per_res = 5
            self.c_z_in = (
                self.ipmp_num_rbf * (atoms_per_res ** 2)  # bb x bb distances
                # + atoms_per_res * 3  # src CA to dst bb direction unit vectors
                + self.ipmp_num_pos_embed  # rel pos embed
            )
            self.embed_ipmp_edge = nn.Sequential(
                nn.Linear(self.c_z_in, c_z),
                nn.ReLU(),
                nn.Linear(c_z, c_z),
                nn.ReLU(),
                nn.Linear(c_z, c_z),
                nn.LayerNorm(c_z)
            )

            self.ipmp_trunk = nn.ModuleList([
                IPMPUpdateLayer(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden
                )
                for _ in range(n_layers)
            ])

        self.seq_logits = nn.Linear(c_s, num_aa)
        self.final_transition = StructureModuleTransition(c_s)


    def _gen_spatial_edge_features(self, rigids, res_mask, batch, self_condition):
        # generate spatial edges
        X_ca = rigids.get_trans()
        masked_X_ca = X_ca.clone()
        masked_X_ca[~res_mask] = torch.inf
        edge_index, knn_edge_select, lrange_edge_select = sample_inv_cubic_edges(
            masked_X_ca, ~res_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        # compute edge features
        edge_features, _ = gen_spatial_graph_features(rigids, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
            self_cond_edge_features, _ = gen_spatial_graph_features(self_cond_rigids, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=0)
            edge_features = torch.cat(
                [edge_features, self_cond_edge_features],
                dim=-1
            )
        elif self.self_conditioning:
            edge_features = F.pad(
                edge_features,
                (0, self.c_z//2 + 4)
            )

        return edge_features, edge_index


    def _gen_seq_edge_features(self, rigids, seq_edge_index, self_condition):
        # compute edge features
        edge_features, _ = gen_spatial_graph_features(rigids, seq_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
            self_cond_edge_features, _ = gen_spatial_graph_features(self_cond_rigids, seq_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=0)
            edge_features = torch.cat(
                [edge_features, self_cond_edge_features],
                dim=-1
            )
        elif self.self_conditioning:
            edge_features = F.pad(
                edge_features,
                (0, self.c_z//2 + 4)
            )
        return edge_features


    def _gen_inital_features(self, data, self_condition):
        res_data = data['residue']
        ## prep features
        # ts = res_data['t']  # (B,)
        ts = data['t']  # (B,)
        res_mask = res_data['res_mask'].bool()
        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'])
        batch = res_data.batch
        device = ts.device

        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, res_data.batch, res_mask)
        rigids_t = rigids_t.translate(-center)

        # # generate sequence edges
        seq_local_edge_index = []
        residx = []
        offset = 0
        for i in range(res_data.batch.max().item() + 1):
            select = (res_data.batch == i)
            subset_num_nodes = select.sum().item()
            local_residx = torch.arange(subset_num_nodes, device=device)
            residx.append(local_residx)
            seq_local_edge_index.append(
                sequence_local_graph(subset_num_nodes, ~res_mask[select]) + offset
            )
            offset += select.sum().item()
        seq_local_edge_index = torch.cat(seq_local_edge_index, dim=-1)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_graph x h_time,)
        fourier_time = batchwise_to_nodewise(fourier_time, res_data.batch)  # (N_node x h_time,)

        # generate node features
        residx = torch.cat(residx, dim=-1)
        res_pos = _node_positional_embeddings(
            residx,
            num_embeddings=self.c_s,
            device=device)
        seq_probs_t = res_data['seq_probs_t']
        chis_t = res_data['chis_t']
        node_input = torch.cat([
            res_pos,
            fourier_time,
            seq_probs_t,
            res_data['noising_mask'].float()[..., None],
            chis_t.view(-1, 8),
        ], dim=-1)

        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
            self_cond_seq_logits_1 = self_condition['decoded_seq_logits']
            self_cond_chis_1 = self_condition['decoded_chis_all']

            trans_rel = self_cond_rigids.get_trans() - rigids_t.get_trans()
            rigids_t_quat = rigids_t.get_rots().get_quats()
            self_cond_quat = self_cond_rigids.get_rots().get_quats()
            quat_rel = ru.quat_multiply(
                ru.invert_quat(rigids_t_quat),
                self_cond_quat
            )

            t7_rel = torch.cat([quat_rel, trans_rel], dim=-1)

            node_input = torch.cat(
                [node_input, t7_rel, self_cond_seq_logits_1, self_cond_chis_1.view(-1, 8 * self.num_aa)],
                dim=-1
            )

        elif self.self_conditioning:
            node_input = F.pad(
                node_input,
                (0, self.node_in - node_input.shape[-1])
            )

        return node_input, seq_local_edge_index


    def _gen_ipmp_edge_features(self, rigids, psis, batch, res_mask, eps=1e-8):
        # edge graph
        masked_X_ca = rigids.get_trans().clone()
        masked_X_ca[~res_mask] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.knn_k, batch)

        # edge features
        bb = compute_backbone(rigids.unsqueeze(0), psis.unsqueeze(0))[-1].squeeze(0)[..., :5, :]
        src = edge_index[1]
        dst = edge_index[0]

        # ## edge distances
        edge_bb_src = bb[src]
        edge_bb_dst = bb[dst]
        edge_bb_dists = torch.linalg.vector_norm(
            edge_bb_src[..., None, :] - edge_bb_dst[..., None, :, :] + eps,
            dim=-1)
        edge_bb_dists = edge_bb_dists.view(edge_index.shape[1], -1, 1)
        edge_rbf = _rbf(edge_bb_dists, D_min=2.0, D_max=22.0, D_count=self.ipmp_num_rbf, device=edge_index.device)
        edge_rbf = edge_rbf.view(edge_index.shape[1], -1)
        # ## edge rel pos embedding
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=self.ipmp_num_pos_embed, device=edge_index.device)
        # edge_features = self.init_edge_embed(graph, edge_index)
        # technically this shouldn't be necessary but just to be safe
        # edge_mask = res_mask[src] & res_mask[dst]
        # edge_features = edge_features * edge_mask[..., None]
        edge_features = torch.cat([edge_rbf, edge_dist_rel_pos], dim=-1)

        return edge_features, edge_index



    def forward(self, data, self_condition=None):
        res_data = data['residue']
        res_mask = (res_data['res_mask']).bool()

        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'])
        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, res_data.batch, res_data['res_mask'].bool())
        rigids_t = rigids_t.translate(-center)

        node_input, seq_edge_index = self._gen_inital_features(data, self_condition=self_condition)
        seq_probs = res_data['seq_probs_t']

        # embed features
        node_features = self.embed_node(node_input)
        node_features = node_features * res_mask[..., None]
        seq_edge_features = torch.zeros((seq_edge_index.shape[-1], self.c_z), device=node_features.device)

        ## denoising
        rigids_t = rigids_t.scale_translation(0.1)
        rigids = rigids_t
        chis = res_data['chis_t']
        chis = chis[..., None, :, :].expand(-1, self.num_aa, -1, -1).contiguous()

        for i, layer in enumerate(self.denoiser):
            # recompute graph
            raw_edge_features, edge_index = self._gen_spatial_edge_features(
                rigids.scale_translation(10),
                res_mask,
                res_data.batch,
                self_condition)
            new_seq_edge_inputs = self._gen_seq_edge_features(
                rigids.scale_translation(10),
                seq_edge_index,
                self_condition)

            node_features, rigids, seq_probs, chis, edge_features, seq_edge_features = layer(
                node_features,
                rigids,
                seq_probs,
                chis,
                raw_edge_features,
                edge_index,
                new_seq_edge_inputs,
                seq_edge_features,
                seq_edge_index,
                res_data,
            )

        node_features = self.final_transition(node_features)
        rigids = rigids.scale_translation(10)

        psi = self.torsion_angles(node_features)
        psi = F.normalize(psi, dim=-1)
        if self.use_ipmp_trunk:
            ipmp_edge_features, ipmp_edge_index = self._gen_ipmp_edge_features(
                rigids, psi, res_data.batch, res_mask
            )
            ipmp_edge_features = self.embed_ipmp_edge(ipmp_edge_features)
            for i, layer in enumerate(self.ipmp_trunk):
                node_features, ipmp_edge_features, chis = layer(
                    node_features,
                    rigids,
                    ipmp_edge_features,
                    ipmp_edge_index,
                    chis,
                    res_data,
                )

        seq_logits = seq_probs

        chis_gt_seq = _collect_from_seq(chis, res_data['seq'], res_data['seq_mask'])
        chis_pred_seq = _collect_from_seq(chis, seq_logits.argmax(dim=-1), res_data['seq_mask'])
        atom14_gt_seq = compute_atom14(rigids, psi[..., None, :], chis_gt_seq, res_data['seq'])
        atom14_pred_seq = compute_atom14(rigids, psi[..., None, :], chis_pred_seq, seq_logits.argmax(dim=-1))

        atom14_mask_dict = make_atom14_masks({"aatype": seq_logits.argmax(dim=-1)})
        atom37, atom37_mask = atom14_to_atom37(atom14_pred_seq, atom14_mask_dict)
        atom37 = adjust_oxygen_pos(atom37, res_mask)
        atom14_gt_seq[:, 3, :] = atom37[:, 4, :]
        atom14_pred_seq[:, 3, :] = atom37[:, 4, :]


        rigids = rigids.translate(center)
        ret = {}
        ret['denoised_frames'] = rigids
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.squeeze(-2).unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)
        denoised_bb = denoised_bb[:, :5]
        ret['denoised_bb'] = denoised_bb
        ret['decoded_seq_logits'] = seq_logits
        ret['decoded_atom14_gt_seq'] = atom14_gt_seq
        ret['decoded_chis_gt_seq'] = chis_gt_seq
        ret['decoded_atom14'] = atom14_pred_seq
        ret['decoded_atom14_mask'] = atom14_mask_dict['atom14_atom_exists']
        ret['decoded_chis'] = chis_pred_seq
        ret['decoded_chis_all'] = chis
        ret['psi'] = psi
        ret['node_features'] = node_features
        ret['edge_index'] = edge_index
        ret['seq_probs'] = seq_probs

        # some dummy variables for hacking smth into the framework rn
        ret['latent_mu'] = torch.zeros_like(seq_logits)
        ret['latent_logvar'] = torch.ones_like(seq_logits)

        return ret
