""" Denoising model """

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import knn_graph

from ligbinddiff.model.modules.common import GaussianRandomFourierBasis
from ligbinddiff.model.modules.layers.edge.sitewise import EdgeTransition
from ligbinddiff.model.modules.layers.node.attention import GraphInvariantPointAttention
from ligbinddiff.model.modules.layers.node.mpnn import IPMP

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.framediff.all_atom import compute_backbone, compute_all_atom14, adjust_oxygen_pos
from ligbinddiff.utils.openfold.feats import atom14_to_atom37

from ligbinddiff.data.datasets.featurize.common import _node_positional_embeddings
from ligbinddiff.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf

from ligbinddiff.model.modules.openfold.frames import BackboneUpdate, StructureModuleTransition
from ligbinddiff.model.modules.layers.edge.embed import MLMPairwiseAtomicEmbedding, SelfConditionPairwiseAtomicEmbedding

from ligbinddiff.model.utils.graph import sample_inv_cubic_edges, sequence_local_graph, gen_spatial_graph_features, batchwise_to_nodewise

from ligbinddiff.model.denoiser.bb.framediff import TorsionAngles
from ligbinddiff.model.modules.openfold.frames import Linear

from ligbinddiff.runtime.loss.common import _collect_from_seq
from ligbinddiff.data.openfold.data_transforms import make_atom14_masks

from torch_geometric.utils import coalesce


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
                 lrange_k=40
                 ):
        """
        Args
        ----
        """
        super().__init__()

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
        self.node_transition = StructureModuleTransition(
            c=c_s
        )

    def forward(
            self,
            node_features,
            rigids,
            edge_features,
            edge_index,
            new_seq_edge_inputs,
            seq_edge_features,
            seq_edge_index,
            res_data,
    ):
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

        res_mask = ~(res_data['res_mask'].bool())
        noising_mask = res_data['noising_mask']

        node_s_update = self.attn_spatial(
            s=node_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=(~res_mask).float()
        )
        node_s_update = node_s_update * (~res_mask)[..., None]
        node_features = self.ln_s1(node_features + node_s_update)

        node_s_update = self.attn_seq(
            s=node_features,
            z=seq_edge_features,
            edge_index=seq_edge_index,
            r=rigids,
            mask=(~res_mask).float()
        )
        node_s_update = node_s_update * (~res_mask)[..., None]
        node_features = self.ln_s2(node_features + node_s_update)

        node_features = self.node_transition(node_features)
        node_features = node_features * (~res_mask)[..., None]
        rigids_update = self.bb_update(
            node_features * noising_mask[..., None])

        rigids = rigids.compose_q_update_vec(
            rigids_update * noising_mask[..., None]
        )
        seq_edge_features = self.seq_edge_transition(node_features, seq_edge_features, seq_edge_index)

        return node_features, rigids, edge_features, seq_edge_features


class DynamicGraphIpaFrameDirichletDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_pts=8,
                 num_v_pts=12,
                 h_time=64,
                 n_layers=4,
                 knn_k=20,
                 lrange_k=40,
                 self_conditioning=True,
                 use_seq_edge=True,
                 use_final_transition=True,
                 impute_oxy=False,
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
        self.node_in = c_s + h_time + 1 + num_aa + self_conditioning * (c_s + 7 + num_aa)

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
            )
            for i in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k

        if use_final_transition:
            self.final_transition = StructureModuleTransition(c_s)
        else:
            self.final_transition = None

        self.torsion_angles = nn.Linear(c_s + num_aa, 81 * 2)
        self.seq_logits = nn.Linear(c_s, num_aa)


    def _gen_spatial_edge_features(self, rigids, res_mask, batch, self_condition):
        res_mask = ~res_mask
        # generate spatial edges
        X_ca = rigids.get_trans()
        masked_X_ca = X_ca.clone()
        masked_X_ca[res_mask] = torch.inf
        edge_index, knn_edge_select, lrange_edge_select = sample_inv_cubic_edges(
            masked_X_ca, res_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

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
        node_input = torch.cat([
                res_pos,
                fourier_time,
                seq_probs_t,
                res_data['noising_mask'].float()[..., None]
            ], dim=-1)

        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
            self_cond_nodes = self_condition['node_features']
            self_cond_seq_logits_1 = self_condition['decoded_seq_logits']

            trans_rel = self_cond_rigids.get_trans() - rigids_t.get_trans()
            rigids_t_quat = rigids_t.get_rots().get_quats()
            self_cond_quat = self_cond_rigids.get_rots().get_quats()
            quat_rel = ru.quat_multiply(
                ru.invert_quat(rigids_t_quat),
                self_cond_quat
            )

            t7_rel = torch.cat([quat_rel, trans_rel], dim=-1)

            node_input = torch.cat(
                [node_input, self_cond_nodes, t7_rel, self_cond_seq_logits_1],
                dim=-1
            )

        elif self.self_conditioning:
            node_input = F.pad(
                node_input,
                (0, self.node_in - node_input.shape[-1])
            )

        return node_input, seq_local_edge_index


    def forward(self, data, self_condition=None):
        res_data = data['residue']
        res_mask = (res_data['res_mask']).bool()

        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'])
        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, res_data.batch, res_data['res_mask'].bool())
        rigids_t = rigids_t.translate(-center)

        node_input, seq_edge_index = self._gen_inital_features(data, self_condition=self_condition)

        # embed features
        node_features = self.embed_node(node_input)
        node_features = node_features * res_mask[..., None]
        seq_edge_features = torch.zeros((seq_edge_index.shape[-1], self.c_z), device=node_features.device)

        ## denoising
        rigids_t = rigids_t.scale_translation(0.1)
        rigids = rigids_t

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

            node_features, rigids, edge_features, seq_edge_features = layer(
                node_features,
                rigids,
                raw_edge_features,
                edge_index,
                new_seq_edge_inputs,
                seq_edge_features,
                seq_edge_index,
                res_data,
            )

        rigids = rigids.scale_translation(10)

        if self.final_transition is not None:
            node_features = self.final_transition(node_features)

        seq_logits = self.seq_logits(node_features)
        torsions = self.torsion_angles(
            torch.cat([node_features, seq_logits], dim=-1)
        )
        torsions = F.normalize(torsions.view(-1, 81, 2), dim=-1, eps=1e-12)
        psi, chis = torsions.split([1, 80], dim=-2)
        chis_per_aatype = chis.view(-1, 20, 4, 2)

        all_atom14 = compute_all_atom14(rigids, psi, chis_per_aatype)
        atom14_gt_seq = _collect_from_seq(all_atom14, res_data['seq'], res_data['seq_mask'])
        atom14_pred_seq = _collect_from_seq(all_atom14, seq_logits.argmax(dim=-1), res_data['seq_mask'])
        chis_gt_seq = _collect_from_seq(chis_per_aatype, res_data['seq'], res_data['seq_mask'])
        chis_pred_seq = _collect_from_seq(chis_per_aatype, seq_logits.argmax(dim=-1), res_data['seq_mask'])

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
        ret['psi'] = psi
        ret['node_features'] = node_features
        ret['edge_index'] = edge_index

        # some dummy variables for hacking smth into the framework rn
        ret['latent_mu'] = seq_logits
        ret['latent_logvar'] = torch.ones_like(seq_logits)

        return ret
