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
from proteinzen.utils.framediff.all_atom import compute_backbone

from proteinzen.data.datasets.featurize.common import _node_positional_embeddings
from proteinzen.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from proteinzen.data.datasets.featurize.common import _edge_positional_embeddings, _rbf

from proteinzen.model.modules.openfold.layers import BackboneUpdate, StructureModuleTransition
from proteinzen.model.modules.layers.edge.embed import MLMPairwiseAtomicEmbedding, SelfConditionPairwiseAtomicEmbedding

from proteinzen.model.utils.graph import sample_inv_cubic_edges, sequence_local_graph, gen_spatial_graph_features, batchwise_to_nodewise

from proteinzen.model.denoiser.bb.framediff import TorsionAngles
from proteinzen.model.modules.openfold.layers import Linear


from torch_geometric.utils import coalesce


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


class EdgeEmbed(nn.Module):
    def __init__(self,
                 edge_in,
                 edge_hidden,
                 edge_out,
                 n_layers=3,
                 gate_edge=False,
                 skip_conn=False,
    ):
        super().__init__()
        assert n_layers > 1
        layers = [Linear(edge_in, edge_hidden), nn.ReLU()]
        for _ in range(n_layers-2):
            layers.append(Linear(edge_hidden, edge_hidden))
            layers.append(nn.ReLU())
        self.last = Linear(edge_hidden, edge_out)

        self.mlp = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(edge_out)

        if gate_edge:
            self.gate = Linear(edge_hidden, edge_out)
        else:
            self.gate = None

        if skip_conn:
            self.skip = Linear(edge_in, edge_out)
        else:
            self.skip = None

    def forward(self, edge_embed):
        x = edge_embed
        x = self.mlp(x)

        if self.gate is not None:
            gate = self.gate(x)
            x = self.last(x)
            x = torch.sigmoid(gate) * x
        else:
            x = self.last(x)

        if self.skip is not None:
            x = x + self.skip(edge_embed)

        x = self.ln(x)

        return x


class BackboneDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_latent,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 self_conditioning=False,
                 gate_spatial_edges=True,
                 skip_conn_spatial_edges=False,
                 sc_edge_unit_vecs=True
                 ):
        """
        Args
        ----
        """
        super().__init__()

        edge_in = c_s*2 + c_latent*2 + c_z + 4 + self_conditioning * (c_z//2 + 4 + 3 * sc_edge_unit_vecs)
        self.edge_embed = EdgeEmbed(
            edge_in=edge_in,
            edge_hidden=c_z,
            edge_out=c_z,
            gate_edge=gate_spatial_edges,
            skip_conn=skip_conn_spatial_edges
        )
        self.seq_edge_update = nn.Sequential(
            nn.Linear(edge_in, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
        )
        self.seq_edge_ln = nn.LayerNorm(c_z)

        self.latent_to_node_update = FeedForward(c_latent + c_s, c_s)
        self.ln_s0 = nn.LayerNorm(c_s)

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
        self.node_to_latent_update = FeedForward(c_s + c_latent, c_latent)

    def forward(
            self,
            node_features,
            rigids,
            latent_features,
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

        edge_inputs = torch.cat([
            edge_features,
            node_features[edge_index[0]],
            node_features[edge_index[1]],
            latent_features[edge_index[0]],
            latent_features[edge_index[1]]
        ], dim=-1)
        edge_features = self.edge_embed(edge_inputs)

        seq_edge_inputs = torch.cat([
            new_seq_edge_inputs,
            node_features[seq_edge_index[0]],
            node_features[seq_edge_index[1]],
            latent_features[seq_edge_index[0]],
            latent_features[seq_edge_index[1]]
        ], dim=-1)
        seq_edge_features = self.seq_edge_ln(
            seq_edge_features +
            self.seq_edge_update(seq_edge_inputs)
        )

        latent_to_node_update = self.latent_to_node_update(
            torch.cat([latent_features, node_features], dim=-1)
        )
        node_features = self.ln_s0(node_features + latent_to_node_update * seq_mask[..., None])

        node_s_update = self.attn_spatial(
            s=node_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=res_mask.float()
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
        latent_features = latent_features + self.node_to_latent_update(
            torch.cat([latent_features, node_features], dim=-1) * seq_mask[..., None]
        ) * seq_noising_mask[..., None]

        return node_features, rigids, edge_features, seq_edge_features, latent_features


class DynamicGraphIpaFrameDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 c_s=256,
                 c_latent=128,
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
                 gate_spatial_edges=True,
                 skip_conn_spatial_edges=False,
                 sc_edge_unit_vecs=True,
                 impute_oxy=False,
                 num_aa=20,
                 masked_latent_feature=False
    ):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.num_aa = num_aa
        self.num_rbf = c_z//2
        self.self_conditioning = self_conditioning
        self.n_layers = n_layers
        self.impute_oxy = impute_oxy
        self.sc_edge_unit_vecs = sc_edge_unit_vecs
        self.masked_latent_feature = masked_latent_feature

        self.h_time = h_time
        self.time_rbf = GaussianRandomFourierBasis(n_basis=h_time//2)

        # node_embedding + time_embedding + fixed_mask + self_conditioning
        self.node_in = c_s + h_time + 1 + self_conditioning * (7 + c_latent) + (1 if masked_latent_feature else 0)

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
            BackboneDenoisingLayer(
                 c_s,
                 c_latent,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 self_conditioning=self_conditioning,
                 gate_spatial_edges=gate_spatial_edges,
                 skip_conn_spatial_edges=skip_conn_spatial_edges,
                 sc_edge_unit_vecs=sc_edge_unit_vecs
            )
            for i in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k
        self.torsion_angles = TorsionAngles(c_s, 1)

    def _gen_spatial_edge_features(self, rigids, res_mask, batch, self_condition):
        res_mask = ~res_mask
        # generate spatial edges
        X_ca = rigids.get_trans()
        masked_X_ca = X_ca.clone()
        masked_X_ca[res_mask] = torch.inf
        edge_index, knn_edge_select, lrange_edge_select = sample_inv_cubic_edges(
            masked_X_ca, res_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        # compute edge features
        edge_features, _ = gen_spatial_graph_features(
            rigids,
            edge_index,
            num_rbf_embed=self.c_z//2,
            num_pos_embed=self.c_z//2)

        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
            self_cond_edge_features, _ = gen_spatial_graph_features(
                self_cond_rigids,
                edge_index,
                num_rbf_embed=self.c_z//2,
                num_pos_embed=0,
                use_unit_vec=self.sc_edge_unit_vecs)
            edge_features = torch.cat(
                [edge_features, self_cond_edge_features],
                dim=-1
            )
        elif self.self_conditioning:
            edge_features = F.pad(
                edge_features,
                (0, self.c_z//2 + 4 + 3 * self.sc_edge_unit_vecs)
            )

        return edge_features, edge_index


    def _gen_seq_edge_features(self, rigids, seq_edge_index, self_condition):
        # compute edge features
        edge_features, _ = gen_spatial_graph_features(rigids, seq_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
            self_cond_edge_features, _ = gen_spatial_graph_features(
                self_cond_rigids,
                seq_edge_index,
                num_rbf_embed=self.c_z//2,
                num_pos_embed=0,
                use_unit_vec=self.sc_edge_unit_vecs)
            edge_features = torch.cat(
                [edge_features, self_cond_edge_features],
                dim=-1
            )
        elif self.self_conditioning:
            edge_features = F.pad(
                edge_features,
                (0, self.c_z//2 + 4 + 3 * self.sc_edge_unit_vecs)
            )
        return edge_features


    def _gen_inital_features(self, data, self_condition, gt_condition=False):
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
        node_input = [
            res_pos,
            fourier_time,
            res_data['noising_mask'].float()[..., None]
        ]
        if self.masked_latent_feature:
            node_input.append(
                res_data['seq_noising_mask'].float()[..., None]
            )
        node_input = torch.cat(node_input, dim=-1)

        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
            self_cond_latent = self_condition['pred_latent_sidechain']

            trans_rel = self_cond_rigids.get_trans() - rigids_t.get_trans()
            rigids_t_quat = rigids_t.get_rots().get_quats()
            self_cond_quat = self_cond_rigids.get_rots().get_quats()
            quat_rel = ru.quat_multiply(
                ru.invert_quat(rigids_t_quat),
                self_cond_quat
            )

            t7_rel = torch.cat([quat_rel, trans_rel], dim=-1)

            node_input = torch.cat(
                [node_input, t7_rel, self_cond_latent],
                dim=-1
            )

        elif self.self_conditioning:
            node_input = F.pad(
                node_input,
                (0, self.node_in - node_input.shape[-1])
            )

        return node_input, seq_local_edge_index


    def forward(self, data, intermediates, self_condition=None):
        res_data = data['residue']
        res_mask = (res_data['res_mask']).bool()

        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'])
        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, res_data.batch, res_data['res_mask'].bool())
        rigids_t = rigids_t.translate(-center)

        node_input, seq_edge_index = self._gen_inital_features(
            data,
            self_condition=self_condition,
        )
        latent_features = intermediates['noised_latent_sidechain']

        # embed features
        node_features = self.embed_node(node_input)
        node_features = node_features * res_mask[..., None]
        seq_edge_features = torch.zeros((seq_edge_index.shape[-1], self.c_z), device=node_features.device)

        ## denoising
        rigids_t = rigids_t.scale_translation(0.1)
        rigids = rigids_t

        rigid_traj = []
        latent_traj = []

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

            node_features, rigids, edge_features, seq_edge_features, latent_features = layer(
                node_features,
                rigids,
                latent_features,
                raw_edge_features,
                edge_index,
                new_seq_edge_inputs,
                seq_edge_features,
                seq_edge_index,
                res_data,
            )
            if i < len(self.denoiser) - 1:
                rigid_traj.append(rigids)
                latent_traj.append(latent_features)

        _, psi = self.torsion_angles(node_features)
        rigids = rigids.scale_translation(10)

        rigids = rigids.translate(center)
        ret = {}
        ret['denoised_frames'] = rigids
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)
        denoised_bb = denoised_bb[:, :5]
        ret['denoised_bb'] = denoised_bb
        ret['psi'] = psi
        ret['node_features'] = node_features
        ret['pred_latent_sidechain'] = latent_features
        ret['edge_index'] = edge_index
        ret['rigid_traj'] = rigid_traj
        ret['latent_traj'] = latent_traj

        return ret
