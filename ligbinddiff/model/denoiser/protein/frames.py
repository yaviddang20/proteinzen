""" Denoising model """

import torch
from torch import nn
import torch.nn.functional as F
from ligbinddiff.model.modules.common import RBF
from ligbinddiff.model.modules.layers.edge.sitewise import EdgeTransition
from ligbinddiff.model.modules.layers.node.attention import GraphInvariantPointAttention
from ligbinddiff.model.modules.layers.node.mpnn import IPMP

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.framediff.all_atom import compute_backbone

from ligbinddiff.data.datasets.featurize.common import _node_positional_embeddings

from ligbinddiff.model.modules.openfold.frames import BackboneUpdate, StructureModuleTransition

from ligbinddiff.model.utils.graph import sample_inv_cubic_edges, sequence_local_graph, gen_spatial_graph_features, batchwise_to_nodewise

from ligbinddiff.model.denoiser.bb.framediff import TorsionAngles
from ligbinddiff.model.modules.openfold.frames import Linear


from torch_geometric.utils import coalesce


class BackboneDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 self_conditioning=False,
                 ):
        """
        Args
        ----
        """
        super().__init__()

        self.attn_spatial = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )
        self.attn_seq = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )
        self.ln_s1 = nn.LayerNorm(c_s)
        self.ln_s2 = nn.LayerNorm(c_s)

        self.bb_update = BackboneUpdate(
            c_s
        )
        self.node_transition = StructureModuleTransition(
            c=c_s
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(c_z + self_conditioning * (c_z//2), c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.LayerNorm(c_z)
        )


    def forward(
            self,
            node_features,
            rigids,
            edge_features,
            edge_index,
            seq_edge_features,
            seq_edge_index,
            res_data,
    ):
        edge_features = self.edge_embed(edge_features)

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
        node_features = self.ln_s1(node_features + node_s_update)

        node_features = self.node_transition(node_features)
        node_features = node_features * (~res_mask)[..., None]
        rigids_update = self.bb_update(
            node_features * noising_mask[..., None])

        rigids = rigids.compose_q_update_vec(
            rigids_update * noising_mask[..., None]
        )

        return node_features, rigids


class SidechainDenoisingLayer(nn.Module):
    def __init__(self,
                 c_s,
                 c_latent,
                 c_z,
                 c_hidden,
                 self_conditioning=True):
        super().__init__()
        self.c_s = c_s
        self.c_latent = c_latent
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.self_conditioning = self_conditioning

        h_dim = c_s + c_latent + c_latent*self_conditioning

        self.ipmp = IPMP(
            c_s=h_dim,
            c_z=c_z,
            c_hidden=c_hidden,
            dropout=0.,
            edge_dropout=0.,
        )

        self.latent_update = Linear(h_dim, c_latent, init='final')

        self.node_update = Linear(
            h_dim,
            c_s,
            init='final')
        self.node_ln = nn.LayerNorm(c_s)

    def forward(self,
                latent_features,
                node_features,
                edge_features,
                edge_index,
                rigids,
                node_mask,
                self_condition=None):

        input_features = [node_features, latent_features]
        if self.self_conditioning and self_condition is not None:
            input_features.append(self_condition['pred_latent_sidechain'])
        elif self.self_conditioning:
            input_features.append(torch.zeros_like(latent_features))

        input_features = torch.cat(input_features, dim=-1)

        joint_features, edge_features = self.ipmp(
            s=input_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=node_mask)

        latent_features = latent_features + self.latent_update(joint_features)
        node_features = self.node_ln(node_features + self.node_update(joint_features))

        return latent_features, node_features, edge_features



class DynamicGraphIpaFrameDenoiser(nn.Module):
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
                 num_vn=4,
                 vn_mode='attn',
                 self_conditioning=False,
                 graph_conditioning=False,
                 impute_oxy=False):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.vn_mode = vn_mode
        self.num_vn = num_vn
        self.self_conditioning = self_conditioning
        if graph_conditioning:
            assert self_conditioning, "graph conditioning requires self-conditioning"
        self.graph_conditioning = graph_conditioning
        self.n_layers = n_layers
        self.impute_oxy = impute_oxy

        self.h_time = h_time
        self.time_rbf = RBF(n_basis=h_time//2)

        self.embed_node = nn.Sequential(
            nn.Linear(
                # node_embedding + time_embedding + fixed_mask + self_conditioning
                c_s + h_time + 1 + self_conditioning * (c_s + 7),
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
        self.embed_vn = nn.Sequential(
            nn.Linear(
                h_time,
                c_s
            ),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s * num_vn),
            nn.LayerNorm(c_s * num_vn)
        )

        self.denoiser = nn.ModuleList([
            BackboneDenoisingLayer(
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 self_conditioning=self_conditioning,
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
        if self.graph_conditioning and self_condition is not None:
            self_cond_X_ca = self_condition['final_rigids'].get_trans()
            masked_X_ca = self_cond_X_ca.clone()
            masked_X_ca[res_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, res_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)
        else:
            masked_X_ca = X_ca.clone()
            masked_X_ca[res_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, res_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        # compute edge features
        edge_features, _ = gen_spatial_graph_features(X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
            self_cond_X_ca = self_cond_rigids.get_trans()
            self_cond_edge_features, _ = gen_spatial_graph_features(self_cond_X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=0)
            edge_features = torch.cat(
                [edge_features, self_cond_edge_features],
                dim=-1
            )
        elif self.self_conditioning:
            edge_features = F.pad(
                edge_features,
                (0, self.c_z//2)
            )

        return edge_features, edge_index


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

        # # generate spatial edges
        X_ca = rigids_t.get_trans()
        seq_edge_features, _ = gen_spatial_graph_features(X_ca, seq_local_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_graph x h_time,)
        fourier_time = batchwise_to_nodewise(fourier_time, res_data.batch)  # (N_node x h_time,)

        # generate node features
        residx = torch.cat(residx, dim=-1)
        res_pos = _node_positional_embeddings(
            residx,
            num_embeddings=self.c_s,
            device=device)
        node_input = torch.cat([
                res_pos,
                fourier_time,
                res_data['noising_mask'].float()[..., None]
            ], dim=-1)

        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
            self_cond_nodes = self_condition['node_features']

            trans_rel = self_cond_rigids.get_trans() - rigids_t.get_trans()
            rigids_t_quat = rigids_t.get_rots().get_quats()
            self_cond_quat = self_cond_rigids.get_rots().get_quats()
            quat_rel = ru.quat_multiply(
                ru.invert_quat(rigids_t_quat),
                self_cond_quat
            )

            t7_rel = torch.cat([quat_rel, trans_rel], dim=-1)

            node_input = torch.cat(
                [node_input, self_cond_nodes, t7_rel],
                dim=-1
            )

        elif self.self_conditioning:
            node_input = F.pad(
                node_input,
                (0, self.c_s + 7)
            )

        edge_features, edge_index = self._gen_spatial_edge_features(rigids_t, res_mask, batch, self_condition)

        return node_input, edge_features, edge_index, seq_edge_features, seq_local_edge_index


    def forward(self, data, self_condition=None):
        res_data = data['residue']
        res_mask = (res_data['res_mask']).bool()

        (
            node_input,
            raw_edge_features,
            edge_index,
            seq_edge_features,
            seq_local_edge_index
        ) = self._gen_inital_features(data, self_condition=self_condition)

        # embed features
        node_features = self.embed_node(node_input)
        node_features = node_features * res_mask[..., None]
        seq_edge_features = self.seq_edge_embed(seq_edge_features)

        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'])
        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, res_data.batch, res_data['res_mask'].bool())
        rigids_t = rigids_t.translate(-center)

        ## denoising
        rigids_t = rigids_t.scale_translation(0.1)
        rigids = rigids_t

        rigids_history = []
        for i, layer in enumerate(self.denoiser):
            if i > 0:
                # recompute graph
                raw_edge_features, edge_index = self._gen_spatial_edge_features(
                    rigids,
                    res_mask,
                    res_data.batch,
                    self_condition)
            node_features, rigids = layer(
                node_features,
                rigids,
                raw_edge_features,
                edge_index,
                seq_edge_features,
                seq_local_edge_index,
                res_data,
            )

            # need to rescale and re-translate rigids to original reference
            rigids_history.append(rigids.scale_translation(10).translate(center))

        _, psi = self.torsion_angles(node_features)

        rigids = rigids.scale_translation(10)
        rigids = rigids.translate(center)
        ret = {}
        ret['denoised_frames'] = rigids
        ret['intermediate_rigids'] = rigids_history
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)
        denoised_bb = denoised_bb[:, :5]
        ret['denoised_bb'] = denoised_bb
        ret['psi'] = psi
        ret['node_features'] = node_features
        ret['edge_index'] = edge_index

        return ret
