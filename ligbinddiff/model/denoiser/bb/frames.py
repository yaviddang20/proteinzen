""" Denoising model """

import torch
from torch import nn
import torch.nn.functional as F
from ligbinddiff.model.modules.common import RBF
from ligbinddiff.model.modules.layers.edge.sitewise import EdgeTransition
from ligbinddiff.model.modules.layers.node.attention import GraphInvariantPointAttention, PointSetAttentionWithEdgeBias
from ligbinddiff.model.modules.layers.node.mpnn import IPMP
from ligbinddiff.model.modules.layers.node.sitewise import BackboneUpdateVectorBias, ChannelwiseVectorGateUpdate, LocalFrameUpdate, NodeTransition, VectorLayerNorm

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.framediff.all_atom import compute_backbone

from ligbinddiff.data.datasets.featurize.common import _node_positional_embeddings

from ligbinddiff.model.modules.openfold.frames import BackboneUpdate, StructureModuleTransition

from ligbinddiff.model.utils.graph import sample_inv_cubic_edges, sequence_local_graph, gen_spatial_graph_features, batchwise_to_nodewise, get_data_lens

from .framediff import TorsionAngles


from ligbinddiff.model.modules.layers.lrange.anchor import LinearPoolUpdate

class GraphIpaFrameDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 use_anchors=True,
                 ):
        """
        Args
        ----
        """
        super().__init__()

        self.attn_seq = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )
        self.attn_spatial = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )
        self.ln_s1 = nn.LayerNorm(c_s)
        self.ln_s2 = nn.LayerNorm(c_s)

        self.use_anchors = use_anchors
        if self.use_anchors:
            self.pool_update = LinearPoolUpdate(c_s, ratio=0.05, connect_dim=3)

        self.bb_update = BackboneUpdate(
            c_s
        )
        self.node_transition = StructureModuleTransition(
            c=c_s
        )
        self.edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )
        self.seq_edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )


    def forward(
            self,
            node_features,
            rigids,
            edge_features,
            edge_index,
            seq_edge_features,
            seq_edge_index,
            data,
            intermediates,
    ):
        noising_mask = intermediates['noising_mask']
        x_mask = data['x_mask']

        node_s_update = self.attn_spatial(
            s=node_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=(~x_mask).float()
        )
        node_s_update = node_s_update * (~x_mask)[..., None]
        node_features = self.ln_s1(node_features + node_s_update)

        node_s_update = self.attn_seq(
            s=node_features,
            z=seq_edge_features,
            edge_index=seq_edge_index,
            r=rigids,
            mask=(~x_mask).float()
        )
        node_s_update = node_s_update * (~x_mask)[..., None]
        node_features = self.ln_s2(node_features + node_s_update)

        if self.use_anchors:
            node_features, anchor_kl, node_kl = self.pool_update(rigids.get_trans(), node_features, edge_index, data.batch)
        else:
            anchor_kl = torch.zeros(data.num_graphs, device=node_features.device)
            node_kl = torch.zeros(data.num_graphs, device=node_features.device)

        node_features = self.node_transition(node_features)
        node_features = node_features * (~x_mask)[..., None]
        rigids_update = self.bb_update(
            node_features * noising_mask[..., None])

        rigids = rigids.compose_q_update_vec(
            rigids_update * noising_mask[..., None]
        )
        edge_features = self.edge_transition(node_features, edge_features, edge_index)
        seq_edge_features = self.seq_edge_transition(node_features, seq_edge_features, seq_edge_index)

        return node_features, rigids, edge_features, seq_edge_features, anchor_kl, node_kl


class GraphIpaFrameDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 h_time=64,
                 n_layers=4,
                 knn_k=20,
                 lrange_k=30,
                 self_conditioning=False,
                 graph_conditioning=False,
                 use_anchors=True):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.self_conditioning = self_conditioning
        if graph_conditioning:
            assert self_conditioning, "graph conditioning requires self-conditioning"
        self.graph_conditioning = graph_conditioning
        self.n_layers = n_layers

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
        self.embed_edge = nn.Sequential(
            nn.Linear(c_z + self_conditioning * (c_z//2), c_z),
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
                 use_anchors=use_anchors
            )
            for _ in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k

        self.torsion_angles = TorsionAngles(c_s, 1)

    def forward(self, data, self_condition=None):
        ## prep features
        data_lens = get_data_lens(data, key='x')
        ts = data['t']  # (B,)
        ts = batchwise_to_nodewise(ts, data_lens)
        x_mask = data['x_mask']
        rigids_t = ru.Rigid.from_tensor_7(data['rigids_t'])
        batch = data.batch
        device = ts.device
        num_nodes = data.num_nodes

        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, data.batch)
        rigids_t = rigids_t.translate(-center)

        # generate sequence edges
        seq_local_edge_index = []
        residx = []
        offset = 0
        for i in range(data.batch.max().item() + 1):
            select = (data.batch == i)
            subset_num_nodes = select.sum().item()
            local_residx = torch.arange(subset_num_nodes, device=device)
            residx.append(local_residx)
            seq_local_edge_index.append(
                sequence_local_graph(subset_num_nodes, data['x_mask'][select]) + offset
            )
            offset += select.sum().item()
        seq_local_edge_index = torch.cat(seq_local_edge_index, dim=-1)

        # generate spatial edges
        X_ca = rigids_t.get_trans()
        if self.graph_conditioning and self_condition is not None:
            self_cond_X_ca = self_condition['final_rigids'].get_trans()
            masked_X_ca = self_cond_X_ca.clone()
            masked_X_ca[x_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)
        else:
            masked_X_ca = X_ca.clone()
            masked_X_ca[x_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        # compute edge features
        edge_features, _ = gen_spatial_graph_features(X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)
        seq_edge_features, _ = gen_spatial_graph_features(X_ca, seq_local_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_node x h_time,)
        # embedded_time = self.time_mlp(fourier_time)  # (N_node x h_time)

        # generate node features
        residx = torch.cat(residx, dim=-1)
        res_pos = _node_positional_embeddings(
            residx,
            num_embeddings=self.c_s,
            device=device)
        node_input = torch.cat([
                res_pos,
                fourier_time,
                data['noising_mask'].float()[..., None]
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

            self_cond_X_ca = self_cond_rigids.get_trans()
            self_cond_edge_features, _ = gen_spatial_graph_features(self_cond_X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=0)
            edge_features = torch.cat(
                [edge_features, self_cond_edge_features],
                dim=-1
            )

        elif self.self_conditioning:
            node_input = F.pad(
                node_input,
                (0, self.c_s + 7)
            )
            edge_features = F.pad(
                edge_features,
                (0, self.c_z//2)
            )
        node_features = self.embed_node(node_input)
        node_features = node_features * (~x_mask)[..., None]
        edge_features = self.embed_edge(edge_features)


        ## denoising
        rigids_t = rigids_t.scale_translation(0.1)
        rigids = rigids_t

        anchor_kl = []
        node_kl = []
        for i, layer in enumerate(self.denoiser):
            node_features, rigids, edge_features, seq_edge_features, a_kl, n_kl = layer(
                node_features,
                rigids,
                edge_features,
                edge_index,
                seq_edge_features,
                seq_local_edge_index,
                data,
                data,
            )
            anchor_kl.append(a_kl)
            node_kl.append(n_kl)

        psi, _ = self.torsion_angles(node_features)

        rigids = rigids.scale_translation(10)
        rigids = rigids.translate(center)
        ret = {}
        ret['denoised_frames'] = rigids
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)[:, :5]
        ret['denoised_bb'] = denoised_bb
        ret['psi'] = psi
        ret['node_features'] = node_features
        ret['anchor_kl'] = anchor_kl
        ret['node_kl'] = node_kl

        return ret


class GraphIpmpFrameDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_pts,
                 ):
        """
        Args
        ----
        """
        super().__init__()

        self.attn_seq = IPMP(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_points=num_pts,
        )
        self.attn_spatial = IPMP(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_points=num_pts,
        )
        self.ln_s1 = nn.LayerNorm(c_s)
        self.ln_s2 = nn.LayerNorm(c_s)

        self.pool_update = LinearPoolUpdate(c_s, ratio=0.05, connect_dim=3)

        self.bb_update = BackboneUpdate(
            c_s
        )
        self.node_transition = StructureModuleTransition(
            c=c_s
        )
        self.edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )
        self.seq_edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )


    def forward(
            self,
            node_features,
            rigids,
            edge_features,
            edge_index,
            seq_edge_features,
            seq_edge_index,
            data,
            intermediates,
    ):
        noising_mask = intermediates['noising_mask']
        x_mask = data['x_mask']

        node_s_update = self.attn_spatial(
            s=node_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=(~x_mask).float()
        )
        node_s_update = node_s_update * (~x_mask)[..., None]
        node_features = self.ln_s1(node_features + node_s_update)

        node_s_update = self.attn_seq(
            s=node_features,
            z=seq_edge_features,
            edge_index=seq_edge_index,
            r=rigids,
            mask=(~x_mask).float()
        )
        node_s_update = node_s_update * (~x_mask)[..., None]
        node_features = self.ln_s2(node_features + node_s_update)

        # node_features, anchor_kl, node_kl = self.pool_update(rigids.get_trans(), node_features, edge_index, data.batch)
        anchor_kl = torch.zeros(data.num_graphs, device=x_mask.device)
        node_kl = torch.zeros(data.num_graphs, device=x_mask.device)

        node_features = self.node_transition(node_features)
        node_features = node_features * (~x_mask)[..., None]
        rigids_update = self.bb_update(
            node_features * noising_mask[..., None])

        rigids = rigids.compose_q_update_vec(
            rigids_update * noising_mask[..., None]
        )
        edge_features = self.edge_transition(node_features, edge_features, edge_index)
        seq_edge_features = self.seq_edge_transition(node_features, seq_edge_features, seq_edge_index)

        return node_features, rigids, edge_features, seq_edge_features, anchor_kl, node_kl


class GraphIpmpFrameDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_pts,
                 h_time=64,
                 n_layers=4,
                 knn_k=20,
                 lrange_k=30,
                 self_conditioning=False,
                 graph_conditioning=True):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.self_conditioning = self_conditioning
        if graph_conditioning:
            assert self_conditioning, "graph conditioning requires self-conditioning"
        self.graph_conditioning = graph_conditioning
        self.n_layers = n_layers

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
        self.embed_edge = nn.Sequential(
            nn.Linear(c_z + self_conditioning * (c_z//2), c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.LayerNorm(c_z)
        )

        self.denoiser = nn.ModuleList([
            GraphIpmpFrameDenoisingLayer(
                 c_s,
                 c_z,
                 c_hidden,
                 num_pts
            )
            for _ in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k

        self.torsion_angles = TorsionAngles(c_s, 1)

    def forward(self, data, self_condition=None):
        ## prep features
        data_lens = get_data_lens(data, key='x')
        ts = data['t']  # (B,)
        ts = batchwise_to_nodewise(ts, data_lens)
        x_mask = data['x_mask']
        rigids_t = ru.Rigid.from_tensor_7(data['rigids_t'])
        batch = data.batch
        device = ts.device
        num_nodes = data.num_nodes

        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, data.batch)
        rigids_t = rigids_t.translate(-center)

        # generate sequence edges
        seq_local_edge_index = []
        residx = []
        offset = 0
        for i in range(data.batch.max().item() + 1):
            select = (data.batch == i)
            subset_num_nodes = select.sum().item()
            local_residx = torch.arange(subset_num_nodes, device=device)
            residx.append(local_residx)
            seq_local_edge_index.append(
                sequence_local_graph(subset_num_nodes, data['x_mask'][select]) + offset
            )
            offset += select.sum().item()
        seq_local_edge_index = torch.cat(seq_local_edge_index, dim=-1)

        # generate spatial edges
        X_ca = rigids_t.get_trans()
        if self.graph_conditioning and self_condition is not None:
            self_cond_X_ca = self_condition['final_rigids'].get_trans()
            masked_X_ca = self_cond_X_ca.clone()
            masked_X_ca[x_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)
        else:
            masked_X_ca = X_ca.clone()
            masked_X_ca[x_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        # compute edge features
        edge_features, _ = gen_spatial_graph_features(X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)
        seq_edge_features, _ = gen_spatial_graph_features(X_ca, seq_local_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_node x h_time,)
        # embedded_time = self.time_mlp(fourier_time)  # (N_node x h_time)

        # generate node features
        residx = torch.cat(residx, dim=-1)
        res_pos = _node_positional_embeddings(
            residx,
            num_embeddings=self.c_s,
            device=device)
        node_input = torch.cat([
                res_pos,
                fourier_time,
                data['noising_mask'].float()[..., None]
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

            self_cond_X_ca = self_cond_rigids.get_trans()
            self_cond_edge_features, _ = gen_spatial_graph_features(self_cond_X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=0)
            edge_features = torch.cat(
                [edge_features, self_cond_edge_features],
                dim=-1
            )

        elif self.self_conditioning:
            node_input = F.pad(
                node_input,
                (0, self.c_s + 7)
            )
            edge_features = F.pad(
                edge_features,
                (0, self.c_z//2)
            )
        node_features = self.embed_node(node_input)
        node_features = node_features * (~x_mask)[..., None]
        edge_features = self.embed_edge(edge_features)


        ## denoising
        rigids_t = rigids_t.scale_translation(0.1)
        rigids = rigids_t

        anchor_kl = []
        node_kl = []
        for i, layer in enumerate(self.denoiser):
            node_features, rigids, edge_features, seq_edge_features, a_kl, n_kl = layer(
                node_features,
                rigids,
                edge_features,
                edge_index,
                seq_edge_features,
                seq_local_edge_index,
                data,
                data,
            )
            anchor_kl.append(a_kl)
            node_kl.append(n_kl)

        psi, _ = self.torsion_angles(node_features)

        rigids = rigids.scale_translation(10)
        rigids = rigids.translate(center)
        ret = {}
        ret['denoised_frames'] = rigids
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)[:, :5]
        ret['denoised_bb'] = denoised_bb
        ret['psi'] = psi
        ret['node_features'] = node_features
        ret['anchor_kl'] = anchor_kl
        ret['node_kl'] = node_kl

        return ret



class PSAEBFrameDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_v,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 gen_vectors=False
                 ):
        """
        Args
        ----
        """
        super().__init__()

        self.attn_seq = PointSetAttentionWithEdgeBias(
            c_s=c_s,
            c_v=c_v,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
            gen_vectors=gen_vectors
        )
        self.attn_spatial = PointSetAttentionWithEdgeBias(
            c_s=c_s,
            c_v=c_v,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
            gen_vectors=False
        )
        self.local_channel_update = ChannelwiseVectorGateUpdate(
            c_s,
            c_v,
            c_hidden
        )
        self.local_rot_update = LocalFrameUpdate(
            c_s,
            c_v,
            c_hidden
        )
        self.ln_s1 = nn.LayerNorm(c_s)
        self.ln_v1 = VectorLayerNorm(c_v)

        self.ln_s2 = nn.LayerNorm(c_s)
        self.ln_v2 = VectorLayerNorm(c_v)

        self.ln_s3 = nn.LayerNorm(c_s)
        self.ln_v3 = VectorLayerNorm(c_v)

        self.bb_update = BackboneUpdateVectorBias(
            c_s, c_v
        )
        self.node_transition = NodeTransition(
            c_s=c_s, c_v=c_v
        )
        self.edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )
        self.seq_edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )


    def forward(
            self,
            node_features,
            rigids,
            edge_features,
            edge_index,
            seq_edge_features,
            seq_edge_index,
            data,
            intermediates,
            node_vectors=None
    ):
        noising_mask = intermediates['noising_mask']
        x_mask = data['x_mask']

        node_s_update, node_v_update = self.attn_seq(
            node_features,
            rigids,
            seq_edge_features,
            seq_edge_index,
            node_vectors
        )
        node_features = self.ln_s1(node_features + node_s_update * (~x_mask)[..., None])
        if node_vectors is None:
            node_vectors = self.ln_v1(node_v_update * (~x_mask)[..., None, None])
            # node_vectors = node_v_update
        else:
            node_vectors = self.ln_v1(node_vectors + node_v_update * (~x_mask)[..., None, None])
            # node_vectors = node_v_update

        node_s_update, node_v_update = self.attn_spatial(
            node_features,
            rigids,
            edge_features,
            edge_index,
            node_vectors
        )
        node_features = self.ln_s2(node_features + node_s_update * (~x_mask)[..., None])
        node_vectors = self.ln_v2(node_vectors + node_v_update * (~x_mask)[..., None, None])

        node_s_update, node_v_update = self.local_channel_update(
            node_features,
            node_vectors,
            rigids
        )
        node_features = node_features + node_s_update * (~x_mask)[..., None]
        node_vectors = node_vectors + node_v_update * (~x_mask)[..., None, None]

        node_s_update, node_v_update = self.local_rot_update(
            node_features,
            node_vectors,
            rigids
        )
        node_features = self.ln_s3(node_features + node_s_update * (~x_mask)[..., None])
        # node_features = node_s_update
        node_vectors = self.ln_v3(node_vectors + node_v_update * (~x_mask)[..., None, None])
        # node_vectors = node_v_update

        node_features, node_vectors = self.node_transition(node_features, node_vectors)
        node_features = node_features  * (~x_mask)[..., None]
        node_vectors = node_vectors  * (~x_mask)[..., None, None]
        rigids_update = self.bb_update(
            node_features * noising_mask[..., None],
            node_vectors * noising_mask[..., None, None],
            rigids)

        rigids = rigids.compose_q_update_vec(
            rigids_update * noising_mask[..., None]
        )
        edge_features = self.edge_transition(node_features, edge_features, edge_index)
        seq_edge_features = self.seq_edge_transition(node_features, seq_edge_features, seq_edge_index)

        return node_features, rigids, edge_features, seq_edge_features, node_vectors


class PSAEBFrameDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 c_s,
                 c_v,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 h_time=64,
                 scalar_h_dim=128,
                 n_layers=4,
                 knn_k=20,
                 lrange_k=30,
                 self_conditioning=False,
                 graph_conditioning=False):
        super().__init__()

        self.c_s = c_s
        self.c_v = c_v
        self.c_z = c_z
        self.self_conditioning = self_conditioning
        self.graph_conditioning = graph_conditioning

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
        self.embed_edge = nn.Sequential(
            nn.Linear(c_z + self_conditioning * (c_z//2), c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.LayerNorm(c_z)
        )


        self.denoiser = nn.ModuleList([
            PSAEBFrameDenoisingLayer(
                 c_s,
                 c_v,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 gen_vectors=(i == 0)
            )
            for i in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k

        self.torsion_angles = TorsionAngles(c_s, 1)

    def forward(self, data, self_condition=None):
        ## prep features
        data_lens = get_data_lens(data, key='x')
        ts = data['t']  # (B,)
        ts = batchwise_to_nodewise(ts, data_lens)
        x_mask = data['x_mask']
        rigids_t = ru.Rigid.from_tensor_7(data['rigids_t'])
        batch = data.batch
        device = ts.device
        num_nodes = data.num_nodes

        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, data.batch)
        rigids_t = rigids_t.translate(-center)

        # generate sequence edges
        seq_local_edge_index = []
        residx = []
        offset = 0
        for i in range(data.batch.max().item() + 1):
            select = (data.batch == i)
            subset_num_nodes = select.sum().item()
            local_residx = torch.arange(subset_num_nodes, device=device)
            residx.append(local_residx)
            seq_local_edge_index.append(
                sequence_local_graph(subset_num_nodes, data['x_mask'][select]) + offset
            )
            offset += select.sum().item()
        seq_local_edge_index = torch.cat(seq_local_edge_index, dim=-1)

        # generate spatial edges
        X_ca = rigids_t.get_trans()
        if self.graph_conditioning and self_condition is not None:
            self_cond_X_ca = self_condition['final_rigids'].get_trans()
            masked_X_ca = self_cond_X_ca.clone()
            masked_X_ca[x_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)
        else:
            masked_X_ca = X_ca.clone()
            masked_X_ca[x_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        # compute edge features
        edge_features, _ = gen_spatial_graph_features(X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)
        seq_edge_features, _ = gen_spatial_graph_features(X_ca, seq_local_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_node x h_time,)
        # embedded_time = self.time_mlp(fourier_time)  # (N_node x h_time)

        # generate node features
        residx = torch.cat(residx, dim=-1)
        res_pos = _node_positional_embeddings(
            residx,
            num_embeddings=self.c_s,
            device=device)
        node_input = torch.cat([
                res_pos,
                fourier_time,
                data['noising_mask'].float()[..., None]
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

            self_cond_X_ca = self_cond_rigids.get_trans()
            self_cond_edge_features, _ = gen_spatial_graph_features(self_cond_X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=0)
            edge_features = torch.cat(
                [edge_features, self_cond_edge_features],
                dim=-1
            )

        elif self.self_conditioning:
            node_input = F.pad(
                node_input,
                (0, self.c_s + 7)
            )
            edge_features = F.pad(
                edge_features,
                (0, self.c_z//2)
            )
        node_features = self.embed_node(node_input)
        node_features = node_features * (~x_mask)[..., None]
        edge_features = self.embed_edge(edge_features)


        ## denoising
        rigids_t = rigids_t.scale_translation(0.1)
        rigids = rigids_t

        node_vectors = None
        for i, layer in enumerate(self.denoiser):
            node_features, rigids, edge_features, seq_edge_features, node_vectors = layer(
                node_features,
                rigids,
                edge_features,
                edge_index,
                seq_edge_features,
                seq_local_edge_index,
                data,
                data,
                node_vectors=node_vectors
            )

        psi, _ = self.torsion_angles(node_features)

        rigids = rigids.scale_translation(10)
        rigids = rigids.translate(center)
        ret = {}
        ret['denoised_frames'] = rigids
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)[:, :5]
        ret['denoised_bb'] = denoised_bb
        ret['node_features'] = node_features

        return ret


from ligbinddiff.model.modules.equiformer_v2.transformer_block import TransBlockV2
from ligbinddiff.model.modules.equiformer_v2.so3 import SO3_Embedding, SO3_Rotation, SO3_Grid, CoefficientMappingModule
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat


class HybridSO3FrameDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 ):
        """
        Args
        ----
        """
        super().__init__()

        self.attn_seq = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )
        self.attn_spatial = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )

        self.ln_s1 = nn.LayerNorm(c_s)
        self.ln_s2 = nn.LayerNorm(c_s)
        self.ln_s3 = nn.LayerNorm(c_s)

        h_equi = c_s // 4

        node_lmax_list = [1]
        self.node_SO3_rotation_list = nn.ModuleList()
        for lmax in node_lmax_list:
            self.node_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        self.node_SO3_grid_list = nn.ModuleList()
        for l in range(max(node_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            self.node_SO3_grid_list.append(SO3_m_grid)
        self.mappingReduced = CoefficientMappingModule(node_lmax_list, node_lmax_list)
        self.trans_compress = nn.Linear(c_s + h_equi, h_equi)
        self.so3_trans = TransBlockV2(
            sphere_channels=h_equi,
            attn_hidden_channels=h_equi,
            num_heads=num_heads,
            attn_alpha_channels=num_qk_pts,
            attn_value_channels=num_v_pts,
            ffn_hidden_channels=h_equi*2,
            output_channels=h_equi,
            lmax_list=node_lmax_list,
            mmax_list=node_lmax_list,
            SO3_rotation=self.node_SO3_rotation_list,
            SO3_grid=self.node_SO3_grid_list,
            mappingReduced=self.mappingReduced,
            edge_channels_list=[c_z, c_z, c_z]
        )
        self.trans_expand = nn.Linear(h_equi, c_s)

        self.bb_update = BackboneUpdate(
            c_s
        )
        self.node_transition = StructureModuleTransition(
            c_s
        )
        self.edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )
        self.seq_edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )


    def forward(
            self,
            node_features,
            rigids,
            edge_features,
            edge_index,
            seq_edge_features,
            seq_edge_index,
            data,
            intermediates,
            node_so3_embed
    ):
        noising_mask = intermediates['noising_mask']
        x_mask = data['x_mask']

        node_s_update = self.attn_spatial(
            s=node_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=(~x_mask).float()
        )
        node_s_update = node_s_update * (~x_mask)[..., None]
        node_features = self.ln_s1(node_features + node_s_update)

        node_s_update = self.attn_seq(
            s=node_features,
            z=seq_edge_features,
            edge_index=seq_edge_index,
            r=rigids,
            mask=(~x_mask).float()
        )
        node_features = self.ln_s2(node_features + node_s_update * (~x_mask)[..., None])

        # set up cache stores
        X_ca = rigids.get_trans()
        edge_distance_vec = X_ca[edge_index[1]] - X_ca[edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for rot in self.node_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)


        so3_scalars = node_so3_embed.get_invariant_features(flat=True)
        embed_scalars = self.trans_compress(
            torch.cat([so3_scalars, node_features], dim=-1)
        )
        node_so3_embed.set_invariant_features(embed_scalars)
        node_so3_embed = self.so3_trans(node_so3_embed, edge_features, edge_index)
        so3_scalars = node_so3_embed.get_invariant_features(flat=True)
        node_s_update = self.trans_expand(so3_scalars)
        node_features = self.ln_s3(node_features + node_s_update * (~x_mask)[..., None])


        node_features = self.node_transition(node_features)
        node_features = node_features  * (~x_mask)[..., None]
        rigids_update = self.bb_update(
            node_features * noising_mask[..., None])

        rigids = rigids.compose_q_update_vec(
            rigids_update * noising_mask[..., None]
        )
        edge_features = self.edge_transition(node_features, edge_features, edge_index)
        seq_edge_features = self.seq_edge_transition(node_features, seq_edge_features, seq_edge_index)

        return node_features, rigids, edge_features, seq_edge_features, node_so3_embed


class HybridSO3FrameDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 h_time=64,
                 n_layers=4,
                 knn_k=20,
                 lrange_k=30,
                 self_conditioning=False,
                 graph_conditioning=True):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.self_conditioning = self_conditioning
        if graph_conditioning:
            assert self_conditioning, "graph conditioning requires self-conditioning"
        self.graph_conditioning = graph_conditioning
        self.n_layers = n_layers

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
        self.embed_edge = nn.Sequential(
            nn.Linear(c_z + self_conditioning * (c_z//2), c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.LayerNorm(c_z)
        )

        self.denoiser = nn.ModuleList([
            HybridSO3FrameDenoisingLayer(
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts
            )
            for _ in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k

        self.torsion_angles = TorsionAngles(c_s, 1)

    def forward(self, data, self_condition=None):
        ## prep features
        data_lens = get_data_lens(data, key='x')
        ts = data['t']  # (B,)
        ts = batchwise_to_nodewise(ts, data_lens)
        x_mask = data['x_mask']
        rigids_t = ru.Rigid.from_tensor_7(data['rigids_t'])
        batch = data.batch
        device = ts.device
        num_nodes = data.num_nodes

        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, data.batch)
        rigids_t = rigids_t.translate(-center)

        # generate sequence edges
        seq_local_edge_index = []
        residx = []
        offset = 0
        for i in range(data.batch.max().item() + 1):
            select = (data.batch == i)
            subset_num_nodes = select.sum().item()
            local_residx = torch.arange(subset_num_nodes, device=device)
            residx.append(local_residx)
            seq_local_edge_index.append(
                sequence_local_graph(subset_num_nodes, data['x_mask'][select]) + offset
            )
            offset += select.sum().item()
        seq_local_edge_index = torch.cat(seq_local_edge_index, dim=-1)

        # generate spatial edges
        X_ca = rigids_t.get_trans()
        if self.graph_conditioning and self_condition is not None:
            self_cond_X_ca = self_condition['final_rigids'].get_trans()
            masked_X_ca = self_cond_X_ca.clone()
            masked_X_ca[x_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)
        else:
            masked_X_ca = X_ca.clone()
            masked_X_ca[x_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        # compute edge features
        edge_features, _ = gen_spatial_graph_features(X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)
        seq_edge_features, _ = gen_spatial_graph_features(X_ca, seq_local_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_node x h_time,)
        # embedded_time = self.time_mlp(fourier_time)  # (N_node x h_time)

        # generate node features
        residx = torch.cat(residx, dim=-1)
        res_pos = _node_positional_embeddings(
            residx,
            num_embeddings=self.c_s,
            device=device)
        node_input = torch.cat([
                res_pos,
                fourier_time,
                data['noising_mask'].float()[..., None]
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

            self_cond_X_ca = self_cond_rigids.get_trans()
            self_cond_edge_features, _ = gen_spatial_graph_features(self_cond_X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=0)
            edge_features = torch.cat(
                [edge_features, self_cond_edge_features],
                dim=-1
            )

        elif self.self_conditioning:
            node_input = F.pad(
                node_input,
                (0, self.c_s + 7)
            )
            edge_features = F.pad(
                edge_features,
                (0, self.c_z//2)
            )
        node_features = self.embed_node(node_input)
        node_features = node_features * (~x_mask)[..., None]
        edge_features = self.embed_edge(edge_features)


        ## denoising
        rigids_t = rigids_t.scale_translation(0.1)
        rigids = rigids_t

        node_so3_embed = SO3_Embedding(
            num_nodes,
            lmax_list=[1],
            num_channels=self.c_s//4,
            device=node_features.device,
            dtype=node_features.dtype
        )
        for i, layer in enumerate(self.denoiser):
            node_features, rigids, edge_features, seq_edge_features, node_so3_embed = layer(
                node_features,
                rigids,
                edge_features,
                edge_index,
                seq_edge_features,
                seq_local_edge_index,
                data,
                data,
                node_so3_embed=node_so3_embed
            )

        psi, _ = self.torsion_angles(node_features)

        rigids = rigids.scale_translation(10)
        rigids = rigids.translate(center)
        ret = {}
        ret['denoised_frames'] = rigids
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)[:, :5]
        ret['denoised_bb'] = denoised_bb
        ret['psi'] = psi
        ret['node_features'] = node_features

        return ret


class SO3FrameDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 ):
        """
        Args
        ----
        """
        super().__init__()

        node_lmax_list = [1]
        self.node_SO3_rotation_list = nn.ModuleList()
        for lmax in node_lmax_list:
            self.node_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        self.node_SO3_grid_list = nn.ModuleList()
        for l in range(max(node_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            self.node_SO3_grid_list.append(SO3_m_grid)
        self.mappingReduced = CoefficientMappingModule(node_lmax_list, node_lmax_list)
        h_equi = c_s // 4

        self.trans_compress = nn.Linear(c_s + h_equi, h_equi)
        self.attn_spatial = TransBlockV2(
            sphere_channels=h_equi,
            attn_hidden_channels=h_equi,
            num_heads=num_heads,
            attn_alpha_channels=num_qk_pts,
            attn_value_channels=num_v_pts,
            ffn_hidden_channels=h_equi*2,
            output_channels=h_equi,
            lmax_list=node_lmax_list,
            mmax_list=node_lmax_list,
            SO3_rotation=self.node_SO3_rotation_list,
            SO3_grid=self.node_SO3_grid_list,
            mappingReduced=self.mappingReduced,
            edge_channels_list=[c_z, c_z, c_z]
        )
        self.attn_seq = TransBlockV2(
            sphere_channels=h_equi,
            attn_hidden_channels=h_equi,
            num_heads=num_heads,
            attn_alpha_channels=num_qk_pts,
            attn_value_channels=num_v_pts,
            ffn_hidden_channels=h_equi*2,
            output_channels=h_equi,
            lmax_list=node_lmax_list,
            mmax_list=node_lmax_list,
            SO3_rotation=self.node_SO3_rotation_list,
            SO3_grid=self.node_SO3_grid_list,
            mappingReduced=self.mappingReduced,
            edge_channels_list=[c_z, c_z, c_z]
        )
        self.trans_expand = nn.Linear(h_equi, c_s)

        self.ln_s1 = nn.LayerNorm(c_s)

        self.bb_update = BackboneUpdateVectorBias(
            c_s, h_equi
        )
        self.node_transition = NodeTransition(
            c_s, h_equi
        )
        self.edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )
        self.seq_edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )


    def forward(
            self,
            node_features,
            rigids,
            edge_features,
            edge_index,
            seq_edge_features,
            seq_edge_index,
            data,
            intermediates,
            node_so3_embed
    ):
        noising_mask = intermediates['noising_mask']
        x_mask = data['x_mask']

        so3_scalars = node_so3_embed.get_invariant_features(flat=True)
        embed_scalars = self.trans_compress(
            torch.cat([so3_scalars, node_features], dim=-1)
        )
        node_so3_embed.set_invariant_features(embed_scalars)

        # set up cache stores
        X_ca = rigids.get_trans()
        edge_distance_vec = X_ca[seq_edge_index[1]] - X_ca[seq_edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for rot in self.node_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        node_embed_update = self.attn_seq(
            node_so3_embed,
            seq_edge_features,
            seq_edge_index,
        )
        node_so3_embed = node_embed_update

        edge_distance_vec = X_ca[edge_index[1]] - X_ca[edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for rot in self.node_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)

        node_embed_update = self.attn_spatial(
            node_so3_embed,
            edge_features,
            edge_index,
        )
        node_so3_embed = node_embed_update

        so3_scalars = node_so3_embed.get_invariant_features(flat=True)
        node_s_update = self.trans_expand(so3_scalars)
        node_features = self.ln_s1(node_features + node_s_update * (~x_mask)[..., None])

        node_vectors = node_embed_update.embedding[..., 1:4, :].transpose(-1, -2)

        node_features, node_vectors = self.node_transition(node_features, node_vectors)
        node_features = node_features  * (~x_mask)[..., None]
        node_vectors =  node_vectors * (~x_mask)[..., None, None]
        rigids_update = self.bb_update(
            node_features * noising_mask[..., None],
            node_vectors * noising_mask[..., None, None],
            rigids)

        rigids = rigids.compose_q_update_vec(
            rigids_update * noising_mask[..., None]
        )
        edge_features = self.edge_transition(node_features, edge_features, edge_index)
        seq_edge_features = self.seq_edge_transition(node_features, seq_edge_features, seq_edge_index)

        return node_features, rigids, edge_features, seq_edge_features, node_so3_embed


class SO3FrameDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 h_time=64,
                 n_layers=4,
                 knn_k=20,
                 lrange_k=30,
                 self_conditioning=False,
                 graph_conditioning=True):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.self_conditioning = self_conditioning
        if graph_conditioning:
            assert self_conditioning, "graph conditioning requires self-conditioning"
        self.graph_conditioning = graph_conditioning
        self.n_layers = n_layers

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
        self.embed_edge = nn.Sequential(
            nn.Linear(c_z + self_conditioning * (c_z//2), c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.LayerNorm(c_z)
        )

        self.denoiser = nn.ModuleList([
            SO3FrameDenoisingLayer(
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts
            )
            for _ in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k

        self.torsion_angles = TorsionAngles(c_s, 1)

    def forward(self, data, self_condition=None):
        ## prep features
        data_lens = get_data_lens(data, key='x')
        ts = data['t']  # (B,)
        ts = batchwise_to_nodewise(ts, data_lens)
        x_mask = data['x_mask']
        rigids_t = ru.Rigid.from_tensor_7(data['rigids_t'])
        batch = data.batch
        device = ts.device
        num_nodes = data.num_nodes

        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, data.batch)
        rigids_t = rigids_t.translate(-center)

        # generate sequence edges
        seq_local_edge_index = []
        residx = []
        offset = 0
        for i in range(data.batch.max().item() + 1):
            select = (data.batch == i)
            subset_num_nodes = select.sum().item()
            local_residx = torch.arange(subset_num_nodes, device=device)
            residx.append(local_residx)
            seq_local_edge_index.append(
                sequence_local_graph(subset_num_nodes, data['x_mask'][select]) + offset
            )
            offset += select.sum().item()
        seq_local_edge_index = torch.cat(seq_local_edge_index, dim=-1)

        # generate spatial edges
        X_ca = rigids_t.get_trans()
        if self.graph_conditioning and self_condition is not None:
            self_cond_X_ca = self_condition['final_rigids'].get_trans()
            masked_X_ca = self_cond_X_ca.clone()
            masked_X_ca[x_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)
        else:
            masked_X_ca = X_ca.clone()
            masked_X_ca[x_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        # compute edge features
        edge_features, _ = gen_spatial_graph_features(X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)
        seq_edge_features, _ = gen_spatial_graph_features(X_ca, seq_local_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_node x h_time,)
        # embedded_time = self.time_mlp(fourier_time)  # (N_node x h_time)

        # generate node features
        residx = torch.cat(residx, dim=-1)
        res_pos = _node_positional_embeddings(
            residx,
            num_embeddings=self.c_s,
            device=device)
        node_input = torch.cat([
                res_pos,
                fourier_time,
                data['noising_mask'].float()[..., None]
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

            self_cond_X_ca = self_cond_rigids.get_trans()
            self_cond_edge_features, _ = gen_spatial_graph_features(self_cond_X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=0)
            edge_features = torch.cat(
                [edge_features, self_cond_edge_features],
                dim=-1
            )

        elif self.self_conditioning:
            node_input = F.pad(
                node_input,
                (0, self.c_s + 7)
            )
            edge_features = F.pad(
                edge_features,
                (0, self.c_z//2)
            )
        node_features = self.embed_node(node_input)
        node_features = node_features * (~x_mask)[..., None]
        edge_features = self.embed_edge(edge_features)


        ## denoising
        rigids_t = rigids_t.scale_translation(0.1)
        rigids = rigids_t

        node_so3_embed = SO3_Embedding(
            num_nodes,
            lmax_list=[1],
            num_channels=self.c_s//4,
            device=node_features.device,
            dtype=node_features.dtype
        )
        for i, layer in enumerate(self.denoiser):
            node_features, rigids, edge_features, seq_edge_features, node_so3_embed = layer(
                node_features,
                rigids,
                edge_features,
                edge_index,
                seq_edge_features,
                seq_local_edge_index,
                data,
                data,
                node_so3_embed=node_so3_embed
            )

        psi, _ = self.torsion_angles(node_features)

        rigids = rigids.scale_translation(10)
        rigids = rigids.translate(center)
        ret = {}
        ret['denoised_frames'] = rigids
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)[:, :5]
        ret['denoised_bb'] = denoised_bb
        ret['psi'] = psi
        ret['node_features'] = node_features

        return ret
