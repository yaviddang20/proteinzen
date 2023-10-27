""" Denoising model """

import torch
from torch import nn
from ligbinddiff.model.modules.common import RBF
from ligbinddiff.model.modules.layers.edge.sitewise import EdgeTransition
from ligbinddiff.model.modules.layers.node.attention import GraphInvariantPointAttention, PointSetAttentionWithEdgeBias
from ligbinddiff.model.modules.layers.node.sitewise import BackboneUpdateVectorBias, ChannelwiseVectorGateUpdate, LocalFrameUpdate, NodeTransition, VectorLayerNorm

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.framediff.all_atom import compute_backbone

from ligbinddiff.data.datasets.featurize.common import _node_positional_embeddings

from ligbinddiff.model.modules.openfold.frames import BackboneUpdate, StructureModuleTransition

from ligbinddiff.model.utils.graph import sample_inv_cubic_edges, sequence_local_graph, gen_spatial_graph_features, batchwise_to_nodewise, get_data_lens

from .framediff import TorsionAngles

class GraphIpaFrameDenoisingLayer(nn.Module):
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


        node_features = self.node_transition(node_features)
        node_features = node_features * (~x_mask)[..., None]
        rigids_update = self.bb_update(
            node_features * noising_mask[..., None])

        rigids = rigids.compose_q_update_vec(
            rigids_update * noising_mask[..., None]
        )
        edge_features = self.edge_transition(node_features, edge_features, edge_index)
        seq_edge_features = self.seq_edge_transition(node_features, seq_edge_features, seq_edge_index)

        return node_features, rigids, edge_features, seq_edge_features

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
                 scalar_h_dim=128,
                 n_layers=4,
                 knn_k=20,
                 lrange_k=30):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z

        self.h_time = h_time
        self.time_rbf = RBF(n_basis=h_time//2)
        self.time_mlp = nn.Sequential(
            nn.Linear(h_time, scalar_h_dim),
            nn.ReLU(),
            nn.Linear(scalar_h_dim, h_time),
            nn.ReLU()
        )

        self.embed_node = nn.Linear(
            c_s + h_time + 1, c_s
        )

        self.denoiser = nn.ModuleList([
            GraphIpaFrameDenoisingLayer(
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
            )
            for i in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k

        self.torsion_angles = TorsionAngles(c_s, 1)

    def forward(self, data):
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
        masked_X_ca = X_ca.clone()
        masked_X_ca[x_mask] = torch.inf
        edge_index = sample_inv_cubic_edges(masked_X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        # compute edge features
        edge_features, _ = gen_spatial_graph_features(X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)
        seq_edge_features, _ = gen_spatial_graph_features(X_ca, seq_local_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_node x h_time,)
        embedded_time = self.time_mlp(fourier_time)  # (N_node x h_time)

        # generate node features
        residx = torch.cat(residx, dim=-1)
        res_pos = _node_positional_embeddings(
            residx,
            num_embeddings=self.c_s,
            device=device)
        node_features = self.embed_node(
            torch.cat([
                res_pos,
                embedded_time,
                data['noising_mask'].float()[..., None]
            ], dim=-1)
        )
        node_features = node_features * (~x_mask)[..., None]

        ## denoising
        rigids_t = rigids_t.scale_translation(0.1)
        rigids = rigids_t

        for i, layer in enumerate(self.denoiser):
            node_features, rigids, edge_features, seq_edge_features = layer(
                node_features,
                rigids,
                edge_features,
                edge_index,
                seq_edge_features,
                seq_local_edge_index,
                data,
                data,
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
                 lrange_k=30):
        super().__init__()

        self.c_s = c_s
        self.c_v = c_v
        self.c_z = c_z

        self.h_time = h_time
        self.time_rbf = RBF(n_basis=h_time//2)
        self.time_mlp = nn.Sequential(
            nn.Linear(h_time, scalar_h_dim),
            nn.ReLU(),
            nn.Linear(scalar_h_dim, h_time),
            nn.ReLU()
        )

        self.embed_node = nn.Linear(
            c_s + h_time + 1, c_s
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

    def forward(self, data):
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
        rigids_t = rigids_t.scale_translation(0.1)

        # generate sequence edges
        seq_local_edge_index = []
        offset = 0
        for i in range(data.batch.max().item() + 1):
            select = (data.batch == i)
            subset_num_nodes = select.sum().item()
            seq_local_edge_index.append(
                sequence_local_graph(subset_num_nodes, data['x_mask'][select]) + offset
            )
            offset += select.sum().item()
        seq_local_edge_index = torch.cat(seq_local_edge_index, dim=-1)

        # generate spatial edges
        X_ca = rigids_t.get_trans()
        masked_X_ca = X_ca.clone()
        masked_X_ca[x_mask] = torch.inf
        edge_index = sample_inv_cubic_edges(masked_X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        # compute edge features
        edge_features, _ = gen_spatial_graph_features(X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)
        seq_edge_features, _ = gen_spatial_graph_features(X_ca, seq_local_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_node x h_time,)
        embedded_time = self.time_mlp(fourier_time)  # (N_node x h_time)

        # generate node features
        residx = torch.arange(num_nodes, device=device)
        res_pos = _node_positional_embeddings(
            residx,
            num_embeddings=self.c_s,
            device=device)
        node_features = self.embed_node(
            torch.cat([
                res_pos,
                embedded_time,
                data['noising_mask'].float()[..., None]
            ], dim=-1)
        )

        ## denoising
        node_vectors = None
        rigids = rigids_t

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
