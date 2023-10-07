""" Denoising model """

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_cluster import knn_graph
from ligbinddiff.data.datasets.featurize.common import _rbf

from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2
from ligbinddiff.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from ligbinddiff.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork, MultiResFeedForwardNetwork, TransBlockV2
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from ligbinddiff.model.modules.common import EdgeUpdate
from ligbinddiff.model.utils.graph import sample_inv_cubic_edges

from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings


class GraphUpdate(nn.Module):
    def __init__(self,
                 bb_lmax_list,
                 edge_channels_list,
                 mappingReduced_bb,
                 bb_SO3_rotation,
                 bb_SO3_grid,
                 num_heads=8,
                 h_channels=32,
                 bb_channels=32,
                 n_bb_atoms=4,
                 ):
        super().__init__()

        self.n_bb_atoms = n_bb_atoms-1
        self.bb_lmax_list = bb_lmax_list
        self.h_channels = h_channels
        self.bb_channels = bb_channels
        self.bb_SO3_rotation = bb_SO3_rotation

        self.attention = TransBlockV2(
            sphere_channels=bb_channels + self.n_bb_atoms,
            attn_hidden_channels=h_channels,
            num_heads=num_heads,
            attn_alpha_channels=h_channels // 2,
            attn_value_channels=h_channels // 4,
            ffn_hidden_channels=h_channels,
            output_channels=bb_channels,
            lmax_list=bb_lmax_list,
            mmax_list=bb_lmax_list,
            SO3_rotation=bb_SO3_rotation,
            SO3_grid=bb_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_bb
        )

        self.edge_update = EdgeUpdate(
            node_lmax_list=bb_lmax_list,
            edge_channels_list=edge_channels_list,
            h_channels=bb_channels
        )


    def forward(self,
                bb_rel: torch.Tensor,
                bb_features: SO3_Embedding,
                edge_features: torch.Tensor,
                edge_index: torch.Tensor,
                noising_mask: torch.Tensor):
        # compute which nodes are being noised and which aren't
        num_nodes = bb_features.length

        # update backbone
        node_features = SO3_Embedding(
            num_nodes,
            self.bb_lmax_list,
            num_channels=self.bb_channels + self.n_bb_atoms,
            device=bb_features.device,
            dtype=bb_features.dtype
        )
        node_features.embedding[..., :self.bb_channels] = bb_features.embedding
        node_features.embedding[..., 1:4, -self.n_bb_atoms:] = bb_rel.transpose(-1, -2)
        node_features.embedding[..., 0:1, -1:] = noising_mask[:, None, None].float() # is node editable or not

        new_bb_features = self.attention(
            node_features,
            edge_features,
            edge_index
        )

        # update edges
        new_edge_features = self.edge_update(
            new_bb_features,
            edge_features,
            edge_index
        )

        return new_bb_features, new_edge_features


class BackboneUpdate(nn.Module):
    def __init__(self,
                 bb_lmax_list,
                 edge_channels_list,
                 mappingReduced_bb,
                 bb_SO3_rotation,
                 bb_SO3_grid,
                 num_heads=8,
                 h_channels=32,
                 bb_channels=32,
                 n_bb_atoms=4,
                 knn_k=20,
                 lrange_k=40):
        super().__init__()

        self.n_bb_atoms = n_bb_atoms-1
        self.bb_lmax_list = bb_lmax_list
        self.h_channels = h_channels
        self.bb_channels = bb_channels
        self.bb_SO3_rotation = bb_SO3_rotation

        self.lrange_attention = TransBlockV2(
            sphere_channels=bb_channels + self.n_bb_atoms,
            attn_hidden_channels=h_channels,
            num_heads=num_heads,
            attn_alpha_channels=h_channels // 2,
            attn_value_channels=h_channels // 4,
            ffn_hidden_channels=h_channels,
            output_channels=bb_channels,
            lmax_list=bb_lmax_list,
            mmax_list=bb_lmax_list,
            SO3_rotation=bb_SO3_rotation,
            SO3_grid=bb_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_bb
        )

        self.local_attention = TransBlockV2(
            sphere_channels=bb_channels + self.n_bb_atoms,
            attn_hidden_channels=h_channels,
            num_heads=num_heads,
            attn_alpha_channels=h_channels // 2,
            attn_value_channels=h_channels // 4,
            ffn_hidden_channels=h_channels,
            output_channels=bb_channels,
            lmax_list=bb_lmax_list,
            mmax_list=bb_lmax_list,
            SO3_rotation=bb_SO3_rotation,
            SO3_grid=bb_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_bb
        )

        self.update_x_ca = SO3_LinearV2(
            in_features=bb_channels,
            out_features=1,
            lmax=max(bb_lmax_list)
        )
        self.update_x_ca_gate = nn.Linear(
            in_features=bb_channels,
            out_features=1
        )

        self.update_bb = SO3_LinearV2(
            in_features=bb_channels,
            out_features=self.n_bb_atoms,
            lmax=max(bb_lmax_list)
        )

        self.knn_k = knn_k
        self.lrange_k = lrange_k


    def forward(self,
                X_ca: torch.Tensor,
                bb_rel: torch.Tensor,
                bb_features: SO3_Embedding,
                batch: torch.Tensor,
                x_mask: torch.Tensor,
                noising_mask: torch.Tensor):
        # compute which nodes are being noised and which aren't
        num_nodes = bb_features.length

        # compute graph with knn + inv cubic edges
        lrange_edge_index = sample_inv_cubic_edges(X_ca, x_mask, batch, self.knn_k, self.lrange_k)
        masked_X_ca = X_ca.clone()
        masked_X_ca[x_mask] = torch.inf
        knn_edge_index = knn_graph(masked_X_ca, self.knn_k, batch)
        edge_index = torch.cat([knn_edge_index, lrange_edge_index], dim=-1)
        edge_dist_vec = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)

        # gen edge features
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device)  # edge_channels_list
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)
        edge_rot_mat = init_edge_rot_mat(edge_dist_vec)

        # update rotation matrices
        knn_edge_rot_mat = edge_rot_mat[:knn_edge_index.shape[1]]
        for rot in self.bb_SO3_rotation:
            rot.set_wigner(knn_edge_rot_mat)

        # update using local information
        node_features = SO3_Embedding(
            num_nodes,
            self.bb_lmax_list,
            num_channels=self.bb_channels + self.n_bb_atoms,
            device=bb_features.device,
            dtype=bb_features.dtype
        )
        node_features.embedding[..., :self.bb_channels] = bb_features.embedding
        node_features.embedding[..., 1:4, -self.n_bb_atoms:] = bb_rel.transpose(-1, -2)
        node_features.embedding[..., 0:1, -1:] = noising_mask[:, None, None].float() # is node editable or not

        knn_edge_features = edge_features[:knn_edge_index.shape[1]]
        updated_bb_features = self.local_attention(
            node_features,
            knn_edge_features,
            knn_edge_index
        )

        # update rotation matrices
        for rot in self.bb_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        # update using lrange information
        node_features = SO3_Embedding(
            num_nodes,
            self.bb_lmax_list,
            num_channels=self.bb_channels + self.n_bb_atoms,
            device=bb_features.device,
            dtype=bb_features.dtype
        )
        node_features.embedding[..., :self.bb_channels] = bb_features.embedding
        node_features.embedding[..., 1:4, -self.n_bb_atoms:] = bb_rel.transpose(-1, -2)
        node_features.embedding[..., 0:1, -1:] = noising_mask[:, None, None].float() # is node editable or not

        updated_bb_features = self.lrange_attention(
            node_features,
            edge_features,
            edge_index
        )

        update_X_ca = self.update_x_ca(updated_bb_features)
        update_X_ca_gate = self.update_x_ca_gate(updated_bb_features.get_invariant_features(flat=True))
        update_bb = self.update_bb(updated_bb_features)

        subset_update_X_ca = update_X_ca.embedding[:, 1:4].squeeze(-1)[noising_mask]
        subset_update_gate = F.softplus(update_X_ca_gate)[noising_mask]
        subset_update_X_ca = subset_update_X_ca * subset_update_gate

        new_X_ca = X_ca.clone()
        new_X_ca[noising_mask] = X_ca[noising_mask] + subset_update_X_ca
        new_bb_rel = bb_rel.clone()
        new_bb_rel[noising_mask] = bb_rel[noising_mask] + update_bb.embedding[:, 1:4].transpose(-1, -2)[noising_mask]

        return new_X_ca, new_bb_rel, updated_bb_features.clone()

class BackboneUpdateFromGraph(nn.Module):
    def __init__(self,
                 bb_lmax_list,
                 edge_channels_list,
                 mappingReduced_bb,
                 bb_SO3_rotation,
                 bb_SO3_grid,
                 num_heads=8,
                 h_channels=32,
                 bb_channels=32,
                 n_bb_atoms=4,
                 knn_k=30,
                 lrange_k=10):
        super().__init__()

        self.n_bb_atoms = n_bb_atoms-1
        self.bb_lmax_list = bb_lmax_list
        self.h_channels = h_channels
        self.bb_channels = bb_channels
        self.bb_SO3_rotation = bb_SO3_rotation

        self.update_x_ca = SO3_LinearV2(
            in_features=bb_channels,
            out_features=1,
            lmax=max(bb_lmax_list)
        )

        self.update_bb = SO3_LinearV2(
            in_features=bb_channels,
            out_features=self.n_bb_atoms,
            lmax=max(bb_lmax_list)
        )

        self.knn_k = knn_k
        self.lrange_k = lrange_k


    def forward(self,
                X_ca: torch.Tensor,
                bb_rel: torch.Tensor,
                bb_features: SO3_Embedding,
                noising_mask: torch.Tensor):
        update_X_ca = self.update_x_ca(bb_features)
        update_bb = self.update_bb(bb_features)

        subset_update_X_ca = update_X_ca.embedding[:, 1:4].squeeze(-1)[noising_mask]

        new_X_ca = X_ca.clone()
        new_X_ca[noising_mask] = X_ca[noising_mask] + subset_update_X_ca
        new_bb_rel = bb_rel.clone()
        new_bb_rel[noising_mask] = bb_rel[noising_mask] + update_bb.embedding[:, 1:4].transpose(-1, -2)[noising_mask]

        return new_X_ca, new_bb_rel


# adapted from https://github.com/jmclong/random-fourier-features-pytorch/blob/main/rff/layers.py
class RBF(nn.Module):
    """ Damped random Fourier Feature encoding layer """
    def __init__(self, n_basis=64):
        super().__init__()
        kappa = torch.randn((n_basis,))
        self.register_buffer('kappa', kappa)

    def forward(self, ts):
        tp = 2 * np.pi * ts * self.kappa
        return torch.cat([torch.cos(tp), torch.sin(tp)], dim=-1)


class BackboneR3Denoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 bb_lmax_list,
                 edge_channels_list,
                 mappingReduced_bb,
                 bb_SO3_rotation,
                 bb_SO3_grid,
                 num_heads=8,
                 h_channels=32,
                 bb_channels=32,
                 h_time=64,
                 scalar_h_dim=128,
                 n_layers=4,
                 knn_k=20,
                 lrange_k=40,
                 device='cpu'):
        super().__init__()

        self.bb_lmax_list = bb_lmax_list
        self.bb_channels = bb_channels

        self.mappingReduced_nodes = mappingReduced_bb
        self.bb_SO3_rotation_list = bb_SO3_rotation
        self.bb_SO3_grid_list = bb_SO3_grid

        self.h_time = h_time
        self.time_rbf = RBF(n_basis=h_time//2)
        self.time_mlp = nn.Sequential(
            nn.Linear(h_time, scalar_h_dim),
            nn.ReLU(),
            nn.Linear(scalar_h_dim, h_time),
            nn.ReLU()
        )

        def time_mlp():
            return nn.Sequential(
            nn.Linear(bb_channels+h_time, scalar_h_dim),
            nn.ReLU(),
            nn.Linear(scalar_h_dim, bb_channels),
            nn.ReLU(),
            nn.LayerNorm(bb_channels)
        )

        self.embed_time = time_mlp()

        # self.embed_time_encoder = nn.ModuleList([
        #     time_mlp() for _ in range(n_layers)
        # ])

        # self.embed_time_denoiser = nn.ModuleList([
        #     time_mlp() for _ in range(n_layers)
        # ])

        self.spatial_encoder = nn.ModuleList([
            GraphUpdate(
                bb_lmax_list=bb_lmax_list,
                edge_channels_list=edge_channels_list,
                mappingReduced_bb=mappingReduced_bb,
                bb_SO3_rotation=bb_SO3_rotation,
                bb_SO3_grid=bb_SO3_grid,
                num_heads=num_heads,
                bb_channels=bb_channels,
                h_channels=h_channels)
            for _ in range(n_layers)
        ])
        self.seq_local_encoder = nn.ModuleList([
            GraphUpdate(
                bb_lmax_list=bb_lmax_list,
                edge_channels_list=edge_channels_list,
                mappingReduced_bb=mappingReduced_bb,
                bb_SO3_rotation=bb_SO3_rotation,
                bb_SO3_grid=bb_SO3_grid,
                num_heads=num_heads,
                bb_channels=bb_channels,
                h_channels=h_channels)
            for _ in range(n_layers)
        ])

        self.denoiser = BackboneUpdateFromGraph(
                bb_lmax_list=bb_lmax_list,
                edge_channels_list=edge_channels_list,
                mappingReduced_bb=mappingReduced_bb,
                bb_SO3_rotation=bb_SO3_rotation,
                bb_SO3_grid=bb_SO3_grid,
                num_heads=num_heads,
                bb_channels=bb_channels,
                h_channels=h_channels)

        self.knn_k = knn_k
        self.lrange_k = lrange_k


    def forward(self, data, intermediates):
        ## prep features
        noised_bb_vecs = intermediates['noised_bb']
        noised_X_ca = noised_bb_vecs[:, 1]
        noised_bb_rel = noised_bb_vecs[:, (0, 2, 3)]
        x_mask = data['residue']['x_mask']
        noising_mask = intermediates['noising_select']
        bb_features = SO3_Embedding(
            data['residue'].num_nodes,
            self.bb_lmax_list,
            self.bb_channels,
            device=noised_X_ca.device,
            dtype=torch.float
        )
        # add residue positional index
        residx = torch.arange(data['residue'].num_nodes, device=bb_features.device)
        bb_features.embedding[:, 0] = _edge_positional_embeddings(residx, num_embeddings=self.bb_channels, device=bb_features.device)
        ts = intermediates['t']  # (B,)

        # compute graph with knn + inv cubic edges
        batch = data['residue'].batch
        masked_X_ca = noised_X_ca.clone()
        masked_X_ca[x_mask] = torch.inf
        edge_index = sample_inv_cubic_edges(masked_X_ca, x_mask, batch, self.knn_k, self.lrange_k)
        edge_dist_vec = noised_X_ca[edge_index[0]] - noised_X_ca[edge_index[1]]
        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)

        # gen edge features
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device)  # edge_channels_list
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)
        edge_rot_mat = init_edge_rot_mat(edge_dist_vec)
        # for rot in self.bb_SO3_rotation_list:
        #     rot.set_wigner(edge_rot_mat)

        seq_local_edge_index = data['residue', 'seq_local', 'residue'].edge_index
        seq_local_edge_dist_vec = noised_X_ca[seq_local_edge_index[0]] - noised_X_ca[seq_local_edge_index[1]]
        seq_local_edge_dist = torch.linalg.vector_norm(seq_local_edge_dist_vec, dim=-1)

        # gen edge features
        seq_local_edge_dist_rbf = _rbf(seq_local_edge_dist, device=edge_dist.device)  # edge_channels_list
        seq_local_edge_dist_rel_pos = _edge_positional_embeddings(seq_local_edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        seq_local_edge_features = torch.cat([seq_local_edge_dist_rbf, seq_local_edge_dist_rel_pos], dim=-1)

        seq_local_edge_rot_mat = init_edge_rot_mat(seq_local_edge_dist_vec)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (B x h_time,)
        embedded_time = self.time_mlp(fourier_time)  # (B x h_time)
        data_splits = data._slice_dict['residue']['x']
        data_lens = data_splits[1:] - data_splits[:-1]
        embedded_time = torch.cat([
            embedded_time[i].view(1, -1).expand(l, -1) for i, l in enumerate(data_lens)
        ])  # n_res x h_time
        # fuse time embedding into node features
        bb_num_l0 = len(self.bb_lmax_list)
        bb_l0 = bb_features.get_invariant_features()  # n_res x (node_num_m0 x h_channels)
        time_expanded = embedded_time.unsqueeze(-2).expand(-1, bb_num_l0, -1)
        bb_update_l0 = self.embed_time(
            torch.cat([bb_l0, time_expanded], dim=-1)
        )
        bb_features.set_invariant_features(bb_l0 + bb_update_l0)

        # center the training example at the mean of the x_cas
        center = []
        for i in range(data['residue'].batch.max().item() + 1):
            select = (data['residue'].batch == i)
            num_nodes = select.long().sum()
            select = select  & (~x_mask)
            subset_x_ca = noised_X_ca[select]
            subset_mean = subset_x_ca.mean(dim=0)
            center.append(subset_mean[None, :].expand(num_nodes, -1))
        center = torch.cat(center, dim=0)

        ## denoising
        X_ca = noised_X_ca - center
        bb_rel = noised_bb_rel

        for spatial_layer, seq_local_layer in zip(self.spatial_encoder, self.seq_local_encoder):
            for rot in self.bb_SO3_rotation_list:
                rot.set_wigner(edge_rot_mat)
            bb_features, edge_features = spatial_layer(
                bb_rel,
                bb_features,
                edge_features,
                edge_index,
                noising_mask
            )

            for rot in self.bb_SO3_rotation_list:
                rot.set_wigner(seq_local_edge_rot_mat)
            bb_features, seq_local_edge_features = seq_local_layer(
                bb_rel,
                bb_features,
                seq_local_edge_features,
                seq_local_edge_index,
                noising_mask
            )

        X_ca, bb_rel = self.denoiser(X_ca, bb_rel, bb_features, noising_mask)

        denoised_bb_vecs = torch.empty_like(noised_bb_vecs)
        denoised_bb_vecs[:, 1] = X_ca + center
        denoised_bb_vecs[:, (0, 2, 3)] = bb_rel
        intermediates['denoised_bb_vecs'] = denoised_bb_vecs
        return intermediates
