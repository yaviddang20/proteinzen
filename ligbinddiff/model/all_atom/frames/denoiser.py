""" Denoising model """

import torch
from torch import nn
import numpy as np
from torch_cluster import knn_graph
from torch_geometric.utils import sort_edge_index

from ligbinddiff.model.modules.equiformer_v2.so2_ops import Nodewise_SO3_Convolution
from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2
from ligbinddiff.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from ligbinddiff.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork, MultiResFeedForwardNetwork, TransBlockV2
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from .autoencoder import ProjectLayer

from ligbinddiff.data.datasets.featurize.sidechain import _rbf, _positional_embeddings
from ligbinddiff.utils.atom_reps import atom91_start_end


def sample_inv_cubic_edges(batched_X_ca, batched_x_mask, batch, knn_k=30, inv_cube_k=10):
    edge_indicies = []
    offset = 0
    for i in range(batch.max().item() + 1):
        X_ca = batched_X_ca[batch == i]
        x_mask = batched_x_mask[batch == i]

        X_ca[x_mask] = torch.inf
        rel_pos_CA = X_ca.unsqueeze(1) - X_ca.unsqueeze(0)  # N x N x 3
        dist_CA = torch.linalg.vector_norm(rel_pos_CA, dim=-1)  # N x N
        sorted_dist, sorted_edges = torch.sort(dist_CA, dim=-1)  # N x N
        knn_edges = sorted_edges[..., :knn_k]

        # remove knn edges
        remaining_dist = sorted_dist[..., knn_k:]  # N x (N - knn_k)
        remaining_edges = sorted_edges[..., knn_k:]  # N x (N - knn_k)

        ## inv cube
        uniform = torch.distributions.Uniform(0,1)
        dist_noise = uniform.sample(remaining_dist.shape).to(batched_X_ca.device)  # N x (N - knn_k)

        logprobs = -3 * torch.log(remaining_dist)  # N x (N - knn_k)
        perturbed_logprobs = logprobs - torch.log(-torch.log(dist_noise))  # N x (N - knn_k)
        _, sampled_edges_relative_idx = torch.topk(perturbed_logprobs, k=inv_cube_k, dim=-1)
        sampled_edges = torch.gather(remaining_edges, -1, sampled_edges_relative_idx)  # N x inv_cube_k

        edge_sinks = torch.cat([knn_edges, sampled_edges], dim=-1)  # B x N x (knn_k + inv_cube_k)
        edge_sources = torch.arange(X_ca.shape[0]).repeat_interleave(knn_k + inv_cube_k).to(edge_sinks.device)
        edge_index = torch.stack([edge_sinks.flatten(), edge_sources], dim=0)
        edge_indicies.append(sort_edge_index(edge_index, sort_by_row=False) + offset)
        offset = offset + (batch == i).long().sum()

    edge_index = torch.cat(edge_indicies, dim=-1)
    edge_dist_vec = batched_X_ca[edge_index[0]] - batched_X_ca[edge_index[1]]
    edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
    # slightly hacky
    # TODO: use x_mask instead
    edge_select = edge_dist.isfinite() & (edge_dist > 0.1)  # mostly arbitrary cutoff
    return edge_index[:, edge_select]


class LatentDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 node_SO3_rotation,
                 node_SO3_grid,
                 num_heads=8,
                 h_channels=32,
                 bb_channels=1,
                 k=30,
                 ):
        """
        Args
        ----
        """
        super().__init__()
        self.node_lmax_list = node_lmax_list

        self.mappingReduced_nodes = mappingReduced_nodes
        self.node_SO3_rotation = node_SO3_rotation
        self.node_SO3_grid = node_SO3_grid

        self.lrange_attention = TransBlockV2(
            sphere_channels=h_channels + bb_channels,
            attn_hidden_channels=h_channels,
            num_heads=num_heads,
            attn_alpha_channels=h_channels // 2,
            attn_value_channels=h_channels // 4,
            ffn_hidden_channels=h_channels,
            output_channels=bb_channels,
            lmax_list=node_lmax_list,
            mmax_list=node_lmax_list,
            SO3_rotation=node_SO3_rotation,
            SO3_grid=node_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_nodes
        )

        self.local_attention = TransBlockV2(
            sphere_channels=h_channels + bb_channels,
            attn_hidden_channels=h_channels,
            num_heads=num_heads,
            attn_alpha_channels=h_channels // 2,
            attn_value_channels=h_channels // 4,
            ffn_hidden_channels=h_channels,
            output_channels=h_channels,
            lmax_list=node_lmax_list,
            mmax_list=node_lmax_list,
            SO3_rotation=node_SO3_rotation,
            SO3_grid=node_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_nodes
        )

        # if bb_lmax_list != node_lmax_list:
        #     # TODO
        #     self.project_bb = ProjectLayer(
        #         in_lmax_list=bb_lmax_list,
        #         in_channels=h_channels,
        #         out_lmax_list=node_lmax_list,
        #         out_channels=h_channels,
        #         edge_channels_list=edge_channels_list,
        #         mappingReduced_super=None,
        #         super_SO3_rotation=None,
        #         super_SO3_grid=None,
        #     )

        self.k = k
        self.h_channels = h_channels
        self.bb_channels = bb_channels

    def forward(
            self,
            node_features: SO3_Embedding,
            data,
            intermediates
    ):
        # compute which nodes are being noised and which aren't
        num_nodes = node_features.length
        noised_residx = intermediates['noised_residx']
        residx_select = torch.arange(num_nodes).to(node_features.device)
        residx_select = (residx_select[:, None] == noised_residx[None, :]).any(dim=-1)
        one_hot_residx_select = residx_select.float()

        # compute graph with knn + inv cubic edges
        X_ca = intermediates['denoised_x']
        x_mask = data['x_mask']
        edge_index = sample_inv_cubic_edges(X_ca, x_mask, data.batch)
        edge_dist_vec = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
        # update rotation matrices
        edge_rot_mat = init_edge_rot_mat(edge_dist_vec)
        for rot in self.node_SO3_rotation:
            rot.set_wigner(edge_rot_mat)

        # update backbone
        bb_node_fused = SO3_Embedding(
            num_nodes,
            self.node_lmax_list,
            num_channels = self.h_channels + self.bb_channels,
            device = node_features.device,
            dtype=node_features.dtype
        )
        bb_node_fused.embedding[..., :self.h_channels] = node_features.embedding
        bb_node_fused.embedding[..., 1:4, -self.bb_channels:] = X_ca.unsqueeze(-1)
        bb_node_fused.embedding[..., 0:1, -self.bb_channels:] = one_hot_residx_select[:, None, None] # is node editable or not
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device)  # edge_channels_list
        edge_dist_rel_pos = _positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        update_X_ca = self.lrange_attention(
            bb_node_fused,
            edge_features,
            edge_index
        )
        update_X_ca = update_X_ca.embedding[..., 1:4, :].squeeze(-1)
        new_X_ca = X_ca.clone()
        new_X_ca[noised_residx] = update_X_ca[noised_residx]

        # compute local knn graph
        edge_index = knn_graph(new_X_ca, self.k, data.batch)
        # update rotation matrices
        edge_dist_vec = new_X_ca[edge_index[0]] - new_X_ca[edge_index[1]]
        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
        # hacky way to filter for bad edges
        # TODO: use x_mask
        edge_select = edge_dist.isfinite() & (edge_dist > 1e-6)  # mostly arbitrary cutoff
        edge_index = edge_index[:, edge_select]
        edge_dist_vec = edge_dist_vec[edge_select]
        edge_dist = edge_dist[edge_select]

        edge_rot_mat = init_edge_rot_mat(edge_dist_vec)
        for rot in self.node_SO3_rotation:
            rot.set_wigner(edge_rot_mat)

        # update latent sidechains
        bb_node_fused = SO3_Embedding(
            num_nodes,
            self.node_lmax_list,
            num_channels = self.h_channels + self.bb_channels,
            device = node_features.device,
            dtype=node_features.dtype
        )
        bb_node_fused.embedding[..., :self.h_channels] = node_features.embedding
        bb_node_fused.embedding[..., 1:4, -self.bb_channels:] = new_X_ca.unsqueeze(-1)
        bb_node_fused.embedding[..., 0:1, -self.bb_channels:] = one_hot_residx_select[:, None, None] # is node editable or not

        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device)  # edge_channels_list
        edge_dist_rel_pos = _positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        update_node_features = self.local_attention(
            bb_node_fused,
            edge_features,
            edge_index
        )
        new_node_features = node_features.clone()
        new_node_features.embedding[noised_residx] = update_node_features.embedding[noised_residx]

        return new_X_ca, new_node_features


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


class LatentDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 node_SO3_rotation,
                 node_SO3_grid,
                 num_heads=8,
                 h_channels=32,
                 h_time=64,
                 scalar_h_dim=128,
                 n_layers=4,
                 device='cpu'):
        super().__init__()

        self.node_lmax_list = node_lmax_list

        self.mappingReduced_nodes = mappingReduced_nodes
        self.node_SO3_rotation_list = node_SO3_rotation
        self.node_SO3_grid_list = node_SO3_grid

        self.h_time = h_time
        self.time_rbf = RBF(n_basis=h_time//2)
        self.time_mlp = nn.Sequential(
            nn.Linear(h_time, scalar_h_dim),
            nn.ReLU(),
            nn.Linear(scalar_h_dim, h_time),
            nn.ReLU()
        )

        self.embed_time = nn.Linear(
            h_channels + h_time, h_channels
        )


        self.denoiser = nn.ModuleList([
            LatentDenoisingLayer(
                node_lmax_list=node_lmax_list,
                edge_channels_list=edge_channels_list,
                mappingReduced_nodes=mappingReduced_nodes,
                node_SO3_rotation=self.node_SO3_rotation_list,
                node_SO3_grid=self.node_SO3_grid_list,
                num_heads=num_heads,
                h_channels=h_channels)
            for _ in range(n_layers)
        ])


    def forward(self, data, intermediates):
        ## prep features
        node_features = intermediates['noised_latent_sidechain']
        ts = intermediates['t']  # (B,)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (B x h_time,)
        embedded_time = self.time_mlp(fourier_time)  # (B x h_time)
        data_splits = data._slice_dict['x']
        data_lens = data_splits[1:] - data_splits[:-1]
        embedded_time = torch.cat([
            embedded_time[i].view(1, -1).expand(l, -1) for i, l in enumerate(data_lens)
        ])  # n_res x h_time

        # fuse time embedding into node features
        node_num_l0 = len(self.node_lmax_list)
        node_l0 = node_features.get_invariant_features()  # n_res x (node_num_m0 x h_channels)
        time_expanded = embedded_time.unsqueeze(-2).expand(-1, node_num_l0, -1)
        node_l0 = self.embed_time(
            torch.cat([node_l0, time_expanded], dim=-1)
        )
        node_features.set_invariant_features(node_l0)


        # center the training example at the mean of the x_cas
        center = []
        for i in range(data.batch.max().item() + 1):
            select = (data.batch == i)
            num_nodes = select.long().sum()
            subset_x_ca = intermediates['noised_x']
            subset_mean = subset_x_ca.mean(dim=0)
            center.append(subset_mean[None, :].expand(num_nodes, -1))
        center = torch.cat(center, dim=0)


        ## denoising
        f_V = node_features
        intermediates['denoised_x'] = intermediates['noised_x'] - center

        for layer in self.denoiser:
            f_ca, f_V = layer(f_V, data, intermediates)
            intermediates['denoised_x'] = f_ca

        intermediates['denoised_x'] = intermediates['denoised_x'] + center

        intermediates['denoised_latent_sidechain'] = f_V
        return intermediates
