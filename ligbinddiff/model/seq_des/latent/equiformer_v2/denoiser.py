""" Denoising model """

import torch
from torch import nn
import numpy as np

from ligbinddiff.model.modules.equiformer_v2.so2_ops import Nodewise_SO3_Convolution
from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2
from ligbinddiff.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from ligbinddiff.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork, MultiResFeedForwardNetwork, TransBlockV2
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat

from ligbinddiff.utils.atom_reps import atom91_start_end


class EdgeUpdate(nn.Module):
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 h_channels=32):
        super().__init__()
        h_dim = edge_channels_list[0]
        num_l0 = len(node_lmax_list)
        self.ff = nn.Sequential(
            nn.Linear(h_dim + h_channels * num_l0 * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)
        )
        self.norm = nn.LayerNorm(h_dim)

    def forward(self, node_features, edge_features, edge_index):
        node_src = node_features.expand_edge(edge_index[0])
        node_src_invariant = node_src.get_invariant_features(flat=True)
        node_dst = node_features.expand_edge(edge_index[1])
        node_dst_invariant = node_dst.get_invariant_features(flat=True)
        in_features = torch.cat([node_src_invariant, node_dst_invariant, edge_features], dim=-1)
        update = self.ff(in_features)

        return edge_features + self.norm(update)


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

        self.attention = TransBlockV2(
            sphere_channels=h_channels,
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

        self.edge_update = EdgeUpdate(
            node_lmax_list,
            edge_channels_list,
            h_channels
        )

    def forward(
            self,
            node_features: SO3_Embedding,
            edge_features: torch.Tensor,
            graph
    ):
        # transformer block
        edge_index = graph.edge_index
        # transformer block
        node_features = self.attention(
            node_features,
            edge_features,
            edge_index=edge_index
        )
        node_features = node_features.distribute_resolutions(len(self.node_lmax_list))

        # update edges
        edge_features = self.edge_update(
            node_features,
            edge_features,
            graph.edge_index
        )
        return node_features, edge_features


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


    def forward(self, graph):
        ## prep features
        num_nodes = graph['x'].shape[0]
        node_features = graph['noised_latent']
        edge_features = graph['edge_s']
        ts = graph['t']  # (B, 1, 1)

        # init SO3_rotation and SO3_grid
        edge_index = graph.edge_index
        X_ca = graph['x']
        edge_distance_vec = X_ca[edge_index[1]] - X_ca[edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for rot in self.node_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (B x h_time,)
        embedded_time = self.time_mlp(fourier_time)  # (B x h_time)
        data_splits = graph._slice_dict['x']
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

        ## denoising
        f_V = node_features
        f_E = edge_features

        for layer in self.denoiser:
            f_V, f_E = layer(f_V, f_E, graph)

        graph['denoised_latent'] = f_V
        # graph['latent_edge'] = f_E
        return graph
