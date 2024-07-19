from typing import Sequence
import numpy as np
import torch
from torch import nn

from proteinzen.model.modules.equiformer_v2.so2_ops import Nodewise_SO3_Convolution

class ProjectLayer(nn.Module):
    """ Layer to interface between different lmax features """
    def __init__(self,
                 in_lmax_list,
                 in_channels,
                 out_lmax_list,
                 out_channels,
                 edge_channels_list,
                 mappingReduced_super,
                 super_SO3_rotation,
                 super_SO3_grid):
        super().__init__()

        self.in_lmax_list = in_lmax_list
        self.out_lmax_list = out_lmax_list
        self.super_lmax_list = [max(l1, l2) for l1, l2 in zip(in_lmax_list, out_lmax_list)]

        self.super_SO3_rotation = super_SO3_rotation
        self.super_SO3_grid = super_SO3_grid

        self.conv = Nodewise_SO3_Convolution(
            sphere_channels=in_channels,
            m_output_channels=out_channels,
            lmax_list=self.super_lmax_list,
            mmax_list=self.super_lmax_list,
            mappingReduced=mappingReduced_super,
            SO3_rotation=super_SO3_rotation,
            edge_channels_list=edge_channels_list
        )

    def forward(self, node_features, edge_features, edge_index):
        node_features = node_features.to_resolutions(self.super_lmax_list, self.super_lmax_list)
        node_features = self.conv(node_features, edge_features, edge_index)
        return node_features.to_resolutions(self.out_lmax_list, self.out_lmax_list)


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


class GaussianRandomFourierBasis(nn.Module):
    """ Damped random Fourier Feature encoding layer """
    def __init__(self, n_basis=64):
        super().__init__()
        kappa = torch.randn((n_basis,))
        self.register_buffer('kappa', kappa)

    def forward(self, ts):
        tp = 2 * np.pi * ts * self.kappa
        return torch.cat([torch.cos(tp), torch.sin(tp)], dim=-1)
