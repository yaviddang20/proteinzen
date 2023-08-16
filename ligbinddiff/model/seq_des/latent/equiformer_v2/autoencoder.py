""" Autoencoder module """

import torch
from torch import nn
import numpy as np

from ligbinddiff.model.modules.equiformer_v2.so2_ops import Nodewise_SO3_Convolution
from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2
from ligbinddiff.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from ligbinddiff.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork, MultiResFeedForwardNetwork, TransBlockV2
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat

from ligbinddiff.model.seq_des.latent.equiformer_v2.denoiser import EdgeUpdate


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

    def forward(self, node_features, edge_features, graph):
        node_features = node_features.to_resolutions(self.super_lmax_list, self.super_lmax_list)
        node_features = self.conv(node_features, edge_features, graph.edges())
        return node_features.to_resolutions(self.out_lmax_list, self.out_lmax_list)


class Encoder(nn.Module):
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 mappingReduced_super,
                 node_SO3_rotation,
                 node_SO3_grid,
                 super_SO3_rotation,
                 super_SO3_grid,
                 in_lmax_list=[1],
                 in_channels=91,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4
                 ):
        super().__init__()

        self.embed = ProjectLayer(
            in_lmax_list=in_lmax_list,
            in_channels=in_channels,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super,
            super_SO3_rotation=super_SO3_rotation,
            super_SO3_grid=super_SO3_grid
        )

        self.transformer = nn.ModuleList(
            [
                TransBlockV2(
                    sphere_channels=h_channels,
                    attn_hidden_channels=h_channels,
                    num_heads=num_heads,
                    attn_alpha_channels=h_channels // 2,
                    attn_value_channels=h_channels // 4,
                    ffn_hidden_channels=h_channels,
                    output_channels=h_channels,
                    lmax_list=node_lmax_list[0:1],
                    mmax_list=node_lmax_list[0:1],
                    SO3_rotation=node_SO3_rotation,
                    SO3_grid=node_SO3_grid,
                    edge_channels_list=edge_channels_list,
                    mappingReduced=mappingReduced_nodes
                )
                for _ in range(num_layers)
            ]
        )

        self.edge_update = nn.ModuleList(
            [
                EdgeUpdate(
                    node_lmax_list=node_lmax_list,
                    edge_channels_list=edge_channels_list,
                    h_channels=h_channels
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, node_features, edge_features, graph):
        node_features = self.embed(node_features, edge_features, graph)

        for node_layer, edge_layer in zip(self.transformer, self.edge_update):
            node_features = node_layer(node_features, edge_features, graph.edges())
            edge_features = edge_layer(node_features, edge_features, graph.edges())

        return node_features, edge_features


class Decoder(nn.Module):
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 mappingReduced_super,
                 node_SO3_rotation,
                 node_SO3_grid,
                 super_SO3_rotation,
                 super_SO3_grid,
                 in_lmax_list=[1],
                 in_channels=91,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4
                 ):
        super().__init__()

        self.project = ProjectLayer(
            in_lmax_list=node_lmax_list,
            in_channels=h_channels,
            out_lmax_list=in_lmax_list,
            out_channels=in_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super,
            super_SO3_rotation=super_SO3_rotation,
            super_SO3_grid=super_SO3_grid
        )

        self.transformer = nn.ModuleList(
            [
                TransBlockV2(
                    sphere_channels=h_channels,
                    attn_hidden_channels=h_channels,
                    num_heads=num_heads,
                    attn_alpha_channels=h_channels // 2,
                    attn_value_channels=h_channels // 4,
                    ffn_hidden_channels=h_channels,
                    output_channels=h_channels,
                    lmax_list=node_lmax_list[0:1],
                    mmax_list=node_lmax_list[0:1],
                    SO3_rotation=node_SO3_rotation,
                    SO3_grid=node_SO3_grid,
                    edge_channels_list=edge_channels_list,
                    mappingReduced=mappingReduced_nodes
                )
                for _ in range(num_layers)
            ]
        )

        self.edge_update = nn.ModuleList(
            [
                EdgeUpdate(
                    node_lmax_list=node_lmax_list,
                    edge_channels_list=edge_channels_list,
                    h_channels=h_channels
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, node_features, edge_features, graph):
        node_features = self.project(node_features, edge_features, graph)

        for node_layer, edge_layer in zip(self.transformer, self.edge_update):
            node_features = node_layer(node_features, edge_features, graph.edges())
            edge_features = edge_layer(node_features, edge_features, graph.edges())

        return node_features, edge_features
