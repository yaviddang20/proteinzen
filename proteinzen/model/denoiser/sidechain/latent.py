""" Denoising model """

import torch
from torch import nn
import numpy as np
from torch_cluster import knn_graph
from proteinzen.model.modules.common import EdgeUpdate, ProjectLayer

from proteinzen.model.modules.equiformer_v2.so2_ops import Nodewise_SO3_Convolution
from proteinzen.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2
from proteinzen.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from proteinzen.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork, MultiResFeedForwardNetwork, TransBlockV2
from proteinzen.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat

from proteinzen.data.datasets.featurize.sidechain import _orientations, _ideal_virtual_Cb, _dihedrals


class LatentDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 node_lmax_list,
                 latent_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 node_SO3_rotation,
                 node_SO3_grid,
                 mappingReduced_super,
                 super_SO3_rotation,
                 super_SO3_grid,
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

        self.node_update = ProjectLayer(
            in_lmax_list=latent_lmax_list,
            in_channels=h_channels,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super,
            super_SO3_rotation=super_SO3_rotation,
            super_SO3_grid=super_SO3_grid
        )

        self.attention = TransBlockV2(
            sphere_channels=h_channels*2,
            attn_hidden_channels=h_channels,
            num_heads=num_heads,
            attn_alpha_channels=h_channels // 2,
            attn_value_channels=h_channels // 2,
            ffn_hidden_channels=h_channels,
            output_channels=h_channels,
            lmax_list=node_lmax_list,
            mmax_list=node_lmax_list,
            SO3_rotation=node_SO3_rotation,
            SO3_grid=node_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_nodes
        )
        self.node_ln = NormSO3(
            lmax_list=node_lmax_list,
            num_channels=h_channels
        )

        self.node_transition = FeedForwardNetwork(
            sphere_channels=h_channels,
            hidden_channels=h_channels*2,
            output_channels=h_channels,
            lmax_list=node_lmax_list,
            mmax_list=node_lmax_list,
            SO3_grid=node_SO3_grid
        )


        self.res_ln = NormSO3(
            lmax_list=node_lmax_list,
            num_channels=h_channels
        )
        self.res_update = ProjectLayer(
            in_lmax_list=node_lmax_list,
            in_channels=h_channels,
            out_lmax_list=latent_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super,
            super_SO3_rotation=super_SO3_rotation,
            super_SO3_grid=super_SO3_grid
        )

        # with torch.no_grad():
        #     self.res_update.weight.fill_(0.0)  # i think this will help?

        self.edge_update = EdgeUpdate(
            node_lmax_list,
            edge_channels_list,
            h_channels
        )

    def forward(
            self,
            node_features: SO3_Embedding,
            res_features: SO3_Embedding,
            edge_features: torch.Tensor,
            edge_index
    ):
        input_features = self.node_update(res_features, edge_features, edge_index)
        input_features.set_embedding(
            torch.cat([input_features.embedding, node_features.embedding], dim=-1)
        )

        # transformer block
        node_update = self.attention(
            input_features,
            edge_features,
            edge_index
        )
        new_node_features = node_features.clone()
        new_node_features.embedding = self.node_ln(new_node_features.embedding + node_update.embedding)

        new_node_features = self.node_transition(new_node_features)

        input_features = new_node_features.clone()
        input_features.embedding = self.res_ln(input_features.embedding)
        res_update = self.res_update(new_node_features, edge_features, edge_index)

        new_res_features = res_features.clone()
        new_res_features.embedding = new_res_features.embedding + res_update.embedding

        # update edges
        edge_features = self.edge_update(
            new_node_features,
            edge_features,
            edge_index
        )
        return new_node_features, new_res_features, edge_features


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


class LatentSidechainDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 node_lmax_list,
                 latent_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 node_SO3_rotation,
                 node_SO3_grid,
                 mappingReduced_super,
                 super_SO3_rotation,
                 super_SO3_grid,
                 num_heads=8,
                 h_channels=32,
                 h_time=64,
                 scalar_h_dim=128,
                 n_layers=4,
                 k=30,
                 device='cpu'):
        super().__init__()

        self.node_lmax_list = node_lmax_list

        self.mappingReduced_nodes = mappingReduced_nodes
        self.node_SO3_rotation_list = node_SO3_rotation
        self.node_SO3_grid_list = node_SO3_grid
        self.mappingReduced_super = mappingReduced_super
        self.super_SO3_rotation_list = super_SO3_rotation
        self.super_SO3_grid_list = super_SO3_grid

        self.h_time = h_time
        self.time_rbf = RBF(n_basis=h_time//2)
        self.time_mlp = nn.Sequential(
            nn.Linear(h_time, scalar_h_dim),
            nn.ReLU(),
            nn.Linear(scalar_h_dim, h_time),
            nn.ReLU()
        )

        self.embed_bb = ProjectLayer(
            in_lmax_list=node_lmax_list,
            in_channels=7,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_nodes,
            super_SO3_rotation=node_SO3_rotation,
            super_SO3_grid=node_SO3_grid
        )
        self.embed_node_l0 = nn.Linear(
            h_channels + h_time, h_channels
        )
        self.embed_edge = nn.Sequential(
            nn.Linear(h_channels, h_channels),
            nn.ReLU(),
            nn.Linear(h_channels, h_channels),
            nn.ReLU(),
            nn.Linear(h_channels, h_channels),
            nn.LayerNorm(h_channels)
        )



        self.denoiser = nn.ModuleList([
            LatentDenoisingLayer(
                node_lmax_list=node_lmax_list,
                latent_lmax_list=latent_lmax_list,
                edge_channels_list=edge_channels_list,
                mappingReduced_nodes=mappingReduced_nodes,
                node_SO3_rotation=self.node_SO3_rotation_list,
                node_SO3_grid=self.node_SO3_grid_list,
                mappingReduced_super=mappingReduced_super,
                super_SO3_rotation=self.super_SO3_rotation_list,
                super_SO3_grid=self.super_SO3_grid_list,
                num_heads=num_heads,
                h_channels=h_channels)
            for _ in range(n_layers)
        ])

        self.k = k


    def forward(self, data, intermediates):
        ## prep features
        res_features = intermediates['noised_latent_sidechain']

        X_ca = data['residue']['x']
        bb = data['residue']['bb']
        orientations = _orientations(X_ca)
        virtual_Cb = _ideal_virtual_Cb(bb) - X_ca
        dihedrals = _dihedrals(bb)
        bb_rel = bb - X_ca.unsqueeze(-2)
        bb_features = {
            0: torch.cat([
                dihedrals,  # 6
            ], dim=-1).unsqueeze(-1),  # total 26
            1: torch.nan_to_num(
                torch.cat([
                    bb_rel,  # 4
                    orientations,  # 2
                    virtual_Cb.unsqueeze(-2)
                ], dim=-2) #1
            )  # total 7
        }
        bb_features = type_l_to_so3(bb_features)
        bb_features.embedding = torch.zeros_like(bb_features.embedding)
        ts = intermediates['t']  # (B, 1, 1)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (B x h_time,)
        embedded_time = self.time_mlp(fourier_time)  # (B x h_time)
        data_splits = data._slice_dict['residue']['x']
        data_lens = data_splits[1:] - data_splits[:-1]
        embedded_time = torch.cat([
            embedded_time[i].view(1, -1).expand(l, -1) for i, l in enumerate(data_lens)
        ])  # n_res x h_time


        # init SO3_rotation and SO3_grid
        X_ca = data['residue']['x']
        masked_X_ca = X_ca.clone()
        masked_X_ca[data['residue']['x_mask']] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, data['residue'].batch)
        edge_distance_vec = X_ca[edge_index[1]] - X_ca[edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for rot in self.node_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        edge_features = data['residue']['edge_s']
        edge_features = self.embed_edge(edge_features)

        node_features = self.embed_bb(bb_features, edge_features, edge_index)

        # fuse time embedding into node features
        node_num_l0 = len(self.node_lmax_list)
        node_l0 = node_features.get_invariant_features()  # n_res x (node_num_m0 x h_channels)
        time_expanded = embedded_time.unsqueeze(-2).expand(-1, node_num_l0, -1)
        node_l0 = self.embed_node_l0(
            torch.cat([node_l0, time_expanded], dim=-1)
        )
        node_features.set_invariant_features(node_l0)

        ## denoising
        f_V = node_features
        f_L = res_features
        f_E = edge_features

        for layer in self.denoiser:
            f_V, f_L, f_E = layer(f_V, f_L, f_E, edge_index)

        intermediates['denoised_latent_sidechain'] = f_L
        return intermediates
