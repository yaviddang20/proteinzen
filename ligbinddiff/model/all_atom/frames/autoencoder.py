""" Autoencoder module """

import torch
from torch import nn
import numpy as np
from torch_cluster import knn_graph

from ligbinddiff.model.modules.equiformer_v2.so2_ops import Nodewise_SO3_Convolution
from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2
from ligbinddiff.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from ligbinddiff.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork, MultiResFeedForwardNetwork, TransBlockV2
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from ligbinddiff.utils.so3_embedding import type_l_to_so3
from ligbinddiff.data.datasets.featurize.sidechain import _rbf, _positional_embeddings, _orientations, _sidechains, _dihedrals

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

    def forward(self, node_features, edge_features, edge_index):
        node_features = node_features.to_resolutions(self.super_lmax_list, self.super_lmax_list)
        node_features = self.conv(node_features, edge_features, edge_index)
        return node_features.to_resolutions(self.out_lmax_list, self.out_lmax_list)


class LatentEncoder(nn.Module):
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 mappingReduced_super_bb,
                 mappingReduced_super_atoms,
                 node_SO3_rotation,
                 node_SO3_grid,
                 bb_super_SO3_rotation,
                 bb_super_SO3_grid,
                 atom_super_SO3_rotation,
                 atom_super_SO3_grid,
                 bb_lmax_list=[1],
                 bb_channels=6,
                 atom_lmax_list=[1],
                 atom_channels=91,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4
                 ):
        super().__init__()
        self.node_lmax_list = node_lmax_list
        self.atom_lmax_list = atom_lmax_list
        self.atom_channels = atom_channels
        self.h_channels = h_channels

        self.node_SO3_rotation = node_SO3_rotation
        self.atom_super_SO3_rotation = atom_super_SO3_rotation
        self.bb_super_SO3_rotation = bb_super_SO3_rotation

        self.embed_bb = ProjectLayer(
            in_lmax_list=bb_lmax_list,
            in_channels=bb_channels,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_bb,
            super_SO3_rotation=bb_super_SO3_rotation,
            super_SO3_grid=bb_super_SO3_grid
        )

        self.embed_atoms = ProjectLayer(
            in_lmax_list=atom_lmax_list,
            in_channels=atom_channels,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_atoms,
            super_SO3_rotation=atom_super_SO3_rotation,
            super_SO3_grid=atom_super_SO3_grid
        )

        self.transformer = nn.ModuleList(
            [
                TransBlockV2(
                    sphere_channels=h_channels * 2,
                    attn_hidden_channels=h_channels,
                    num_heads=num_heads,
                    attn_alpha_channels=h_channels // 2,
                    attn_value_channels=h_channels // 4,
                    ffn_hidden_channels=h_channels,
                    output_channels=h_channels * 2,
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
                    h_channels=h_channels * 2
                )
                for _ in range(num_layers)
            ]
        )

        self.output_mu = TransBlockV2(
            sphere_channels=h_channels * 2,
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

        self.output_logvar = TransBlockV2(
            sphere_channels=h_channels * 2,
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

    def forward(self, graph):
        ## prep features
        num_nodes = graph['x'].shape[0]
        bb_features = {
            0: graph['bb_s'].unsqueeze(-1),
            1: graph['bb_v']
        }
        bb_features = type_l_to_so3(bb_features)

        edge_features = graph['edge_s']

        # atom_features = graph['noised_atom91']
        atom_features = SO3_Embedding(
            num_nodes,
            lmax_list=self.atom_lmax_list,
            num_channels=self.atom_channels,
            device=graph['x'].device,
            dtype=torch.float
        )
        atom_features.embedding[:, 1:4] = graph['atom91_centered'].transpose(-1, -2)

        edge_index = graph.edge_index
        X_ca = graph['x']
        edge_distance_vec = X_ca[edge_index[1]] - X_ca[edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for rot in self.node_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        for rot in self.atom_super_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        for rot in self.bb_super_SO3_rotation:
            rot.set_wigner(edge_rot_mat)

        bb_features = self.embed_bb(bb_features, edge_features, edge_index)
        atom_features = self.embed_atoms(atom_features, edge_features, edge_index)

        res_features = SO3_Embedding(
            num_nodes,
            lmax_list=self.node_lmax_list,
            num_channels=self.h_channels*2,
            device=bb_features.device,
            dtype=bb_features.dtype
        )
        res_features.set_embedding(
            torch.cat([bb_features.embedding, atom_features.embedding], dim=-1)
        )

        for node_layer, edge_layer in zip(self.transformer, self.edge_update):
            res_features = node_layer(res_features, edge_features, graph.edge_index)
            edge_features = edge_layer(res_features, edge_features, graph.edge_index)

        latent_mu = self.output_mu(res_features, edge_features, graph.edge_index)
        latent_logvar = self.output_logvar(res_features, edge_features, graph.edge_index)

        out_dict = {}
        out_dict['latent_mu'] = latent_mu
        out_dict['latent_logvar'] = latent_logvar
        return out_dict


class LatentDecoder(nn.Module):
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 mappingReduced_super_bb,
                 mappingReduced_super_atoms,
                 node_SO3_rotation,
                 node_SO3_grid,
                 bb_super_SO3_rotation,
                 bb_super_SO3_grid,
                 atom_super_SO3_rotation,
                 atom_super_SO3_grid,
                 bb_lmax_list=[1],
                 bb_channels=6,
                 atom_lmax_list=[1],
                 atom_channels=91,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4,
                 k=30,
                 ):
        super().__init__()
        self.node_lmax_list = node_lmax_list
        self.atom_lmax_list = atom_lmax_list
        self.atom_channels = atom_channels
        self.h_channels = h_channels

        self.embed_bb = ProjectLayer(
            in_lmax_list=bb_lmax_list,
            in_channels=bb_channels,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_bb,
            super_SO3_rotation=bb_super_SO3_rotation,
            super_SO3_grid=bb_super_SO3_grid
        )

        self.transformer = nn.ModuleList(
            [
                TransBlockV2(
                    sphere_channels=h_channels * 2,
                    attn_hidden_channels=h_channels,
                    num_heads=num_heads,
                    attn_alpha_channels=h_channels // 2,
                    attn_value_channels=h_channels // 4,
                    ffn_hidden_channels=h_channels,
                    output_channels=h_channels * 2,
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
                    h_channels=h_channels * 2
                )
                for _ in range(num_layers)
            ]
        )

        self.project = TransBlockV2(
            sphere_channels=h_channels * 2,
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

        self.output_atoms = ProjectLayer(
            in_lmax_list=node_lmax_list,
            in_channels=h_channels,
            out_lmax_list=atom_lmax_list,
            out_channels=atom_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_atoms,
            super_SO3_rotation=atom_super_SO3_rotation,
            super_SO3_grid=atom_super_SO3_grid
        )

        self.node_SO3_rotation_list = node_SO3_rotation
        self.bb_super_SO3_rotation_list = bb_super_SO3_rotation
        self.atom_super_SO3_rotation_list = atom_super_SO3_rotation
        self.atom_SO3_rotation_list = nn.ModuleList()
        for lmax in atom_lmax_list:
            self.atom_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )

        self.seq_head = nn.Sequential(
            nn.LayerNorm(atom_channels),
            nn.Linear(atom_channels, 20),
            nn.LogSoftmax(dim=-1)
        )

        self.k = k


    def forward(self, graph, intermediates):
        ## prep features
        num_nodes = graph['x'].shape[0]
        node_features = intermediates['latent_sidechain']
        bb = graph['bb']
        X_ca = bb[..., 1, :]
        masked_X_ca = X_ca.clone()
        masked_X_ca[graph['x_mask']] = torch.inf

        edge_index = knn_graph(masked_X_ca, self.k, graph.batch)
        edge_distance_vec = X_ca[edge_index[1]] - X_ca[edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for rot in self.atom_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.node_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.bb_super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.atom_super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)

        orientations = _orientations(X_ca)
        sidechains = _sidechains(bb)
        dihedrals = _dihedrals(bb)
        bb_features = {
            0: dihedrals.unsqueeze(-1),
            1: torch.nan_to_num(
                torch.cat([
                    orientations,
                    sidechains.unsqueeze(-2)], dim=-2)
            )
        }
        bb_features = type_l_to_so3(bb_features)

        edge_dist = torch.linalg.vector_norm(edge_distance_vec, dim=-1)
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device)  # edge_channels_list
        edge_dist_rel_pos = _positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        bb_features = self.embed_bb(bb_features, edge_features, edge_index)
        res_features = SO3_Embedding(
            num_nodes,
            lmax_list=self.node_lmax_list,
            num_channels=self.h_channels*2,
            device=bb_features.device,
            dtype=bb_features.dtype
        )
        res_features.set_embedding(
            torch.cat([bb_features.embedding, node_features.embedding], dim=-1)
        )

        for node_layer, edge_layer in zip(self.transformer, self.edge_update):
            res_features = node_layer(res_features, edge_features, edge_index)
            edge_features = edge_layer(res_features, edge_features, edge_index)

        res_features = self.project(res_features, edge_features, edge_index)
        atom_features = self.output_atoms(res_features, edge_dist_rbf, edge_index)

        atom_invariants = atom_features.get_invariant_features(flat=True)
        seq_logits = self.seq_head(atom_invariants)

        out_dict = {}
        out_dict['decoded_latent'] = atom_features.embedding[..., 1:4, :].transpose(-1, -2)
        out_dict['decoded_seq_logits'] = seq_logits
        return out_dict
