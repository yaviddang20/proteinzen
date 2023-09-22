""" Autoencoder module """

import torch
from torch import nn
import numpy as np
from torch_cluster import knn, knn_graph
from torch_geometric.utils import sort_edge_index

from ligbinddiff.model.modules.equiformer_v2.so2_ops import Nodewise_SO3_Convolution
from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2
from ligbinddiff.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from ligbinddiff.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork, MultiResFeedForwardNetwork, TransBlockV2
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from ligbinddiff.utils.so3_embedding import type_l_to_so3
from ligbinddiff.utils.frames import backbone_frames_to_bb_atoms
from ligbinddiff.data.datasets.featurize.sidechain import _rbf, _positional_embeddings, _orientations, _ideal_virtual_Cb, _dihedrals

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


def directional_knn(X_ca, k, batch, dst_mask):
    src_X_ca = X_ca
    dst_X_ca = X_ca.clone()
    dst_X_ca[dst_mask] = torch.inf
    row, col = knn(src_X_ca, dst_X_ca, k, batch_x=batch, batch_y=batch)
    return torch.stack([col, row], dim=0)


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
                 bb_channels=7,
                 atom_lmax_list=[1],
                 num_aa=20,
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

        self.num_aa = num_aa
        embed_aa = torch.eye(num_aa)
        self.register_buffer("embed_aa", embed_aa)

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
        self.k = k

    def forward(self, graph, noising_mask=None, condition_seq=True):
        ## prep features
        num_nodes = graph.num_nodes
        bb = graph['bb']
        X_ca = bb[..., 1, :]
        masked_X_ca = X_ca.clone()
        masked_X_ca[graph['x_mask']] = torch.inf
        if noising_mask is None:
            noising_mask = torch.zeros(num_nodes, device=X_ca.device).bool()

        if condition_seq:
            conditional_seq = self.embed_aa[graph['seq']]
            conditional_seq[noising_mask] = 0
        else:
            conditional_seq = torch.zeros(graph.num_nodes, self.num_aa, device=X_ca.device)

        orientations = _orientations(X_ca)
        virtual_Cb = _ideal_virtual_Cb(bb)
        dihedrals = _dihedrals(bb)
        bb_rel = bb - X_ca.unsqueeze(-2)
        bb_features = {
            0: dihedrals.unsqueeze(-1),  # 6
            1: torch.nan_to_num(
                torch.cat([
                    bb_rel,  # 4
                    orientations,  # 2
                    virtual_Cb.unsqueeze(-2)], dim=-2) #1
            )  # total 7
        }
        bb_features = type_l_to_so3(bb_features)

        # edge_index = directional_knn(masked_X_ca, self.k, graph.batch, noising_mask)
        edge_index = knn_graph(masked_X_ca, self.k, graph.batch)
        if noising_mask.all() or not noising_mask.any():
            noised_residx = torch.arange(num_nodes, device=X_ca.device)[noising_mask]
            noise_src_edges = (edge_index[1][:, None] == noised_residx[None, :]).any(dim=-1)
            noise_dst_edges = (edge_index[0][:, None] == noised_residx[None, :]).any(dim=-1)
            noise2noise_edges = noise_src_edges & noise_dst_edges
            noise2unnoise_edges = noise_src_edges & ~noise_dst_edges
            unnoise2unnoise_edges = ~noise_src_edges & ~noise_dst_edges
            edge_index = torch.cat([
                edge_index[:, unnoise2unnoise_edges],
                edge_index[:, noise2unnoise_edges],
                edge_index[:, noise2noise_edges],
            ], dim=0)
            edge_index = sort_edge_index(edge_index, sort_by_row=False)


        edge_distance_vec = X_ca[edge_index[1]] - X_ca[edge_index[0]]
        edge_dist = torch.linalg.vector_norm(edge_distance_vec, dim=-1)
        # hacky way to get rid of bad edges

        # edge_select = edge_dist.isfinite() & (edge_dist > 1e-3)  # mostly arbitrary cutoff
        # edge_index = edge_index[:, edge_select]
        # edge_distance_vec = edge_distance_vec[edge_select]
        # edge_dist = edge_dist[edge_select]

        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for rot in self.node_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        for rot in self.atom_super_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        for rot in self.bb_super_SO3_rotation:
            rot.set_wigner(edge_rot_mat)

        # atom_features = graph['noised_atom91']
        atom_features = SO3_Embedding(
            num_nodes,
            lmax_list=self.atom_lmax_list,
            num_channels=self.atom_channels,
            device=graph['x'].device,
            dtype=torch.float
        )
        atom_features.embedding[:, 1:4] = graph['atom91_centered'].transpose(-1, -2)
        atom_features.embedding[..., :self.num_aa] = conditional_seq

        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device)  # edge_channels_list
        edge_dist_rel_pos = _positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

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
            res_features = node_layer(res_features, edge_features, edge_index)
            edge_features = edge_layer(res_features, edge_features, edge_index)

        latent_mu = self.output_mu(res_features, edge_features, edge_index)
        latent_logvar = self.output_logvar(res_features, edge_features, edge_index)

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
                 num_aa=20,
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

        self.num_aa = num_aa
        embed_aa = torch.eye(num_aa)
        self.register_buffer("embed_aa", embed_aa)

        self.embed_bb = ProjectLayer(
            in_lmax_list=bb_lmax_list,
            in_channels=bb_channels + num_aa,
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

        self.seq_head = nn.Sequential(
            nn.LayerNorm(atom_channels),
            nn.Linear(atom_channels, 20),
            nn.LogSoftmax(dim=-1)
        )

        self.k = k


    def forward(self, graph, intermediates, denoiser_output=None, condition_seq=True):
        ## prep features
        num_nodes = graph.num_nodes
        node_features = intermediates['latent'] # CHANGED intermediates['latent_sidechain']
        if denoiser_output is not None:
            bb = denoiser_output['denoised_bb']
        else:
            bb = graph['bb']
        X_ca = bb[..., 1, :]
        noising_mask = torch.ones(num_nodes, device=X_ca.device).bool() # CHANGED intermediates['noising_mask']
        if condition_seq:
            conditional_seq = self.embed_aa[graph['seq']]
            conditional_seq[noising_mask] = 0
        else:
            conditional_seq = torch.zeros(graph.num_nodes, self.num_aa, device=X_ca.device)

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

        masked_X_ca = X_ca.clone()
        masked_X_ca[graph['x_mask']] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph.batch)
        edge_distance_vec = X_ca[edge_index[1]] - X_ca[edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        edge_dist = torch.linalg.vector_norm(edge_distance_vec, dim=-1)
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device)  # edge_channels_list
        edge_dist_rel_pos = _positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        for rot in self.atom_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.bb_super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.atom_super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)

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
        atom_features = self.output_atoms(res_features, edge_features, edge_index)
        # seq_logits = self.seq_head(atom_features, edge_features, graph.edge_index)

        atom_invariants = atom_features.get_invariant_features(flat=True)
        seq_logits = self.seq_head(atom_invariants)

        graph['decoded_latent'] = atom_features.embedding[..., 1:4, :].transpose(-1, -2)
        graph['decoded_seq_logits'] = seq_logits
        return graph


class LatentEncoder2(nn.Module):
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
                 bb_channels=7,
                 atom_lmax_list=[1],
                 atom_channels=91,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4,
                 k=30
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
        self.k = k

    def forward(self, graph):
        ## prep features
        num_nodes = graph.num_nodes
        X_ca = graph['x']
        bb = graph['bb']
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

        masked_X_ca = X_ca.clone()
        masked_X_ca[graph['x_mask']] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph.batch)
        edge_distance_vec = X_ca[edge_index[1]] - X_ca[edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        edge_dist = torch.linalg.vector_norm(edge_distance_vec, dim=-1)
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device)  # edge_channels_list
        edge_dist_rel_pos = _positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        # atom_features = graph['noised_atom91']
        atom_features = SO3_Embedding(
            num_nodes,
            lmax_list=self.atom_lmax_list,
            num_channels=self.atom_channels,
            device=graph['x'].device,
            dtype=torch.float
        )
        atom_features.embedding[:, 1:4] = graph['atom91_centered'].transpose(-1, -2)

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
            res_features = node_layer(res_features, edge_features, edge_index)
            edge_features = edge_layer(res_features, edge_features, edge_index)

        latent_mu = self.output_mu(res_features, edge_features, edge_index)
        latent_logvar = self.output_logvar(res_features, edge_features, edge_index)

        out_dict = {}

        out_dict['latent_mu'] = latent_mu
        out_dict['latent_logvar'] = latent_logvar
        return out_dict


class LatentDecoder2(nn.Module):
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
                 bb_channels=7,
                 atom_lmax_list=[1],
                 atom_channels=91,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4,
                 k=30
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

        self.seq_head = nn.Sequential(
            nn.LayerNorm(atom_channels),
            nn.Linear(atom_channels, atom_channels),
            nn.ReLU(),
            nn.Linear(atom_channels, 20),
            nn.LogSoftmax(dim=-1)
        )

        self.k = k


    def forward(self, graph, intermediates, denoiser_output=None):
        ## prep features
        num_nodes = graph.num_nodes
        X_ca = graph['x']
        bb = graph['bb']
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

        masked_X_ca = X_ca.clone()
        masked_X_ca[graph['x_mask']] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph.batch)
        edge_distance_vec = X_ca[edge_index[1]] - X_ca[edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        edge_dist = torch.linalg.vector_norm(edge_distance_vec, dim=-1)
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device)  # edge_channels_list
        edge_dist_rel_pos = _positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        node_features = intermediates['latent_sidechain']

        for rot in self.node_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.bb_super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.atom_super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)

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
        atom_features = self.output_atoms(res_features, edge_features, edge_index)
        # seq_logits = self.seq_head(atom_features, edge_features, graph.edge_index)

        atom_invariants = atom_features.get_invariant_features(flat=True)
        seq_logits = self.seq_head(atom_invariants)

        out_dict = {}
        out_dict['decoded_latent'] = atom_features.embedding[..., 1:4, :].transpose(-1, -2)
        out_dict['decoded_seq_logits'] = seq_logits
        return out_dict


class AtomicSidechainEncoder(nn.Module):
    def __init__(self,
                 atom_lmax_list,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_atoms,
                 mappingReduced_super_atoms,
                 mappingReduced_nodes,
                 node_SO3_rotation,
                 node_SO3_grid,
                 atom_SO3_rotation,
                 atom_SO3_grid,
                 atom_super_SO3_rotation,
                 atom_super_SO3_grid,
                 atom_channels=18+5,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4
                 ):
        super().__init__()
        self.atom_lmax_list = atom_lmax_list
        self.node_lmax_list = node_lmax_list
        self.atom_channels = atom_channels
        self.h_channels = h_channels

        self.atom_SO3_rotation = atom_SO3_rotation
        self.atom_super_SO3_rotation = atom_super_SO3_rotation

        self.embed_atoms = SO3_LinearV2(
            in_features=atom_channels,
            out_features=h_channels,
            lmax=max(atom_lmax_list)
        )

        self.atom_radius_interactions = nn.ModuleList(
            [
                TransBlockV2(
                    sphere_channels=h_channels,
                    attn_hidden_channels=h_channels,
                    num_heads=num_heads,
                    attn_alpha_channels=h_channels // 2,
                    attn_value_channels=h_channels // 4,
                    ffn_hidden_channels=h_channels,
                    output_channels=h_channels,
                    lmax_list=atom_lmax_list[0:1],
                    mmax_list=atom_lmax_list[0:1],
                    SO3_rotation=atom_SO3_rotation,
                    SO3_grid=atom_SO3_grid,
                    edge_channels_list=edge_channels_list,
                    mappingReduced=mappingReduced_atoms
                )
                for _ in range(num_layers)
            ]
        )

        self.atomic_radius_edge_update = nn.ModuleList(
            [
                EdgeUpdate(
                    node_lmax_list=atom_lmax_list,
                    edge_channels_list=edge_channels_list,
                    h_channels=h_channels
                )
                for _ in range(num_layers)
            ]
        )

        self.residue_aggregation = TransBlockV2(
            sphere_channels=h_channels,
            attn_hidden_channels=h_channels,
            num_heads=num_heads,
            attn_alpha_channels=h_channels // 2,
            attn_value_channels=h_channels // 4,
            ffn_hidden_channels=h_channels,
            output_channels=h_channels,
            lmax_list=atom_lmax_list[0:1],
            mmax_list=atom_lmax_list[0:1],
            SO3_rotation=atom_SO3_rotation,
            SO3_grid=atom_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_atoms
        )

        self.project_to_node = ProjectLayer(
            in_lmax_list=atom_lmax_list,
            in_channels=h_channels,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_atoms,
            super_SO3_rotation=atom_super_SO3_rotation,
            super_SO3_grid=atom_super_SO3_grid
        )

        self.output_mu = TransBlockV2(
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

        self.output_logvar = TransBlockV2(
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


    def forward(self, graph, residue_noising_select=None):
        ## prep features
        num_atoms = graph['atomic'].num_nodes
        num_res = graph['residue'].num_nodes

        atom_features = SO3_Embedding(
            num_atoms,
            lmax_list=self.atom_lmax_list,
            num_channels=self.atom_channels,
            device=graph['atomic']['x'].device,
            dtype=torch.float
        )
        atom_features.set_invariant_features(graph['atomic'].atom_embedding)
        atom_features = self.embed_atoms(atom_features)

        # update atomic embeddings using atomic radius-based graph
        radius_edge_index = graph['atomic'].atomic_radius_edge_index

        atom_x = graph['atomic'].x
        edge_distance_vec = atom_x[radius_edge_index[1]] - atom_x[radius_edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for rot in self.atom_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        edge_dist = torch.linalg.vector_norm(edge_distance_vec, dim=-1)
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device, D_count=32)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf], dim=-1)

        for node_layer, edge_layer in zip(self.atom_radius_interactions, self.atomic_radius_edge_update):
            atom_features = node_layer(atom_features, edge_features, radius_edge_index)
            edge_features = edge_layer(atom_features, edge_features, radius_edge_index)


        # aggregate atom features to residue level
        agg_edge_index = graph['atomic'].atom14_to_ca_edge_index

        atom_x = graph['atomic'].x
        edge_distance_vec = atom_x[agg_edge_index[1]] - atom_x[agg_edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for rot in self.atom_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        edge_dist = torch.linalg.vector_norm(edge_distance_vec, dim=-1)
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device, D_count=32)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf], dim=-1)

        atom_agg_features = self.residue_aggregation(atom_features, edge_features, agg_edge_index)
        atom_agg_bb_embedding = atom_agg_features.embedding[graph['atomic'].backbone_atoms_select]
        embedding_CA = atom_agg_bb_embedding.view(
            num_res, 4,
            atom_agg_features.num_coeffs, atom_agg_features.num_channels
        )[:, 1]

        res_features = SO3_Embedding(
            0,
            lmax_list=self.atom_lmax_list,
            num_channels=self.h_channels,
            device=atom_features.device,
            dtype=torch.float
        )
        res_features.set_embedding(embedding_CA)
        res_edge_index = graph['residue'].edge_index
        res_features = self.project_to_node(res_features, edge_features, res_edge_index)

        latent_mu = self.output_mu(res_features, edge_features, res_edge_index)
        latent_logvar = self.output_logvar(res_features, edge_features, res_edge_index)

        out_dict = {}
        out_dict['latent_mu'] = latent_mu
        out_dict['latent_logvar'] = latent_logvar

        return out_dict


class SidechainDecoder(nn.Module):
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
                 bb_channels=7,
                 atom_lmax_list=[1],
                 atom_channels=91,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4,
                 k=30
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

        self.seq_head = nn.Sequential(
            nn.LayerNorm(atom_channels),
            nn.Linear(atom_channels, atom_channels),
            nn.ReLU(),
            nn.Linear(atom_channels, 20),
            nn.LogSoftmax(dim=-1)
        )

        self.k = k


    def forward(self, graph, intermediates, denoiser_output=None):
        ## prep features
        num_nodes = graph.num_nodes
        bb = graph['residue']['bb']
        X_ca = bb[:, 1]
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

        masked_X_ca = X_ca.clone()
        masked_X_ca[graph['residue']['x_mask']] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph['residue'].batch)
        edge_distance_vec = X_ca[edge_index[1]] - X_ca[edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        edge_dist = torch.linalg.vector_norm(edge_distance_vec, dim=-1)
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device)  # edge_channels_list
        edge_dist_rel_pos = _positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        node_features = intermediates['latent_sidechain']

        for rot in self.node_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.bb_super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.atom_super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)

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
        atom_features = self.output_atoms(res_features, edge_features, edge_index)

        atom_invariants = atom_features.get_invariant_features(flat=True)
        seq_logits = self.seq_head(atom_invariants)

        out_dict = {}
        out_dict['decoded_latent'] = atom_features.embedding[..., 1:4, :].transpose(-1, -2)
        out_dict['decoded_seq_logits'] = seq_logits
        return out_dict
