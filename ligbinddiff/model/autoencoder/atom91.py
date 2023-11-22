""" Autoencoder module """

import torch
from torch import nn
import numpy as np
from torch_cluster import knn, knn_graph
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf
from ligbinddiff.model.modules.common import ProjectLayer

from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid
from ligbinddiff.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from ligbinddiff.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork, MultiResFeedForwardNetwork, TransBlockV2
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from ligbinddiff.utils.so3_embedding import type_l_to_so3
from ligbinddiff.utils.frames import backbone_frames_to_bb_atoms
from ligbinddiff.data.datasets.featurize.sidechain import _orientations, _ideal_virtual_Cb, _dihedrals

from ligbinddiff.model.modules.common import EdgeUpdate


class Atom91Encoder(nn.Module):
    def __init__(self,
                 node_lmax_list,
                 latent_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 mappingReduced_super_bb,
                 mappingReduced_super_atoms,
                 mappingReduced_super_latent,
                 node_SO3_rotation,
                 node_SO3_grid,
                 bb_super_SO3_rotation,
                 bb_super_SO3_grid,
                 atom_super_SO3_rotation,
                 atom_super_SO3_grid,
                 latent_super_SO3_rotation,
                 latent_super_SO3_grid,
                 bb_lmax_list=[1],
                 bb_channels=7,
                 atom_lmax_list=[1],
                 atom_channels=91,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4,
                 k=30,
                 n_aa=20,
                 ):
        super().__init__()
        self.node_lmax_list = node_lmax_list
        self.atom_lmax_list = atom_lmax_list
        self.atom_channels = atom_channels
        self.h_channels = h_channels

        self.node_SO3_rotation = node_SO3_rotation
        self.atom_super_SO3_rotation = atom_super_SO3_rotation
        self.bb_super_SO3_rotation = bb_super_SO3_rotation
        self.latent_super_SO3_rotation = latent_super_SO3_rotation

        self.embed_seq = nn.Embedding(n_aa, n_aa)

        self.embed_bb = ProjectLayer(
            in_lmax_list=bb_lmax_list,
            in_channels=6+n_aa,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels // 2,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_bb,
            super_SO3_rotation=bb_super_SO3_rotation,
            super_SO3_grid=bb_super_SO3_grid
        )

        self.embed_atoms = ProjectLayer(
            in_lmax_list=atom_lmax_list,
            in_channels=atom_channels,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels // 2,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_atoms,
            super_SO3_rotation=atom_super_SO3_rotation,
            super_SO3_grid=atom_super_SO3_grid
        )

        self.transformer = nn.ModuleList(
            [
                TransBlockV2(
                    sphere_channels=h_channels,
                    attn_hidden_channels=h_channels,
                    num_heads=num_heads,
                    attn_alpha_channels=h_channels // 2,
                    attn_value_channels=h_channels // 2,
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
        self.res_ln = NormSO3(
            lmax_list=node_lmax_list,
            num_channels=h_channels
        )


        self.mu_ln = NormSO3(
            lmax_list=latent_lmax_list,
            num_channels=h_channels
        )
        self.output_mu = ProjectLayer(
            in_lmax_list=node_lmax_list,
            in_channels=h_channels,
            out_lmax_list=latent_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_latent,
            super_SO3_rotation=latent_super_SO3_rotation,
            super_SO3_grid=latent_super_SO3_grid
        )

        self.logvar_ln = NormSO3(
            lmax_list=latent_lmax_list,
            num_channels=h_channels
        )
        self.output_logvar = ProjectLayer(
            in_lmax_list=node_lmax_list,
            in_channels=h_channels,
            out_lmax_list=latent_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_latent,
            super_SO3_rotation=latent_super_SO3_rotation,
            super_SO3_grid=latent_super_SO3_grid
        )
        self.k = k

    def forward(self, graph):
        ## prep features
        num_nodes = graph['residue'].num_nodes
        X_ca = graph['residue']['x']
        bb = graph['residue']['bb']
        orientations = _orientations(X_ca)
        virtual_Cb = _ideal_virtual_Cb(bb) - X_ca
        dihedrals = _dihedrals(bb)
        bb_rel = bb - X_ca.unsqueeze(-2)
        bb_features = {
            0: torch.cat([
                dihedrals,  # 6
                torch.zeros_like(self.embed_seq(graph['residue']['seq']))
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
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        # atom_features = graph['noised_atom91']
        atom_features = SO3_Embedding(
            num_nodes,
            lmax_list=self.atom_lmax_list,
            num_channels=self.atom_channels,
            device=graph['residue']['x'].device,
            dtype=torch.float
        )
        atom_features.embedding[:, 1:4] = graph['residue']['atom91_centered'].transpose(-1, -2)
        atom_features.embedding = torch.zeros_like(atom_features.embedding)

        for rot in self.node_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        for rot in self.atom_super_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        for rot in self.bb_super_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        for rot in self.latent_super_SO3_rotation:
            rot.set_wigner(edge_rot_mat)

        bb_features = self.embed_bb(bb_features, edge_features, edge_index)
        atom_features = self.embed_atoms(atom_features, edge_features, edge_index)

        res_features = SO3_Embedding(
            num_nodes,
            lmax_list=self.node_lmax_list,
            num_channels=self.h_channels,
            device=bb_features.device,
            dtype=bb_features.dtype
        )
        res_features.set_embedding(
            torch.cat([bb_features.embedding, atom_features.embedding], dim=-1)
        )

        for node_layer, edge_layer in zip(self.transformer, self.edge_update):
            res_features = node_layer(res_features, edge_features, edge_index)
            edge_features = edge_layer(res_features, edge_features, edge_index)

        res_features.embedding = self.res_ln(res_features.embedding)
        latent_mu = self.output_mu(res_features, edge_features, edge_index)
        latent_mu.embedding = self.mu_ln(latent_mu.embedding)
        latent_logvar = self.output_logvar(res_features, edge_features, edge_index)
        latent_logvar.embedding = self.logvar_ln(latent_logvar.embedding)

        out_dict = {}

        out_dict['latent_mu'] = latent_mu
        out_dict['latent_logvar'] = latent_logvar
        return out_dict


class Atom91Decoder(nn.Module):
    def __init__(self,
                 node_lmax_list,
                 latent_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 mappingReduced_super_bb,
                 mappingReduced_super_atoms,
                 mappingReduced_super_latent,
                 node_SO3_rotation,
                 node_SO3_grid,
                 bb_super_SO3_rotation,
                 bb_super_SO3_grid,
                 atom_super_SO3_rotation,
                 atom_super_SO3_grid,
                 latent_super_SO3_rotation,
                 latent_super_SO3_grid,
                 bb_lmax_list=[1],
                 bb_channels=7,
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
        self.bb_channels = bb_channels
        if latent_lmax_list is None:
            latent_lmax_list = node_lmax_list

        self.embed_bb = ProjectLayer(
            in_lmax_list=bb_lmax_list,
            in_channels=7,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels // 2,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_bb,
            super_SO3_rotation=bb_super_SO3_rotation,
            super_SO3_grid=bb_super_SO3_grid
        )

        self.latent_ln = NormSO3(
            lmax_list=latent_lmax_list,
            num_channels=h_channels
        )
        self.embed_latent = ProjectLayer(
            in_lmax_list=latent_lmax_list,
            in_channels=h_channels,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels // 2,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_latent,
            super_SO3_rotation=latent_super_SO3_rotation,
            super_SO3_grid=latent_super_SO3_grid
        )

        self.transformer = nn.ModuleList(
            [
                TransBlockV2(
                    sphere_channels=h_channels,
                    attn_hidden_channels=h_channels,
                    num_heads=num_heads,
                    attn_alpha_channels=h_channels // 2,
                    attn_value_channels=h_channels // 2,
                    ffn_hidden_channels=h_channels,
                    output_channels=h_channels,
                    lmax_list=node_lmax_list[0:1],
                    mmax_list=node_lmax_list[0:1],
                    SO3_rotation=node_SO3_rotation,
                    SO3_grid=node_SO3_grid,
                    edge_channels_list=edge_channels_list,
                    mappingReduced=mappingReduced_nodes,
                    alpha_drop=0.1,
                    proj_drop=0.1,
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

        # self.project = TransBlockV2(
        #     sphere_channels=h_channels * 2,
        #     attn_hidden_channels=h_channels,
        #     num_heads=num_heads,
        #     attn_alpha_channels=h_channels // 2,
        #     attn_value_channels=h_channels // 4,
        #     ffn_hidden_channels=h_channels,
        #     output_channels=h_channels,
        #     lmax_list=node_lmax_list[0:1],
        #     mmax_list=node_lmax_list[0:1],
        #     SO3_rotation=node_SO3_rotation,
        #     SO3_grid=node_SO3_grid,
        #     edge_channels_list=edge_channels_list,
        #     mappingReduced=mappingReduced_nodes
        # )

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
        self.latent_super_SO3_rotation_list = latent_super_SO3_rotation

        self.seq_head = nn.Sequential(
            nn.Linear(h_channels, h_channels*2),
            nn.ReLU(),
            nn.Linear(h_channels*2, h_channels),
            nn.ReLU(),
            nn.Linear(h_channels, 20),
            nn.LogSoftmax(dim=-1)
        )

        self.k = k


    def forward(self, graph, intermediates, denoiser_output=None):
        ## prep features
        num_res = graph['residue'].num_nodes
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
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        node_features = intermediates['latent_sidechain'].clone()
        node_features.embedding = self.latent_ln(node_features.embedding)

        for rot in self.node_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.bb_super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.atom_super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)

        embed_bb = self.embed_bb(bb_features, edge_features, edge_index)
        embed_latent = self.embed_latent(node_features, edge_features, edge_index)

        res_features = SO3_Embedding(
            num_res,
            lmax_list=self.node_lmax_list,
            num_channels=self.h_channels + self.bb_channels,
            device=bb_features.device,
            dtype=bb_features.dtype
        )
        res_features.set_embedding(
            torch.cat([embed_bb.embedding, embed_latent.embedding], dim=-1)
        )

        for node_layer, edge_layer in zip(self.transformer, self.edge_update):
            res_features = node_layer(res_features, edge_features, edge_index)
            edge_features = edge_layer(res_features, edge_features, edge_index)


        # res_features = self.project(res_features, edge_features, edge_index)
        atom_features = self.output_atoms(res_features, edge_features, edge_index)

        seq_logits = self.seq_head(res_features.get_invariant_features(flat=True))

        out_dict = {}
        out_dict['decoded_latent'] = atom_features.embedding[..., 1:4, :].transpose(-1, -2)
        out_dict['decoded_seq_logits'] = seq_logits
        return out_dict
