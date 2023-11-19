""" Autoencoder module """

import torch
from torch import nn
import numpy as np
from torch_cluster import knn, knn_graph
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf
from ligbinddiff.model.modules.common import ProjectLayer

from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid
from ligbinddiff.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from ligbinddiff.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork, SO3_LinearV2, TransBlockV2
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from ligbinddiff.utils.so3_embedding import type_l_to_so3
from ligbinddiff.utils.frames import backbone_frames_to_bb_atoms
from ligbinddiff.data.datasets.featurize.sidechain import _orientations, _ideal_virtual_Cb, _dihedrals

from ligbinddiff.model.modules.common import EdgeUpdate


class Atom91SeqLayer(nn.Module):
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced,
                 node_SO3_rotation,
                 node_SO3_grid,
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

        self.node_l0_fuse = nn.Linear(h_channels*3, h_channels*2) 

        self.transformer = TransBlockV2(
            sphere_channels=h_channels * 2,
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
            mappingReduced=mappingReduced
        )

        self.res_update = SO3_LinearV2(
            in_features=h_channels,
            out_features=h_channels,
            lmax=max(node_lmax_list)
        )
        self.res_ln = NormSO3(
            lmax_list=node_lmax_list,
            num_channels=h_channels
        )

        self.seq_update = nn.Linear(h_channels, h_channels)
        self.seq_ln = nn.LayerNorm(h_channels)

        self.edge_update = EdgeUpdate(
            node_lmax_list,
            edge_channels_list,
            h_channels
        )

    def forward(self, 
                node_features: SO3_Embedding, 
                res_features: SO3_Embedding, 
                seq_features: torch.Tensor, 
                edge_features: torch.Tensor, 
                edge_index: torch.Tensor):
        
        # fuse together inputs for transformer block
        trans_input = node_features.clone()
        trans_input.set_embedding(
            torch.cat([node_features.embedding, res_features.embedding], dim=-1)
        )
        input_invariants = trans_input.get_invariant_features(flat=True)
        trans_input.set_invariant_features(
            self.node_l0_fuse(
                torch.cat([input_invariants, seq_features], dim=-1)
            )
        )

        node_features = self.transformer(trans_input, edge_features, edge_index)

        res_update = self.res_update(node_features)
        res_features.embedding = self.res_ln(res_features.embedding + res_update.embedding)

        seq_update = self.seq_update(node_features.get_invariant_features(flat=True))
        seq_features = self.seq_ln(seq_features + seq_update)

        edge_features = self.edge_update(
            node_features,
            edge_features,
            edge_index
        )

        return node_features, res_features, seq_features, edge_features


class Atom91SeqEncoder(nn.Module):
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
                 k=30,
                 n_aa=20,
                 ):
        super().__init__()
        self.node_lmax_list = node_lmax_list
        self.atom_lmax_list = atom_lmax_list
        self.atom_channels = atom_channels
        self.bb_channels = bb_channels
        self.h_channels = h_channels

        self.node_SO3_rotation = node_SO3_rotation
        self.atom_super_SO3_rotation = atom_super_SO3_rotation
        self.bb_super_SO3_rotation = bb_super_SO3_rotation

        self.embed_seq = nn.Embedding(n_aa, h_channels)

        self.embed_res = ProjectLayer(
            in_lmax_list=bb_lmax_list,
            in_channels=atom_channels + bb_channels,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_bb,
            super_SO3_rotation=bb_super_SO3_rotation,
            super_SO3_grid=bb_super_SO3_grid
        )
        self.embed_node = ProjectLayer(
            in_lmax_list=atom_lmax_list,
            in_channels=atom_channels + bb_channels,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_atoms,
            super_SO3_rotation=atom_super_SO3_rotation,
            super_SO3_grid=atom_super_SO3_grid
        )

        self.update = nn.ModuleList(
            [
                Atom91SeqLayer(
                    node_lmax_list=node_lmax_list,
                    edge_channels_list=edge_channels_list,
                    mappingReduced=mappingReduced_nodes,
                    node_SO3_rotation=node_SO3_rotation,
                    node_SO3_grid=node_SO3_grid,
                    h_channels=h_channels
                )
                for _ in range(num_layers)
            ]
        )

        self.res_mu = TransBlockV2(
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

        self.res_logvar = TransBlockV2(
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

        self.seq_dist_params = nn.Linear(h_channels, h_channels*2)

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

        seq_features = torch.zeros_like(self.embed_seq(graph['residue']['seq']))

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

        for rot in self.node_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        for rot in self.atom_super_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        for rot in self.bb_super_SO3_rotation:
            rot.set_wigner(edge_rot_mat)

        input_features = SO3_Embedding(
            num_nodes,
            lmax_list=self.node_lmax_list,
            num_channels=self.bb_channels + self.atom_channels,
            device=bb_features.device,
            dtype=bb_features.dtype
        )
        input_features.set_embedding(
            torch.cat([bb_features.embedding, atom_features.embedding], dim=-1)
        )
        node_features = self.embed_node(input_features, edge_features, edge_index)
        res_features = self.embed_res(input_features, edge_features, edge_index)

        for layer in self.update:
            node_features, res_features, seq_features, edge_features = layer(
                node_features,
                res_features,
                seq_features,
                edge_features,
                edge_index
            )

        latent_mu = self.res_mu(res_features, edge_features, edge_index)
        latent_logvar = self.res_logvar(res_features, edge_features, edge_index)
        seq_params = self.seq_dist_params(seq_features)
        seq_mu, seq_logvar = seq_params.split(self.h_channels, dim=-1)

        out_dict = {}

        out_dict['latent_mu'] = latent_mu
        out_dict['latent_logvar'] = latent_logvar
        out_dict['seq_mu'] = seq_mu
        out_dict['seq_logvar'] = seq_logvar
        return out_dict


class Atom91SeqDecoder(nn.Module):
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
        self.bb_channels = bb_channels

        self.embed_node = ProjectLayer(
            in_lmax_list=bb_lmax_list,
            in_channels=bb_channels,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_bb,
            super_SO3_rotation=bb_super_SO3_rotation,
            super_SO3_grid=bb_super_SO3_grid
        )

        self.update= nn.ModuleList(
            [
                Atom91SeqLayer(
                    node_lmax_list=node_lmax_list,
                    edge_channels_list=edge_channels_list,
                    mappingReduced=mappingReduced_nodes,
                    node_SO3_rotation=node_SO3_rotation,
                    node_SO3_grid=node_SO3_grid,
                    h_channels=h_channels
                )
                for _ in range(num_layers)
            ]
        )
        self.res_ln = NormSO3(
            lmax_list=node_lmax_list,
            num_channels=h_channels
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
            nn.LayerNorm(h_channels),
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

        for rot in self.node_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.bb_super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.atom_super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)

        res_features = intermediates['latent_sidechain']
        seq_features = intermediates['latent_seq']
        dummy_seq_features = seq_features.clone()
        node_features = self.embed_node(bb_features, edge_features, edge_index)

        for layer in self.update:
            node_features, res_features, dummy_seq_features, edge_features = layer(
                node_features,
                res_features,
                dummy_seq_features,
                edge_features,
                edge_index
            )

        res_features.embedding = self.res_ln(res_features.embedding)
        atom_features = self.output_atoms(res_features, edge_features, edge_index)
        seq_logits = self.seq_head(seq_features)

        out_dict = {}
        out_dict['decoded_latent'] = atom_features.embedding[..., 1:4, :].transpose(-1, -2)
        out_dict['decoded_seq_logits'] = seq_logits
        return out_dict
