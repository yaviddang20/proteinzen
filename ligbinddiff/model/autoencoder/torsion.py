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
from ligbinddiff.data.datasets.featurize.sidechain import _orientations, _ideal_virtual_Cb, _dihedrals

from ligbinddiff.utils.framediff.all_atom import compute_all_atom14 , compute_backbone
from ligbinddiff.utils.atom_reps import restype_1to3, atom91_start_end
from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.data.openfold.residue_constants import restypes

from ligbinddiff.model.modules.common import EdgeUpdate


class Atom91Decoder(nn.Module):
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

        self.embed = ProjectLayer(
            in_lmax_list=bb_lmax_list,
            in_channels=bb_channels + h_channels,
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

        self.project_conv = ProjectLayer(
            in_lmax_list=node_lmax_list,
            in_channels=h_channels,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_atoms,
            super_SO3_rotation=atom_super_SO3_rotation,
            super_SO3_grid=atom_super_SO3_grid
        )
        self.torsion_pred = nn.Linear(h_channels, 20 * 4 * 2 + 2)

        self.node_SO3_rotation_list = node_SO3_rotation
        self.bb_super_SO3_rotation_list = bb_super_SO3_rotation
        self.atom_super_SO3_rotation_list = atom_super_SO3_rotation

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

        node_features = intermediates['latent_sidechain']

        for rot in self.node_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.bb_super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.atom_super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)

        res_features = SO3_Embedding(
            num_res,
            lmax_list=self.node_lmax_list,
            num_channels=self.h_channels + self.bb_channels,
            device=bb_features.device,
            dtype=bb_features.dtype
        )
        res_features.set_embedding(
            torch.cat([bb_features.embedding, node_features.embedding], dim=-1)
        )
        res_features = self.embed(res_features, edge_features, edge_index)

        for node_layer, edge_layer in zip(self.transformer, self.edge_update):
            res_features = node_layer(res_features, edge_features, edge_index)
            edge_features = edge_layer(res_features, edge_features, edge_index)

        # res_features = self.project(res_features, edge_features, edge_index)
        torsion_features = self.project_conv(res_features, edge_features, edge_index)
        unnorm_torsions = self.torsion_pred(torsion_features.get_invariant_features(flat=True))
        chi_per_aatype = unnorm_torsions.view(-1, 81, 2)
        chi_per_aatype = chi_per_aatype / torch.linalg.vector_norm(chi_per_aatype + 1e-8, dim=-1)[..., None]
        psi_torsions, chi_per_aatype = chi_per_aatype.split([1, 80], dim=-2)
        chi_per_aatype = chi_per_aatype.view(-1, 20, 4, 2)

        rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)
        all_atom14 = compute_all_atom14(rigids, psi_torsions, chi_per_aatype)
        atom91 = torch.zeros((num_res, 91, 3), device=all_atom14.device)
        atom91[..., :4, :] = all_atom14[..., 0, :4, :]
        for i in range(20):
            aa = restype_1to3[restypes[i]] 
            start, end = atom91_start_end[aa]
            atom91[..., start:end, :] = all_atom14[..., i, 4:4+(end-start), :]
        
        atom91 = atom91 - rigids.get_trans()[..., None, :]
        
        seq_logits = self.seq_head(res_features.get_invariant_features(flat=True))

        out_dict = {}
        out_dict['decoded_latent'] = atom91 
        out_dict['decoded_seq_logits'] = seq_logits
        return out_dict
