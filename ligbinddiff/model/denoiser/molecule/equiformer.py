""" Molecule denoiser """

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

from ligbinddiff.model.modules.common import EdgeUpdate


class RefineLayer(nn.Module):
    def __init__(self,
                 lmax,
                 n_channels):
        self.lmax_list = [lmax]
        self.n_channels = n_channels

        self.transformer = TransBlockV2(
            sphere_channels,
            attn_hidden_channels,
            num_heads,
            attn_alpha_channels,
            attn_value_channels,
            ffn_hidden_channels,
            output_channels,
            lmax_list,
            mmax_list,
            SO3_rotation,
            mappingReduced,
            SO3_grid,
            edge_channels_list,
        )

    def forward(self,
                atom_pos,
                atom_features,
                bond_features,
                bond_edge_index,
                atom_mask):


class MoleculeDenoiser(nn.Module):
    def __init__(self,
                 lmax,
                 n_channels):
        pass

    @torch.no_grad()
    def _prep_features(self, graph, eps=1e-8):
        atom_data = graph['ligand']
        bond_data = graph['ligand', 'bonds', 'ligand']
