""" Molecule denoiser """

import torch
from torch import nn
import numpy as np
from torch_cluster import knn, knn_graph
from proteinzen.data.datasets.featurize.common import _edge_positional_embeddings, _rbf
from proteinzen.model.modules.common import ProjectLayer
from proteinzen.model.modules.openfold.layers import Linear

from proteinzen.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid
from proteinzen.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from proteinzen.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork, MultiResFeedForwardNetwork, TransBlockV2
from proteinzen.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from proteinzen.utils.so3_embedding import type_l_to_so3

from proteinzen.model.modules.common import EdgeUpdate


class RefineLayer(nn.Module):
    def __init__(self,
                 lmax,
                 n_channels,
                 edge_hidden,
                 SO3_rotation,
                 SO3_grid,
                 mappingReduced,
                 update_edge=False,
                 n_heads=4,
                 n_qk=8,
                 n_v=12):
        super().__init__()
        self.lmax_list = [lmax]
        self.n_channels = n_channels

        self.transformer = TransBlockV2(
            sphere_channels=n_channels,
            attn_hidden_channels=n_channels // n_heads,
            num_heads=n_heads,
            attn_alpha_channels=n_qk,
            attn_value_channels=n_v,
            ffn_hidden_channels=n_channels,
            output_channels=n_channels,
            lmax_list=self.lmax_list,
            mmax_list=self.lmax_list,
            SO3_rotation=SO3_rotation,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            edge_channels_list=[edge_hidden, edge_hidden],
        )
        self.pos_update = Linear(
            n_channels,
            n_channels,
            init='final'
        )
        self.update_edge = update_edge
        if self.update_edge:
            self.edge_update = EdgeUpdate(
                self.lmax_list,
                edge_channels_list=[edge_hidden, edge_hidden],
                h_channels=n_channels
            )


    def forward(self,
                atom_features: SO3_Embedding,
                edge_features: torch.Tensor,
                edge_index: torch.Tensor,
                atom_mask=None):

        if atom_mask is None:
            atom_mask = torch.ones(atom_features.embedding.shape[0], device=atom_features.device)

        atom_features.embedding = atom_features.embedding * atom_mask[..., None, None]
        atom_features = self.transformer(
            atom_features,
            edge_features,
            edge_index
        )
        if self.update_edge:
            edge_features = self.edge_update(
                atom_features,
                edge_features,
                edge_index
            )

        return atom_features, edge_features


class PosUpdate(nn.Module):
    def __init__(self,
                 n_channels):
        super().__init__()
        self.n_channels = n_channels

        self.pos_update = Linear(
            n_channels,
            n_channels,
            init='final'
        )

    def forward(self,
                atom_pos: torch.Tensor,
                atom_features: SO3_Embedding,
                atom_mask: torch.Tensor):
        atom_vecs = atom_features.embedding[:, 1:4].transpose(-1, -2)
        atom_scalars = atom_features.get_invariant_features(flat=True)
        vec_gate = self.pos_update(atom_scalars)
        atom_pos_update = torch.sum(atom_vecs * vec_gate[..., None], dim=-2)
        atom_pos = atom_pos + atom_pos_update * atom_mask[..., None]

        return atom_pos


class MoleculeDenoiser(nn.Module):
    def __init__(self,
                 lmax=1,
                 n_channels=32,
                 edge_hidden=32,
                 n_rbf=32,
                 n_heads=4,
                 n_qk=8,
                 n_v=12,
                 n_atom_in=43,
                 n_bond_in=7-4,
                 n_layers=4,
                 k=10,
                 self_conditioning=False):
        super().__init__()
        self.lmax = lmax
        self.lmax_list = [lmax]
        self.n_channels = n_channels
        self.edge_hidden = edge_hidden
        self.n_rbf = n_rbf
        self.self_conditioning = self_conditioning

        self.spatial_SO3_rotation = nn.ModuleList()
        self.bond_SO3_rotation = nn.ModuleList()
        for lmax in self.lmax_list:
            self.spatial_SO3_rotation.append(
                SO3_Rotation(lmax)
            )
        for lmax in self.lmax_list:
            self.bond_SO3_rotation.append(
                SO3_Rotation(lmax)
            )

        self.spatial_SO3_grid = nn.ModuleList()
        self.bond_SO3_grid = nn.ModuleList()
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            self.spatial_SO3_grid.append(SO3_m_grid)
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            self.bond_SO3_grid.append(SO3_m_grid)

        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.lmax_list)

        self.embed_atoms = nn.Sequential(
            nn.Linear(n_atom_in, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels),
            nn.ReLU(),
            nn.Linear(n_channels, n_channels),
            nn.LayerNorm(n_channels)
        )
        self.embed_bonds = nn.Sequential(
            nn.Linear(n_bond_in + n_rbf, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.LayerNorm(edge_hidden)
        )

        self.refine_bonds = nn.ModuleList(
            [
                RefineLayer(
                    lmax=lmax,
                    n_channels=n_channels,
                    edge_hidden=edge_hidden,
                    SO3_rotation=self.bond_SO3_rotation,
                    SO3_grid=self.bond_SO3_grid,
                    mappingReduced=self.mappingReduced,
                    update_edge=True,
                    n_heads=n_heads,
                    n_qk=n_qk,
                    n_v=n_v
                )
                for _ in range(n_layers)
            ]
        )

        self.embed_spatial_edge = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_rbf, edge_hidden),
                    nn.ReLU(),
                    nn.Linear(edge_hidden, edge_hidden),
                    nn.ReLU(),
                    nn.Linear(edge_hidden, edge_hidden),
                    nn.LayerNorm(edge_hidden)
                )
                for _ in range(n_layers)
            ]
        )

        self.refine_spatial = nn.ModuleList(
            [
                RefineLayer(
                    lmax=lmax,
                    n_channels=n_channels,
                    edge_hidden=edge_hidden,
                    SO3_rotation=self.spatial_SO3_rotation,
                    SO3_grid=self.spatial_SO3_grid,
                    mappingReduced=self.mappingReduced,
                    update_edge=False,
                    n_heads=n_heads,
                    n_qk=n_qk,
                    n_v=n_v
                )
                for _ in range(n_layers)
            ]
        )
        self.pos_update = nn.ModuleList(
            [
                PosUpdate(
                    n_channels=n_channels,
                )
                for _ in range(n_layers)
            ]
        )

        self.k = k


    @torch.no_grad()
    def _prep_features(self, graph, eps=1e-8):
        print(graph)
        atom_data = graph['ligand']
        bond_data = graph['ligand', 'bonds', 'ligand']

        atom_pos = atom_data['noised_atom_pos']

        atom_features = [
            atom_data[key].view(atom_data.num_nodes, -1)
            for key in (
                "atom_period",                 # N, categorical
                "atom_row",                       # N, categorical
                "atom_chirality",           # N, ordinal
                "atom_hybridization",   # N, categorical
                "atom_implicit_hs",       # N, ordinal
                "atom_formal_charge",   # N, ordinal
                "atom_aromatic",             # N, bool
                "atom_ring_props",         # N x 7, 1-6 bool, 7 ordinal
                "atom_degree",                 # N, ordinal
        )]

        atom_features = torch.cat(atom_features, dim=-1).float()

        bond_edge_index = bond_data.edge_index
        bond_features = [
            bond_data[key].view(bond_edge_index.shape[-1], -1) for key in (
                "bond_order",                   # E, ordinal
                "bond_aromatic",             # E, bool
                # "bond_type",                     # E, categorical
                "bond_conjugated",         # E, bool
        )]
        bond_len = torch.linalg.vector_norm(
            atom_pos[bond_edge_index[0]] - atom_pos[bond_edge_index[1]] + eps,
            dim=-1
        )
        bond_len_rbf = _rbf(bond_len, D_min=0., D_max=3., D_count=self.n_rbf, device=bond_len.device)
        bond_features.append(bond_len_rbf)

        bond_features = torch.cat(bond_features, dim=-1).float()


        return atom_pos, atom_features, bond_features, bond_edge_index,

    def _gen_spatial_edges(self, atom_pos, batch):
        spatial_edge_index = knn_graph(atom_pos, self.k, batch=batch)
        edge_vecs = atom_pos[spatial_edge_index[0]] - atom_pos[spatial_edge_index[1]]
        print(edge_vecs)
        edge_rot_mats = init_edge_rot_mat(edge_vecs)

        spatial_edge_len = torch.linalg.vector_norm(edge_vecs + eps, dim=-1)
        spatial_edge_features = _rbf(spatial_edge_len, D_min=0., D_max=10., D_count=self.n_rbf, device=spatial_edge_index.device)

        return spatial_edge_features, spatial_edge_index, edge_rot_mats


    def forward(self, data, self_condition=None):
        atom_pos, atom_scalars, bond_features, bond_edge_index = self._prep_features(data)

        atom_scalars = self.embed_atoms(atom_scalars)
        atom_features = SO3_Embedding(
            atom_scalars.shape[0],
            lmax_list=self.lmax_list,
            num_channels=self.n_channels,
            device=atom_scalars.device,
            dtype=atom_scalars.dtype
        )
        atom_features.set_invariant_features(atom_scalars)
        bond_features = self.embed_bonds(bond_features)

        for bond_layer, embed_spatial_edge, spatial_layer, pos_update in zip(
            self.refine_bonds, self.embed_spatial_edge, self.refine_spatial, self.pos_update
        ):
            print(atom_pos[bond_edge_index[0]] - atom_pos[bond_edge_index[1]])
            bond_rot_mats = init_edge_rot_mat(
                atom_pos[bond_edge_index[0]] - atom_pos[bond_edge_index[1]]
            )
            for rot in self.bond_SO3_rotation:
                rot.set_wigner(bond_rot_mats)
            atom_features, bond_features = bond_layer(atom_features, bond_features, bond_edge_index)

            spatial_edges, spatial_edge_index, spatial_edge_rot_mats = self._gen_spatial_edges(atom_pos, data['ligand'].batch)
            for rot in self.spatial_SO3_rotation:
                rot.set_wigner(spatial_edge_rot_mats)
            spatial_edge_features = embed_spatial_edge(spatial_edges)

            atom_features, _ = spatial_layer(atom_features, spatial_edge_features, spatial_edge_index)

            atom_pos = pos_update(atom_pos, atom_features)

        return {
            "pred_atom_pos": atom_pos
        }
