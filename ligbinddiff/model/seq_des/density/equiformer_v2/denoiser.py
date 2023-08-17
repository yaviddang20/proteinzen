""" Denoising model """

import torch
from torch import nn
import numpy as np

from ligbinddiff.model.modules.equiformer_v2.so2_ops import Nodewise_SO3_Convolution
from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2
from ligbinddiff.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from ligbinddiff.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork, MultiResFeedForwardNetwork, TransBlockV2
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat

from ligbinddiff.utils.atom_reps import atom91_start_end
from ligbinddiff.utils.so3_embedding import type_l_to_so3, density_to_so3, so3_to_density


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


class DensityDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 node_lmax_list,
                 density_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 mappingReduced_nodes_condensed,
                 mappingReduced_density,
                 mappingReduced_super,
                 density_SO3_rotation,
                 node_SO3_rotation,
                 density_SO3_grid,
                 node_SO3_grid,
                 super_SO3_rotation,
                 num_heads=8,
                 density_channels=4,
                 h_channels=32,
                 norm_type="rms_norm_sh"
                 ):
        """
        Args
        ----
        """
        super().__init__()
        self.density_lmax_list = density_lmax_list
        self.node_lmax_list = node_lmax_list
        self.super_lmax_list = [max(l1, l2) for l1, l2 in zip(node_lmax_list, density_lmax_list)]

        self.mappingReduced_nodes = mappingReduced_nodes
        self.mappingReduced_nodes_condensed = mappingReduced_nodes_condensed
        self.mappingReduced_density = mappingReduced_density

        self.density_SO3_rotation = density_SO3_rotation
        self.node_SO3_rotation = node_SO3_rotation
        self.density_SO3_grid = density_SO3_grid
        self.node_SO3_grid = node_SO3_grid

        self.density_to_nodes = Nodewise_SO3_Convolution(
            sphere_channels=density_channels,
            m_output_channels=h_channels,
            lmax_list=self.super_lmax_list,
            mmax_list=self.super_lmax_list,
            mappingReduced=mappingReduced_super,
            SO3_rotation=super_SO3_rotation,
        )
        self.norm_node_update = NormSO3(
            lmax_list=node_lmax_list,
            num_channels=h_channels)

        self.node_ff = MultiResFeedForwardNetwork(
            sphere_channels=h_channels,
            hidden_channels=h_channels * 2,
            output_channels=h_channels,
            lmax_list=node_lmax_list,
            mmax_list=node_lmax_list.copy(),
            SO3_grid=node_SO3_grid
        )
        self.norm_node = NormSO3(
            lmax_list=node_lmax_list,
            num_channels=h_channels)

        condensed_h_channels = h_channels * len(self.node_lmax_list)

        self.attention = TransBlockV2(
            sphere_channels=condensed_h_channels,
            attn_hidden_channels=condensed_h_channels,
            num_heads=num_heads,
            attn_alpha_channels=condensed_h_channels // 2,
            attn_value_channels=condensed_h_channels // 4,
            ffn_hidden_channels=condensed_h_channels,
            output_channels=condensed_h_channels,
            lmax_list=node_lmax_list[0:1],
            mmax_list=node_lmax_list[0:1],
            SO3_rotation=node_SO3_rotation,
            SO3_grid=node_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_nodes_condensed
        )

        self.edge_update = EdgeUpdate(
            node_lmax_list,
            edge_channels_list,
            h_channels
        )

        self.nodes_to_density = Nodewise_SO3_Convolution(
            sphere_channels=h_channels,
            m_output_channels=density_channels,
            lmax_list=self.super_lmax_list,
            mmax_list=self.super_lmax_list,
            mappingReduced=mappingReduced_super,
            SO3_rotation=super_SO3_rotation
        )
        self.norm_density_update = NormSO3(
            lmax_list=density_lmax_list,
            num_channels=density_channels)

        self.density_ff = MultiResFeedForwardNetwork(
            sphere_channels=density_channels,
            hidden_channels=density_channels * 2,
            output_channels=density_channels,
            lmax_list=density_lmax_list,
            mmax_list=density_lmax_list.copy(),
            SO3_grid=density_SO3_grid
        )
        self.norm_density = NormSO3(
            lmax_list=density_lmax_list,
            num_channels=density_channels)


    def forward(
            self,
            density_features: SO3_Embedding,
            node_features: SO3_Embedding,
            edge_features: torch.Tensor,
            edge_index
    ):
        # update nodes from density
        density_features_super = density_features.to_resolutions(self.super_lmax_list, self.super_lmax_list)
        density_to_node_update = self.density_to_nodes(
            density_features_super,
            edge_features,
            edge_index
        )
        density_to_node_update = density_to_node_update.to_resolutions(self.node_lmax_list, self.node_lmax_list)
        norm_d2n_update = self.norm_node_update(density_to_node_update.embedding)
        node_features.embedding = node_features.embedding + norm_d2n_update
        node_update = self.norm_node(self.node_ff(node_features).embedding)
        node_features.embedding = node_features.embedding + node_update

        # transformer block
        node_features = node_features.condense_resolutions()
        # transformer block
        node_features = self.attention(
            node_features,
            edge_features,
            edge_index=edge_index
        )
        node_features = node_features.distribute_resolutions(len(self.node_lmax_list))

        # update edges
        edge_features = self.edge_update(
            node_features,
            edge_features,
            edge_index
        )

        # update density from nodes
        node_features_super = node_features.to_resolutions(self.super_lmax_list, self.super_lmax_list)
        density_update = self.nodes_to_density(
            node_features_super,
            edge_features,
            edge_index
        )

        density_update = density_update.to_resolutions(self.density_lmax_list, self.density_lmax_list)
        norm_density_update = self.norm_density_update(density_update.embedding)
        density_features.embedding = density_features.embedding + norm_density_update
        density_update_2 = self.norm_density(self.density_ff(density_features).embedding)
        density_features.embedding = density_features.embedding + density_update_2

        return density_features, node_features, edge_features


class SequenceHead(nn.Module):
    """ Layer to predict AA identity from density """
    def __init__(self,
                 density_channels,
                 mappingReduced_density,
                 density_lmax_list,
                 edge_channels_list,
                 h_dim,
                 SO3_rotation,
                 num_aa=20):
        super().__init__()
        self.collapse = Nodewise_SO3_Convolution(
            sphere_channels=density_channels,
            m_output_channels=1,
            lmax_list=density_lmax_list,
            mmax_list=density_lmax_list,
            mappingReduced=mappingReduced_density,
            edge_channels_list=edge_channels_list,
            extra_m0_output_channels=h_dim,
            SO3_rotation=SO3_rotation
        )
        self.norm = nn.LayerNorm(h_dim)
        self.transition = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)
        )
        self.out = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_aa)
        )

    def forward(self, density_features, edge_features, edge_index):
        _, density_scalars = self.collapse(density_features, edge_features, edge_index)
        hidden = self.transition(density_scalars.squeeze(-1))
        hidden = self.norm(hidden)
        prelogits = self.out(hidden)
        logits = torch.log_softmax(prelogits, dim=-1)
        return logits, hidden


class Atom91Head(nn.Module):
    """ Layer to predict sidechain atom positions from density """
    def __init__(self,
                 density_channels,
                 mappingReduced_density,
                 mappingReduced_atoms,
                 density_lmax_list,
                 edge_channels_list,
                 density_SO3_rotation,
                 density_SO3_grid,
                 atom_SO3_rotation,
                 atom_SO3_grid,
                 scalar_h_dim,
                 num_atoms=91,
                 num_heads=8,
                 num_layers=3,
                 h_channels=32):
        super().__init__()
        self.num_atoms = num_atoms
        self.density_lmax_list = density_lmax_list

        self.atom_lmax_list = [1]
        output_lmax_list = [1] + [0 for _ in density_lmax_list[1:]]
        self.super_lmax_list = [max(l1, l2) for l1, l2 in zip(output_lmax_list, density_lmax_list)]
        assert self.super_lmax_list == self.density_lmax_list  # idek how this would not be the case...

        self.density_SO3_rotation = density_SO3_rotation
        self.density_SO3_grid = density_SO3_grid
        self.atom_SO3_rotation = atom_SO3_rotation
        self.atom_SO3_grid = atom_SO3_grid

        self.mappingReduced_density = mappingReduced_density
        self.mappingReduced_atoms = mappingReduced_atoms

        self.fuse_seq_to_density = nn.Linear(
            density_channels + scalar_h_dim, density_channels
        )

        self.collapse = Nodewise_SO3_Convolution(
            sphere_channels=density_channels,
            m_output_channels=num_atoms,
            lmax_list=density_lmax_list,
            mmax_list=density_lmax_list,
            mappingReduced=mappingReduced_density,
            edge_channels_list=edge_channels_list,
            SO3_rotation=density_SO3_rotation
        )
        self.collapse_norm = NormSO3(
            lmax_list=self.density_lmax_list,
            num_channels=num_atoms
        )

        self.refine = nn.ModuleList(
            [
                TransBlockV2(
                    sphere_channels=num_atoms,
                    attn_hidden_channels=h_channels,
                    num_heads=num_heads,
                    attn_alpha_channels=h_channels // 2,
                    attn_value_channels=h_channels // 4,
                    ffn_hidden_channels=h_channels,
                    output_channels=num_atoms,
                    lmax_list=self.atom_lmax_list,
                    mmax_list=self.atom_lmax_list,
                    SO3_rotation=self.atom_SO3_rotation,
                    SO3_grid=self.atom_SO3_grid,
                    edge_channels_list=edge_channels_list,
                    mappingReduced=mappingReduced_atoms
                )
                for _ in range(num_layers)
            ]
        )
        # self.refine = nn.ModuleList(
        #     [
        #         FeedForwardNetwork(
        #             sphere_channels=num_atoms,
        #             hidden_channels=num_atoms*2,
        #             output_channels=num_atoms,
        #             lmax_list=self.atom_lmax_list,
        #             mmax_list=self.atom_lmax_list,
        #             SO3_grid=self.atom_SO3_grid
        #         )
        #         for _ in range(num_layers)
        #     ]
        # )
        self.proj = SO3_LinearV2(
            num_atoms,
            num_atoms,
            lmax=max(self.atom_lmax_list)
        )


    def forward(self, density_features, seq_features, edge_features, edge_index):
        # fuse seq features into density features
        num_l0 = len(self.density_lmax_list)
        density_l0 = density_features.get_invariant_features()
        seq_features = seq_features.unsqueeze(-2).expand(-1, num_l0, -1)
        fused_features = torch.cat([density_l0, seq_features], dim=-1)
        density_l0 = self.fuse_seq_to_density(fused_features)
        density_features.set_invariant_features(density_l0)

        # collapse density to atoms
        density_collapsed = self.collapse(density_features, edge_features, edge_index)
        density_collapsed.embedding = self.collapse_norm(density_collapsed.embedding)
        atoms = SO3_Embedding(
            0,
            lmax_list=self.atom_lmax_list,
            num_channels=self.num_atoms,
            dtype=density_features.dtype,
            device=density_features.device
        )
        atom_subset_index = int((self.atom_lmax_list[0] + 1) ** 2)
        atom_subset = density_collapsed.embedding[:, :atom_subset_index]

        cumsums = []
        for start, end in atom91_start_end.values():
            chunk = atom_subset[..., start:end]
            chunk_cumsum = torch.cumsum(chunk, dim=-1)
            cumsums.append(chunk_cumsum)
        atom_subset[..., 4:] = torch.cat(cumsums, dim=-1)

        atoms.set_embedding(atom_subset)
        # print(atoms.embedding.shape, edge_features.shape)

        # refine atoms
        for layer in self.refine:
            # atoms.embedding = atoms.embedding + layer(atoms).embedding
            atoms = layer(atoms, edge_features, edge_index)

        return self.proj(atoms)


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


class DensityDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 node_in_channels,
                 node_in_lmax_list,
                 node_lmax_list,
                 density_lmax_list,
                 edge_channels_list,
                 num_heads=8,
                 density_channels=4,
                 h_channels=32,
                 h_time=64,
                 scalar_h_dim=128,
                 n_layers=4,
                 detach_aux_grads=False,#True,
                 device='cpu'):
        super().__init__()

        super_lmax_list = [max(l1, l2) for l1, l2 in zip(node_lmax_list, density_lmax_list)]
        atoms_lmax_list = [1]
        self.density_lmax_list = density_lmax_list
        self.node_lmax_list = node_lmax_list
        self.h_channels = h_channels
        self.detach_aux_grads = detach_aux_grads

        mappingReduced_density = CoefficientMappingModule(density_lmax_list, density_lmax_list)
        mappingReduced_nodes = CoefficientMappingModule(node_lmax_list, node_lmax_list)
        mappingReduced_nodes_condensed = CoefficientMappingModule(node_lmax_list[0:1], node_lmax_list[0:1])
        mappingReduced_super = CoefficientMappingModule(super_lmax_list, super_lmax_list)
        mappingReduced_atoms = CoefficientMappingModule(atoms_lmax_list, atoms_lmax_list)
        self.mappingReduced_density = mappingReduced_density
        self.mappingReduced_nodes = mappingReduced_nodes
        self.mappingReduced_nodes_condensed = mappingReduced_nodes_condensed
        self.mappingReduced_super = mappingReduced_super
        self.mappingReduced_atoms = mappingReduced_atoms

        self.density_SO3_rotation_list = nn.ModuleList()
        self.node_SO3_rotation_list = nn.ModuleList()
        self.super_SO3_rotation_list = nn.ModuleList()
        for lmax in density_lmax_list:
            self.density_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in node_lmax_list:
            self.node_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in super_lmax_list:
            self.super_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )

        self.density_SO3_grid_list = nn.ModuleList()
        self.node_SO3_grid_list = nn.ModuleList()
        for l in range(max(density_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(density_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            self.density_SO3_grid_list.append(SO3_m_grid)

        for l in range(max(node_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(node_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            self.node_SO3_grid_list.append(SO3_m_grid)

        self.atom_SO3_rotation_list = nn.ModuleList(
            [SO3_Rotation(lmax=1)]
        )
        self.atom_SO3_grid_list = nn.ModuleList()
        for l in range(max(atoms_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(atoms_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            self.atom_SO3_grid_list.append(SO3_m_grid)
        # print("node", self.node_SO3_grid_list)
        # print("density", self.density_SO3_grid_list)
        # print("atom", self.atom_SO3_grid_list)

        self.embed_node = Nodewise_SO3_Convolution(
            sphere_channels=node_in_channels,
            m_output_channels=h_channels,
            lmax_list=node_lmax_list,
            mmax_list=node_lmax_list,
            mappingReduced=mappingReduced_nodes,
            edge_channels_list=edge_channels_list,
            SO3_rotation=self.node_SO3_rotation_list
        )

        self.h_time = h_time
        self.time_rbf = RBF(n_basis=h_time//2)
        self.time_mlp = nn.Sequential(
            nn.Linear(h_time, scalar_h_dim),
            nn.ReLU(),
            nn.Linear(scalar_h_dim, h_time),
            nn.ReLU()
        )

        self.embed_time = nn.Linear(
            h_channels + h_time, h_channels
        )


        self.denoiser = nn.ModuleList([
            DensityDenoisingLayer(
                node_lmax_list=node_lmax_list,
                density_lmax_list=density_lmax_list,
                edge_channels_list=edge_channels_list,
                mappingReduced_nodes=mappingReduced_nodes,
                mappingReduced_nodes_condensed=mappingReduced_nodes_condensed,
                mappingReduced_density=mappingReduced_density,
                mappingReduced_super=mappingReduced_super,
                density_SO3_rotation=self.density_SO3_rotation_list,
                density_SO3_grid=self.density_SO3_grid_list,
                node_SO3_rotation=self.node_SO3_rotation_list,
                node_SO3_grid=self.node_SO3_grid_list,
                super_SO3_rotation=self.super_SO3_rotation_list,
                num_heads=num_heads,
                density_channels=density_channels,
                h_channels=h_channels)
            for _ in range(n_layers)
        ])

        self.seq_head = SequenceHead(
            density_channels * 2,
            mappingReduced_density,
            density_lmax_list,
            edge_channels_list,
            scalar_h_dim,
            SO3_rotation=self.density_SO3_rotation_list
        )

        self.atom91 = Atom91Head(
            density_channels * 2,
            mappingReduced_density,
            mappingReduced_atoms,
            density_lmax_list,
            edge_channels_list,
            self.density_SO3_rotation_list,
            self.density_SO3_grid_list,
            self.atom_SO3_rotation_list,
            self.atom_SO3_grid_list,
            scalar_h_dim=scalar_h_dim,
            h_channels=h_channels
        )


    def forward(self, graph):
        ## prep features
        node_features = {
            0: graph['bb_s'].unsqueeze(-1),
            1: graph['bb_v']
        }
        node_features = type_l_to_so3(node_features)
        edge_features = graph['edge_s']
        density_features = density_to_so3(graph['noised_density'])
        ts = graph['t']

        # init SO3_rotation and SO3_grid
        edge_index = graph.edge_index
        X_ca = graph['x']
        X_cb = graph['x_cb']
        node_edge_distance_vec = X_ca[edge_index[1]] - X_ca[edge_index[0]]
        density_edge_distance_vec = X_cb[edge_index[1]] - X_cb[edge_index[0]]
        node_edge_rot_mat = init_edge_rot_mat(node_edge_distance_vec)
        density_edge_rot_mat = init_edge_rot_mat(density_edge_distance_vec)
        for rot in self.density_SO3_rotation_list:
            rot.set_wigner(density_edge_rot_mat)
        for rot in self.node_SO3_rotation_list:
            rot.set_wigner(node_edge_rot_mat)
        for rot in self.super_SO3_rotation_list:
            rot.set_wigner(node_edge_rot_mat)
        for rot in self.atom_SO3_rotation_list:
            rot.set_wigner(density_edge_rot_mat)

        # embed node features
        node_features = node_features.to_resolutions(self.node_lmax_list, self.node_lmax_list)
        node_features = self.embed_node(node_features, edge_features, edge_index)

        ## create time embedding
        fourier_time = self.time_rbf(ts)  # (h_time,)
        num_nodes = node_features.embedding.shape[0]
        embedded_time = self.time_mlp(fourier_time.unsqueeze(0))  # (1 x h_time)
        embedded_time = fourier_time.expand(num_nodes, -1)

        # fuse time embedding into node features
        node_num_l0 = len(self.node_lmax_list)
        node_l0 = node_features.get_invariant_features()  # n_res x (node_num_m0 x h_channels)
        time_expanded = embedded_time.unsqueeze(-2).expand(-1, node_num_l0, -1)
        node_l0 = self.embed_time(
            torch.cat([node_l0, time_expanded], dim=-1)
        )
        node_features.set_invariant_features(node_l0)

        ## denoising
        f_D = density_features
        f_E = edge_features
        f_V = node_features

        for layer in self.denoiser:
            f_D, f_V, f_E = layer(f_D, f_V, f_E, edge_index)

        ## aux heads
        density = f_D.clone()
        if self.detach_aux_grads:
            f_D.embedding = f_D.embedding.detach()
            f_E = f_E.detach()

        # denoising assistance formulation as inspired by https://arxiv.org/abs/2306.09192
        density_da_features = SO3_Embedding(
            0,
            density_features.lmax_list,
            density_features.num_channels * 2,
            density_features.device,
            density_features.dtype
        )
        density_da_features.set_embedding(
            torch.cat([density_features.embedding, f_D.embedding], dim=-1)
        )
        seq_logits, seq_features = self.seq_head(
            density_da_features,
            f_E,
            edge_index
        )
        atom91 = self.atom91(
            density_da_features,
            seq_features,
            f_E,
            edge_index
        )

        return {
            "density": so3_to_density(density),
            "seq_logits": seq_logits,
            "atom91": atom91.embedding[..., 1:, :].transpose(-1, -2)
        }
