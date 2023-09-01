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


class SuperpositionDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain superposition """
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 mappingReduced_atoms,
                 mappingReduced_super,
                 atom_SO3_rotation,
                 node_SO3_rotation,
                 atom_SO3_grid,
                 node_SO3_grid,
                 super_SO3_rotation,
                 atom_lmax_list=[1],
                 atom_channels=91,
                 num_heads=8,
                 h_channels=32,
                 norm_type="rms_norm_sh"
                 ):
        """
        Args
        ----
        """
        super().__init__()
        self.atom_lmax_list = atom_lmax_list
        self.node_lmax_list = node_lmax_list
        self.super_lmax_list = [max(l1, l2) for l1, l2 in zip(node_lmax_list, atom_lmax_list)]

        self.mappingReduced_nodes = mappingReduced_nodes
        self.mappingReduced_atoms = mappingReduced_atoms

        self.atom_SO3_rotation = atom_SO3_rotation
        self.node_SO3_rotation = node_SO3_rotation
        self.atom_SO3_grid = atom_SO3_grid
        self.node_SO3_grid = node_SO3_grid

        self.atoms_to_nodes = Nodewise_SO3_Convolution(
            sphere_channels=atom_channels,
            m_output_channels=h_channels,
            lmax_list=self.super_lmax_list,
            mmax_list=self.super_lmax_list,
            mappingReduced=mappingReduced_super,
            SO3_rotation=super_SO3_rotation,
        )
        self.norm_node_update = NormSO3(
            lmax_list=node_lmax_list,
            num_channels=h_channels)

        self.node_ff = FeedForwardNetwork(
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

        self.attention = TransBlockV2(
            sphere_channels=h_channels,
            attn_hidden_channels=h_channels,
            num_heads=num_heads,
            attn_alpha_channels=h_channels // 2,
            attn_value_channels=h_channels // 4,
            ffn_hidden_channels=h_channels,
            output_channels=h_channels,
            lmax_list=node_lmax_list,
            mmax_list=node_lmax_list,
            SO3_rotation=node_SO3_rotation,
            SO3_grid=node_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_nodes
        )

        self.edge_update = EdgeUpdate(
            node_lmax_list,
            edge_channels_list,
            h_channels
        )

        self.nodes_to_atoms = Nodewise_SO3_Convolution(
            sphere_channels=h_channels,
            m_output_channels=atom_channels,
            lmax_list=self.super_lmax_list,
            mmax_list=self.super_lmax_list,
            mappingReduced=mappingReduced_super,
            SO3_rotation=super_SO3_rotation
        )
        self.norm_atom_update = NormSO3(
            lmax_list=atom_lmax_list,
            num_channels=atom_channels)

        self.atom_ff = FeedForwardNetwork(
            sphere_channels=atom_channels,
            hidden_channels=atom_channels * 2,
            output_channels=atom_channels,
            lmax_list=atom_lmax_list,
            mmax_list=atom_lmax_list.copy(),
            SO3_grid=atom_SO3_grid
        )
        self.norm_atom = NormSO3(
            lmax_list=atom_lmax_list,
            num_channels=atom_channels)


    def forward(
            self,
            atom_features: SO3_Embedding,
            node_features: SO3_Embedding,
            edge_features: torch.Tensor,
            edge_index
    ):
        # print("atom_features", atom_features.embedding)
        # update nodes from atom
        atom_features_super = atom_features.to_resolutions(self.super_lmax_list, self.super_lmax_list)
        atom_to_node_update = self.atoms_to_nodes(
            atom_features_super,
            edge_features,
            edge_index
        )
        # print("atom_to_node_update", atom_to_node_update.embedding)
        atom_to_node_update = atom_to_node_update.to_resolutions(self.node_lmax_list, self.node_lmax_list)
        norm_d2n_update = self.norm_node_update(atom_to_node_update.embedding)
        # print("norm_d2n_node_update", norm_d2n_update)
        node_features.embedding = node_features.embedding + norm_d2n_update
        node_update = self.norm_node(self.node_ff(node_features).embedding)
        # print("norm_node_update", node_update)
        node_features.embedding = node_features.embedding + node_update

        # transformer block
        node_features = self.attention(
            node_features,
            edge_features,
            edge_index=edge_index
        )
        # print("node_attn", node_features.embedding)

        # update edges
        edge_features = self.edge_update(
            node_features,
            edge_features,
            edge_index
        )
        # print("edge_update", edge_features)

        # update atom from nodes
        node_features_super = node_features.to_resolutions(self.super_lmax_list, self.super_lmax_list)
        atom_update = self.nodes_to_atoms(
            node_features_super,
            edge_features,
            edge_index
        )
        # print("node_to_atoms_update", atom_update.embedding)

        atom_update = atom_update.to_resolutions(self.atom_lmax_list, self.atom_lmax_list)
        norm_atom_update = self.norm_atom_update(atom_update.embedding)
        # print("norm_atom_update", norm_atom_update)
        atom_features.embedding = atom_features.embedding + norm_atom_update
        atom_update_2 = self.norm_atom(self.atom_ff(atom_features).embedding)
        # print("atom_update_2", norm_atom_update)
        atom_features.embedding = atom_features.embedding + atom_update_2

        return atom_features, node_features, edge_features


class SequenceHead(nn.Module):
    """ Layer to predict AA identity from atom """
    def __init__(self,
                 atom_channels,
                 mappingReduced_atom,
                 atom_lmax_list,
                 edge_channels_list,
                 h_dim,
                 SO3_rotation,
                 num_aa=20):
        super().__init__()
        self.collapse = Nodewise_SO3_Convolution(
            sphere_channels=atom_channels,
            m_output_channels=1,
            lmax_list=atom_lmax_list,
            mmax_list=atom_lmax_list,
            mappingReduced=mappingReduced_atom,
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

    def forward(self, atom_features, edge_features, edge_index):
        _, atom_scalars = self.collapse(atom_features, edge_features, edge_index)
        hidden = self.transition(atom_scalars.squeeze(-1))
        hidden = self.norm(hidden)
        prelogits = self.out(hidden)
        logits = torch.log_softmax(prelogits, dim=-1)
        return logits


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


class SuperpositionDenoiser(nn.Module):
    """ Denoising model on sidechain superposition """
    def __init__(self,
                 node_in_channels,
                 node_lmax_list,
                 edge_channels_list,
                 atom_lmax_list=[1],
                 atom_channels=91,
                 num_heads=8,
                 h_channels=32,
                 h_time=64,
                 scalar_h_dim=128,
                 n_layers=4,
                 detach_aux_grads=True,
                 device='cpu'):
        super().__init__()

        super_lmax_list = [max(l1, l2) for l1, l2 in zip(node_lmax_list, atom_lmax_list)]
        self.atom_lmax_list = atom_lmax_list
        self.node_lmax_list = node_lmax_list
        self.h_channels = h_channels
        self.atom_channels = atom_channels
        self.detach_aux_grads = detach_aux_grads

        mappingReduced_atoms = CoefficientMappingModule(atom_lmax_list, atom_lmax_list)
        mappingReduced_nodes = CoefficientMappingModule(node_lmax_list, node_lmax_list)
        mappingReduced_super = CoefficientMappingModule(super_lmax_list, super_lmax_list)
        self.mappingReduced_atoms = mappingReduced_atoms
        self.mappingReduced_nodes = mappingReduced_nodes
        self.mappingReduced_super = mappingReduced_super

        self.atom_SO3_rotation_list = nn.ModuleList()
        self.node_SO3_rotation_list = nn.ModuleList()
        self.super_SO3_rotation_list = nn.ModuleList()
        for lmax in atom_lmax_list:
            self.atom_SO3_rotation_list.append(
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

        self.atom_SO3_grid_list = nn.ModuleList()
        self.node_SO3_grid_list = nn.ModuleList()
        for l in range(max(atom_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(atom_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            self.atom_SO3_grid_list.append(SO3_m_grid)

        for l in range(max(node_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(node_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            self.node_SO3_grid_list.append(SO3_m_grid)


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
            SuperpositionDenoisingLayer(
                node_lmax_list=node_lmax_list,
                atom_lmax_list=atom_lmax_list,
                edge_channels_list=edge_channels_list,
                mappingReduced_nodes=mappingReduced_nodes,
                mappingReduced_atoms=mappingReduced_atoms,
                mappingReduced_super=mappingReduced_super,
                atom_SO3_rotation=self.atom_SO3_rotation_list,
                atom_SO3_grid=self.atom_SO3_grid_list,
                node_SO3_rotation=self.node_SO3_rotation_list,
                node_SO3_grid=self.node_SO3_grid_list,
                super_SO3_rotation=self.super_SO3_rotation_list,
                num_heads=num_heads,
                atom_channels=atom_channels,
                h_channels=h_channels)
            for _ in range(n_layers)
        ])

        # self.seq_head = SequenceHead(
        #     atom_channels * 2,
        #     mappingReduced_atoms,
        #     atom_lmax_list,
        #     edge_channels_list,
        #     scalar_h_dim,
        #     SO3_rotation=self.atom_SO3_rotation_list
        # )

        self.seq_head = nn.Sequential(
            nn.LayerNorm(atom_channels),
            nn.Linear(atom_channels, 20),
            nn.LogSoftmax(dim=-1)
        )


    def forward(self, graph):
        ## prep features
        num_nodes = graph['x'].shape[0]
        node_features = {
            0: graph['bb_s'].unsqueeze(-1),
            1: graph['bb_v']
        }
        node_features = type_l_to_so3(node_features)

        edge_features = graph['edge_s']

        # atom_features = graph['noised_atom91']
        atom_features = SO3_Embedding(
            num_nodes,
            lmax_list=self.atom_lmax_list,
            num_channels=self.atom_channels,
            device=graph['x'].device,
            dtype=torch.float
        )
        atom_features.embedding[:, 1:4] = graph['noised_atom91'].transpose(-1, -2)
        # print("atom embed")
        # print(atom_features.embedding)

        ts = graph['t']  # (B, 1, 1)

        # init SO3_rotation and SO3_grid
        edge_index = graph.edge_index
        X_ca = graph['x']
        edge_distance_vec = X_ca[edge_index[1]] - X_ca[edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for rot in self.atom_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.node_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)
        for rot in self.super_SO3_rotation_list:
            rot.set_wigner(edge_rot_mat)

        # embed node features
        node_features = node_features.to_resolutions(self.node_lmax_list, self.node_lmax_list)
        node_features = self.embed_node(node_features, edge_features, edge_index)
        # print("node embed")
        # print(node_features.embedding)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (B x h_time,)
        embedded_time = self.time_mlp(fourier_time)  # (B x h_time)
        data_splits = graph._slice_dict['x']
        data_lens = data_splits[1:] - data_splits[:-1]
        embedded_time = torch.cat([
            embedded_time[i].view(1, -1).expand(l, -1) for i, l in enumerate(data_lens)
        ])  # n_res x h_time

        # fuse time embedding into node features
        node_num_l0 = len(self.node_lmax_list)
        node_l0 = node_features.get_invariant_features()  # n_res x (node_num_m0 x h_channels)
        time_expanded = embedded_time.unsqueeze(-2).expand(-1, node_num_l0, -1)
        node_l0 = self.embed_time(
            torch.cat([node_l0, time_expanded], dim=-1)
        )
        node_features.set_invariant_features(node_l0)

        # print("time embed", node_features.embedding)

        ## denoising
        f_S = atom_features
        f_E = edge_features
        f_V = node_features
        # print("denoise", f_S.embedding, f_E, f_V.embedding)

        for layer in self.denoiser:
            f_S, f_V, f_E = layer(f_S, f_V, f_E, edge_index)
            # print("layer", f_S.embedding)

        ## aux heads
        atom91 = f_S.clone()
        if self.detach_aux_grads:
            f_S.embedding = f_S.embedding.detach()
            f_E = f_E.detach()

        # denoising assistance formulation as inspired by https://arxiv.org/abs/2306.09192
        atom_da_features = SO3_Embedding(
            0,
            atom_features.lmax_list,
            atom_features.num_channels * 2,
            atom_features.device,
            atom_features.dtype
        )
        atom_da_features.set_embedding(
            torch.cat([atom_features.embedding, f_S.embedding], dim=-1)
        )
        seq_logits = self.seq_head(f_S.get_invariant_features(flat=True))
        # seq_logits = self.seq_head(
        #     atom_da_features,
        #     f_E,
        #     edge_index
        # )

        graph['denoised_atom91'] = atom91.embedding[..., 1:, :].transpose(-1, -2)
        graph['seq_logits'] = seq_logits
        return graph
        # return {
        #     "atom91_centered": atom91.embedding[..., 1:, :].transpose(-1, -2),
        #     "seq_logits": seq_logits
        # }
