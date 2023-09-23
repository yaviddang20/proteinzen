""" Denoising model """

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_cluster import knn_graph
from torch_geometric.utils import sort_edge_index

from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2
from ligbinddiff.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from ligbinddiff.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork, MultiResFeedForwardNetwork, TransBlockV2
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat

from ligbinddiff.data.datasets.featurize.sidechain import _rbf, _positional_embeddings


def sample_inv_cubic_edges(batched_X_ca, batched_x_mask, batch, knn_k=30, inv_cube_k=10):
    edge_indicies = []
    offset = 0
    for i in range(batch.max().item() + 1):
        X_ca = batched_X_ca[batch == i]
        x_mask = batched_x_mask[batch == i]

        X_ca[x_mask] = torch.inf
        rel_pos_CA = X_ca.unsqueeze(1) - X_ca.unsqueeze(0)  # N x N x 3
        dist_CA = torch.linalg.vector_norm(rel_pos_CA, dim=-1)  # N x N
        sorted_dist, sorted_edges = torch.sort(dist_CA, dim=-1)  # N x N
        knn_edges = sorted_edges[..., :knn_k]

        # remove knn edges
        remaining_dist = sorted_dist[..., knn_k:]  # N x (N - knn_k)
        remaining_edges = sorted_edges[..., knn_k:]  # N x (N - knn_k)

        ## inv cube
        uniform = torch.distributions.Uniform(0,1)
        dist_noise = uniform.sample(remaining_dist.shape).to(batched_X_ca.device)  # N x (N - knn_k)

        logprobs = -3 * torch.log(remaining_dist)  # N x (N - knn_k)
        perturbed_logprobs = logprobs - torch.log(-torch.log(dist_noise))  # N x (N - knn_k)
        _, sampled_edges_relative_idx = torch.topk(perturbed_logprobs, k=inv_cube_k, dim=-1)
        sampled_edges = torch.gather(remaining_edges, -1, sampled_edges_relative_idx)  # N x inv_cube_k

        edge_sinks = torch.cat([knn_edges, sampled_edges], dim=-1)  # B x N x (knn_k + inv_cube_k)
        edge_sources = torch.arange(X_ca.shape[0]).repeat_interleave(knn_k + inv_cube_k).to(edge_sinks.device)
        edge_index = torch.stack([edge_sinks.flatten(), edge_sources], dim=0)
        edge_indicies.append(sort_edge_index(edge_index, sort_by_row=False) + offset)
        offset = offset + (batch == i).long().sum()

    edge_index = torch.cat(edge_indicies, dim=-1)
    edge_dist_vec = batched_X_ca[edge_index[0]] - batched_X_ca[edge_index[1]]
    edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
    # slightly hacky
    # TODO: use x_mask instead
    edge_select = edge_dist.isfinite() & (edge_dist > 0.1)  # mostly arbitrary cutoff
    return edge_index[:, edge_select]


class BackboneUpdate(nn.Module):
    def __init__(self,
                 bb_lmax_list,
                 edge_channels_list,
                 mappingReduced_bb,
                 bb_SO3_rotation,
                 bb_SO3_grid,
                 num_heads=8,
                 h_channels=32,
                 bb_channels=32,
                 n_bb_atoms=4,
                 knn_k=30,
                 lrange_k=30):
        super().__init__()

        self.n_bb_atoms = n_bb_atoms-1
        self.bb_lmax_list = bb_lmax_list
        self.h_channels = h_channels
        self.bb_channels = bb_channels
        self.bb_SO3_rotation = bb_SO3_rotation

        self.lrange_attention = TransBlockV2(
            sphere_channels=bb_channels + self.n_bb_atoms,
            attn_hidden_channels=h_channels,
            num_heads=num_heads,
            attn_alpha_channels=h_channels // 2,
            attn_value_channels=h_channels // 4,
            ffn_hidden_channels=h_channels,
            output_channels=bb_channels,
            lmax_list=bb_lmax_list,
            mmax_list=bb_lmax_list,
            SO3_rotation=bb_SO3_rotation,
            SO3_grid=bb_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_bb
        )

        self.update_x_ca = SO3_LinearV2(
            in_features=bb_channels,
            out_features=1,
            lmax=max(bb_lmax_list)
        )
        self.update_x_ca_gate = nn.Linear(
            in_features=bb_channels,
            out_features=1
        )

        self.update_bb = SO3_LinearV2(
            in_features=bb_channels,
            out_features=self.n_bb_atoms,
            lmax=max(bb_lmax_list)
        )

        self.knn_k = knn_k
        self.lrange_k = lrange_k


    def forward(self,
                X_ca: torch.Tensor,
                bb_rel: torch.Tensor,
                bb_features: SO3_Embedding,
                batch: torch.Tensor,
                x_mask: torch.Tensor,
                noising_mask: torch.Tensor):
        # compute which nodes are being noised and which aren't
        num_nodes = bb_features.length

        # compute graph with knn + inv cubic edges
        edge_index = sample_inv_cubic_edges(X_ca, x_mask, batch)
        edge_dist_vec = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
        # update rotation matrices
        edge_rot_mat = init_edge_rot_mat(edge_dist_vec)
        for rot in self.bb_SO3_rotation:
            rot.set_wigner(edge_rot_mat)

        # update backbone
        node_features = SO3_Embedding(
            num_nodes,
            self.bb_lmax_list,
            num_channels=self.bb_channels + self.n_bb_atoms,
            device=bb_features.device,
            dtype=bb_features.dtype
        )
        node_features.embedding[..., :self.bb_channels] = bb_features.embedding
        node_features.embedding[..., 1:4, -self.n_bb_atoms:] = bb_rel.transpose(-1, -2)
        node_features.embedding[..., 0:1, -1:] = noising_mask[:, None, None].float() # is node editable or not

        # gen edge features
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device)  # edge_channels_list
        edge_dist_rel_pos = _positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        updated_bb_features = self.lrange_attention(
            node_features,
            edge_features,
            edge_index
        )

        update_X_ca = self.update_x_ca(updated_bb_features)
        update_X_ca_gate = self.update_x_ca_gate(updated_bb_features.get_invariant_features(flat=True))
        update_bb = self.update_bb(updated_bb_features)

        subset_update_X_ca = update_X_ca.embedding[:, 1:4].squeeze(-1)[noising_mask]
        subset_update_gate = F.softplus(update_X_ca_gate)[noising_mask]
        subset_update_X_ca = subset_update_X_ca * subset_update_gate

        new_X_ca = X_ca.clone()
        new_X_ca[noising_mask] = X_ca[noising_mask] + subset_update_X_ca
        new_bb_rel = bb_rel.clone()
        new_bb_rel[noising_mask] = bb_rel[noising_mask] + update_bb.embedding[:, 1:4].transpose(-1, -2)[noising_mask]

        return new_X_ca, new_bb_rel, updated_bb_features.clone()


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


class BackboneR3Denoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 bb_lmax_list,
                 edge_channels_list,
                 mappingReduced_bb,
                 bb_SO3_rotation,
                 bb_SO3_grid,
                 num_heads=8,
                 h_channels=32,
                 bb_channels=32,
                 h_time=64,
                 scalar_h_dim=128,
                 n_layers=4,
                 device='cpu'):
        super().__init__()

        self.bb_lmax_list = bb_lmax_list
        self.bb_channels = bb_channels

        self.mappingReduced_nodes = mappingReduced_bb
        self.bb_SO3_rotation_list = bb_SO3_rotation
        self.bb_SO3_grid_list = bb_SO3_grid

        self.h_time = h_time
        self.time_rbf = RBF(n_basis=h_time//2)
        self.time_mlp = nn.Sequential(
            nn.Linear(h_time, scalar_h_dim),
            nn.ReLU(),
            nn.Linear(scalar_h_dim, h_time),
            nn.ReLU()
        )

        self.embed_time = nn.Linear(
            bb_channels + h_time, bb_channels
        )

        self.denoiser = nn.ModuleList([
            BackboneUpdate(
                bb_lmax_list=bb_lmax_list,
                edge_channels_list=edge_channels_list,
                mappingReduced_bb=mappingReduced_bb,
                bb_SO3_rotation=bb_SO3_rotation,
                bb_SO3_grid=bb_SO3_grid,
                num_heads=num_heads,
                bb_channels=bb_channels,
                h_channels=h_channels)
            for _ in range(n_layers)
        ])


    def forward(self, data, intermediates):
        ## prep features
        noised_bb_vecs = intermediates['noised_bb']
        noised_X_ca = noised_bb_vecs[:, 1]
        noised_bb_rel = noised_bb_vecs[:, (0, 2, 3)]
        x_mask = data['residue']['x_mask']
        noising_mask = intermediates['noising_select']
        bb_features = SO3_Embedding(
            data['residue'].num_nodes,
            self.bb_lmax_list,
            self.bb_channels,
            device=noised_X_ca.device,
            dtype=torch.float
        )
        ts = intermediates['t']  # (B,)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (B x h_time,)
        embedded_time = self.time_mlp(fourier_time)  # (B x h_time)
        data_splits = data._slice_dict['residue']['x']
        data_lens = data_splits[1:] - data_splits[:-1]
        embedded_time = torch.cat([
            embedded_time[i].view(1, -1).expand(l, -1) for i, l in enumerate(data_lens)
        ])  # n_res x h_time


        # center the training example at the mean of the x_cas
        center = []
        for i in range(data['residue'].batch.max().item() + 1):
            select = (data['residue'].batch == i)
            num_nodes = select.long().sum()
            select = select  & (~x_mask)
            subset_x_ca = noised_X_ca[select]
            subset_mean = subset_x_ca.mean(dim=0)
            center.append(subset_mean[None, :].expand(num_nodes, -1))
        center = torch.cat(center, dim=0)


        ## denoising
        X_ca = noised_X_ca - center
        bb_rel = noised_bb_rel

        for layer in self.denoiser:
            # fuse time embedding into node features
            bb_num_l0 = len(self.bb_lmax_list)
            bb_l0 = bb_features.get_invariant_features()  # n_res x (node_num_m0 x h_channels)
            time_expanded = embedded_time.unsqueeze(-2).expand(-1, bb_num_l0, -1)
            bb_l0 = self.embed_time(
                torch.cat([bb_l0, time_expanded], dim=-1)
            )
            bb_features.set_invariant_features(bb_l0)

            X_ca, bb_rel, bb_features = layer(
                X_ca,
                bb_rel,
                bb_features,
                data['residue'].batch,
                x_mask,
                noising_mask)

        denoised_bb_vecs = torch.empty_like(noised_bb_vecs)
        denoised_bb_vecs[:, 1] = X_ca + center
        denoised_bb_vecs[:, (0, 2, 3)] = bb_rel
        intermediates['denoised_bb_vecs'] = denoised_bb_vecs
        return intermediates
