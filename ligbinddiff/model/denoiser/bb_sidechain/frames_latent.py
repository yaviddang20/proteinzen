""" Denoising model """

import torch
from torch import nn
import numpy as np
from torch_cluster import knn_graph
from torch_geometric.utils import sort_edge_index

from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2
from ligbinddiff.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from ligbinddiff.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork, MultiResFeedForwardNetwork, TransBlockV2
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from ligbinddiff.model.modules.openfold import rigid_utils as ru
from ligbinddiff.diffusion.noisers.se3 import _extract_rots_trans, _assemble_rigid

from ligbinddiff.data.datasets.featurize.sidechain import _rbf, _positional_embeddings

from ligbinddiff.utils.frames import backbone_frames_to_bb_atoms


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


def mask_rigids(rigid_unmasked, rigid_masked, mask):
    rots_unmasked, trans_unmasked = _extract_rots_trans(rigid_unmasked)
    rots_masked, trans_masked = _extract_rots_trans(rigid_masked)

    rots = mask.long()[:, None, None] * rots_masked + (1 - mask.long())[:, None, None] * rots_unmasked
    trans = mask.long()[:, None] * trans_masked + (1 - mask.long())[:, None] * trans_unmasked
    return _assemble_rigid(rots, trans)


class FrameUpdate(nn.Module):
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 node_SO3_rotation,
                 node_SO3_grid,
                 frame_SO3_rotation,
                 num_heads=8,
                 h_channels=32,
                 rigid_h_channels=32,
                 sidechain_h_channels=32,
                 frame_channels=3,
                 knn_k=30,
                 lrange_k=30):
        super().__init__()

        self.node_lmax_list = node_lmax_list
        self.h_channels = h_channels
        self.rigid_h_channels = rigid_h_channels
        self.sidechain_h_channels = sidechain_h_channels
        self.node_SO3_rotation = node_SO3_rotation
        self.frame_channels = frame_channels
        self.frame_SO3_rotation = frame_SO3_rotation

        self.lrange_attention = TransBlockV2(
            sphere_channels=rigid_h_channels + sidechain_h_channels + frame_channels,
            attn_hidden_channels=h_channels,
            num_heads=num_heads,
            attn_alpha_channels=h_channels // 2,
            attn_value_channels=h_channels // 4,
            ffn_hidden_channels=h_channels,
            output_channels=rigid_h_channels,
            lmax_list=node_lmax_list,
            mmax_list=node_lmax_list,
            SO3_rotation=node_SO3_rotation,
            SO3_grid=node_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_nodes
        )
        self.frame_rot_update = nn.Sequential(
            nn.Linear(rigid_h_channels, h_channels * 2),
            nn.ReLU(),
            nn.Linear(h_channels*2, 6),
        )
        self.frame_trans_update = SO3_LinearV2(
            in_features=rigid_h_channels,
            out_features=1,
            lmax=max(node_lmax_list)
        )
        self.knn_k = knn_k
        self.lrange_k = lrange_k


    def forward(self,
                rigids: ru.Rigid,
                rigid_features: SO3_Embedding,
                sidechain_features: SO3_Embedding,
                batch: torch.Tensor,
                x_mask: torch.Tensor,
                noising_mask: torch.Tensor):
        num_nodes = noising_mask.shape[0]

        X_ca = rigids.get_trans()
        edge_index = sample_inv_cubic_edges(X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)
        edge_dist_vec = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
        # hacky way to filter for bad edges
        # TODO: use x_mask
        edge_select = edge_dist.isfinite() & (edge_dist > 1e-3)  # mostly arbitrary cutoff
        edge_index = edge_index[:, edge_select]
        edge_dist_vec = edge_dist_vec[edge_select]
        edge_dist = edge_dist[edge_select]
        # update rotation matrices
        edge_rot_mat = init_edge_rot_mat(edge_dist_vec)
        for rot in self.node_SO3_rotation:
            rot.set_wigner(edge_rot_mat)

        # compute edge features
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device)  # edge_channels_list
        edge_dist_rel_pos = _positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        # we can treat the rotation matrix as the three basis vectors of the frame
        frame_atoms = backbone_frames_to_bb_atoms(rigids)
        bb_node_fused = SO3_Embedding(
            num_nodes,
            self.node_lmax_list,
            num_channels=self.rigid_h_channels + self.sidechain_h_channels + self.frame_channels,
            device=rigid_features.device,
            dtype=rigid_features.dtype
        )
        bb_node_fused.embedding[..., :self.rigid_h_channels] = rigid_features.embedding
        bb_node_fused.embedding[..., self.rigid_h_channels:self.rigid_h_channels + self.sidechain_h_channels] = sidechain_features.embedding
        bb_node_fused.embedding[..., 1:4, -self.frame_channels:] = frame_atoms.transpose(-1, -2)
        editable = noising_mask.clone()
        editable[x_mask] = False
        bb_node_fused.embedding[..., 0:1, -1:] = editable[..., None, None].long() # is node editable or not

        bb_update = self.lrange_attention(
            bb_node_fused,
            edge_features,
            edge_index
        )
        bb_invariants = bb_update.get_invariant_features(flat=True)
        q_vec_update = self.frame_rot_update(bb_invariants)

        # compute rotation and translation
        quat_update = self._pure_vec_to_unit_quat(q_vec_update[..., :3])
        rot_update = ru.quat_to_rot(quat_update)
        trans_update = self.frame_trans_update(bb_update)
        trans_update = trans_update.embedding[:, 1:4].squeeze(-1)

        # update rigids
        rots, trans = _extract_rots_trans(rigids)
        rots_updated = torch.einsum("...ij,...jk->...ik", rots, rot_update)
        trans_updated = trans + trans_update
        rigids_updated = _assemble_rigid(rots_updated, trans_updated)

        # apply noising mask
        # True = noise, False = fixed
        rigids_updated = mask_rigids(rigids, rigids_updated, noising_mask)
        rigid_features_updated = rigid_features.clone()
        rigid_features_updated.embedding[~noising_mask] = bb_update.embedding[~noising_mask]

        return rigids_updated, rigid_features_updated

    def _pure_vec_to_unit_quat(self, pure_vec):
        ones = torch.ones(list(pure_vec.shape[:-1]) + [1], device=pure_vec.device)
        unnormed_quat = torch.cat([ones, pure_vec], dim=-1)
        norm = torch.linalg.vector_norm(unnormed_quat, dim=-1, keepdims=True)
        return unnormed_quat / norm


class LatentUpdate(nn.Module):
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 node_SO3_rotation,
                 node_SO3_grid,
                 frame_SO3_rotation,
                 num_heads=8,
                 h_channels=32,
                 rigid_h_channels=32,
                 sidechain_h_channels=32,
                 frame_channels=3,
                 k=30):
        super().__init__()

        self.node_lmax_list = node_lmax_list
        self.h_channels = h_channels
        self.rigid_h_channels = rigid_h_channels
        self.sidechain_h_channels = sidechain_h_channels
        self.node_SO3_rotation = node_SO3_rotation
        self.frame_channels = frame_channels
        self.frame_SO3_rotation = frame_SO3_rotation

        self.local_attention = TransBlockV2(
            sphere_channels=rigid_h_channels + sidechain_h_channels + frame_channels,
            attn_hidden_channels=h_channels,
            num_heads=num_heads,
            attn_alpha_channels=h_channels // 2,
            attn_value_channels=h_channels // 4,
            ffn_hidden_channels=h_channels,
            output_channels=sidechain_h_channels,
            lmax_list=node_lmax_list,
            mmax_list=node_lmax_list,
            SO3_rotation=node_SO3_rotation,
            SO3_grid=node_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_nodes
        )
        self.k = k

    def forward(self,
                rigids: ru.Rigid,
                rigid_features: SO3_Embedding,
                sidechain_features: SO3_Embedding,
                batch: torch.Tensor,
                x_mask: torch.Tensor,
                noising_mask: torch.Tensor):
        num_nodes = noising_mask.shape[0]

        # compute local knn graph
        X_ca = rigids.get_trans()
        edge_index = knn_graph(X_ca, self.k, batch)
        # update rotation matrices
        edge_dist_vec = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
        # hacky way to filter for bad edges
        # TODO: use x_mask
        edge_select = edge_dist.isfinite() & (edge_dist > 1e-3)  # mostly arbitrary cutoff
        edge_index = edge_index[:, edge_select]
        edge_dist_vec = edge_dist_vec[edge_select]
        edge_dist = edge_dist[edge_select]

        edge_rot_mat = init_edge_rot_mat(edge_dist_vec)
        for rot in self.node_SO3_rotation:
            rot.set_wigner(edge_rot_mat)

        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device)  # edge_channels_list
        edge_dist_rel_pos = _positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        # update latent sidechains
        # we can treat the rotation matrix as the three basis vectors of the frame
        frame_atoms = backbone_frames_to_bb_atoms(rigids)
        bb_node_fused = SO3_Embedding(
            num_nodes,
            self.node_lmax_list,
            num_channels=self.rigid_h_channels + self.sidechain_h_channels + self.frame_channels,
            device=sidechain_features.device,
            dtype=sidechain_features.dtype
        )
        bb_node_fused.embedding[..., :self.rigid_h_channels] = rigid_features.embedding
        bb_node_fused.embedding[..., self.rigid_h_channels:self.rigid_h_channels + self.sidechain_h_channels] = sidechain_features.embedding
        bb_node_fused.embedding[..., 1:4, -self.frame_channels:] = frame_atoms.transpose(-1, -2)

        editable = noising_mask.clone()
        editable[x_mask] = False
        bb_node_fused.embedding[..., 0:1, -1:] = editable[..., None, None].long() # is node editable or not

        update_sidechain_features = self.local_attention(
            bb_node_fused,
            edge_features,
            edge_index
        )
        new_sidechain_features = sidechain_features.clone()
        new_sidechain_features.embedding[noising_mask] = update_sidechain_features.embedding[noising_mask]
        return new_sidechain_features


class LatentDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 node_SO3_rotation,
                 node_SO3_grid,
                 frame_SO3_rotation,
                 num_heads=8,
                 h_channels=32,
                 frame_channels=3,
                 knn_k=30,
                 lrange_k=10,
                 ):
        """
        Args
        ----
        """
        super().__init__()
        self.node_lmax_list = node_lmax_list

        self.mappingReduced_nodes = mappingReduced_nodes
        self.node_SO3_rotation = node_SO3_rotation
        self.node_SO3_grid = node_SO3_grid

        self.frame_update = FrameUpdate(
            node_lmax_list=node_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_nodes=mappingReduced_nodes,
            node_SO3_rotation=node_SO3_rotation,
            node_SO3_grid=node_SO3_grid,
            frame_SO3_rotation=frame_SO3_rotation,
            num_heads=num_heads,
            h_channels=h_channels,
            frame_channels=frame_channels,
            knn_k=knn_k,
            lrange_k=lrange_k
        )

        self.latent_update = LatentUpdate(
            node_lmax_list=node_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_nodes=mappingReduced_nodes,
            node_SO3_rotation=node_SO3_rotation,
            node_SO3_grid=node_SO3_grid,
            frame_SO3_rotation=frame_SO3_rotation,
            num_heads=num_heads,
            h_channels=h_channels,
            frame_channels=frame_channels,
            k=knn_k
        )

        self.h_channels = h_channels
        self.bb_channels = frame_channels

    def forward(
            self,
            rigid_features: SO3_Embedding,
            data,
            intermediates
    ):
        rigids = intermediates['denoised_frames']
        sidechain_features = intermediates['denoised_latent_sidechain']
        noising_mask = intermediates['noising_mask']
        x_mask = data['x_mask']
        batch = data.batch

        new_rigids, new_rigid_features = self.frame_update(
            rigids,
            rigid_features,
            sidechain_features,
            batch,
            x_mask,
            noising_mask,
        )
        new_sidechain_features = self.latent_update(
            new_rigids,
            new_rigid_features,
            sidechain_features,
            batch,
            x_mask,
            noising_mask,
        )

        intermediates['denoised_frames'] = new_rigids
        intermediates['denoised_latent_sidechain'] = new_sidechain_features

        return new_rigid_features


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


class LatentDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 node_SO3_rotation,
                 node_SO3_grid,
                 num_heads=8,
                 h_channels=32,
                 rigid_h_channels=32,
                 h_time=64,
                 scalar_h_dim=128,
                 n_layers=4,
                 device='cpu'):
        super().__init__()

        self.node_lmax_list = node_lmax_list
        self.rigid_h_channels = rigid_h_channels
        self.h_channels = h_channels

        self.mappingReduced_nodes = mappingReduced_nodes
        self.node_SO3_rotation_list = node_SO3_rotation
        self.node_SO3_grid_list = node_SO3_grid

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

        self.frame_SO3_rotation = nn.ModuleList()
        for lmax in node_lmax_list:
            self.frame_SO3_rotation.append(
                SO3_Rotation(lmax)
            )

        self.denoiser = nn.ModuleList([
            LatentDenoisingLayer(
                node_lmax_list=node_lmax_list,
                edge_channels_list=edge_channels_list,
                mappingReduced_nodes=mappingReduced_nodes,
                node_SO3_rotation=self.node_SO3_rotation_list,
                node_SO3_grid=self.node_SO3_grid_list,
                frame_SO3_rotation=self.frame_SO3_rotation,
                num_heads=num_heads,
                h_channels=h_channels)
            for _ in range(n_layers)
        ])

    def _translate_rigids(self, rigids, shift):
        rots = rigids.get_rots()
        trans = rigids.get_trans()
        return ru.Rigid(rots, trans + shift)

    def forward(self, data, intermediates):
        ## prep features
        rigid_features = SO3_Embedding(
            data.num_nodes,
            self.node_lmax_list,
            self.rigid_h_channels,
            device=data['x'].device,
            dtype=torch.float
        )
        ts = intermediates['t']  # (B,)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (B x h_time,)
        embedded_time = self.time_mlp(fourier_time)  # (B x h_time)
        data_splits = data._slice_dict['x']
        data_lens = data_splits[1:] - data_splits[:-1]
        embedded_time = torch.cat([
            embedded_time[i].view(1, -1).expand(l, -1) for i, l in enumerate(data_lens)
        ])  # n_res x h_time

        # fuse time embedding into node features
        rigid_num_l0 = len(self.node_lmax_list)
        rigid_l0 = rigid_features.get_invariant_features()  # n_res x (node_num_m0 x h_channels)
        time_expanded = embedded_time.unsqueeze(-2).expand(-1, rigid_num_l0, -1)
        node_l0 = self.embed_time(
            torch.cat([rigid_l0, time_expanded], dim=-1)
        )
        rigid_features.set_invariant_features(node_l0)

        # center the training example at the mean of the x_cas
        center = []
        for i in range(data.batch.max().item() + 1):
            select = (data.batch == i)
            num_nodes = select.long().sum()
            subset_x_ca = intermediates['noised_frames'].get_trans()[select]
            subset_mean = subset_x_ca.mean(dim=0)
            center.append(subset_mean[None, :].expand(num_nodes, -1))
        center = torch.cat(center, dim=0)

        ## denoising
        f_V = rigid_features
        intermediates['denoised_frames'] = self._translate_rigids(intermediates['noised_frames'], -center)
        intermediates['denoised_latent_sidechain'] = intermediates['noised_latent_sidechain']

        for layer in self.denoiser:
            f_V = layer(f_V, data, intermediates)

        intermediates['denoised_frames'] = self._translate_rigids(intermediates['denoised_frames'], center)
        intermediates['denoised_bb'] = backbone_frames_to_bb_atoms(intermediates['denoised_frames'])

        return intermediates
