""" Denoising model """

import torch
from torch import nn
from torch_cluster import knn_graph
from torch_geometric.utils import sort_edge_index
from ligbinddiff.data.datasets.featurize.common import _rbf
from ligbinddiff.model.modules.common import RBF

from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2
from ligbinddiff.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from ligbinddiff.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork, MultiResFeedForwardNetwork, TransBlockV2
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.diffusion.noisers.se3 import _extract_rots_trans, _assemble_rigid

from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings

from ligbinddiff.utils.frames import backbone_frames_to_bb_atoms

from ligbinddiff.model.modules.frames import PointSetAttentionWithEdgeBias, EdgeTransition, NodeTransition, BackboneUpdateVectorBias, VectorLayerNorm, LocalFrameUpdate

from ligbinddiff.model.utils.graph import sample_inv_cubic_edges, sequence_local_graph

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
                 frame_channels=3,
                 knn_k=30,
                 lrange_k=30):
        super().__init__()

        self.node_lmax_list = node_lmax_list
        self.h_channels = h_channels
        self.node_SO3_rotation = node_SO3_rotation
        self.frame_channels = frame_channels
        self.frame_SO3_rotation = frame_SO3_rotation

        self.lrange_attention = TransBlockV2(
            sphere_channels=h_channels + frame_channels,
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
        self.frame_rot_update = nn.Sequential(
            nn.Linear(h_channels, h_channels * 2),
            nn.ReLU(),
            nn.Linear(h_channels*2, h_channels),
            nn.ReLU(),
            nn.Linear(h_channels, 6)

        )
        self.frame_trans_update = SO3_LinearV2(
            in_features=h_channels,
            out_features=1,
            lmax=max(node_lmax_list)
        )
        self.knn_k = knn_k
        self.lrange_k = lrange_k

    def forward(self,
                rigids: ru.Rigid,
                node_features: SO3_Embedding,
                batch: torch.Tensor,
                x_mask: torch.Tensor,
                noising_mask: torch.Tensor,
                seq_local_edge_index):
        num_nodes = noising_mask.shape[0]

        X_ca = rigids.get_trans()
        edge_index = sample_inv_cubic_edges(X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)
        edge_index = torch.cat([edge_index, seq_local_edge_index], dim=-1)
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
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        # we can treat the rotation matrix as the three basis vectors of the frame
        frame_atoms = backbone_frames_to_bb_atoms(rigids)
        bb_node_fused = SO3_Embedding(
            num_nodes,
            self.node_lmax_list,
            num_channels=self.h_channels + self.frame_channels,
            device = node_features.device,
            dtype=node_features.dtype
        )
        bb_node_fused.embedding[..., :self.h_channels] = node_features.embedding
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

        # rotate associated node features
        quat_update = self._pure_vec_to_unit_quat(q_vec_update[..., :3])
        rot_update = ru.quat_to_rot(quat_update)
        # for rot in self.frame_SO3_rotation:
        #     rot.set_wigner(rot_update)
        # node_features_updated._rotate(self.frame_SO3_rotation, self.node_lmax_list, self.node_lmax_list)

        trans_update = self.frame_trans_update(bb_update)
        trans_update = trans_update.embedding[:, 1:4].squeeze(-1)

        # update rigids
        rots, trans = _extract_rots_trans(rigids)
        rots_updated = torch.einsum("...ij,...jk->...ik", rots, rot_update)
        # trans_updated = trans + torch.einsum("...ij,...j->...i", rots, q_vec_update[..., 3:])  # transform translation in local frame to global frame
        trans_updated = trans + trans_update
        rigids_updated = _assemble_rigid(rots_updated, trans_updated)

        # apply noising mask
        # True = noise, False = fixed
        rigids_updated = mask_rigids(rigids, rigids_updated, noising_mask)

        return rigids_updated, bb_update

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
                 frame_channels=3,
                 k=30):
        super().__init__()

        self.node_lmax_list = node_lmax_list
        self.h_channels = h_channels
        self.node_SO3_rotation = node_SO3_rotation
        self.frame_channels = frame_channels
        self.frame_SO3_rotation = frame_SO3_rotation

        self.local_attention = TransBlockV2(
            sphere_channels=h_channels + frame_channels,
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
        self.k = k

    def forward(self,
                rigids: ru.Rigid,
                node_features: SO3_Embedding,
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
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        # update latent sidechains
        # we can treat the rotation matrix as the three basis vectors of the frame
        frame_atoms = backbone_frames_to_bb_atoms(rigids)
        bb_node_fused = SO3_Embedding(
            num_nodes,
            self.node_lmax_list,
            num_channels=self.h_channels + self.frame_channels,
            device = node_features.device,
            dtype=node_features.dtype
        )
        bb_node_fused.embedding[..., :self.h_channels] = node_features.embedding
        bb_node_fused.embedding[..., 1:4, -self.frame_channels:] = frame_atoms.transpose(-1, -2)

        editable = noising_mask.clone()
        editable[x_mask] = False
        bb_node_fused.embedding[..., 0:1, -1:] = editable[..., None, None].long() # is node editable or not

        update_node_features = self.local_attention(
            bb_node_fused,
            edge_features,
            edge_index
        )
        return update_node_features


class FrameDenoisingLayer(nn.Module):
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
            frame_features: SO3_Embedding,
            data,
            intermediates
    ):
        rigids = intermediates['denoised_frames']
        noising_mask = intermediates['noising_mask']
        seq_local_edge_index = intermediates['seq_local_edge_index']
        x_mask = data['x_mask']
        batch = data.batch

        new_rigids, new_frame_features = self.frame_update(
            rigids,
            frame_features,
            batch,
            x_mask,
            noising_mask,
            seq_local_edge_index
        )
        # new_frame_features = self.latent_update(
        #     new_rigids,
        #     rotated_frame_features,
        #     batch,
        #     x_mask,
        #     noising_mask,
        # )

        return new_rigids, new_frame_features.clone()


# adapted from https://github.com/jmclong/random-fourier-features-pytorch/blob/main/rff/layers.py
class FrameDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_nodes,
                 node_SO3_rotation,
                 node_SO3_grid,
                 num_heads=8,
                 h_channels=32,
                 h_time=64,
                 scalar_h_dim=128,
                 n_layers=4,
                 device='cpu'):
        super().__init__()

        self.node_lmax_list = node_lmax_list
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
            FrameDenoisingLayer(
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
        node_features = SO3_Embedding(
           data.num_nodes,
           self.node_lmax_list,
           self.h_channels,
           device=data['x'].device,
           dtype=torch.float
        )
        ts = intermediates['t']  # (B,)

        # add residue positional index
        residx = []
        seq_local_edge_index = []
        offset = 0
        for i in range(data.batch.max().item() + 1):
            select = (data.batch == i)
            local_residx = torch.arange(select.sum().item(), device=node_features.device) + offset
            residx.append(local_residx)
            seq_local_edge_index.append(
                sequence_local_graph(select.sum().item(), data['x_mask'][select]) + offset
            )
            offset += select.sum().item()
        seq_local_edge_index = torch.cat(seq_local_edge_index, dim=-1)
        intermediates['seq_local_edge_index'] = seq_local_edge_index

        residx = torch.cat(residx, dim=-1)
        node_features.embedding[:, 0] = _edge_positional_embeddings(residx, num_embeddings=self.h_channels, device=node_features.device)

        # ## create time embedding
        # fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (B x h_time,)
        # embedded_time = self.time_mlp(fourier_time)  # (B x h_time)

        # # fuse time embedding into node features
        # node_num_l0 = len(self.node_lmax_list)
        # node_l0 = node_features.get_invariant_features()  # n_res x (node_num_m0 x h_channels)
        # time_expanded = embedded_time.unsqueeze(-2).expand(-1, node_num_l0, -1)
        # node_l0 = self.embed_time(
        #     torch.cat([node_l0, time_expanded], dim=-1)
        # )
        # node_features.set_invariant_features(node_l0)

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
        f_V = node_features

        intermediates['denoised_frames'] = self._translate_rigids(intermediates['noised_frames'], -center)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (B x h_time,)
        embedded_time = self.time_mlp(fourier_time)  # (B x h_time)
        for i, layer in enumerate(self.denoiser):
            # fuse time embedding into node features
            node_num_l0 = len(self.node_lmax_list)
            node_l0 = f_V.get_invariant_features()  # n_res x (node_num_m0 x h_channels)
            time_expanded = embedded_time.unsqueeze(-2).expand(-1, node_num_l0, -1)
            node_l0 = self.embed_time(
                torch.cat([node_l0, time_expanded], dim=-1)
            )
            f_V.set_invariant_features(node_l0)

            f_ca, f_V = layer(f_V, data, intermediates)
            intermediates['denoised_frames'] = f_ca

        intermediates['denoised_frames'] = self._translate_rigids(intermediates['denoised_frames'], center)
        intermediates['denoised_bb'] = backbone_frames_to_bb_atoms(intermediates['denoised_frames'])
        intermediates['node_features'] = f_V

        return intermediates


class FrameDenoisingLayer2(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_v,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 gen_vectors=False
                 ):
        """
        Args
        ----
        """
        super().__init__()

        self.attn_seq = PointSetAttentionWithEdgeBias(
            c_s=c_s,
            c_v=c_v,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
            gen_vectors=gen_vectors
        )
        self.attn_spatial = PointSetAttentionWithEdgeBias(
            c_s=c_s,
            c_v=c_v,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
            gen_vectors=False
        )
        self.local_update = LocalFrameUpdate(
            c_s,
            c_v,
            c_hidden
        )
        self.ln_s1 = nn.LayerNorm(c_s)
        self.ln_v1 = VectorLayerNorm(c_v)

        self.ln_s2 = nn.LayerNorm(c_s)
        self.ln_v2 = VectorLayerNorm(c_v)

        self.ln_s3 = nn.LayerNorm(c_s)
        self.ln_v3 = VectorLayerNorm(c_v)

        self.bb_update = BackboneUpdateVectorBias(
            c_s, c_v
        )
        self.node_transition = NodeTransition(
            c_s=c_s, c_v=c_v
        )
        self.edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )
        self.seq_edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )


    def forward(
            self,
            node_features,
            rigids,
            edge_features,
            edge_index,
            seq_edge_features,
            seq_edge_index,
            data,
            intermediates,
            node_vectors=None
    ):
        noising_mask = intermediates['noising_mask']
        x_mask = data['x_mask']

        edge_features = self.edge_transition(node_features, edge_features, edge_index)
        seq_edge_features = self.seq_edge_transition(node_features, seq_edge_features, seq_edge_index)

        node_s_update, node_v_update = self.attn_seq(
            node_features,
            rigids,
            seq_edge_features,
            seq_edge_index,
            node_vectors
        )
        node_features = self.ln_s1(node_features + node_s_update * (~x_mask)[..., None])
        if node_vectors is None:
            node_vectors = self.ln_v1(node_v_update * (~x_mask)[..., None, None])
            # node_vectors = node_v_update
        else:
            node_vectors = self.ln_v1(node_vectors + node_v_update * (~x_mask)[..., None, None])
            # node_vectors = node_v_update

        node_s_update, node_v_update = self.attn_spatial(
            node_features,
            rigids,
            edge_features,
            edge_index,
            node_vectors
        )
        node_features = self.ln_s2(node_features + node_s_update * (~x_mask)[..., None])
        node_vectors = self.ln_v2(node_vectors + node_v_update * (~x_mask)[..., None, None])

        node_s_update, node_v_update = self.local_update(
            node_features,
            node_vectors,
            rigids
        )
        node_features = self.ln_s3(node_features + node_s_update * (~x_mask)[..., None])
        # node_features = node_s_update
        node_vectors = self.ln_v3(node_vectors + node_v_update * (~x_mask)[..., None, None])
        # node_vectors = node_v_update

        node_features, node_vectors = self.node_transition(node_features, node_vectors)
        node_features = node_features  * (~x_mask)[..., None]
        node_vectors = node_vectors  * (~x_mask)[..., None, None]
        rigids_update = self.bb_update(
            node_features * noising_mask[..., None],
            node_vectors * noising_mask[..., None, None])

        rigids = rigids.compose_q_update_vec(
            rigids_update * noising_mask[..., None]
        )


        return node_features, rigids, edge_features, seq_edge_features, node_vectors


class FrameDenoiser2(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 c_s,
                 c_v,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 h_time=64,
                 scalar_h_dim=128,
                 n_layers=4,
                 device='cpu',
                 knn_k=20,
                 lrange_k=30):
        super().__init__()

        self.c_s = c_s
        self.c_v = c_v
        self.c_z = c_z

        self.h_time = h_time
        self.time_rbf = RBF(n_basis=h_time//2)
        self.time_mlp = nn.Sequential(
            nn.Linear(h_time, scalar_h_dim),
            nn.ReLU(),
            nn.Linear(scalar_h_dim, h_time),
            nn.ReLU()
        )

        self.embed_node = nn.Linear(
            c_s + h_time + 1, c_s
        )

        self.denoiser = nn.ModuleList([
            FrameDenoisingLayer2(
                 c_s,
                 c_v,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 gen_vectors=(i == 0)
            )
            for i in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k

    def _translate_rigids(self, rigids, shift):
        rots = rigids.get_rots()
        trans = rigids.get_trans()
        return ru.Rigid(rots, trans + shift)

    def forward(self, data, intermediates):
        ## prep features
        ts = intermediates['t']  # (B,)
        x_mask = data['x_mask']
        batch = data.batch
        device = ts.device
        num_nodes = ts.shape[0]

        # center the training example at the mean of the x_cas
        center = []
        for i in range(data.batch.max().item() + 1):
            select = (data.batch == i)
            num_nodes = select.long().sum()
            subset_x_ca = intermediates['noised_frames'].get_trans()[select]
            subset_mean = subset_x_ca.mean(dim=0)
            center.append(subset_mean[None, :].expand(num_nodes, -1))
        center = torch.cat(center, dim=0)
        rigids = self._translate_rigids(intermediates['noised_frames'], -center)
        rigids = scale_rigids(rigids, 0.1)

        # generate sequence edges
        residx = []
        seq_local_edge_index = []
        offset = 0
        for i in range(data.batch.max().item() + 1):
            select = (data.batch == i)
            local_residx = torch.arange(select.sum().item(), device=device) + offset
            residx.append(local_residx)
            seq_local_edge_index.append(
                sequence_local_graph(select.sum().item(), data['x_mask'][select]) + offset
            )
            offset += select.sum().item()
        seq_local_edge_index = torch.cat(seq_local_edge_index, dim=-1)

        # generate spatial edges
        X_ca = rigids.get_trans()
        masked_X_ca = X_ca.clone()
        masked_X_ca[x_mask] = torch.inf
        edge_index = sample_inv_cubic_edges(X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        # compute edge features
        edge_dist_vec = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device, D_count=self.c_z//2)  # edge_channels_list
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=self.c_z//2, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        seq_edge_dist_vec = X_ca[seq_local_edge_index[0]] - X_ca[seq_local_edge_index[1]]
        seq_edge_dist = torch.linalg.vector_norm(seq_edge_dist_vec, dim=-1)
        seq_edge_dist_rbf = _rbf(seq_edge_dist, device=seq_edge_dist.device, D_count=self.c_z//2)  # edge_channels_list
        seq_edge_dist_rel_pos = _edge_positional_embeddings(seq_local_edge_index, num_embeddings=self.c_z//2, device=seq_edge_dist.device)  # edge_channels_list
        seq_edge_features = torch.cat([seq_edge_dist_rbf, seq_edge_dist_rel_pos], dim=-1)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_node x h_time,)
        embedded_time = self.time_mlp(fourier_time)  # (N_node x h_time)

        # generate node features
        residx = torch.cat(residx, dim=-1)
        res_pos = _edge_positional_embeddings(
            torch.stack([residx, torch.zeros_like(residx)]),  # yea this is hacky
            num_embeddings=self.c_s, device=device)
        node_features = self.embed_node(
            torch.cat([
                res_pos,
                embedded_time,
                intermediates['noising_mask'].float()[..., None]
            ], dim=-1)
        )

        ## denoising
        node_vectors = None

        for i, layer in enumerate(self.denoiser):
            node_features, rigids, edge_features, seq_edge_features, node_vectors = layer(
                node_features,
                rigids,
                edge_features,
                edge_index,
                seq_edge_features,
                seq_local_edge_index,
                data,
                intermediates,
                node_vectors=node_vectors
            )

        rigids = scale_rigids(rigids, 10)
        intermediates['denoised_frames'] = self._translate_rigids(rigids, center)
        intermediates['denoised_bb'] = backbone_frames_to_bb_atoms(intermediates['denoised_frames'])
        intermediates['node_features'] = node_features

        return intermediates

def scale_rigids(rigids, scale):
    trans = rigids.get_trans()
    scaled_trans = trans * scale
    return ru.Rigid(rots=rigids.get_rots(), trans=scaled_trans)


class FrameDenoiser2p5(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 c_s,
                 c_v,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 h_time=64,
                 scalar_h_dim=128,
                 n_layers=4,
                 device='cpu',
                 knn_k=20,
                 lrange_k=30):
        super().__init__()

        self.c_s = c_s
        self.c_v = c_v
        self.c_z = c_z

        self.h_time = h_time
        self.time_rbf = RBF(n_basis=h_time//2)
        self.time_mlp = nn.Sequential(
            nn.Linear(h_time, scalar_h_dim),
            nn.ReLU(),
            nn.Linear(scalar_h_dim, h_time),
            nn.ReLU()
        )

        self.embed_node = nn.Linear(
            c_s + h_time + 1, c_s
        )

        self.denoiser = nn.ModuleList([
            FrameDenoisingLayer2(
                 c_s,
                 c_v,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 gen_vectors=(i == 0)
            )
            for i in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k

    def _translate_rigids(self, rigids, shift):
        rots = rigids.get_rots()
        trans = rigids.get_trans()
        return ru.Rigid(rots, trans + shift)

    def forward(self, data):
        ## prep features
        ts = data['t']  # (B,)
        expanded_ts = []

        for i, t in enumerate(ts.view(-1).tolist()):
            select = (data.batch == i)
            num_nodes = select.long().sum()
            expanded_ts.append(t * torch.ones(num_nodes, device=select.device))
        ts = torch.cat(expanded_ts, dim=0)

        x_mask = data['x_mask']
        rigids_t = ru.Rigid.from_tensor_7(data['rigids_t'])
        batch = data.batch
        device = ts.device
        num_nodes = ts.shape[0]

        # center the training example at the mean of the x_cas
        center = []
        for i in range(data.batch.max().item() + 1):
            select = (data.batch == i)
            num_nodes = select.long().sum()
            subset_x_ca = rigids_t.get_trans()[select]
            subset_mean = subset_x_ca.mean(dim=0)
            center.append(subset_mean[None, :].expand(num_nodes, -1))
        center = torch.cat(center, dim=0)
        rigids = self._translate_rigids(rigids_t, -center)
        rigids = scale_rigids(rigids, 0.1)

        # generate sequence edges
        residx = []
        seq_local_edge_index = []
        offset = 0
        for i in range(data.batch.max().item() + 1):
            select = (data.batch == i)
            local_residx = torch.arange(select.sum().item(), device=device) + offset
            residx.append(local_residx)
            seq_local_edge_index.append(
                sequence_local_graph(select.sum().item(), data['x_mask'][select]) + offset
            )
            offset += select.sum().item()
        seq_local_edge_index = torch.cat(seq_local_edge_index, dim=-1)

        # generate spatial edges
        X_ca = rigids.get_trans()
        masked_X_ca = X_ca.clone()
        masked_X_ca[x_mask] = torch.inf
        edge_index = sample_inv_cubic_edges(masked_X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        # compute edge features
        edge_dist_vec = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device, D_count=self.c_z//2)  # edge_channels_list
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=self.c_z//2, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        seq_edge_dist_vec = X_ca[seq_local_edge_index[0]] - X_ca[seq_local_edge_index[1]]
        seq_edge_dist = torch.linalg.vector_norm(seq_edge_dist_vec, dim=-1)
        seq_edge_dist_rbf = _rbf(seq_edge_dist, device=seq_edge_dist.device, D_count=self.c_z//2)  # edge_channels_list
        seq_edge_dist_rel_pos = _edge_positional_embeddings(seq_local_edge_index, num_embeddings=self.c_z//2, device=seq_edge_dist.device)  # edge_channels_list
        seq_edge_features = torch.cat([seq_edge_dist_rbf, seq_edge_dist_rel_pos], dim=-1)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_node x h_time,)
        embedded_time = self.time_mlp(fourier_time)  # (N_node x h_time)

        # generate node features
        residx = torch.cat(residx, dim=-1)
        res_pos = _edge_positional_embeddings(
            torch.stack([residx, torch.zeros_like(residx)]),  # yea this is hacky
            num_embeddings=self.c_s, device=device)
        # print(res_pos.shape, embedded_time.shape, data['noising_mask'].shape)
        node_features = self.embed_node(
            torch.cat([
                res_pos,
                embedded_time,
                data['noising_mask'].float()[..., None]
            ], dim=-1)
        )

        ## denoising
        node_vectors = None

        for i, layer in enumerate(self.denoiser):
            node_features, rigids, edge_features, seq_edge_features, node_vectors = layer(
                node_features,
                rigids,
                edge_features,
                edge_index,
                seq_edge_features,
                seq_local_edge_index,
                data,
                data,
                node_vectors=node_vectors
            )

        rigids = scale_rigids(rigids, 10)
        ret = {}
        ret['denoised_frames'] = self._translate_rigids(rigids, center)
        ret['final_rigids'] = self._translate_rigids(rigids, center)
        ret['denoised_bb'] = backbone_frames_to_bb_atoms(ret['denoised_frames'])
        ret['node_features'] = node_features

        return ret
