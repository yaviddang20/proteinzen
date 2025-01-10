import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy

import functools as fn

from proteinzen.model.modules.layers.node.attention import FlashTransformerEncoder
from proteinzen.model.modules.openfold.layers import InvariantPointAttention
from proteinzen.model.modules.openfold.layers_v2 import Linear, ConditionedInvariantPointAttention, ConditionedTransition, BackboneUpdate, TorsionAngles, Transition, LayerNorm, AdaLN
import proteinzen.utils.openfold.rigid_utils as ru
from proteinzen.utils.openfold.rigid_utils import Rigid, batchwise_center
from proteinzen.utils.framediff.all_atom import compute_backbone
from proteinzen.utils.coarse_grain import compute_atom14_from_cg_frames

from ._attn import ConditionedPairUpdate, MultiRigidPairEmbedder
from ._frame_transformer import (
    SequenceFrameTransformerUpdate, ConditionedSequenceFrameTransformerUpdate,
    get_indexing_matrix, single_to_keys, single_to_weighted_keys, pairs_to_framepairs,
    KnnFrameTransformerUpdate, KnnInvariantPointAttention
)
from ._v5_helper import BroadcastZInvariantPointAttention

def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


class TorsionAngles(nn.Module):
    def __init__(self, c, num_torsions, eps=1e-8):
        super(TorsionAngles, self).__init__()

        self.c = c
        self.eps = eps
        self.num_torsions = num_torsions

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        # TODO: Remove after published checkpoint is updated without these weights.
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.linear_final = Linear(
            self.c, self.num_torsions * 2, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)

        s = s + s_initial
        unnormalized_s = self.linear_final(s)
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        normalized_s = unnormalized_s / norm_denom

        return unnormalized_s, normalized_s


class SeqPredictor(nn.Module):
    def __init__(self, c_frame, n_aa=21):
        super().__init__()
        self.proj = nn.Sequential(
            LayerNorm(c_frame),
            Linear(c_frame, c_frame, bias=False)
        )
        self.out = Linear(c_frame, n_aa)
        self.c_frame = c_frame

    def forward(self, frame_embed):
        seq_embed = self.proj(frame_embed)
        out = self.out(seq_embed.mean(dim=-2))
        return out


class SeqPairPredictor(nn.Module):
    def __init__(self, c_z, n_aa=21):
        super().__init__()
        self.proj = nn.Sequential(
            Linear(c_z, n_aa * n_aa, bias=False)
        )

    def forward(self, edge_embed):
        two_body_logits = self.proj(edge_embed)
        return two_body_logits


class EdgeDistPredictor(nn.Module):
    def __init__(self, c_z, n_bins=64):
        super().__init__()
        self.proj = Linear(c_z, c_z)
        self.out = nn.Sequential(
            LayerNorm(c_z),
            Linear(c_z, n_bins),
        )

    def forward(self, edge_features):
        z = self.proj(edge_features)
        z = z + z.transpose(-2, -3)
        return self.out(z)


class FramepairDistPredictor(nn.Module):
    def __init__(self, c_framepair, n_bins=22):
        super().__init__()
        self.n_bins = n_bins
        self.proj = Linear(c_framepair, n_bins, bias=False)

    def forward(self, framepair_embed):
        return self.proj(framepair_embed)


class Embedder(nn.Module):

    def __init__(self,
                 c_s,
                 c_z,
                 c_frame,
                 index_embed_size=32,
                 use_init_distogram=False,
    ):
        super(Embedder, self).__init__()
        self.c_sidechain = c_frame

        # Time step embedding
        t_embed_size = index_embed_size
        node_embed_dims = (
            t_embed_size    # time
            + 1             # noised state
            + index_embed_size  # res index
        )

        node_embed_size = c_s
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            LayerNorm(node_embed_size),
        )

        edge_in = (
            t_embed_size    # time
            + 1             # noised state
        ) * 2
        edge_in += index_embed_size
        edge_in += 22

        self.use_init_distogram = use_init_distogram
        if use_init_distogram:
            edge_in += 22

        edge_embed_size = c_z
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=index_embed_size
        )

        self.node_to_sidechain_frame = Linear(c_s, c_frame, bias=False)
        self.pos_embedder = nn.Embedding(5, c_frame)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

    def forward(
            self,
            *,
            seq_idx,
            t,
            fixed_mask,
            sc_rigids=None,
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        num_batch, num_res = seq_idx.shape
        node_feats = []

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))

        prot_t_embed = torch.cat([
            prot_t_embed,
            fixed_mask,
        ], dim=-1)
        node_feats = [prot_t_embed]
        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])

        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]
        pair_feats.append(self.index_embedder(rel_seq_offset))

        if sc_rigids is not None:
            sc_dgram = calc_distogram(
                sc_rigids[..., 0].get_trans(),
                1e-5,
                20,
                22
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))

        else:
            pair_feats.append(torch.zeros([num_batch, num_res**2, 22], device=prot_t_embed.device))

        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())

        sidechain_frame_embed = self.node_to_sidechain_frame(node_embed[..., None, :]).tile((1, 1, 5, 1))
        sidechain_frame_embed = sidechain_frame_embed + self.pos_embedder(
            torch.arange(5, device=sidechain_frame_embed.device)
        )[None, None]

        return node_embed, edge_embed, sidechain_frame_embed


class EmbedderV2(nn.Module):

    def __init__(self,
                 c_s,
                 c_z,
                 c_frame,
                 c_framepair,
                 c_hidden=64,
                 num_heads=4,
                 block_q=32,
                 block_k=128,
                 n_tfmr_blocks=3,
                 n_pair_embed_blocks=2,
                 index_embed_size=32,
                 break_symmetry=True,
                 rigids_per_residue=3,
                 use_sc_rigid_transformer=False,
                 rigid_transformer_add_vanilla_transformer=False,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_sidechain = c_frame
        self.block_q = block_q
        self.block_k = block_k
        self.break_symmetry = break_symmetry
        self.rigids_per_residue = rigids_per_residue

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=index_embed_size
        )
        self.node_init = Linear(index_embed_size*2, self.c_s, bias=False)

        self.frame_tfmr = SequenceFrameTransformerUpdate(
            c_s,
            c_z,
            c_frame,
            c_framepair,
            num_heads=num_heads,
            n_blocks=n_tfmr_blocks,
            block_q=block_q,
            block_k=block_k,
            do_rigid_updates=False,
            broadcast_singles=True,
            broadcast_pairs=True,
            framepair_init=True,
            framepair_ffn=True,
            add_vanilla_transformer=rigid_transformer_add_vanilla_transformer
        )
        self.use_sc_rigid_transformer = use_sc_rigid_transformer
        if use_sc_rigid_transformer:
            self.sc_frame_tfmr = SequenceFrameTransformerUpdate(
                c_s,
                c_z,
                c_frame,
                c_framepair,
                num_heads=num_heads,
                n_blocks=n_tfmr_blocks,
                block_q=block_q,
                block_k=block_k,
                do_rigid_updates=False,
                broadcast_singles=True,
                broadcast_pairs=True,
                framepair_init=True,
                framepair_ffn=True,
                add_vanilla_transformer=rigid_transformer_add_vanilla_transformer
            )
            self.node_adaln = AdaLN(c_s, c_s)
            self.frame_adaln = AdaLN(c_frame, c_frame)
            self.framepair_adaln = AdaLN(c_framepair, c_framepair)


        self.pair_embedder = MultiRigidPairEmbedder(
            c_z,
            c_hidden,
            no_blocks=n_pair_embed_blocks
        )

        self.node_to_sidechain_frame = Linear(c_s, c_frame, bias=False)
        self.pos_embedder = nn.Embedding(rigids_per_residue, c_frame)

    def forward(
            self,
            *,
            seq_idx,
            t,
            fixed_mask,
            node_mask,
            rigids,
            sc_rigids=None,
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        n_batch = rigids.shape[0]
        n_rigids = rigids.shape[1] * rigids.shape[2]
        n_padding = (self.block_q - n_rigids % self.block_q) % self.block_q
        n_attn_blocks = (n_rigids + n_padding) // self.block_q

        indexing_matrix = get_indexing_matrix(
            n_attn_blocks,
            W=self.block_q,
            H=self.block_k,
            device=t.device
        )
        to_queries = lambda x: x.view(n_batch, n_attn_blocks, self.block_q, -1)
        to_keys = fn.partial(
            single_to_keys, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )
        to_pairs = fn.partial(
            pairs_to_framepairs, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )
        edge_embed = self.pair_embedder(rigids, node_mask, sc_rigids=sc_rigids)

        num_batch, num_res = seq_idx.shape
        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        seq_idx_embed = self.index_embedder(seq_idx)
        node_feats = [t_embed, seq_idx_embed]
        node_embed = torch.cat(node_feats, dim=-1)
        node_embed = self.node_init(node_embed)
        node_init = node_embed

        rigids_embed = self.node_to_sidechain_frame(node_embed[..., None, :]).tile((1, 1, self.rigids_per_residue, 1))
        if self.break_symmetry:
            rigids_embed = rigids_embed + self.pos_embedder(
                torch.arange(self.rigids_per_residue, device=rigids_embed.device)
            )[None, None]
        else:
            rigids_embed = rigids_embed + self.pos_embedder(
                torch.tensor([0] + [1 for _ in range(self.rigids_per_residue-1)], device=rigids_embed.device)
            )[None, None]
        rigids_init = rigids_embed

        rigids_embed, node_embed, framepair_embed, _ = self.frame_tfmr(
            node_embed,
            edge_embed,
            rigids,
            rigids_embed,
            to_queries,
            to_keys,
            to_pairs,
            framepair_embed=None
        )

        if self.use_sc_rigid_transformer:
            if sc_rigids is not None:
                sc_rigids_embed, sc_node_embed, sc_framepair_embed, _ = self.sc_frame_tfmr(
                    node_init,
                    edge_embed,
                    sc_rigids,
                    rigids_init,
                    to_queries,
                    to_keys,
                    to_pairs,
                    framepair_embed=None
                )
            else:
                sc_rigids_embed = torch.zeros_like(rigids_init)
                sc_node_embed = torch.zeros_like(node_init)
                sc_framepair_embed = torch.zeros_like(framepair_embed)

            node_embed = self.node_adaln(node_embed, sc_node_embed)
            rigids_embed = self.frame_adaln(rigids_embed, sc_rigids_embed)
            framepair_embed = self.framepair_adaln(framepair_embed, sc_framepair_embed)

        return node_embed, edge_embed, rigids_embed, framepair_embed


class IpaScore(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=64,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 use_traj_predictions=False,
                 traj_framepair_predictor=False,
                 traj_seqpair_predictor=False,
                 force_flash_transformer=False,
                 block_q=32,
                 block_k=128,
                 propagate_framepair_embed=False,
                 use_knn=False,
                 k=30,
                 rigid_transformer_num_blocks=1,
                 rigid_transformer_rigid_updates=False,
                 rigid_transformer_agg_embed=True,
                 rel_quat_pair_updates=False,
                 z_broadcast=False,
                 compile_ipa=False
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.use_traj_predictions = use_traj_predictions
        self.traj_framepair_predictor = traj_framepair_predictor
        self.traj_seqpair_predictor = traj_seqpair_predictor
        self.force_flash_transformer = force_flash_transformer
        self.propagate_framepair_embed = propagate_framepair_embed
        self.rigid_transformer_rigid_updates = rigid_transformer_rigid_updates
        self.use_knn = use_knn
        self.k = k

        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        self.num_blocks = num_blocks
        self.block_q = block_q
        self.block_k = block_k

        for b in range(num_blocks):
            self.trunk[f'ipa_{b}'] = InvariantPointAttention(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                pre_ln=True,
                lin_bias=False
            )
            if compile_ipa:
                self.trunk[f'ipa_{b}'].compile()

            if self.use_knn:
                self.trunk[f'knn_ipa_{b}'] = KnnInvariantPointAttention(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    num_heads=num_heads,
                    num_qk_points=num_qk_points,
                    num_v_points=num_v_points,
                )

            if force_flash_transformer:
                self.trunk[f'tfmr_{b}'] = FlashTransformerEncoder(
                    h_dim=c_s,
                    n_layers=2,
                    no_heads=4,
                    h_ff=c_s,
                    dropout=0.0,
                    ln_first=True,
                    dtype=None
                )
            else:
                tfmr_layer = torch.nn.TransformerEncoderLayer(
                    d_model=c_s,
                    nhead=4,
                    dim_feedforward=c_s,
                    batch_first=True,
                    dropout=0.0,
                    norm_first=True,
                )
                self.trunk[f'tfmr_{b}'] = torch.nn.TransformerEncoder(
                    tfmr_layer, 2)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                c_s, c_s, init="final", bias=False)
            self.trunk[f'transition_{b}'] = Transition(
                c=c_s,
            )

            self.trunk[f'rigids_tfmr_{b}'] = SequenceFrameTransformerUpdate(
                c_s=c_s,
                c_z=c_z,
                c_frame=c_frame,
                c_framepair=c_framepair,
                num_heads=4, # num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                block_q=block_q,
                block_k=block_k,
                n_blocks=rigid_transformer_num_blocks,
                do_rigid_updates=rigid_transformer_rigid_updates,
                agg_rigid_embed=rigid_transformer_agg_embed,
                broadcast_pairs=z_broadcast
            )
            if not rigid_transformer_rigid_updates:
                self.trunk[f'rigids_update_{b}'] = BackboneUpdate(c_frame)

            if use_traj_predictions:
                self.trunk[f'seq_pred_{b}'] = SeqPredictor(c_frame)
                self.trunk[f'dist_pred_{b}'] = EdgeDistPredictor(c_z)
                if traj_framepair_predictor:
                    self.trunk[f'rigid_dist_pred_{b}'] = FramepairDistPredictor(c_framepair)
                if traj_seqpair_predictor:
                    self.trunk[f'seqpair_pred_{b}'] = SeqPairPredictor(c_z)

            if b < num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ConditionedPairUpdate(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_z//4,
                    no_heads=4,
                    include_rel_quat=rel_quat_pair_updates
                )

        self.torsion_pred = TorsionAngles(c_s, 1)

    @torch.no_grad()
    def _construct_knn_graph(self, rigids):
        trans = rigids.get_trans()
        bb_trans = trans[..., 0, :]
        dist_mat = torch.cdist(bb_trans, bb_trans)
        _, res_edge_index = torch.topk(dist_mat, k=self.k, dim=-1, largest=False)
        return res_edge_index

    def forward(self, init_node_embed, edge_embed, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = input_feats['rigids_t'].type(torch.float32)
        rigids_embed = input_feats['rigids_embed']
        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))

        n_batch = curr_rigids.shape[0]
        n_rigids = curr_rigids.shape[1] * curr_rigids.shape[2]
        n_padding = (self.block_q - n_rigids % self.block_q) % self.block_q
        n_attn_blocks = (n_rigids + n_padding) // self.block_q

        indexing_matrix = get_indexing_matrix(
            n_attn_blocks,
            W=self.block_q,
            H=self.block_k,
            device=rigids_embed.device
        )
        to_queries = lambda x: x.view(n_batch, n_attn_blocks, self.block_q, -1)
        to_keys = fn.partial(
            single_to_keys, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )
        to_pairs = fn.partial(
            pairs_to_framepairs, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]

        traj_data = {
            b: {}
            for b in range(self.num_blocks)
        }

        framepair_embed = input_feats['framepair_embed']
        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                s=node_embed,
                z=edge_embed,
                r=curr_rigids[..., 0],
                mask=node_mask)
            node_embed = (node_embed + ipa_embed) * node_mask[..., None]

            if self.use_knn:
                edge_index = self._construct_knn_graph(curr_rigids)
                knn_edge_embed = torch.gather(
                    edge_embed,
                    2,
                    edge_index[..., None].tile(1, 1, 1, edge_embed.shape[-1])
                )
                knn_ipa_embed = self.trunk[f'knn_ipa_{b}'](
                    s=node_embed,
                    edge_index=edge_index,
                    z=knn_edge_embed,
                    r=curr_rigids[..., 0],
                    s_mask=node_mask)
                node_embed = (node_embed + knn_ipa_embed) * node_mask[..., None]

            if self.force_flash_transformer:
                seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                    node_embed, node_mask)
            else:
                seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                    node_embed, src_key_padding_mask=1 - node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = node_embed * node_mask[..., None]

            node_embed = node_embed + self.trunk[f'transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]

            rigids_embed, node_embed, framepair_embed, rigids_out = self.trunk[f'rigids_tfmr_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                rigids_embed,
                to_queries,
                to_keys,
                to_pairs,
                framepair_embed=framepair_embed
            )
            if not self.propagate_framepair_embed:
                framepair_embed = input_feats['framepair_embed']

            if not self.rigid_transformer_rigid_updates:
                rigid_update = self.trunk[f'rigids_update_{b}'](
                    rigids_embed * diffuse_mask[..., None, None])
                curr_rigids = curr_rigids.compose_q_update_vec(
                    rigid_update * diffuse_mask[..., None, None])
            else:
                curr_rigids = rigids_out

            if b < self.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, curr_rigids[..., 0], edge_mask)
                edge_embed *= edge_mask[..., None]

            if self.use_traj_predictions:
                traj_data[b]['seq_logits'] = self.trunk[f"seq_pred_{b}"](rigids_embed)
                traj_data[b]['dist_logits'] = self.trunk[f"dist_pred_{b}"](edge_embed)
                traj_data[b]['rigids'] = self.unscale_rigids(curr_rigids)
                if self.traj_framepair_predictor:
                    traj_data[b]['framepair_logits_dict'] = {
                        'logits': self.trunk[f'rigid_dist_pred_{b}'](framepair_embed),
                        'to_queries': to_queries,
                        'to_keys': to_keys,
                        'n_padding': n_padding
                    }
                if self.traj_seqpair_predictor:
                    traj_data[b]['seqpair_logits'] = self.trunk[f'seqpair_pred_{b}'](edge_embed)

        curr_rigids = self.unscale_rigids(curr_rigids)
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'node_embed': node_embed,
            'traj_data': traj_data
        }
        return model_out


class IpaScoreV2(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=64,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 use_traj_predictions=False,
                 traj_framepair_predictor=False,
                 traj_seqpair_predictor=False,
                 force_flash_transformer=False,
                 block_q=32,
                 block_k=128,
                 propagate_framepair_embed=False,
                 use_knn=False,
                 k=30,
                 rigid_transformer_num_blocks=1,
                 rigid_transformer_rigid_updates=False,
                 rigid_transformer_agg_embed=True,
                 rigid_embed_skip_sc=False,
                 rel_quat_pair_updates=False,
                 z_broadcast=False,
                 compile_ipa=False
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.use_traj_predictions = use_traj_predictions
        self.traj_framepair_predictor = traj_framepair_predictor
        self.traj_seqpair_predictor = traj_seqpair_predictor
        self.force_flash_transformer = force_flash_transformer
        self.propagate_framepair_embed = propagate_framepair_embed
        self.use_knn = use_knn
        self.rigid_transformer_rigid_updates = rigid_transformer_rigid_updates
        self.k = k

        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        self.num_blocks = num_blocks
        self.block_q = block_q
        self.block_k = block_k

        for b in range(num_blocks):
            self.trunk[f'ipa_{b}'] = InvariantPointAttention(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                pre_ln=True,
                lin_bias=False
            )
            if compile_ipa:
                self.trunk[f'ipa_{b}'].compile()
            if self.use_knn:
                self.trunk[f'knn_ipa_{b}'] = KnnInvariantPointAttention(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    num_heads=num_heads,
                    num_qk_points=num_qk_points,
                    num_v_points=num_v_points,
                )

            if force_flash_transformer:
                self.trunk[f'tfmr_{b}'] = FlashTransformerEncoder(
                    h_dim=c_s,
                    n_layers=2,
                    no_heads=4,
                    h_ff=c_s,
                    dropout=0.0,
                    ln_first=True,
                    dtype=None
                )
            else:
                tfmr_layer = torch.nn.TransformerEncoderLayer(
                    d_model=c_s,
                    nhead=4,
                    dim_feedforward=c_s,
                    batch_first=True,
                    dropout=0.0,
                    norm_first=True,
                )
                self.trunk[f'tfmr_{b}'] = torch.nn.TransformerEncoder(
                    tfmr_layer, 2)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                c_s, c_s, init="final", bias=False)
            self.trunk[f'transition_{b}'] = Transition(
                c=c_s,
            )

            self.trunk[f'rigids_tfmr_{b}'] = SequenceFrameTransformerUpdate(
                c_s=c_s,
                c_z=c_z,
                c_frame=c_frame,
                c_framepair=c_framepair,
                num_heads=4, # num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                block_q=block_q,
                block_k=block_k,
                n_blocks=rigid_transformer_num_blocks,
                do_rigid_updates=rigid_transformer_rigid_updates,
                agg_rigid_embed=rigid_transformer_agg_embed,
                broadcast_pairs=z_broadcast
            )
            if not rigid_transformer_rigid_updates:
                self.trunk[f'rigids_update_{b}'] = BackboneUpdate(c_frame)

            if use_traj_predictions:
                self.trunk[f'seq_pred_{b}'] = SeqPredictor(c_frame)
                self.trunk[f'dist_pred_{b}'] = EdgeDistPredictor(c_z)
                if traj_framepair_predictor:
                    self.trunk[f'rigid_dist_pred_{b}'] = FramepairDistPredictor(c_framepair)
                if traj_seqpair_predictor:
                    self.trunk[f'seqpair_pred_{b}'] = SeqPairPredictor(c_z)

            if b < num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ConditionedPairUpdate(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_z//4,
                    no_heads=4,
                    include_rel_quat=rel_quat_pair_updates
                )

        self.torsion_pred = TorsionAngles(c_s, 1)

    @torch.no_grad()
    def _construct_knn_graph(self, rigids):
        trans = rigids.get_trans()
        bb_trans = trans[..., 0, :]
        dist_mat = torch.cdist(bb_trans, bb_trans)
        _, res_edge_index = torch.topk(dist_mat, k=self.k, dim=-1, largest=False)
        return res_edge_index

    def forward(self, init_node_embed, edge_embed, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = input_feats['rigids_t'].type(torch.float32)
        rigids_embed = input_feats['rigids_embed']
        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))

        n_batch = curr_rigids.shape[0]
        n_rigids = curr_rigids.shape[1] * curr_rigids.shape[2]
        n_padding = (self.block_q - n_rigids % self.block_q) % self.block_q
        n_attn_blocks = (n_rigids + n_padding) // self.block_q

        indexing_matrix = get_indexing_matrix(
            n_attn_blocks,
            W=self.block_q,
            H=self.block_k,
            device=rigids_embed.device
        )
        to_queries = lambda x: x.view(n_batch, n_attn_blocks, self.block_q, -1)
        to_keys = fn.partial(
            single_to_keys, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )
        to_pairs = fn.partial(
            pairs_to_framepairs, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )

        # Main trunk
        rigids_skip = self.scale_rigids(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        rigids_embed_skip = rigids_embed

        traj_data = {
            b: {}
            for b in range(self.num_blocks)
        }

        framepair_embed_skip = input_feats['framepair_embed']

        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                s=node_embed,
                z=edge_embed,
                r=rigids_skip[..., 0],
                mask=node_mask)
            node_embed = (node_embed + ipa_embed) * node_mask[..., None]

            if self.use_knn:
                edge_index = self._construct_knn_graph(curr_rigids)
                knn_edge_embed = torch.gather(
                    edge_embed,
                    2,
                    edge_index[..., None].tile(1, 1, 1, edge_embed.shape[-1])
                )
                knn_ipa_embed = self.trunk[f'knn_ipa_{b}'](
                    s=node_embed,
                    edge_index=edge_index,
                    z=knn_edge_embed,
                    r=rigids_skip[..., 0],
                    s_mask=node_mask)
                node_embed = (node_embed + knn_ipa_embed) * node_mask[..., None]

            if self.force_flash_transformer:
                seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                    node_embed, node_mask)
            else:
                seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                    node_embed, src_key_padding_mask=1 - node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = node_embed * node_mask[..., None]

            node_embed = node_embed + self.trunk[f'transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]

            rigids_embed, node_embed, framepair_embed, rigids_out = self.trunk[f'rigids_tfmr_{b}'](
                node_embed,
                edge_embed,
                rigids_skip,
                rigids_embed_skip,
                to_queries,
                to_keys,
                to_pairs,
                framepair_embed=framepair_embed_skip
            )
            if self.propagate_framepair_embed:
                framepair_embed_skip = framepair_embed

            if not self.rigid_transformer_rigid_updates:
                rigid_update = self.trunk[f'rigids_update_{b}'](
                    rigids_embed * diffuse_mask[..., None, None])
                curr_rigids = rigids_skip.compose_q_update_vec(
                    rigid_update * diffuse_mask[..., None, None])
            else:
                curr_rigids = rigids_out

            if b < self.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, curr_rigids[..., 0], edge_mask)
                edge_embed *= edge_mask[..., None]

            if self.use_traj_predictions:
                traj_data[b]['seq_logits'] = self.trunk[f"seq_pred_{b}"](rigids_embed)
                traj_data[b]['dist_logits'] = self.trunk[f"dist_pred_{b}"](edge_embed)
                traj_data[b]['rigids'] = self.unscale_rigids(curr_rigids)
                if self.traj_framepair_predictor:
                    traj_data[b]['framepair_logits_dict'] = {
                        'logits': self.trunk[f'rigid_dist_pred_{b}'](framepair_embed),
                        'to_queries': to_queries,
                        'to_keys': to_keys,
                        'n_padding': n_padding
                    }
                if self.traj_seqpair_predictor:
                    traj_data[b]['seqpair_logits'] = self.trunk[f'seqpair_pred_{b}'](edge_embed)

        curr_rigids = self.unscale_rigids(curr_rigids)
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'node_embed': node_embed,
            'traj_data': traj_data
        }
        return model_out


class IpaScoreV3(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=64,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 use_traj_predictions=False,
                 traj_framepair_predictor=False,
                 traj_seqpair_predictor=False,
                 force_flash_transformer=False,
                 block_q=32,
                 block_k=128,
                 propagate_framepair_embed=False,
                 use_knn=False,
                 k=30,
                 rigid_transformer_num_blocks=1,
                 rigid_transformer_rigid_updates=False,
                 rigid_transformer_agg_embed=True,
                 rigid_embed_skip_sc=False,
                 rel_quat_pair_updates=False,
                 z_broadcast=False,
                 compile_ipa=False
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.use_traj_predictions = use_traj_predictions
        self.traj_framepair_predictor = traj_framepair_predictor
        self.traj_seqpair_predictor = traj_seqpair_predictor
        self.force_flash_transformer = force_flash_transformer
        self.propagate_framepair_embed = propagate_framepair_embed
        self.use_knn = use_knn
        self.rigid_transformer_rigid_updates = rigid_transformer_rigid_updates
        self.rigid_embed_skip_sc = rigid_embed_skip_sc
        self.k = k

        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        self.num_blocks = num_blocks
        self.block_q = block_q
        self.block_k = block_k

        for b in range(num_blocks):
            self.trunk[f'ipa_{b}'] = InvariantPointAttention(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                pre_ln=True,
                lin_bias=False
            )
            if compile_ipa:
                self.trunk[f'ipa_{b}'].compile()
            if self.use_knn:
                self.trunk[f'knn_ipa_{b}'] = KnnInvariantPointAttention(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    num_heads=num_heads,
                    num_qk_points=num_qk_points,
                    num_v_points=num_v_points,
                )

            if force_flash_transformer:
                self.trunk[f'tfmr_{b}'] = FlashTransformerEncoder(
                    h_dim=c_s,
                    n_layers=2,
                    no_heads=4,
                    h_ff=c_s,
                    dropout=0.0,
                    ln_first=True,
                    dtype=None
                )
            else:
                tfmr_layer = torch.nn.TransformerEncoderLayer(
                    d_model=c_s,
                    nhead=4,
                    dim_feedforward=c_s,
                    batch_first=True,
                    dropout=0.0,
                    norm_first=True,
                )
                self.trunk[f'tfmr_{b}'] = torch.nn.TransformerEncoder(
                    tfmr_layer, 2)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                c_s, c_s, init="final", bias=False)
            self.trunk[f'transition_{b}'] = Transition(
                c=c_s,
            )

            self.trunk[f'rigids_tfmr_{b}'] = SequenceFrameTransformerUpdate(
                c_s=c_s,
                c_z=c_z,
                c_frame=c_frame,
                c_framepair=c_framepair,
                num_heads=4, # num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                block_q=block_q,
                block_k=block_k,
                n_blocks=rigid_transformer_num_blocks,
                do_rigid_updates=rigid_transformer_rigid_updates,
                agg_rigid_embed=rigid_transformer_agg_embed,
                broadcast_pairs=z_broadcast
            )
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(c_s)

            if use_traj_predictions:
                self.trunk[f'seq_pred_{b}'] = SeqPredictor(c_frame)
                self.trunk[f'dist_pred_{b}'] = EdgeDistPredictor(c_z)
                if traj_framepair_predictor:
                    self.trunk[f'rigid_dist_pred_{b}'] = FramepairDistPredictor(c_framepair)
                if traj_seqpair_predictor:
                    self.trunk[f'seqpair_pred_{b}'] = SeqPairPredictor(c_z)

            if b < num_blocks-1:
                if self.rigid_embed_skip_sc:
                    self.trunk[f'rigid_embed_skip_sc_{b}'] = nn.Sequential(
                        LayerNorm(c_s),
                        Linear(c_s, c_frame, bias=False, init='final')
                    )
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ConditionedPairUpdate(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_z//4,
                    no_heads=4,
                    include_rel_quat=rel_quat_pair_updates
                )

        self.torsion_pred = TorsionAngles(c_s, 1)

    @torch.no_grad()
    def _construct_knn_graph(self, rigids):
        trans = rigids.get_trans()
        bb_trans = trans[..., 0, :]
        dist_mat = torch.cdist(bb_trans, bb_trans)
        _, res_edge_index = torch.topk(dist_mat, k=self.k, dim=-1, largest=False)
        return res_edge_index

    def forward(self, init_node_embed, edge_embed, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = input_feats['rigids_t'].type(torch.float32)
        rigids_embed = input_feats['rigids_embed']
        init_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        bb_rigids = Rigid.from_tensor_7(torch.clone(init_frames[..., 0, :]))

        n_batch = init_rigids.shape[0]
        n_rigids = init_rigids.shape[1] * init_rigids.shape[2]
        n_padding = (self.block_q - n_rigids % self.block_q) % self.block_q
        n_attn_blocks = (n_rigids + n_padding) // self.block_q

        indexing_matrix = get_indexing_matrix(
            n_attn_blocks,
            W=self.block_q,
            H=self.block_k,
            device=rigids_embed.device
        )
        to_queries = lambda x: x.view(n_batch, n_attn_blocks, self.block_q, -1)
        to_keys = fn.partial(
            single_to_keys, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )
        to_pairs = fn.partial(
            pairs_to_framepairs, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )

        # Main trunk
        rigids_skip = self.scale_rigids(init_rigids)
        bb_rigids = self.scale_rigids(bb_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        rigids_embed_skip = rigids_embed

        traj_data = {
            b: {}
            for b in range(self.num_blocks)
        }

        framepair_embed_skip = input_feats['framepair_embed']
        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                s=node_embed,
                z=edge_embed,
                r=bb_rigids,
                mask=node_mask)
            node_embed = (node_embed + ipa_embed) * node_mask[..., None]

            if self.force_flash_transformer:
                seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                    node_embed, node_mask)
            else:
                seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                    node_embed, src_key_padding_mask=1 - node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = node_embed * node_mask[..., None]

            node_embed = node_embed + self.trunk[f'transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]

            rigids_embed, node_embed, framepair_embed, curr_rigids = self.trunk[f'rigids_tfmr_{b}'](
                node_embed,
                edge_embed,
                rigids_skip,
                rigids_embed_skip,
                to_queries,
                to_keys,
                to_pairs,
                framepair_embed=framepair_embed_skip
            )

            bb_rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * diffuse_mask[..., None])
            bb_rigids = bb_rigids.compose_q_update_vec(
                bb_rigid_update * diffuse_mask[..., None])

            if b < self.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, curr_rigids[..., 0], edge_mask)
                edge_embed *= edge_mask[..., None]

            if self.use_traj_predictions:
                traj_data[b]['seq_logits'] = self.trunk[f"seq_pred_{b}"](rigids_embed)
                traj_data[b]['dist_logits'] = self.trunk[f"dist_pred_{b}"](edge_embed)
                traj_data[b]['rigids'] = self.unscale_rigids(curr_rigids)
                traj_data[b]['bb_rigids'] = self.unscale_rigids(bb_rigids)
                if self.traj_framepair_predictor:
                    traj_data[b]['framepair_logits_dict'] = {
                        'logits': self.trunk[f'rigid_dist_pred_{b}'](framepair_embed),
                        'to_queries': to_queries,
                        'to_keys': to_keys,
                        'n_padding': n_padding
                    }
                if self.traj_seqpair_predictor:
                    traj_data[b]['seqpair_logits'] = self.trunk[f'seqpair_pred_{b}'](edge_embed)

            if self.propagate_framepair_embed:
                framepair_embed_skip = framepair_embed

        curr_rigids = self.unscale_rigids(curr_rigids)
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'node_embed': node_embed,
            'traj_data': traj_data
        }
        return model_out


class IpaScoreV4(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=64,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 use_traj_predictions=False,
                 traj_framepair_predictor=False,
                 traj_seqpair_predictor=False,
                 force_flash_transformer=False,
                 block_q=32,
                 block_k=128,
                 propagate_framepair_embed=False,
                 use_knn=False,
                 k=30,
                 rigid_transformer_num_blocks=1,
                 rigid_transformer_rigid_updates=False,
                 rigid_transformer_agg_embed=True,
                 rigid_transformer_framepair_init=False,
                 rigid_embed_skip_sc=False,
                 rel_quat_pair_updates=False,
                 z_broadcast=False,
                 compile_ipa=False
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.use_traj_predictions = use_traj_predictions
        self.traj_framepair_predictor = traj_framepair_predictor
        self.traj_seqpair_predictor = traj_seqpair_predictor
        self.force_flash_transformer = force_flash_transformer
        self.propagate_framepair_embed = propagate_framepair_embed
        self.use_knn = use_knn
        self.rigid_transformer_rigid_updates = rigid_transformer_rigid_updates
        self.rigid_embed_skip_sc = rigid_embed_skip_sc
        self.k = k

        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        self.num_blocks = num_blocks
        self.block_q = block_q
        self.block_k = block_k

        for b in range(num_blocks):
            self.trunk[f'ipa_{b}'] = InvariantPointAttention(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                pre_ln=True,
                lin_bias=False
            )
            if compile_ipa:
                self.trunk[f'ipa_{b}'].compile()
            if self.use_knn:
                self.trunk[f'knn_ipa_{b}'] = KnnInvariantPointAttention(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    num_heads=num_heads,
                    num_qk_points=num_qk_points,
                    num_v_points=num_v_points,
                )

            if force_flash_transformer:
                self.trunk[f'tfmr_{b}'] = FlashTransformerEncoder(
                    h_dim=c_s,
                    n_layers=2,
                    no_heads=4,
                    h_ff=c_s,
                    dropout=0.0,
                    ln_first=True,
                    dtype=None
                )
            else:
                tfmr_layer = torch.nn.TransformerEncoderLayer(
                    d_model=c_s,
                    nhead=4,
                    dim_feedforward=c_s,
                    batch_first=True,
                    dropout=0.0,
                    norm_first=True,
                )
                self.trunk[f'tfmr_{b}'] = torch.nn.TransformerEncoder(
                    tfmr_layer, 2)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                c_s, c_s, init="final", bias=False)
            self.trunk[f'transition_{b}'] = Transition(
                c=c_s,
            )

            self.trunk[f'rigids_tfmr_{b}'] = ConditionedSequenceFrameTransformerUpdate(
                c_s=c_s,
                c_z=c_z,
                c_frame=c_frame,
                c_framepair=c_framepair,
                num_heads=4, # num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                block_q=block_q,
                block_k=block_k,
                n_blocks=rigid_transformer_num_blocks,
                do_rigid_updates=rigid_transformer_rigid_updates,
                framepair_init=rigid_transformer_framepair_init,
                agg_rigid_embed=rigid_transformer_agg_embed,
                broadcast_pairs=z_broadcast,
            )
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(c_s)

            if use_traj_predictions:
                self.trunk[f'seq_pred_{b}'] = SeqPredictor(c_frame)
                self.trunk[f'dist_pred_{b}'] = EdgeDistPredictor(c_z)
                if traj_framepair_predictor:
                    self.trunk[f'rigid_dist_pred_{b}'] = FramepairDistPredictor(c_framepair)
                if traj_seqpair_predictor:
                    self.trunk[f'seqpair_pred_{b}'] = SeqPairPredictor(c_z)

            if b < num_blocks-1:
                if self.rigid_embed_skip_sc:
                    self.trunk[f'rigid_embed_skip_sc_{b}'] = nn.Sequential(
                        LayerNorm(c_s),
                        Linear(c_s, c_frame, bias=False, init='final')
                    )
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ConditionedPairUpdate(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_z//4,
                    no_heads=4,
                    include_rel_quat=rel_quat_pair_updates
                )

        self.torsion_pred = TorsionAngles(c_s, 1)

    @torch.no_grad()
    def _construct_knn_graph(self, rigids):
        trans = rigids.get_trans()
        bb_trans = trans[..., 0, :]
        dist_mat = torch.cdist(bb_trans, bb_trans)
        _, res_edge_index = torch.topk(dist_mat, k=self.k, dim=-1, largest=False)
        return res_edge_index

    def forward(self, init_node_embed, edge_embed, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = input_feats['rigids_t'].type(torch.float32)
        rigids_embed = input_feats['rigids_embed']
        init_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        bb_rigids = Rigid.from_tensor_7(torch.clone(init_frames[..., 0, :]))

        n_batch = init_rigids.shape[0]
        n_rigids = init_rigids.shape[1] * init_rigids.shape[2]
        n_padding = (self.block_q - n_rigids % self.block_q) % self.block_q
        n_attn_blocks = (n_rigids + n_padding) // self.block_q

        indexing_matrix = get_indexing_matrix(
            n_attn_blocks,
            W=self.block_q,
            H=self.block_k,
            device=rigids_embed.device
        )
        to_queries = lambda x: x.view(n_batch, n_attn_blocks, self.block_q, -1)
        to_keys = fn.partial(
            single_to_keys, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )
        to_pairs = fn.partial(
            pairs_to_framepairs, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )

        # Main trunk
        rigids_skip = self.scale_rigids(init_rigids)
        bb_rigids = self.scale_rigids(bb_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        rigids_embed_skip = rigids_embed

        traj_data = {
            b: {}
            for b in range(self.num_blocks)
        }

        framepair_embed_skip = input_feats['framepair_embed']
        latest_rigids_embed = rigids_embed_skip

        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                s=node_embed,
                z=edge_embed,
                r=bb_rigids,
                mask=node_mask)
            node_embed = (node_embed + ipa_embed) * node_mask[..., None]

            if self.force_flash_transformer:
                seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                    node_embed, node_mask)
            else:
                seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                    node_embed, src_key_padding_mask=1 - node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = node_embed * node_mask[..., None]

            node_embed = node_embed + self.trunk[f'transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]

            latest_rigids_embed, node_embed, framepair_embed, curr_rigids = self.trunk[f'rigids_tfmr_{b}'](
                node_embed,
                edge_embed,
                rigids_skip,
                rigids_embed_skip,
                latest_rigids_embed,
                to_queries,
                to_keys,
                to_pairs,
                framepair_embed=framepair_embed_skip
            )
            if self.propagate_framepair_embed:
                framepair_embed_skip = framepair_embed

            bb_rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * diffuse_mask[..., None])
            bb_rigids = bb_rigids.compose_q_update_vec(
                bb_rigid_update * diffuse_mask[..., None])

            if b < self.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, curr_rigids[..., 0], edge_mask)
                edge_embed *= edge_mask[..., None]


            if self.use_traj_predictions:
                traj_data[b]['seq_logits'] = self.trunk[f"seq_pred_{b}"](rigids_embed)
                traj_data[b]['dist_logits'] = self.trunk[f"dist_pred_{b}"](edge_embed)
                traj_data[b]['rigids'] = self.unscale_rigids(curr_rigids)
                traj_data[b]['bb_rigids'] = self.unscale_rigids(bb_rigids)
                if self.traj_framepair_predictor:
                    traj_data[b]['framepair_logits_dict'] = {
                        'logits': self.trunk[f'rigid_dist_pred_{b}'](framepair_embed),
                        'to_queries': to_queries,
                        'to_keys': to_keys,
                        'n_padding': n_padding
                    }
                if self.traj_seqpair_predictor:
                    traj_data[b]['seqpair_logits'] = self.trunk[f'seqpair_pred_{b}'](edge_embed)

        curr_rigids = self.unscale_rigids(curr_rigids)
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'node_embed': node_embed,
            'traj_data': traj_data
        }
        return model_out


class EmbedderForV5(nn.Module):

    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden=64,
                 num_heads=4,
                 block_q=32,
                 block_k=128,
                 n_tfmr_blocks=3,
                 n_pair_embed_blocks=2,
                 index_embed_size=32,
                 break_symmetry=True,
                 rigids_per_residue=3
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.block_q = block_q
        self.block_k = block_k
        self.break_symmetry = break_symmetry
        self.rigids_per_residue = rigids_per_residue

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=index_embed_size
        )

        self.node_init = Linear(index_embed_size*2, self.c_s, bias=False)
        self.pos_embedder = nn.Embedding(rigids_per_residue, c_s)

        self.ipa = BroadcastZInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=16,
            num_heads=8,
            num_qk_points=8,
            num_v_points=12,
            pre_ln=True,
            lin_bias=False
        )

        tfmr_layer = torch.nn.TransformerEncoderLayer(
            d_model=c_s,
            nhead=4,
            dim_feedforward=c_s,
            batch_first=True,
            dropout=0.0,
            norm_first=True,
        )
        self.tfmr = torch.nn.TransformerEncoder(
            tfmr_layer, 3)

        self.pair_embedder = MultiRigidPairEmbedder(
            c_z,
            c_hidden,
            no_blocks=n_pair_embed_blocks
        )


    def forward(
            self,
            *,
            seq_idx,
            t,
            fixed_mask,
            node_mask,
            rigids,
            sc_rigids=None,
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        n_batch = rigids.shape[0]
        n_rigids = rigids.shape[1] * rigids.shape[2]
        n_padding = (self.block_q - n_rigids % self.block_q) % self.block_q
        n_attn_blocks = (n_rigids + n_padding) // self.block_q

        edge_embed = self.pair_embedder(rigids, node_mask, sc_rigids=sc_rigids)

        num_batch, num_res = seq_idx.shape
        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        seq_idx_embed = self.index_embedder(seq_idx)
        node_feats = [t_embed, seq_idx_embed]
        node_embed = torch.cat(node_feats, dim=-1)
        node_embed = self.node_init(node_embed)

        rigids_embed = node_embed[..., None, :].tile((1, 1, self.rigids_per_residue, 1))
        rigids_embed = rigids_embed + self.pos_embedder(
            torch.arange(self.rigids_per_residue, device=rigids_embed.device)
        )[None, None]
        rigids_embed = rigids_embed.view(n_batch, -1, self.c_s)

        rigid_to_res_idx = torch.arange(rigids.shape[1], device=rigids_embed.device)[None, :, None].expand(
            n_batch, -1, rigids.shape[-1]
        ).reshape(
            n_batch, -1
        )
        rigids_mask = node_mask.repeat_interleave(rigids.shape[-1], dim=-1).float()
        rigids_update = self.ipa(
            s=rigids_embed,
            z=edge_embed,
            r=rigids.view(n_batch, -1),
            mask=rigids_mask,
            rigid_to_res_idx=rigid_to_res_idx)
        rigids_embed = rigids_embed + rigids_update * rigids_mask[..., None]

        rigids_embed = self.tfmr(
            rigids_embed,
            src_key_padding_mask=1 - rigids_mask.float(),
        )
        rigids_embed = rigids_embed.view(n_batch, -1, self.rigids_per_residue, self.c_s)

        return node_embed, edge_embed, rigids_embed


class IpaScoreV5(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 use_traj_predictions=False,
                 traj_framepair_predictor=False,
                 traj_seqpair_predictor=False,
                 force_flash_transformer=False,
                 block_q=32,
                 block_k=128,
                 propagate_framepair_embed=False,
                 rel_quat_pair_updates=False,
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.c_s = c_s
        self.use_traj_predictions = use_traj_predictions
        self.traj_framepair_predictor = traj_framepair_predictor
        self.traj_seqpair_predictor = traj_seqpair_predictor
        self.force_flash_transformer = force_flash_transformer
        self.propagate_framepair_embed = propagate_framepair_embed

        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        self.num_blocks = num_blocks
        self.block_q = block_q
        self.block_k = block_k

        for b in range(num_blocks):
            self.trunk[f'ipa_{b}'] = BroadcastZInvariantPointAttention(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                pre_ln=True,
                lin_bias=False
            )

            # self.trunk[f'tfmr_{b}'] = FlashTransformerEncoder(
            #     h_dim=c_s,
            #     n_layers=2,
            #     no_heads=4,
            #     h_ff=c_s,
            #     dropout=0.0,
            #     ln_first=True,
            #     dtype=None
            # )
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=c_s,
                nhead=4,
                dim_feedforward=c_s,
                batch_first=True,
                dropout=0.0,
                norm_first=True,
            )
            self.trunk[f'tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, 2)

            self.trunk[f'post_tfmr_{b}'] = Linear(
                c_s, c_s, init="final", bias=False)
            self.trunk[f'transition_{b}'] = Transition(
                c=c_s,
            )
            self.trunk[f'rigids_update_{b}'] = BackboneUpdate(c_s)

            if use_traj_predictions:
                self.trunk[f'seq_pred_{b}'] = SeqPredictor(c_s)
                self.trunk[f'dist_pred_{b}'] = EdgeDistPredictor(c_z)
                if traj_seqpair_predictor:
                    self.trunk[f'seqpair_pred_{b}'] = SeqPairPredictor(c_z)

            if b < num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ConditionedPairUpdate(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_z//4,
                    no_heads=4,
                    include_rel_quat=rel_quat_pair_updates
                )

        self.torsion_pred = TorsionAngles(c_s, 1)

    def forward(self, init_node_embed, edge_embed, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        rigids_embed = input_feats['rigids_embed']
        init_frames = input_feats['rigids_t'].type(torch.float32)
        init_rigids = Rigid.from_tensor_7(init_frames)
        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        rigids_mask = node_mask.repeat_interleave(init_rigids.shape[-1], dim=-1)
        diffuse_mask = diffuse_mask.repeat_interleave(init_rigids.shape[-1], dim=-1)

        batch_size = curr_rigids.shape[0]
        curr_rigids = curr_rigids.view(batch_size, -1)
        init_rigids_embed = rigids_embed

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)
        n_res = rigids_embed.shape[1]
        rigid_to_res_idx = torch.arange(n_res, device=rigids_embed.device)[None, :, None].expand(
            batch_size, -1, init_rigids.shape[-1]
        ).reshape(
            batch_size, -1
        )

        traj_data = {
            b: {}
            for b in range(self.num_blocks)
        }

        for b in range(self.num_blocks):
            rigids_embed = rigids_embed.view(batch_size, -1, rigids_embed.shape[-1])
            ipa_embed = self.trunk[f'ipa_{b}'](
                s=rigids_embed,
                z=edge_embed,
                r=curr_rigids,
                mask=rigids_mask,
                rigid_to_res_idx=rigid_to_res_idx)
            rigids_embed = (rigids_embed + ipa_embed) * rigids_mask[..., None]

            seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                rigids_embed, src_key_padding_mask=1 - rigids_mask.float())

            rigids_embed = rigids_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            rigids_embed = rigids_embed * rigids_mask[..., None]

            rigids_embed = rigids_embed + self.trunk[f'transition_{b}'](rigids_embed)
            rigids_embed = rigids_embed * rigids_mask[..., None]

            rigid_update = self.trunk[f'rigids_update_{b}'](
                rigids_embed * diffuse_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update * diffuse_mask[..., None])

            if b < self.num_blocks-1:
                res_embed = rigids_embed.view(init_rigids_embed.shape).mean(dim=-2)
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    res_embed, edge_embed, curr_rigids.view(init_rigids.shape)[..., 0], edge_mask)
                edge_embed *= edge_mask[..., None]

            if self.use_traj_predictions:
                traj_data[b]['seq_logits'] = self.trunk[f"seq_pred_{b}"](rigids_embed.view(init_rigids_embed.shape))
                traj_data[b]['dist_logits'] = self.trunk[f"dist_pred_{b}"](edge_embed)
                traj_data[b]['rigids'] = self.unscale_rigids(curr_rigids.view(init_rigids.shape))


        curr_rigids = self.unscale_rigids(curr_rigids.view(init_rigids.shape))
        _, psi_pred = self.torsion_pred(rigids_embed.view(init_rigids_embed.shape)[..., 0, :])
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'node_embed': rigids_embed,
            'traj_data': traj_data
        }
        return model_out


class IpaScoreV6(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=64,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 use_traj_predictions=False,
                 traj_framepair_predictor=False,
                 traj_seqpair_predictor=False,
                 force_flash_transformer=False,
                 block_q=32,
                 block_k=128,
                 propagate_framepair_embed=False,
                 use_knn=False,
                 k=30,
                 rigid_transformer_num_blocks=1,
                 rigid_transformer_rigid_updates=False,
                 rigid_embed_skip_sc=False,
                 rigid_transformer_add_vanilla_transformer=False,
                 rel_quat_pair_updates=False,
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.use_traj_predictions = use_traj_predictions
        self.traj_framepair_predictor = traj_framepair_predictor
        self.traj_seqpair_predictor = traj_seqpair_predictor
        self.force_flash_transformer = force_flash_transformer
        self.propagate_framepair_embed = propagate_framepair_embed
        self.use_knn = use_knn
        self.rigid_transformer_rigid_updates = rigid_transformer_rigid_updates
        self.rigid_embed_skip_sc = rigid_embed_skip_sc
        self.k = k

        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        self.num_blocks = num_blocks
        self.block_q = block_q
        self.block_k = block_k

        for b in range(num_blocks):
            self.trunk[f'ipa_{b}'] = InvariantPointAttention(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                pre_ln=True,
                lin_bias=False
            )

            if force_flash_transformer:
                self.trunk[f'tfmr_{b}'] = FlashTransformerEncoder(
                    h_dim=c_s,
                    n_layers=2,
                    no_heads=4,
                    h_ff=c_s,
                    dropout=0.0,
                    ln_first=True,
                    dtype=None
                )
            else:
                tfmr_layer = torch.nn.TransformerEncoderLayer(
                    d_model=c_s,
                    nhead=4,
                    dim_feedforward=c_s,
                    batch_first=True,
                    dropout=0.0,
                    norm_first=True,
                )
                self.trunk[f'tfmr_{b}'] = torch.nn.TransformerEncoder(
                    tfmr_layer, 2)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                c_s, c_s, init="final", bias=False)
            self.trunk[f'transition_{b}'] = Transition(
                c=c_s,
            )
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(c_s)

            if use_traj_predictions:
                self.trunk[f'seq_pred_{b}'] = SeqPredictor(c_frame)
                self.trunk[f'dist_pred_{b}'] = EdgeDistPredictor(c_z)
                if traj_seqpair_predictor:
                    self.trunk[f'seqpair_pred_{b}'] = SeqPairPredictor(c_z)

            if b < num_blocks-1:
                if self.rigid_embed_skip_sc:
                    self.trunk[f'rigid_embed_skip_sc_{b}'] = nn.Sequential(
                        LayerNorm(c_s),
                        Linear(c_s, c_frame, bias=False, init='final')
                    )
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ConditionedPairUpdate(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_z//4,
                    no_heads=4,
                    include_rel_quat=rel_quat_pair_updates
                )

        self.node_to_rigid_cond = Linear(c_s, c_frame*3, bias=False)

        self.rigids_tfmr = ConditionedSequenceFrameTransformerUpdate(
            c_s=c_s,
            c_z=c_z,
            c_frame=c_frame,
            c_framepair=c_framepair,
            num_heads=4, # num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            block_q=block_q,
            block_k=block_k,
            n_blocks=rigid_transformer_num_blocks,
            do_rigid_updates=True,
            agg_rigid_embed=True,
            broadcast_pairs=True,
            add_vanilla_transformer=rigid_transformer_add_vanilla_transformer
        )
        if traj_framepair_predictor:
            self.rigid_dist_pred = FramepairDistPredictor(c_framepair)

        self.torsion_pred = TorsionAngles(c_s, 1)

    @torch.no_grad()
    def _construct_knn_graph(self, rigids):
        trans = rigids.get_trans()
        bb_trans = trans[..., 0, :]
        dist_mat = torch.cdist(bb_trans, bb_trans)
        _, res_edge_index = torch.topk(dist_mat, k=self.k, dim=-1, largest=False)
        return res_edge_index

    def forward(self, init_node_embed, edge_embed, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = input_feats['rigids_t'].type(torch.float32)
        rigids_embed = input_feats['rigids_embed']
        init_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        bb_rigids = Rigid.from_tensor_7(torch.clone(init_frames[..., 0, :]))

        n_batch = init_rigids.shape[0]
        n_rigids = init_rigids.shape[1] * init_rigids.shape[2]
        n_padding = (self.block_q - n_rigids % self.block_q) % self.block_q
        n_attn_blocks = (n_rigids + n_padding) // self.block_q

        indexing_matrix = get_indexing_matrix(
            n_attn_blocks,
            W=self.block_q,
            H=self.block_k,
            device=rigids_embed.device
        )
        to_queries = lambda x: x.view(n_batch, n_attn_blocks, self.block_q, -1)
        to_keys = fn.partial(
            single_to_keys, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )
        to_pairs = fn.partial(
            pairs_to_framepairs, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )

        # Main trunk
        rigids_skip = self.scale_rigids(init_rigids)
        bb_rigids = self.scale_rigids(bb_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        rigids_embed_skip = rigids_embed
        framepair_embed_skip = input_feats['framepair_embed']

        traj_data = {
            b: {}
            for b in range(self.num_blocks)
        }

        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                s=node_embed,
                z=edge_embed,
                r=bb_rigids,
                mask=node_mask)
            node_embed = (node_embed + ipa_embed) * node_mask[..., None]

            if self.force_flash_transformer:
                seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                    node_embed, node_mask)
            else:
                seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                    node_embed, src_key_padding_mask=1 - node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = node_embed * node_mask[..., None]

            node_embed = node_embed + self.trunk[f'transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]


            bb_rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * diffuse_mask[..., None])
            bb_rigids = bb_rigids.compose_q_update_vec(
                bb_rigid_update * diffuse_mask[..., None])

            if b < self.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, bb_rigids, edge_mask)
                edge_embed *= edge_mask[..., None]

            if self.use_traj_predictions:
                traj_data[b]['seq_logits'] = self.trunk[f"seq_pred_{b}"](rigids_embed)
                traj_data[b]['dist_logits'] = self.trunk[f"dist_pred_{b}"](edge_embed)
                # dummy entry
                traj_data[b]['rigids'] = self.unscale_rigids(init_rigids)
                traj_data[b]['bb_rigids'] = self.unscale_rigids(bb_rigids)
                if self.traj_seqpair_predictor:
                    traj_data[b]['seqpair_logits'] = self.trunk[f'seqpair_pred_{b}'](edge_embed)
                if self.traj_framepair_predictor:
                    traj_data[self.num_blocks-1]['framepair_logits_dict'] = {
                        'logits': torch.zeros(
                            [*framepair_embed_skip.shape[:-1], self.rigid_dist_pred.n_bins],
                            device=framepair_embed_skip.device
                        ),
                        'to_queries': to_queries,
                        'to_keys': to_keys,
                        'n_padding': n_padding
                    }

        rigids_cond = self.node_to_rigid_cond(node_embed)
        rigids_cond = rigids_cond.view(rigids_embed_skip.shape)

        rigids_embed, node_embed, framepair_embed, curr_rigids = self.rigids_tfmr(
            node_embed,
            edge_embed,
            rigids_skip,
            rigids_embed_skip,
            rigids_cond,
            to_queries,
            to_keys,
            to_pairs,
            framepair_embed=framepair_embed_skip
        )
        if self.traj_framepair_predictor:
            traj_data[self.num_blocks-1]['framepair_logits_dict'] = {
                'logits': self.rigid_dist_pred(framepair_embed),
                'to_queries': to_queries,
                'to_keys': to_keys,
                'n_padding': n_padding
            }

        curr_rigids = self.unscale_rigids(curr_rigids)
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'node_embed': node_embed,
            'traj_data': traj_data
        }
        return model_out



class IpaMultiRigidDenoiser(nn.Module):

    def __init__(self,
                 # diffuser,
                 c_s=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=16,
                 c_hidden=16,
                 num_heads=16,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=8,
                 use_init_dgram=False,
                 use_traj_predictions=False,
                 use_traj_framepair_predictor=False,
                 use_traj_seqpair_predictor=False,
                 break_rigid_symmetry=True,
                 force_flash_transformer=False,
                 trans_preconditioning=False,
                 block_q=16,
                 block_k=64,
                 propagate_framepair_embed=False,
                 use_knn=False,
                 rigid_transformer_num_blocks=1,
                 rigid_transformer_rigid_updates=False,
                 rigid_transformer_agg_embed=True,
                 rigid_transformer_add_vanilla_transformer=False,
                 rigid_embed_skip_sc=False,
                 use_v2=False,
                 use_v3=False,
                 use_v4=False,
                 use_v5=False,
                 use_v6=False,
                 use_embedder_v2=False,
                 use_embedder_sc_rigid_transformer=False,
                 rel_quat_pair_updates=False,
                 z_broadcast=False,
                 compile_ipa=False,
                 rigids_per_residue=3
                 ):
        super().__init__()
        # some compatibility code
        self.self_conditioning = True
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000

        self.rigids_per_residue = rigids_per_residue
        self.use_v5 = use_v5

        if use_v2:
            self.ipa_score = IpaScoreV2(
                c_s=c_s,
                c_z=c_z,
                c_frame=c_frame,
                c_framepair=c_framepair,
                c_hidden=c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                num_blocks=num_blocks,
                use_traj_predictions=use_traj_predictions,
                traj_framepair_predictor=use_traj_framepair_predictor,
                traj_seqpair_predictor=use_traj_seqpair_predictor,
                force_flash_transformer=force_flash_transformer,
                coordinate_scaling=1 if trans_preconditioning else 0.1,
                block_q=block_q,
                block_k=block_k,
                propagate_framepair_embed=propagate_framepair_embed,
                use_knn=use_knn,
                rigid_transformer_num_blocks=rigid_transformer_num_blocks,
                rigid_transformer_rigid_updates=rigid_transformer_rigid_updates,
                rigid_transformer_agg_embed=rigid_transformer_agg_embed,
                rigid_embed_skip_sc=rigid_embed_skip_sc,
                rel_quat_pair_updates=rel_quat_pair_updates,
                z_broadcast=z_broadcast,
                compile_ipa=compile_ipa
            )
        elif use_v3:
            self.ipa_score = IpaScoreV3(
                c_s=c_s,
                c_z=c_z,
                c_frame=c_frame,
                c_framepair=c_framepair,
                c_hidden=c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                num_blocks=num_blocks,
                use_traj_predictions=use_traj_predictions,
                traj_framepair_predictor=use_traj_framepair_predictor,
                traj_seqpair_predictor=use_traj_seqpair_predictor,
                force_flash_transformer=force_flash_transformer,
                coordinate_scaling=1 if trans_preconditioning else 0.1,
                block_q=block_q,
                block_k=block_k,
                propagate_framepair_embed=propagate_framepair_embed,
                use_knn=use_knn,
                rigid_transformer_num_blocks=rigid_transformer_num_blocks,
                rigid_transformer_rigid_updates=rigid_transformer_rigid_updates,
                rigid_transformer_agg_embed=rigid_transformer_agg_embed,
                rigid_embed_skip_sc=rigid_embed_skip_sc,
                rel_quat_pair_updates=rel_quat_pair_updates,
                z_broadcast=z_broadcast,
                compile_ipa=compile_ipa
            )
        elif use_v4:
            self.ipa_score = IpaScoreV4(
                c_s=c_s,
                c_z=c_z,
                c_frame=c_frame,
                c_framepair=c_framepair,
                c_hidden=c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                num_blocks=num_blocks,
                use_traj_predictions=use_traj_predictions,
                traj_framepair_predictor=use_traj_framepair_predictor,
                traj_seqpair_predictor=use_traj_seqpair_predictor,
                force_flash_transformer=force_flash_transformer,
                coordinate_scaling=1 if trans_preconditioning else 0.1,
                block_q=block_q,
                block_k=block_k,
                propagate_framepair_embed=propagate_framepair_embed,
                use_knn=use_knn,
                rigid_transformer_num_blocks=rigid_transformer_num_blocks,
                rigid_transformer_rigid_updates=rigid_transformer_rigid_updates,
                rigid_transformer_agg_embed=rigid_transformer_agg_embed,
                rigid_transformer_framepair_init=(not use_embedder_v2),
                rigid_embed_skip_sc=rigid_embed_skip_sc,
                rel_quat_pair_updates=rel_quat_pair_updates,
                z_broadcast=z_broadcast,
                compile_ipa=compile_ipa
            )
        elif use_v5:
            self.ipa_score = IpaScoreV5(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                num_blocks=num_blocks,
                use_traj_predictions=use_traj_predictions,
                coordinate_scaling=1 if trans_preconditioning else 0.1,
                block_q=block_q,
                block_k=block_k,
                propagate_framepair_embed=propagate_framepair_embed,
                rel_quat_pair_updates=rel_quat_pair_updates,
            )
        elif use_v6:
            self.ipa_score = IpaScoreV6(
                c_s=c_s,
                c_z=c_z,
                c_frame=c_frame,
                c_framepair=c_framepair,
                c_hidden=c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                num_blocks=num_blocks,
                use_traj_predictions=use_traj_predictions,
                traj_framepair_predictor=use_traj_framepair_predictor,
                traj_seqpair_predictor=use_traj_seqpair_predictor,
                force_flash_transformer=force_flash_transformer,
                coordinate_scaling=1 if trans_preconditioning else 0.1,
                block_q=block_q,
                block_k=block_k,
                propagate_framepair_embed=propagate_framepair_embed,
                use_knn=use_knn,
                rigid_transformer_num_blocks=rigid_transformer_num_blocks,
                rigid_transformer_add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
                rigid_embed_skip_sc=rigid_embed_skip_sc,
                rel_quat_pair_updates=rel_quat_pair_updates,
            )
        else:
            self.ipa_score = IpaScore(
                c_s=c_s,
                c_z=c_z,
                c_frame=c_frame,
                c_framepair=c_framepair,
                c_hidden=c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                num_blocks=num_blocks,
                use_traj_predictions=use_traj_predictions,
                traj_framepair_predictor=use_traj_framepair_predictor,
                traj_seqpair_predictor=use_traj_seqpair_predictor,
                force_flash_transformer=force_flash_transformer,
                coordinate_scaling=1 if trans_preconditioning else 0.1,
                block_q=block_q,
                block_k=block_k,
                propagate_framepair_embed=propagate_framepair_embed,
                use_knn=use_knn,
                rigid_transformer_num_blocks=rigid_transformer_num_blocks,
                rigid_transformer_rigid_updates=rigid_transformer_rigid_updates,
                rigid_transformer_agg_embed=rigid_transformer_agg_embed,
                rel_quat_pair_updates=rel_quat_pair_updates,
                z_broadcast=z_broadcast,
                compile_ipa=compile_ipa
            )

        self.use_embedder_v2 = use_embedder_v2
        if use_v5:
            self.embedder = EmbedderForV5(
                c_s=c_s,
                c_z=c_z,
                block_q=block_q,
                block_k=block_k,
                break_symmetry=break_rigid_symmetry,
                rigids_per_residue=rigids_per_residue
            )

        elif use_embedder_v2:
            self.embedder = EmbedderV2(
                c_s=c_s,
                c_z=c_z,
                c_frame=c_frame,
                c_framepair=c_framepair,
                # c_hidden=32,
                block_q=block_q,
                block_k=block_k,
                break_symmetry=break_rigid_symmetry,
                rigids_per_residue=rigids_per_residue,
                use_sc_rigid_transformer=use_embedder_sc_rigid_transformer,
                rigid_transformer_add_vanilla_transformer=rigid_transformer_add_vanilla_transformer
            )
        else:
            self.embedder = Embedder(
                c_s=c_s,
                c_z=c_z,
                c_frame=c_frame,
                use_init_distogram=use_init_dgram,
            )
        self.c_s = c_s
        self.trans_preconditioning = trans_preconditioning

    def forward(self, data, self_condition=None):
        res_data = data['residue']
        res_mask = (res_data['res_mask']).bool()

        data_list = data.to_data_list()
        for d in data_list:
            assert d.num_nodes == data_list[0].num_nodes

        seq_idx = [torch.arange(data_list[0].num_nodes, device=res_mask.device) for _ in data_list]
        seq_idx = torch.stack(seq_idx)
        t = data['t']
        batch_size = t.shape[0]
        fixed_mask = torch.zeros_like(res_mask).view(batch_size, -1)

        if self_condition is not None:
            sc_rigids = self_condition['final_rigids'].view(batch_size, -1, self.rigids_per_residue)
        else:
            sc_rigids = None

        # center the training example at the mean of the x_cas
        rigids_t = Rigid.from_tensor_7(res_data['rigids_t'])
        rigids_t = rigids_t.view([t.shape[0], -1, rigids_t.shape[-1]])
        center = rigids_t.get_trans().mean(dim=(-2, -3))
        rigids_t = rigids_t.translate(-center[..., None, None, :])

        if self.trans_preconditioning:
            rigids_in = rigids_t.apply_trans_fn(lambda x: x * data['trans_c_in'][..., None, None, None])
        else:
            rigids_in = rigids_t

        if self.use_v5:
            node_embed, edge_embed, rigids_embed = self.embedder(
                seq_idx=seq_idx,
                t=t,
                rigids=rigids_in,
                node_mask=res_mask.view(batch_size, -1),
                fixed_mask=fixed_mask,
                sc_rigids=sc_rigids,
            )
            framepair_embed = None
        elif self.use_embedder_v2:
            node_embed, edge_embed, rigids_embed, framepair_embed = self.embedder(
                seq_idx=seq_idx,
                t=t,
                rigids=rigids_in,
                node_mask=res_mask.view(batch_size, -1),
                fixed_mask=fixed_mask,
                sc_rigids=sc_rigids,
            )
        else:
            node_embed, edge_embed, rigids_embed = self.embedder(
                seq_idx=seq_idx,
                t=t,
                fixed_mask=fixed_mask,
                sc_rigids=sc_rigids,
            )
            framepair_embed = None


        input_feats = {
            'fixed_mask': fixed_mask,
            'res_mask': res_mask.view(batch_size, -1),
            'rigids_t': rigids_in.to_tensor_7(),
            't': t,
            "rigids_embed": rigids_embed,
            "framepair_embed": framepair_embed
        }

        score_dict = self.ipa_score(node_embed, edge_embed, input_feats)
        traj_data = score_dict['traj_data']
        rigids_out = score_dict['final_rigids']
        if self.trans_preconditioning:
            rigids_out = rigids_out.apply_trans_fn(
                lambda x: x * data['trans_c_out'][..., None, None, None] + rigids_t.get_trans() * data['trans_c_skip'][..., None, None, None]
            )

        rigids_out = rigids_out.translate(center[..., None, None, :])
        seq_logits = traj_data[max(traj_data.keys())]['seq_logits']
        denoised_atom14_gt_seq = compute_atom14_from_cg_frames(rigids_out, res_mask, res_data['seq'].view(batch_size, -1))
        denoised_atom14_pred_seq = compute_atom14_from_cg_frames(rigids_out, res_mask, seq_logits.argmax(dim=-1))
        psi = score_dict['psi'].view(-1, 2)

        ret = {}
        ret['denoised_frames'] = rigids_out.view(-1, rigids_out.shape[-1])
        ret['final_rigids'] = rigids_out.view(-1, rigids_out.shape[-1])
        ret['denoised_atom14'] = denoised_atom14_pred_seq.flatten(0, 1)
        ret['denoised_atom14_gt_seq'] = denoised_atom14_gt_seq.flatten(0, 1)
        ret['denoised_bb'] = compute_backbone(rigids_out[..., 0], score_dict['psi'], impute_O=True)[-1][..., :5, :].flatten(0, 1)
        ret['psi'] = psi
        ret['traj_data'] = traj_data
        ret['decoded_seq_logits'] = seq_logits.flatten(0, 1)

        return ret