import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import copy
import functools as fn

import torch_geometric.utils as pygu

from proteinzen.model.modules.layers.node.attention import ConditionedTransformerPairBias
from proteinzen.model.modules.openfold.layers import InvariantPointAttention, Dropout
from proteinzen.model.modules.openfold.layers_v2 import (
    Linear, ConditionedInvariantPointAttention, BackboneUpdate, TorsionAngles, Transition, LayerNorm, AdaLN, ConditionedTransition
)
from proteinzen.data.openfold import residue_constants as rc
import proteinzen.utils.openfold.rigid_utils as ru
from proteinzen.utils.openfold.rigid_utils import Rigid, batchwise_center
from proteinzen.utils.framediff.all_atom import compute_backbone
from proteinzen.utils.coarse_grain import compute_atom14_from_cg_frames
from proteinzen.stoch_interp.so3_utils import geodesic_t, rotquat_to_rotvec, _rotquat_to_axis_angle

from ._attn import PairUpdate, ConditionedPairUpdate, MultiRigidPairEmbedder
from ._frame_transformer import (
    ScatterUpdate, FramepairEmbedder,
    SequenceFrameTransformerUpdate, ConditionedSequenceFrameTransformerUpdate,
    get_indexing_matrix, single_to_keys, pairs_to_framepairs,
    pad_and_flatten_rigids, unflatten_rigids
)

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


def get_timestep_embedding_flexshape(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[..., None] * emb.view(*[1 for _ in timesteps.shape], -1)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
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
    def __init__(self, c_s, c_frame, n_aa=21):
        super().__init__()
        self.ln = LayerNorm(c_frame)
        self.scatter_update = ScatterUpdate(c_s, c_frame)
        self.out = Linear(c_s, n_aa)
        self.c_s = c_s

    def forward(self, rigids_embed_flat, rigids_to_res_idx, rigids_mask, out):
        rigids_embed_flat = self.ln(rigids_embed_flat)
        seq_embed = self.scatter_update(
            rigids_embed_flat,
            out,
            rigids_to_res_idx,
            rigids_mask
        )
        return self.out(seq_embed)


class Embedder(nn.Module):
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
                 index_embed_size=256,
                 break_symmetry=True,
                 rigids_per_residue=3,
                 use_sc_rigid_transformer=False,
                 rigid_transformer_add_vanilla_transformer=False,
                 rigid_transformer_add_full_transformer=False,
                 use_ipa_gating=False,
                 ablate_ipa_down_z=False,
                 use_qk_norm=False,
                 restype_dict=rc.restype_order_with_x
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
            get_timestep_embedding_flexshape,
            embedding_dim=index_embed_size
        )

        self.restype_dict = restype_dict
        self.num_aa = len(restype_dict)
        self.mask_token = restype_dict['X']
        self.seq_embedder = Linear(
            self.num_aa, self.c_s, bias=False
        )
        self.time_init = Linear(index_embed_size, c_frame, bias=False)
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=index_embed_size
        )
        self.node_init = Linear(index_embed_size, self.c_s, bias=False)

        self.framepair_init = FramepairEmbedder(
            c_framepair
        )
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
            add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
            add_full_transformer=rigid_transformer_add_full_transformer,
            use_ipa_gating=use_ipa_gating,
            ablate_ipa_down_z=ablate_ipa_down_z,
            use_qk_norm=use_qk_norm,
        )
        self.use_sc_rigid_transformer = use_sc_rigid_transformer
        if use_sc_rigid_transformer:
            self.sc_framepair_init = FramepairEmbedder(
                c_framepair
            )
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
                add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
                use_ipa_gating=use_ipa_gating,
                ablate_ipa_down_z=ablate_ipa_down_z,
                use_qk_norm=use_qk_norm,
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
            seq,
            seq_noising_mask,
            seq_mask,
            chain_idx,
            t,
            node_mask,
            rigids,
            rigids_noising_mask,
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
        edge_embed = self.pair_embedder(
            rigids,
            node_mask,
            seq_idx,
            chain_idx,
            sc_rigids=sc_rigids
        )

        seq_idx_embed = self.index_embedder(seq_idx)
        node_feats = [seq_idx_embed]
        node_init = torch.cat(node_feats, dim=-1)
        node_init = self.node_init(node_init)
        total_seq_mask = (~seq_mask) & seq_noising_mask
        visible_seq = seq * (~total_seq_mask) + self.mask_token * total_seq_mask
        seq_embed = F.one_hot(visible_seq, num_classes=self.num_aa).float()
        node_init = node_init + self.seq_embedder(seq_embed)

        rigids_init = self.node_to_sidechain_frame(node_init[..., None, :]).tile((1, 1, self.rigids_per_residue, 1))
        if self.break_symmetry:
            rigids_init = rigids_init + self.pos_embedder(
                torch.arange(self.rigids_per_residue, device=rigids_init.device)
            )[None, None]
        else:
            rigids_init = rigids_init + self.pos_embedder(
                torch.tensor([0] + [1 for _ in range(self.rigids_per_residue-1)], device=rigids_init.device)
            )[None, None]
        rigids_init = rigids_init + self.time_init(self.timestep_embedder(t))

        (
            rigids_flat,
            rigids_init_flat,
            rigids_to_res_idx,
            rigids_flat_mask,
            rigids_noising_mask_flat,
            bb_rigids_mask,
            n_padding
        ) = pad_and_flatten_rigids(
            rigids,
            rigids_init,
            rigids_noising_mask,
            self.block_q
        )

        framepair_embed = self.framepair_init(
            rigids_flat,
            rigids_to_res_idx,
            rigids_flat_mask,
            to_queries,
            to_keys
        )

        rigids_embed_flat, node_embed, framepair_embed, _ = self.frame_tfmr(
            node_init,
            edge_embed,
            framepair_embed,
            rigids_flat,
            rigids_init_flat,
            rigids_to_res_idx,
            rigids_flat_mask,
            None,
            to_queries,
            to_keys,
            to_pairs,
        )

        if self.use_sc_rigid_transformer:
            if sc_rigids is not None:
                sc_framepair_embed = self.sc_framepair_init(
                    rigids_flat,
                    rigids_to_res_idx,
                    rigids_flat_mask,
                    to_queries,
                    to_keys
                )
                sc_rigids_embed_flat, sc_node_embed, sc_framepair_embed, _ = self.sc_frame_tfmr(
                    node_init,
                    edge_embed,
                    sc_framepair_embed,
                    rigids_flat,
                    rigids_init_flat,
                    rigids_to_res_idx,
                    rigids_flat_mask,
                    None,
                    to_queries,
                    to_keys,
                    to_pairs,
                )
            else:
                sc_rigids_embed_flat = torch.zeros_like(rigids_init_flat)
                sc_node_embed = torch.zeros_like(node_init)
                sc_framepair_embed = torch.zeros_like(framepair_embed)

            node_embed = self.node_adaln(node_embed, sc_node_embed)
            rigids_embed_flat = self.frame_adaln(rigids_embed_flat, sc_rigids_embed_flat)
            framepair_embed = self.framepair_adaln(framepair_embed, sc_framepair_embed)

        return {
            "node_embed": node_embed,
            "rigids_flat": rigids_flat,
            "rigids_embed_flat": rigids_embed_flat,
            "rigids_to_res_idx": rigids_to_res_idx,
            "bb_rigids_mask": bb_rigids_mask,
            "rigids_mask_flat": rigids_flat_mask,
            "rigids_noising_mask_flat": rigids_noising_mask_flat,
            "n_padding": n_padding,
            "edge_embed": edge_embed,
            "framepair_embed": framepair_embed,
            "to_queries": to_queries,
            "to_keys": to_keys,
            "to_pairs": to_pairs,
        }


class IpaScore(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_cond=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=64,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 block_q=32,
                 block_k=128,
                 rigid_transformer_num_blocks=1,
                 rigid_transformer_rigid_updates=False,
                 rigid_transformer_agg_embed=True,
                 rigid_transformer_add_vanilla_transformer=False,
                 rigid_transformer_add_full_transformer=False,
                 rel_quat_pair_updates=False,
                 z_broadcast=False,
                 compile_ipa=False,
                 use_ipa_gating=False,
                 ablate_ipa_down_z=False,
                 ipa_row_dropout_r=0.,
                 tfmr_row_dropout_r=0.,
                 use_qk_norm=False,
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.rigid_transformer_rigid_updates = rigid_transformer_rigid_updates

        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        self.num_blocks = num_blocks
        self.block_q = block_q
        self.block_k = block_k

        for b in range(num_blocks):
            self.trunk[f'ipa_{b}'] = ConditionedInvariantPointAttention(
                c_s=c_s,
                c_cond=c_cond,
                c_z=c_z,
                c_hidden=c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
            )
            if compile_ipa:
                self.trunk[f'ipa_{b}'].compile()
            self.trunk[f'ipa_row_dropout_{b}'] = Dropout(ipa_row_dropout_r, -2)

            self.trunk[f'tfmr_{b}'] = ConditionedTransformerPairBias(
                c_s=c_s,
                c_cond=c_cond,
                c_z=c_z,
                no_heads=4,
                n_layers=2,
                row_dropout=tfmr_row_dropout_r,
                use_qk_norm=use_qk_norm,
            )
            self.trunk[f'post_tfmr_{b}'] = Linear(
                c_s, c_s, init="final", bias=False)
            self.trunk[f'transition_{b}'] = ConditionedTransition(
                c_s=c_s,
                c_cond=c_cond
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
                broadcast_pairs=z_broadcast,
                add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
                add_full_transformer=rigid_transformer_add_full_transformer,
                use_ipa_gating=use_ipa_gating,
                ablate_ipa_down_z=ablate_ipa_down_z,
                use_qk_norm=use_qk_norm,
            )

            if not rigid_transformer_rigid_updates:
                self.trunk[f'rigids_update_{b}'] = BackboneUpdate(c_frame)

            if b < num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ConditionedPairUpdate(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_z//4,
                    no_heads=4,
                    include_rel_quat=rel_quat_pair_updates,
                    use_qk_norm=use_qk_norm
                )

        self.torsion_pred = TorsionAngles(c_s, 1)
        self.seq_pred = SeqPredictor(c_s, c_frame)


    def forward(self, input_feats):
        node_embed = input_feats['node_embed']
        node_mask = input_feats['res_mask'].type(torch.float32)
        condition_embed = input_feats['condition_embed']

        edge_embed = input_feats['edge_embed']
        edge_mask = node_mask[..., None] * node_mask[..., None, :]

        framepair_embed = input_feats['framepair_embed']

        curr_rigids = input_feats['rigids_flat']
        rigids_embed_flat = input_feats['rigids_embed_flat']
        bb_rigids_mask = input_feats['bb_rigids_mask']
        rigids_to_res_idx = input_feats['rigids_to_res_idx']
        rigids_mask_flat = input_feats['rigids_mask_flat']
        rigids_noising_mask_flat = input_feats['rigids_noising_mask_flat']

        to_queries = input_feats['to_queries']
        to_keys = input_feats['to_keys']
        to_pairs = input_feats['to_pairs']

        curr_rigids = self.scale_rigids(curr_rigids)
        node_embed = node_embed * node_mask[..., None]

        # Main trunk
        for b in range(self.num_blocks):
            bb_rigids = curr_rigids[bb_rigids_mask].view(node_embed.shape[:-1])
            ipa_embed = self.trunk[f'ipa_{b}'](
                s=node_embed,
                cond=condition_embed,
                z=edge_embed,
                r=bb_rigids,
                mask=node_mask)
            node_embed = (node_embed + ipa_embed) * node_mask[..., None]

            seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                node_embed, condition_embed, edge_embed, node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = node_embed * node_mask[..., None]

            node_embed = node_embed + self.trunk[f'transition_{b}'](node_embed, condition_embed)
            node_embed = node_embed * node_mask[..., None]

            rigids_embed_flat, node_embed, framepair_embed, curr_rigids = self.trunk[f'rigids_tfmr_{b}'](
                node_embed,
                edge_embed,
                framepair_embed,
                curr_rigids,
                rigids_embed_flat,
                rigids_to_res_idx,
                rigids_mask_flat,
                rigids_noising_mask_flat,
                to_queries,
                to_keys,
                to_pairs,
            )

            if b < self.num_blocks-1:
                bb_rigids = curr_rigids[bb_rigids_mask].view(node_embed.shape[:-1])
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, bb_rigids, edge_mask)
                edge_embed *= edge_mask[..., None]

        seq_logits = self.seq_pred(
            rigids_embed_flat,
            rigids_to_res_idx,
            rigids_mask_flat,
            out=torch.zeros_like(node_embed)
        )
        curr_rigids = unflatten_rigids(curr_rigids, input_feats['n_padding'])
        curr_rigids = self.unscale_rigids(curr_rigids)
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'node_embed': node_embed,
            'seq_logits': seq_logits
        }
        return model_out


class GaussianRandomFourierBasis(nn.Module):
    """ Damped random Fourier Feature encoding layer """
    def __init__(self, n_basis=64, c_out=256):
        super().__init__()
        kappa = torch.randn((1, n_basis,))
        self.register_buffer('kappa', kappa)
        self.proj = nn.Sequential(
            LayerNorm(n_basis * 2),
            Linear(n_basis*2, c_out, bias=False)
        )

    def forward(self, ts):
        tp = 2 * np.pi * ts[..., None] * self.kappa
        n = torch.cat([torch.cos(tp), torch.sin(tp)], dim=-1)
        return self.proj(n)


def to_internal_chain_numbering(chain_idx, batch):
    idx_out = []
    current_chain_idx = 0
    for i in range(batch.max().item() + 1):
        select = (batch == i)
        chain_idx_ = chain_idx[select]
        _idx = torch.arange(chain_idx_.shape[0], device=chain_idx_.device)
        chain_idx_change = (chain_idx_[:-1] != chain_idx_[1:])
        chain_ends = _idx[:-1][chain_idx_change].tolist()
        chain_ends.append(_idx.numel()-1)
        chain_starts = _idx[1:][chain_idx_change].tolist()
        chain_starts.insert(0, 0)

        for start, end in zip(chain_starts, chain_ends):
            idx_out.append(
                torch.full(
                    (end-start+1,),
                    int(current_chain_idx),
                    device=chain_idx.device
                )
            )
            current_chain_idx += 1

    return torch.cat(idx_out, dim=0)


class IpaMultiRigidDenoiser(nn.Module):

    def __init__(self,
                 # diffuser,
                 c_s=256,
                 c_cond=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=16,
                 c_hidden=16,
                 num_heads=16,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=8,
                 use_traj_predictions=False,
                 break_rigid_symmetry=True,
                 trans_preconditioning=False,
                 rot_preconditioning=False,
                 rot_capping=False,
                 block_q=16,
                 block_k=64,
                 rigid_transformer_num_blocks=1,
                 rigid_transformer_rigid_updates=False,
                 rigid_transformer_agg_embed=True,
                 rigid_transformer_add_vanilla_transformer=False,
                 rigid_transformer_add_full_transformer=False,
                 use_embedder_sc_rigid_transformer=False,
                 rel_quat_pair_updates=False,
                 z_broadcast=False,
                 compile_ipa=False,
                 rigids_per_residue=3,
                 use_ipa_gating=False,
                 ablate_ipa_down_z=False,
                 ipa_row_dropout_r=0.,
                 tfmr_row_dropout_r=0.,
                 cg_version=2,
                 use_qk_norm=False,
                 use_amp=False
                 ):
        super().__init__()

        c_hidden = c_s // num_heads

        # some compatibility code
        self.self_conditioning = True
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000

        self.rigids_per_residue = rigids_per_residue
        self.use_amp = use_amp

        self.ipa_score = IpaScore(
            c_s=c_s,
            c_cond=c_cond,
            c_z=c_z,
            c_frame=c_frame,
            c_framepair=c_framepair,
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            num_blocks=num_blocks,
            coordinate_scaling=1 if trans_preconditioning else 0.1,
            block_q=block_q,
            block_k=block_k,
            rigid_transformer_num_blocks=rigid_transformer_num_blocks,
            rigid_transformer_rigid_updates=rigid_transformer_rigid_updates,
            rigid_transformer_agg_embed=rigid_transformer_agg_embed,
            rigid_transformer_add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
            rigid_transformer_add_full_transformer=rigid_transformer_add_full_transformer,
            rel_quat_pair_updates=rel_quat_pair_updates,
            z_broadcast=z_broadcast,
            compile_ipa=compile_ipa,
            use_ipa_gating=use_ipa_gating,
            ablate_ipa_down_z=ablate_ipa_down_z,
            ipa_row_dropout_r=ipa_row_dropout_r,
            tfmr_row_dropout_r=tfmr_row_dropout_r,
            use_qk_norm=use_qk_norm,
        )

        self.embedder = Embedder(
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
            rigid_transformer_add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
            rigid_transformer_add_full_transformer=rigid_transformer_add_full_transformer,
            use_ipa_gating=use_ipa_gating,
            ablate_ipa_down_z=ablate_ipa_down_z,
            use_qk_norm=use_qk_norm,
        )

        self.c_s = c_s
        self.trans_preconditioning = trans_preconditioning
        self.rot_preconditioning = rot_preconditioning
        self.rot_capping = rot_capping
        self.cg_version = cg_version
        self.use_traj_predictions = use_traj_predictions

    def forward(self, data, self_condition=None):
        res_data = data['residue']
        res_mask = (res_data['res_mask']).bool()
        batch_size = data.num_graphs # t.shape[0]
        rigidwise_t = data['rigidwise_t']

        data_list = data.to_data_list()
        for d in data_list:
            assert d.num_nodes == data_list[0].num_nodes

        chain_idx = to_internal_chain_numbering(res_data['chain_idx'], res_data.batch)
        res_per_chain = pygu.scatter(
            torch.ones_like(chain_idx),
            chain_idx
        )
        chain_offset = torch.cumsum(res_per_chain, dim=-1)
        chain_offset = F.pad(chain_offset[:-1], (1, 0), value=0)

        seq_idx = torch.arange(res_mask.numel(), device=res_mask.device) - chain_offset[chain_idx]
        seq_idx = seq_idx.reshape(batch_size, -1)
        chain_idx = chain_idx.reshape(batch_size, -1)

        if self_condition is not None:
            sc_rigids = self_condition['final_rigids'].view(batch_size, -1, self.rigids_per_residue)
        else:
            sc_rigids = None

        # center the training example at the mean of the x_cas
        rigids_t = Rigid.from_tensor_7(res_data['rigids_t'])
        rigids_t = rigids_t.view([batch_size, -1, rigids_t.shape[-1]])
        center = rigids_t.get_trans().mean(dim=(-2, -3))
        rigids_t = rigids_t.translate(-center[..., None, None, :])

        if self.trans_preconditioning:
            rigids_in = rigids_t.apply_trans_fn(lambda x: x * data['trans_c_in'][..., None, None, None])
        else:
            rigids_in = rigids_t

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            input_feats = self.embedder(
                seq_idx=seq_idx,
                seq=res_data['seq'].view(batch_size, -1),
                seq_mask=res_data['seq_mask'].view(batch_size, -1),
                seq_noising_mask=res_data['seq_noising_mask'].view(batch_size, -1),
                chain_idx=chain_idx,
                t=rigidwise_t.unflatten(0, (batch_size, seq_idx.shape[1])),
                rigids=rigids_in,
                rigids_noising_mask=res_data['rigids_noising_mask'].unflatten(0, (batch_size, -1)),
                node_mask=res_mask.view(batch_size, -1),
                sc_rigids=sc_rigids,
            )
            # input_feats['condition_embed'] = torch.zeros_like(input_feats['node_embed'])
            input_feats['condition_embed'] = torch.ones_like(input_feats['node_embed'])
            input_feats['res_mask'] = res_data['res_mask'].view(batch_size, -1)

            score_dict = self.ipa_score(input_feats)

        rigids_out = score_dict['final_rigids']

        if self.rot_preconditioning:
            rigidwise_t = data['rigidwise_t'].unflatten(0, (batch_size, -1))
            def scale_rot(rot_in, rot_out):
                rel_rot = rot_out.compose_q(rot_in.invert())
                rel_rotquat = rel_rot.get_quats()
                rel_rotvec = rotquat_to_rotvec(rel_rotquat.view(-1, 4)).view(*rel_rotquat.shape[:-1], -1)
                angle = torch.linalg.vector_norm(rel_rotvec + 1e-8, dim=-1)
                scaled_angle = angle * (1 - rigidwise_t)
                axis = F.normalize(rel_rotvec, dim=-1)
                scaled_rotquat = torch.cat([
                    torch.cos(scaled_angle/2)[..., None], torch.sin(scaled_angle/2)[..., None] * axis
                ], dim=-1)
                scaled_rot = ru.Rotation(quats=scaled_rotquat)
                new_rot = scaled_rot.compose_q(rot_in)
                return new_rot

            rots_in = rigids_in.get_rots()
            rots_out = rigids_out.get_rots()
            rigids_out = Rigid(
                rots=scale_rot(rots_in, rots_out),
                trans=rigids_out.get_trans()
            )

        rigids_out = rigids_out.translate(center[..., None, None, :])
        seq_logits = score_dict['seq_logits']

        denoised_atom14_gt_seq = compute_atom14_from_cg_frames(
            rigids_out,
            res_mask,
            res_data['seq'].view(batch_size, -1),
            cg_version=self.cg_version
        )
        denoised_atom14_pred_seq = compute_atom14_from_cg_frames(
            rigids_out,
            res_mask,
            seq_logits.argmax(dim=-1),
            cg_version=self.cg_version
        )

        ret = {}
        ret['denoised_frames'] = rigids_out.view(-1, rigids_out.shape[-1])
        ret['final_rigids'] = rigids_out.view(-1, rigids_out.shape[-1])
        ret['denoised_atom14'] = denoised_atom14_pred_seq.flatten(0, 1)
        ret['denoised_atom14_gt_seq'] = denoised_atom14_gt_seq.flatten(0, 1)
        _psi = score_dict['psi']
        ret['denoised_bb'] = compute_backbone(rigids_out[..., 0], _psi, impute_O=True)[-1][..., :5, :].flatten(0, 1)
        ret['psi'] = _psi
        ret['decoded_seq_logits'] = seq_logits.flatten(0, 1)

        return ret