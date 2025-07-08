import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import copy
import functools as fn

import torch_geometric.utils as pygu

from boltz.data import const

from proteinzen.model.modules.layers.node.attention import ConditionedTransformerPairBias
from proteinzen.model.modules.openfold.layers import InvariantPointAttention, Dropout, TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming, permute_final_dims
from proteinzen.model.modules.openfold.layers_v2 import (
    Linear, ConditionedInvariantPointAttention, BackboneUpdate, TorsionAngles, LayerNorm, AdaLN, ConditionedTransition,
)
from proteinzen.data.openfold import residue_constants as rc

from proteinzen.data.datasets.featurize.rigid_assembler import rigids_to_atom14
import proteinzen.utils.openfold.rigid_utils as ru
from proteinzen.utils.openfold.rigid_utils import Rigid
from proteinzen.utils.openfold.tensor_utils import batched_gather
from proteinzen.utils.framediff.all_atom import compute_backbone
from proteinzen.utils.coarse_grain import compute_atom14_from_cg_frames
from proteinzen.stoch_interp.so3_utils import rotquat_to_rotvec

from ._attn import ConditionedPairUpdate, MultiRigidPairEmbedder, MultiRigidPairEmbedderV2
from ._attn import TriangleAttentionCore, SwishTransition, cuet_supported
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
                 max_rigids_idx=3+14,
                 use_sc_rigid_transformer=False,
                 rigid_transformer_add_vanilla_transformer=False,
                 rigid_transformer_add_second_transformer=False,
                 rigid_transformer_add_full_transformer=False,
                 use_ipa_gating=False,
                 ablate_ipa_down_z=False,
                 use_qk_norm=False,
                 restype_dict=const.token_ids,
                 num_elements=const.num_elements,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_sidechain = c_frame
        self.block_q = block_q
        self.block_k = block_k

        self.timestep_embedder = fn.partial(
            get_timestep_embedding_flexshape,
            embedding_dim=index_embed_size
        )
        self.time_condition_embed = Linear(index_embed_size, c_s, bias=False)
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=index_embed_size
        )

        self.restype_dict = restype_dict
        self.num_aa = len(restype_dict)
        self.mask_token = restype_dict['UNK']
        self.node_init = Linear(index_embed_size, self.c_s, bias=False)
        self.node_seq_embed = Linear(
            self.num_aa, self.c_s, bias=False
        )
        # self.node_is_atomized_embed = Linear(1, c_s, bias=False)
        self.node_is_unindexed_embed = Linear(1, c_s, bias=False)

        self.rigid_init = Linear(self.c_s, c_frame, bias=False)
        self.rigid_time_embed = Linear(index_embed_size, c_frame, bias=False)
        self.rigid_idx_embed = nn.Embedding(max_rigids_idx, c_frame)
        self.rigid_is_atomized_embed = Linear(1, c_frame, bias=False)
        self.rigid_element_embed = nn.Embedding(num_elements, c_frame)
        self.rigid_charge_embed = Linear(1, c_frame, bias=False)
        # self.rigid_is_unindexed_embed = Linear(1, c_frame, bias=False)

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
            add_second_transformer=rigid_transformer_add_second_transformer,
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
                add_second_transformer=rigid_transformer_add_second_transformer,
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
            no_blocks=n_pair_embed_blocks,
            use_qk_norm=use_qk_norm
        )

    def _gen_node_features(
        self,
        seq,
        seq_idx,
        seq_noising_mask,
        seq_mask,
        rigids,
        token_gather_idx,
        is_unindexed_mask,
        # is_atomized_mask,
        sc_rigids=None
    ):
        rigids_to_nodes = fn.partial(gather_helper, token_gather_idx=token_gather_idx)
        # print(token_gather_idx, rigids.shape)

        node_rigids = rigids_to_nodes(rigids)
        if sc_rigids is not None:
            # node_sc_rigids = rigids_to_nodes(sc_rigids)
            node_sc_rigids = ru.Rigid.from_tensor_7(rigids_to_nodes(sc_rigids.to_tensor_7()))
        else:
            node_sc_rigids = None

        node_seq_idx_embed = self.index_embedder(seq_idx)
        node_init = self.node_init(node_seq_idx_embed) * (~is_unindexed_mask[..., None])
        visible_seq = seq * seq_mask + self.mask_token * (~seq_mask)
        visible_seq = visible_seq * (~seq_noising_mask) + self.mask_token * seq_noising_mask
        # print(seq_idx)
        # print(seq, seq_mask, seq_noising_mask, visible_seq)
        # print(is_unindexed_mask)
        seq_embed = F.one_hot(visible_seq, num_classes=self.num_aa).float()
        node_init = (
            node_init
            + self.node_seq_embed(seq_embed)
            # + self.node_is_atomized_embed(is_atomized_mask[..., None].float())
            + self.node_is_unindexed_embed(is_unindexed_mask[..., None].float())
        )

        return {
            "node_init": node_init,
            "node_rigids": ru.Rigid.from_tensor_7(node_rigids),
            "node_sc_rigids": node_sc_rigids,
        }

    def _gen_rigid_features(
        self,
        node_init,
        rigids_element,
        rigids_charge,
        t,
        rigids_token_uid,
        rigids_idx,
        rigids_is_atomized_mask,
        # rigids_is_unindexed_mask,
    ):
        nodes_to_rigids = fn.partial(gather_helper, token_gather_idx=rigids_token_uid)

        rigids_init = self.rigid_init(nodes_to_rigids(node_init))
        time_embed = self.rigid_time_embed(self.timestep_embedder(t))
        rigids_idx_embed = self.rigid_idx_embed(rigids_idx)
        is_atomized_embed = self.rigid_is_atomized_embed(rigids_is_atomized_mask[..., None].float())
        element_mask = (rigids_element != -1)
        element_embed = self.rigid_element_embed(rigids_element * element_mask) * element_mask[..., None]
        charge_embed = self.rigid_charge_embed(rigids_charge.unsqueeze(-1))
        # is_unindexed_embed = self.rigid_is_unindexed_embed(rigids_is_unindexed_mask[..., None].float())

        rigids_init = (
            rigids_init
            + time_embed
            + rigids_idx_embed
            + is_atomized_embed
            + element_embed
            + charge_embed
            # + is_unindexed_embed
        )
        return rigids_init


    def _pad_rigid_features(
        self,
        rigids_data,
        n_padding: int
    ):
        padded_data = {}
        for key, value in rigids_data.items():
            if key in ("rigids_t", "sc_rigids"):
                rigids = value
                if rigids is not None:
                    n_batch = rigids.shape[0]
                    rigids_padding = ru.Rigid.identity(shape=(n_batch, n_padding), device=rigids._trans.device, fmt="quat")
                    # we don't use ru.Rigid.cat cuz this automatically converts stuff to rotmats...
                    # we need to stay in quats
                    rigids = ru.Rigid(
                        trans=torch.cat([rigids.get_trans(), rigids_padding.get_trans()], dim=-2),
                        rots=ru.Rotation(
                            quats=torch.cat([
                                rigids.get_rots().get_quats(), rigids_padding.get_rots().get_quats()
                            ], dim=-2)
                        )
                    )
                    padded_data[key] = rigids
                else:
                    padded_data[key] = None
            elif key == 'rigids_init':
                padded_data[key] = F.pad(value, (0, 0, 0, n_padding), value=False)
            elif value.dtype == torch.bool:
                padded_data[key] = F.pad(value, (0, n_padding), value=False)
            elif value.dtype == torch.long or value.dtype == torch.float:
                padded_data[key] = F.pad(value, (0, n_padding), value=0)
        return padded_data


    def forward(
            self,
            *,
            token_mask,
            token_seq,
            token_seq_idx,
            token_seq_noising_mask,
            token_seq_mask,
            token_chain_idx,
            # token_is_atomized_mask,
            token_is_unindexed_mask,
            token_gather_idx,
            t,
            rigids,
            rigids_element,
            rigids_charge,
            rigids_token_uid,
            rigids_idx,
            rigids_mask,
            rigids_noising_mask,
            rigids_is_atomized_mask,
            # rigids_is_token_rigid_mask,
            # rigids_is_unindexed_mask,
            token_bonds,
            sc_rigids=None,
        ):

        # build the indexing matrices
        n_batch = rigids.shape[0]
        n_rigids = rigids.shape[1]
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

        # compute node embeddings
        node_data = self._gen_node_features(
            token_seq,
            token_seq_idx,
            token_seq_noising_mask,
            token_seq_mask,
            rigids,
            token_gather_idx,
            token_is_unindexed_mask,
            # token_is_atomized_mask,
            sc_rigids
        )
        node_init = node_data['node_init']

        edge_embed = self.pair_embedder(
            node_data['node_rigids'],
            token_mask,
            token_seq_idx,
            token_chain_idx,
            token_is_unindexed_mask,
            token_bonds,
            sc_rigids=node_data['node_sc_rigids'],
        )

        rigids_init = self._gen_rigid_features(
            node_init,
            rigids_element,
            rigids_charge,
            t,
            rigids_token_uid,
            rigids_idx,
            rigids_is_atomized_mask,
            # rigids_is_unindexed_mask
        )

        rigids_inputs = {
            "rigids_t": ru.Rigid.from_tensor_7(rigids),
            "sc_rigids": sc_rigids if sc_rigids is not None else None,
            "rigids_init": rigids_init,
            "rigids_mask": rigids_mask,
            "rigids_token_uid": rigids_token_uid,
            # "rigids_center_mask": rigids_is_token_rigid_mask,
            "rigids_is_atomized_mask": rigids_is_atomized_mask,
            # "rigids_is_unindexed_mask": rigids_is_unindexed_mask,
            "rigids_noising_mask": rigids_noising_mask
        }
        # print({k: v.shape for k,v in rigids_data.items() if v is not None})
        # we need to pad this so we can use the efficient seq-local blocks
        rigids_data = self._pad_rigid_features(
            rigids_inputs,
            n_padding
        )
        rigids = rigids_data['rigids_t']
        rigids_init = rigids_data['rigids_init']
        rigids_token_uid = rigids_data['rigids_token_uid']
        rigids_mask = rigids_data['rigids_mask']

        framepair_init = self.framepair_init(
            rigids,
            rigids_token_uid,
            rigids_mask,
            to_queries,
            to_keys
        )

        rigids_embed, node_embed, framepair_embed, _ = self.frame_tfmr(
            node_init,
            edge_embed,
            framepair_init,
            rigids,
            rigids_init,
            rigids_token_uid,
            rigids_mask,
            None,
            to_queries,
            to_keys,
            to_pairs,
        )

        if self.use_sc_rigid_transformer:
            sc_rigids = rigids_data['sc_rigids']
            if sc_rigids is not None:
                sc_framepair_init = self.sc_framepair_init(
                    sc_rigids,
                    rigids_token_uid,
                    rigids_mask,
                    to_queries,
                    to_keys
                )
                sc_rigids_embed_flat, sc_node_embed, sc_framepair_embed, _ = self.sc_frame_tfmr(
                    node_init,
                    edge_embed,
                    sc_framepair_init,
                    sc_rigids,
                    rigids_init,
                    rigids_token_uid,
                    rigids_mask,
                    None,
                    to_queries,
                    to_keys,
                    to_pairs,
                )
            else:
                sc_rigids_embed_flat = torch.zeros_like(rigids_init)
                sc_node_embed = torch.zeros_like(node_init)
                sc_framepair_embed = torch.zeros_like(framepair_embed)

            node_embed = self.node_adaln(node_embed, sc_node_embed)
            rigids_embed = self.frame_adaln(rigids_embed, sc_rigids_embed_flat)
            framepair_embed = self.framepair_adaln(framepair_embed, sc_framepair_embed)

        ret = {
            "node_init": node_init,
            "node_embed": node_embed,
            "edge_embed": edge_embed,
            "time_condition_embed": self.time_condition_embed(self.timestep_embedder(t)),
            "rigids_embed": rigids_embed,
            "framepair_embed": framepair_embed,
            "framepair_init": framepair_init,
            "to_queries": to_queries,
            "to_keys": to_keys,
            "to_pairs": to_pairs,
            "n_padding": n_padding,
        }
        ret.update(rigids_data)

        return ret



def batched_mask_select(data, mask, no_batch_dims):
    remaining_mask_dims = [i for i in range(mask.dim()) if i < no_batch_dims]
    assert len(remaining_mask_dims) > 0
    selected_per_batch = mask.sum(dim=remaining_mask_dims)
    assert (selected_per_batch == selected_per_batch.view(-1)[0]).all(), selected_per_batch

    batched_mask_dims = mask.shape[:no_batch_dims]
    return data[mask].view(*batched_mask_dims, *data.shape[no_batch_dims:])


def gather_helper(tensor, token_gather_idx):
    new_dims = tensor.dim() - token_gather_idx.dim()
    idx_expand = token_gather_idx.view(
        *token_gather_idx.shape, *[1 for _ in range(new_dims)]
    ).expand(
        *[-1 for _ in token_gather_idx.shape],
        *tensor.shape[-new_dims:]
    ).long()
    return torch.gather(
        tensor,
        1,
        idx_expand
    )

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
                 index_embed_size=256,
                 max_rigids_idx=3+14,
                 use_sc_rigid_transformer=False,
                 rigid_transformer_add_vanilla_transformer=False,
                 rigid_transformer_add_second_transformer=False,
                 rigid_transformer_add_full_transformer=False,
                 use_ipa_gating=False,
                 ablate_ipa_down_z=False,
                 use_qk_norm=False,
                 restype_dict=rc.restype_order_with_x,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_sidechain = c_frame
        self.block_q = block_q
        self.block_k = block_k

        self.timestep_embedder = fn.partial(
            get_timestep_embedding_flexshape,
            embedding_dim=index_embed_size
        )
        self.time_condition_embed = Linear(index_embed_size, c_s, bias=False)
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=index_embed_size
        )

        self.restype_dict = restype_dict
        self.num_aa = len(restype_dict)
        self.mask_token = restype_dict['X']
        self.node_init = Linear(index_embed_size, self.c_s, bias=False)
        self.node_seq_embed = Linear(
            self.num_aa, self.c_s, bias=False
        )
        self.node_is_atomized_embed = Linear(1, c_s, bias=False)
        self.node_is_unindexed_embed = Linear(1, c_s, bias=False)

        self.rigid_init = Linear(self.c_s, c_frame, bias=False)
        self.rigid_time_embed = Linear(index_embed_size, c_frame, bias=False)
        self.rigid_idx_embed = nn.Embedding(max_rigids_idx, c_frame)
        self.rigid_is_atomized_embed = Linear(1, c_frame, bias=False)
        self.rigid_is_unindexed_embed = Linear(1, c_frame, bias=False)

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
            add_second_transformer=rigid_transformer_add_second_transformer,
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
                add_second_transformer=rigid_transformer_add_second_transformer,
                use_ipa_gating=use_ipa_gating,
                ablate_ipa_down_z=ablate_ipa_down_z,
                use_qk_norm=use_qk_norm,
            )
            self.node_adaln = AdaLN(c_s, c_s)
            self.frame_adaln = AdaLN(c_frame, c_frame)
            self.framepair_adaln = AdaLN(c_framepair, c_framepair)

        self.pair_embedder = MultiRigidPairEmbedderV2(
            c_z,
            c_hidden,
            no_blocks=n_pair_embed_blocks,
            use_qk_norm=use_qk_norm
        )

    def _gen_node_features(
        self,
        seq,
        seq_idx,
        seq_noising_mask,
        seq_mask,
        rigids,
        token_gather_idx,
        is_unindexed_mask,
        is_atomized_mask,
        sc_rigids=None
    ):
        rigids_to_nodes = fn.partial(gather_helper, token_gather_idx=token_gather_idx)

        node_rigids = rigids_to_nodes(rigids)
        if sc_rigids is not None:
            # node_sc_rigids = rigids_to_nodes(sc_rigids)
            node_sc_rigids = ru.Rigid.from_tensor_7(rigids_to_nodes(sc_rigids.to_tensor_7()))
        else:
            node_sc_rigids = None

        node_seq_idx_embed = self.index_embedder(seq_idx)
        node_init = self.node_init(node_seq_idx_embed) * (~is_unindexed_mask[..., None])
        visible_seq = seq * seq_mask + self.mask_token * (~seq_mask)
        visible_seq = visible_seq * (~seq_noising_mask) + self.mask_token * seq_noising_mask
        # print(seq_idx)
        # print(seq, seq_mask, seq_noising_mask, visible_seq)
        # print(is_unindexed_mask)
        seq_embed = F.one_hot(visible_seq, num_classes=self.num_aa).float()
        node_init = (
            node_init
            + self.node_seq_embed(seq_embed)
            + self.node_is_atomized_embed(is_atomized_mask[..., None].float())
            + self.node_is_unindexed_embed(is_unindexed_mask[..., None].float())
        )

        return {
            "node_init": node_init,
            "node_rigids": ru.Rigid.from_tensor_7(node_rigids),
            "node_sc_rigids": node_sc_rigids,
        }

    def _gen_rigid_features(
        self,
        node_init,
        t,
        rigids_token_uid,
        rigids_idx,
        rigids_is_atomized_mask,
        rigids_is_unindexed_mask,
    ):
        nodes_to_rigids = fn.partial(gather_helper, token_gather_idx=rigids_token_uid)

        rigids_init = self.rigid_init(nodes_to_rigids(node_init))
        time_embed = self.rigid_time_embed(self.timestep_embedder(t))
        rigids_idx_embed = self.rigid_idx_embed(rigids_idx)
        is_atomized_embed = self.rigid_is_atomized_embed(rigids_is_atomized_mask[..., None].float())
        is_unindexed_embed = self.rigid_is_unindexed_embed(rigids_is_unindexed_mask[..., None].float())

        rigids_init = (
            rigids_init
            + time_embed
            + rigids_idx_embed
            + is_atomized_embed
            + is_unindexed_embed
        )
        return rigids_init


    def _pad_rigid_features(
        self,
        rigids_data,
        n_padding: int
    ):
        padded_data = {}
        for key, value in rigids_data.items():
            if key in ("rigids_t", "sc_rigids"):
                rigids = value
                if rigids is not None:
                    n_batch = rigids.shape[0]
                    rigids_padding = ru.Rigid.identity(shape=(n_batch, n_padding), device=rigids._trans.device, fmt="quat")
                    # we don't use ru.Rigid.cat cuz this automatically converts stuff to rotmats...
                    # we need to stay in quats
                    rigids = ru.Rigid(
                        trans=torch.cat([rigids.get_trans(), rigids_padding.get_trans()], dim=-2),
                        rots=ru.Rotation(
                            quats=torch.cat([
                                rigids.get_rots().get_quats(), rigids_padding.get_rots().get_quats()
                            ], dim=-2)
                        )
                    )
                    padded_data[key] = rigids
                else:
                    padded_data[key] = None
            elif key == 'rigids_init':
                padded_data[key] = F.pad(value, (0, 0, 0, n_padding), value=False)
            elif value.dtype == torch.bool:
                padded_data[key] = F.pad(value, (0, n_padding), value=False)
            elif value.dtype == torch.long or value.dtype == torch.float:
                padded_data[key] = F.pad(value, (0, n_padding), value=0)
        return padded_data


    def forward(
            self,
            *,
            token_mask,
            token_seq,
            token_seq_idx,
            token_seq_noising_mask,
            token_seq_mask,
            token_chain_idx,
            token_is_atomized_mask,
            token_is_unindexed_mask,
            token_gather_idx,
            t,
            rigids,
            rigids_token_uid,
            rigids_idx,
            rigids_mask,
            rigids_noising_mask,
            rigids_is_atomized_mask,
            rigids_is_token_rigid_mask,
            rigids_is_unindexed_mask,
            sc_rigids=None,
        ):

        # build the indexing matrices
        n_batch = rigids.shape[0]
        n_rigids = rigids.shape[1]
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

        # compute node embeddings
        node_data = self._gen_node_features(
            token_seq,
            token_seq_idx,
            token_seq_noising_mask,
            token_seq_mask,
            rigids,
            token_gather_idx,
            token_is_unindexed_mask,
            token_is_atomized_mask,
            sc_rigids
        )
        node_init = node_data['node_init']

        edge_embed = self.pair_embedder(
            node_data['node_rigids'],
            token_mask,
            token_seq_idx,
            token_chain_idx,
            token_is_unindexed_mask,
            sc_rigids=node_data['node_sc_rigids'],
        )

        rigids_init = self._gen_rigid_features(
            node_init,
            t,
            rigids_token_uid,
            rigids_idx,
            rigids_is_atomized_mask,
            rigids_is_unindexed_mask
        )

        rigids_inputs = {
            "rigids_t": ru.Rigid.from_tensor_7(rigids),
            "sc_rigids": sc_rigids if sc_rigids is not None else None,
            "rigids_init": rigids_init,
            "rigids_mask": rigids_mask,
            "rigids_token_uid": rigids_token_uid,
            "rigids_center_mask": rigids_is_token_rigid_mask,
            "rigids_is_atomized_mask": rigids_is_atomized_mask,
            "rigids_is_unindexed_mask": rigids_is_unindexed_mask,
            "rigids_noising_mask": rigids_noising_mask
        }

        # TODO: in theory this is already handled by the featurization pipeline but keeping it here for now
        # we need to pad this so we can use the efficient seq-local blocks
        rigids_data = self._pad_rigid_features(
            rigids_inputs,
            n_padding
        )
        rigids = rigids_data['rigids_t']
        rigids_init = rigids_data['rigids_init']
        rigids_token_uid = rigids_data['rigids_token_uid']
        rigids_mask = rigids_data['rigids_mask']

        framepair_init = self.framepair_init(
            rigids,
            rigids_token_uid,
            rigids_mask,
            to_queries,
            to_keys
        )

        rigids_embed, node_embed, framepair_embed, _ = self.frame_tfmr(
            node_init,
            edge_embed,
            framepair_init,
            rigids,
            rigids_init,
            rigids_token_uid,
            rigids_mask,
            None,
            to_queries,
            to_keys,
            to_pairs,
        )

        if self.use_sc_rigid_transformer:
            sc_rigids = rigids_data['sc_rigids']
            if sc_rigids is not None:
                sc_framepair_init = self.sc_framepair_init(
                    sc_rigids,
                    rigids_token_uid,
                    rigids_mask,
                    to_queries,
                    to_keys
                )
                sc_rigids_embed_flat, sc_node_embed, sc_framepair_embed, _ = self.sc_frame_tfmr(
                    node_init,
                    edge_embed,
                    sc_framepair_init,
                    sc_rigids,
                    rigids_init,
                    rigids_token_uid,
                    rigids_mask,
                    None,
                    to_queries,
                    to_keys,
                    to_pairs,
                )
            else:
                sc_rigids_embed_flat = torch.zeros_like(rigids_init)
                sc_node_embed = torch.zeros_like(node_init)
                sc_framepair_embed = torch.zeros_like(framepair_embed)

            node_embed = self.node_adaln(node_embed, sc_node_embed)
            rigids_embed = self.frame_adaln(rigids_embed, sc_rigids_embed_flat)
            framepair_embed = self.framepair_adaln(framepair_embed, sc_framepair_embed)

        ret = {
            "node_init": node_init,
            "node_embed": node_embed,
            "edge_embed": edge_embed,
            "time_condition_embed": self.time_condition_embed(self.timestep_embedder(t)),
            "rigids_embed": rigids_embed,
            "framepair_embed": framepair_embed,
            "framepair_init": framepair_init,
            "to_queries": to_queries,
            "to_keys": to_keys,
            "to_pairs": to_pairs,
            "n_padding": n_padding,
        }
        ret.update(rigids_data)

        return ret


class IpaScore(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_cond=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=64,
                 c_hidden=256,
                 c_hidden_trig=32,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 block_q=32,
                 block_k=128,
                 use_conditioned_ipa=True,
                 use_conditioned_rigid_transformer=False,
                 rigid_transformer_num_blocks=1,
                 rigid_transformer_num_heads=4,
                 rigid_transformer_rigid_updates=False,
                 rigid_transformer_agg_embed=True,
                 rigid_transformer_add_vanilla_transformer=False,
                 rigid_transformer_add_second_transformer=False,
                 rigid_transformer_add_full_transformer=False,
                 rel_quat_pair_updates=False,
                 z_broadcast=False,
                 compile_ipa=False,
                 use_ipa_gating=False,
                 ablate_ipa_down_z=False,
                 ipa_row_dropout_r=0.,
                 tfmr_row_dropout_r=0.,
                 use_qk_norm=False,
                 ipa_qk_norm=None
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.use_conditioned_ipa = use_conditioned_ipa
        self.use_conditioned_rigid_transformer = use_conditioned_rigid_transformer
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
            if use_conditioned_ipa:
                self.trunk[f'ipa_{b}'] = ConditionedInvariantPointAttention(
                    c_s=c_s,
                    c_cond=c_cond,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    num_heads=num_heads,
                    num_qk_points=num_qk_points,
                    num_v_points=num_v_points,
                    use_qk_norm=(ipa_qk_norm if ipa_qk_norm is not None else use_qk_norm),
                )
            else:
                self.trunk[f'ipa_{b}'] = InvariantPointAttention(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    num_heads=num_heads,
                    num_qk_points=num_qk_points,
                    num_v_points=num_v_points,
                    use_qk_norm=use_qk_norm
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
            # self.trunk[f'tfmr_{b}'] = TransformerPairBias(
            #     c_s=c_s,
            #     c_z=c_z,
            #     no_heads=4,
            #     n_layers=2,
            #     row_dropout=tfmr_row_dropout_r,
            #     use_qk_norm=use_qk_norm,
            # )
            self.trunk[f'post_tfmr_{b}'] = Linear(
                c_s, c_s, init="final", bias=False)
            self.trunk[f'transition_{b}'] = ConditionedTransition(
                c_s=c_s,
                c_cond=c_cond
            )
            # self.trunk[f'transition_{b}'] = Transition(c_s)

            if self.use_conditioned_rigid_transformer:
                self.trunk[f'rigids_tfmr_{b}'] = ConditionedSequenceFrameTransformerUpdate(
                    c_s=c_s,
                    c_z=c_z,
                    c_frame=c_frame,
                    c_framepair=c_framepair,
                    num_heads=rigid_transformer_num_heads,
                    num_qk_points=num_qk_points,
                    num_v_points=num_v_points,
                    block_q=block_q,
                    block_k=block_k,
                    n_blocks=rigid_transformer_num_blocks,
                    do_rigid_updates=rigid_transformer_rigid_updates,
                    agg_rigid_embed=rigid_transformer_agg_embed,
                    broadcast_pairs=z_broadcast,
                    add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
                    add_second_transformer=rigid_transformer_add_second_transformer,
                    use_ipa_gating=use_ipa_gating,
                    ablate_ipa_down_z=ablate_ipa_down_z,
                    use_qk_norm=use_qk_norm,
                    use_conditioned_ipa=use_conditioned_ipa,
                )
            else:
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
                    add_second_transformer=rigid_transformer_add_second_transformer,
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
                    c_hidden=c_hidden_trig,
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
            if self.use_conditioned_ipa:
                ipa_embed = self.trunk[f'ipa_{b}'](
                    s=node_embed,
                    cond=condition_embed,
                    z=edge_embed,
                    r=bb_rigids,
                    mask=node_mask)
            else:
                ipa_embed = self.trunk[f'ipa_{b}'](
                    s=node_embed,
                    z=edge_embed,
                    r=bb_rigids,
                    mask=node_mask)
            node_embed = (node_embed + ipa_embed) * node_mask[..., None]

            seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                node_embed, condition_embed, edge_embed, node_mask)
            # seq_tfmr_out = self.trunk[f'tfmr_{b}'](
            #     node_embed, edge_embed, node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = node_embed * node_mask[..., None]

            node_embed = node_embed + self.trunk[f'transition_{b}'](node_embed, condition_embed)
            # node_embed = node_embed + self.trunk[f'transition_{b}'](node_embed)
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
            if not self.rigid_transformer_rigid_updates:
                rigid_update = self.trunk[f'rigids_update_{b}'](
                    rigids_embed_flat * rigids_noising_mask_flat[..., None])
                curr_rigids = curr_rigids.compose_q_update_vec(
                    rigid_update *  rigids_noising_mask_flat[..., None])

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


class IpaScoreV2(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_cond=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=64,
                 c_hidden=256,
                 c_hidden_trig=32,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 block_q=32,
                 block_k=128,
                 use_conditioned_ipa=True,
                 use_conditioned_rigid_transformer=False,
                 rigid_transformer_num_blocks=1,
                 rigid_transformer_num_heads=4,
                 rigid_transformer_rigid_updates=False,
                 rigid_transformer_agg_embed=True,
                 rigid_transformer_add_vanilla_transformer=False,
                 rigid_transformer_add_second_transformer=False,
                 rigid_transformer_add_full_transformer=False,
                 rel_quat_pair_updates=False,
                 z_broadcast=False,
                 compile_ipa=False,
                 use_ipa_gating=False,
                 ablate_ipa_down_z=False,
                 ipa_row_dropout_r=0.,
                 tfmr_row_dropout_r=0.,
                 use_qk_norm=False,
                 predict_final_rot=False,
                 direct_rot_vf_output=False,
                 detach_grad_pre_seq_pred=False,
                 num_aa=21
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.use_conditioned_ipa = use_conditioned_ipa
        self.use_conditioned_rigid_transformer = use_conditioned_rigid_transformer
        self.rigid_transformer_rigid_updates = rigid_transformer_rigid_updates

        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        self.num_blocks = num_blocks
        self.block_q = block_q
        self.block_k = block_k
        self.num_aa = num_aa

        for b in range(num_blocks):
            if use_conditioned_ipa:
                self.trunk[f'ipa_{b}'] = ConditionedInvariantPointAttention(
                    c_s=c_s,
                    c_cond=c_cond,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    num_heads=num_heads,
                    num_qk_points=num_qk_points,
                    num_v_points=num_v_points,
                    use_qk_norm=use_qk_norm,
                )
            else:
                self.trunk[f'ipa_{b}'] = InvariantPointAttention(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    num_heads=num_heads,
                    num_qk_points=num_qk_points,
                    num_v_points=num_v_points,
                    use_qk_norm=use_qk_norm
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

            if self.use_conditioned_rigid_transformer:
                self.trunk[f'rigids_tfmr_{b}'] = ConditionedSequenceFrameTransformerUpdate(
                    c_s=c_s,
                    c_z=c_z,
                    c_frame=c_frame,
                    c_framepair=c_framepair,
                    num_heads=rigid_transformer_num_heads,
                    num_qk_points=num_qk_points,
                    num_v_points=num_v_points,
                    block_q=block_q,
                    block_k=block_k,
                    n_blocks=rigid_transformer_num_blocks,
                    do_rigid_updates=rigid_transformer_rigid_updates,
                    agg_rigid_embed=rigid_transformer_agg_embed,
                    broadcast_pairs=z_broadcast,
                    add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
                    add_second_transformer=rigid_transformer_add_second_transformer,
                    use_ipa_gating=use_ipa_gating,
                    ablate_ipa_down_z=ablate_ipa_down_z,
                    use_qk_norm=use_qk_norm,
                    use_conditioned_ipa=use_conditioned_ipa,
                )
            else:
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
                    add_second_transformer=rigid_transformer_add_second_transformer,
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
                    c_hidden=c_hidden_trig,
                    no_heads=4,
                    include_rel_quat=rel_quat_pair_updates,
                    use_qk_norm=use_qk_norm
                )

        self.torsion_pred = TorsionAngles(c_s, 1)
        self.seq_pred = SeqPredictor(c_s, c_frame, n_aa=num_aa)
        self.detach_grad_pre_seq_pred = detach_grad_pre_seq_pred

        if predict_final_rot:
            self.final_rot_head = nn.Sequential(
                LayerNorm(c_frame),
                Linear(c_frame, c_frame, bias=False),
                nn.ReLU(),
                Linear(c_frame, 3, bias=False)
            )
        else:
            self.final_rot_head = None

        if direct_rot_vf_output:
            self.rot_vf_head = nn.Sequential(
                LayerNorm(c_frame),
                Linear(c_frame, c_frame, bias=False),
                nn.ReLU(),
                Linear(c_frame, 3, bias=False, init='final')
            )
        else:
            self.rot_vf_head = None

    def forward(self, input_feats):
        node_embed = input_feats['node_embed']
        node_mask = input_feats['token_mask'].type(torch.float32)
        condition_embed = input_feats['condition_embed']

        edge_embed = input_feats['edge_embed']
        edge_mask = node_mask[..., None] * node_mask[..., None, :]

        framepair_embed = input_feats['framepair_embed']

        init_rigids = input_feats['rigids_t']
        rigids_embed_flat = input_feats['rigids_embed']
        rigids_token_uid = input_feats['rigids_token_uid']
        rigids_mask_flat = input_feats['rigids_mask']
        rigids_noising_mask_flat = input_feats['rigids_noising_mask']

        to_queries = input_feats['to_queries']
        to_keys = input_feats['to_keys']
        to_pairs = input_feats['to_pairs']

        curr_rigids = self.scale_rigids(init_rigids)
        node_embed = node_embed * node_mask[..., None]
        rigids_to_nodes = fn.partial(gather_helper, token_gather_idx=input_feats['token_gather_idx'])

        # Main trunk
        for b in range(self.num_blocks):
            curr_rigids_tensor_7 = curr_rigids.to_tensor_7()
            token_rigids = ru.Rigid.from_tensor_7(rigids_to_nodes(curr_rigids_tensor_7))
            if self.use_conditioned_ipa:
                ipa_embed = self.trunk[f'ipa_{b}'](
                    s=node_embed,
                    cond=condition_embed,
                    z=edge_embed,
                    r=token_rigids,
                    mask=node_mask)
            else:
                ipa_embed = self.trunk[f'ipa_{b}'](
                    s=node_embed,
                    z=edge_embed,
                    r=token_rigids,
                    mask=node_mask)
            node_embed = (node_embed + ipa_embed) * node_mask[..., None]

            seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                node_embed, condition_embed, edge_embed, node_mask)
            # seq_tfmr_out = self.trunk[f'tfmr_{b}'](
            #     node_embed, edge_embed, node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = node_embed * node_mask[..., None]

            node_embed = node_embed + self.trunk[f'transition_{b}'](node_embed, condition_embed)
            # node_embed = node_embed + self.trunk[f'transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]

            rigids_embed_flat, node_embed, framepair_embed, curr_rigids = self.trunk[f'rigids_tfmr_{b}'](
                node_embed,
                edge_embed,
                framepair_embed,
                curr_rigids,
                rigids_embed_flat,
                rigids_token_uid,
                rigids_mask_flat,
                rigids_noising_mask_flat,
                to_queries,
                to_keys,
                to_pairs,
            )
            if not self.rigid_transformer_rigid_updates:
                rigid_update = self.trunk[f'rigids_update_{b}'](
                    rigids_embed_flat * rigids_noising_mask_flat[..., None])
                curr_rigids = curr_rigids.compose_q_update_vec(
                    rigid_update *  rigids_noising_mask_flat[..., None])

            if b < self.num_blocks-1:
                curr_rigids_tensor_7 = curr_rigids.to_tensor_7()
                token_rigids = ru.Rigid.from_tensor_7(rigids_to_nodes(curr_rigids_tensor_7))
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, token_rigids, edge_mask)
                edge_embed *= edge_mask[..., None]

        if self.detach_grad_pre_seq_pred:
            seq_logits = self.seq_pred(
                rigids_embed_flat.detach(),
                rigids_token_uid,
                rigids_mask_flat,
                out=torch.zeros_like(node_embed)
            )
        else:
            seq_logits = self.seq_pred(
                rigids_embed_flat,
                rigids_token_uid,
                rigids_mask_flat,
                out=torch.zeros_like(node_embed)
            )
        # print(curr_rigids.shape, input_feats['n_padding'])
        if self.final_rot_head is not None:
            rotvec = self.final_rot_head(rigids_embed_flat)
            axis = F.normalize(rotvec, dim=-1)
            angle = F.sigmoid(torch.linalg.vector_norm(rotvec + 1e-8, dim=-1)) * 2 * torch.pi
            angle = angle * rigids_noising_mask_flat * rigids_mask_flat
            rotquat = torch.cat([
                torch.cos(angle/2)[..., None], torch.sin(angle/2)[..., None] * axis
            ], dim=-1)
            rel_rot = ru.Rotation(quats=rotquat)
            new_rot = rel_rot.compose_q(init_rigids.get_rots())
            curr_rigids = ru.Rigid(
                rots=new_rot,
                trans=curr_rigids.get_trans()
            )

        if self.rot_vf_head is not None:
            pred_rot_vf = self.rot_vf_head(rigids_embed_flat)
            axis = F.normalize(pred_rot_vf, dim=-1)
            angle = torch.linalg.vector_norm(pred_rot_vf + 1e-8, dim=-1)
            angle = angle * rigids_noising_mask_flat * rigids_mask_flat
            angle = angle * (1 - input_feats['t'])
            rotquat = torch.cat([
                torch.cos(angle/2)[..., None], torch.sin(angle/2)[..., None] * axis
            ], dim=-1)
            rel_rot = ru.Rotation(quats=rotquat)
            new_rot = rel_rot.compose_q(init_rigids.get_rots())
            curr_rigids = ru.Rigid(
                rots=new_rot,
                trans=curr_rigids.get_trans()
            )

            if input_feats['n_padding'] > 0:
                pred_rot_vf = pred_rot_vf[..., :-input_feats['n_padding'], :]
        else:
            pred_rot_vf = None

        if input_feats['n_padding'] > 0:
            curr_rigids = curr_rigids[..., :-input_feats['n_padding']]
        curr_rigids = self.unscale_rigids(curr_rigids)


        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'pred_rot_vf': pred_rot_vf,
            'node_embed': node_embed,
            'seq_logits': seq_logits
        }
        return model_out


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


class Pairformer(nn.Module):
    def __init__(
        self,
        c_s,
        c_cond,
        c_z,
        num_blocks=8,
        num_heads=8,
        num_trig_heads=4,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.inf = 1e8
        c_head = c_z // num_trig_heads

        self.lin_edge_cond = nn.Sequential(
            LayerNorm(c_s),
            Linear(c_s, c_z, bias=False)
        )
        self.edge_transition = ConditionedTransition(c_z, c_z)
        self.trunk = nn.ModuleDict()
        for i in range(num_blocks):
            self.trunk[f'mult_out_{i}'] = TriangleMultiplicationOutgoing(c_z, c_z)
            self.trunk[f'mult_in_{i}'] = TriangleMultiplicationIncoming(c_z, c_z)
            self.trunk[f'edge_bias_{i}'] = nn.Sequential(
                LayerNorm(c_z),
                Linear(c_z, num_trig_heads, bias=False)
            )
            self.trunk[f'attn_start_{i}'] = TriangleAttentionCore(c_z, c_head, num_trig_heads, starting=True, use_qk_norm=True)
            self.trunk[f'attn_end_{i}'] = TriangleAttentionCore(c_z, c_head, num_trig_heads, starting=False, use_qk_norm=True)
            self.trunk[f'transition_{i}'] = SwishTransition(c_z)

            self.trunk[f'token_attn_{i}'] = ConditionedTransformerPairBias(
                c_s,
                c_cond,
                c_z,
                num_heads,
                n_layers=1,
                use_qk_norm=True
            )

    def forward(
        self,
        s,
        z,
        s_cond,
        node_mask,
    ):
        edge_cond = self.lin_edge_cond(s)
        num_res = s.shape[-2]
        cond = (
            torch.tile(edge_cond[..., None, :], (1, 1, num_res, 1))
            + torch.tile(edge_cond[..., None, :, :], (1, num_res, 1, 1))
        )
        z = z + self.edge_transition(z, cond)

        edge_mask = node_mask[..., None] & node_mask[..., None, :]
        global cuet_supported
        if cuet_supported:
            mask_bias = edge_mask[..., :, None, None, :]
        else:
            mask_bias = (edge_mask[..., :, None, None, :].float() - 1) * self.inf

        for i in range(self.num_blocks):
            z = z + self.trunk[f'mult_out_{i}'](z, edge_mask)
            z = z + self.trunk[f'mult_in_{i}'](z, edge_mask)
            edge_bias = self.trunk[f'edge_bias_{i}'](z)
            edge_bias = permute_final_dims(edge_bias.unsqueeze(-4), (2, 0, 1))
            z = z + self.trunk[f'attn_start_{i}'](z, mask_bias=mask_bias, edge_bias=edge_bias)
            z = z + self.trunk[f'attn_end_{i}'](z, mask_bias=mask_bias, edge_bias=edge_bias)
            z = z + self.trunk[f'transition_{i}'](z)

            s = self.trunk[f'token_attn_{i}'](
                s, s_cond, z, node_mask)

        return s, z


class IpaMultiRigidDenoiser(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_cond=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=16,
                 c_hidden=16,
                 c_hidden_trig=32,
                 c_hidden_trig_embedder=64,
                 num_heads=16,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=8,
                 trans_preconditioning=False,
                 rot_preconditioning=False,
                 block_q=16,
                 block_k=64,
                 use_conditioned_ipa=True,
                 use_conditioned_rigid_transformer=False,
                 rigid_transformer_num_blocks=1,
                 rigid_transformer_num_heads=4,
                 rigid_transformer_rigid_updates=False,
                 rigid_transformer_agg_embed=True,
                 rigid_transformer_add_vanilla_transformer=False,
                 rigid_transformer_add_second_transformer=False,
                 rigid_transformer_add_full_transformer=False,
                 use_embedder_sc_rigid_transformer=False,
                 rel_quat_pair_updates=False,
                 z_broadcast=False,
                 compile_ipa=False,
                 use_ipa_gating=False,
                 ablate_ipa_down_z=False,
                 ipa_row_dropout_r=0.,
                 tfmr_row_dropout_r=0.,
                 cg_version=1,
                 use_qk_norm=False,
                 use_amp=False,
                 predict_final_rot=False,
                 direct_rot_vf_output=False,
                 learnable_noise_schedule=False,
                 use_pairformer=False,
                 rot_vf_scaling=1,
                 self_conditioning=True
                 ):
        super().__init__()

        # some compatibility code
        self.self_conditioning = self_conditioning
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000

        self.use_amp = use_amp

        self.ipa_score = IpaScoreV2(
            c_s=c_s,
            c_cond=c_s,
            c_z=c_z,
            c_frame=c_frame,
            c_framepair=c_framepair,
            c_hidden=c_hidden // num_heads,
            c_hidden_trig=c_hidden_trig,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            num_blocks=num_blocks,
            coordinate_scaling=1 if trans_preconditioning else 0.1,
            block_q=block_q,
            block_k=block_k,
            use_conditioned_ipa=use_conditioned_ipa,
            use_conditioned_rigid_transformer=use_conditioned_rigid_transformer,
            rigid_transformer_num_blocks=rigid_transformer_num_blocks,
            rigid_transformer_num_heads=rigid_transformer_num_heads,
            rigid_transformer_rigid_updates=rigid_transformer_rigid_updates,
            rigid_transformer_agg_embed=rigid_transformer_agg_embed,
            rigid_transformer_add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
            rigid_transformer_add_second_transformer=rigid_transformer_add_second_transformer,
            rigid_transformer_add_full_transformer=rigid_transformer_add_full_transformer,
            rel_quat_pair_updates=rel_quat_pair_updates,
            z_broadcast=z_broadcast,
            compile_ipa=compile_ipa,
            use_ipa_gating=use_ipa_gating,
            ablate_ipa_down_z=ablate_ipa_down_z,
            ipa_row_dropout_r=ipa_row_dropout_r,
            tfmr_row_dropout_r=tfmr_row_dropout_r,
            use_qk_norm=use_qk_norm,
            predict_final_rot=predict_final_rot,
            direct_rot_vf_output=direct_rot_vf_output,
            detach_grad_pre_seq_pred=learnable_noise_schedule,
            num_aa=len(const.tokens)
        )

        self.embedder = Embedder(
            c_s=c_s,
            c_z=c_z,
            c_frame=c_frame,
            c_framepair=c_framepair,
            c_hidden=c_hidden_trig_embedder,
            block_q=block_q,
            block_k=block_k,
            use_sc_rigid_transformer=use_embedder_sc_rigid_transformer,
            rigid_transformer_add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
            rigid_transformer_add_second_transformer=rigid_transformer_add_second_transformer,
            rigid_transformer_add_full_transformer=rigid_transformer_add_full_transformer,
            use_ipa_gating=use_ipa_gating,
            ablate_ipa_down_z=ablate_ipa_down_z,
            use_qk_norm=use_qk_norm,
        )

        if use_pairformer:
            self.pairformer = Pairformer(
                c_s=c_s,
                c_cond=c_cond,
                c_z=c_z,
            )
        else:
            self.pairformer = None

        self.c_s = c_s
        self.trans_preconditioning = trans_preconditioning
        self.rot_preconditioning = rot_preconditioning
        self.cg_version = cg_version
        self.direct_rot_vf_output = direct_rot_vf_output
        self.rot_vf_scaling = rot_vf_scaling

        if learnable_noise_schedule:
            # self.trans_gamma_t = lambda x: x # MonotonicIncreasingFn()
            self.rot_gamma_t = MonotonicIncreasingFn()
            self.trans_gamma_t = MonotonicIncreasingFn()
            # self.rot_gamma_t = lambda x: x # MonotonicIncreasingFn()

    def forward(self, data, self_condition=None):
        token_data = data['token']
        rigids_data = data['rigids']
        # print(rigids_data['rigids_t'].shape, token_data['token_seq_idx'].shape)
        # print(token_data['token_is_protein_output_mask'])
        # print(token_data['token_gather_idx'], rigids_data['rigids_token_uid'])

        if self_condition is not None:
            sc_rigids = self_condition['denoised_rigids']
        else:
            sc_rigids = None

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            input_feats = self.embedder(
                token_mask=token_data['token_mask'],
                token_seq_idx=token_data['token_seq_idx'],
                token_seq=token_data['res_type'],
                token_seq_mask=token_data['token_mask'],
                token_seq_noising_mask=token_data['seq_noising_mask'],
                token_chain_idx=token_data['asym_id'],
                # token_is_atomized_mask=token_data['token_is_atomized_mask'],
                token_is_unindexed_mask=token_data['token_is_unindexed_mask'],
                token_gather_idx=token_data['token_to_rep_rigid'],
                t=data['t'],
                rigids=rigids_data['rigids_t'],
                rigids_element=rigids_data['rigids_ref_element'],
                rigids_charge=rigids_data['rigids_ref_charge'],
                rigids_token_uid=rigids_data['rigids_to_token'],
                rigids_idx=rigids_data['rigids_sidechain_idx'],
                rigids_mask=rigids_data['rigids_mask'],
                rigids_is_atomized_mask=rigids_data['rigids_is_atom_mask'],
                # rigids_is_token_rigid_mask=rigids_data['rigids_is_token_rigid_mask'],
                # rigids_is_unindexed_mask=rigids_data['rigids_is_unindexed_mask'],
                rigids_noising_mask=rigids_data['rigids_noising_mask'],
                token_bonds=token_data['token_bonds'],
                sc_rigids=sc_rigids,
            )

            # for k, v in input_feats.items():
            #     if isinstance(v, torch.Tensor) and v.isnan().any():
            #         print(k, "is nan")

            input_feats['condition_embed'] = input_feats['time_condition_embed']
            # input_feats['condition_embed'] = torch.ones_like(input_feats['node_embed'])
            input_feats['token_mask'] = token_data['token_mask']
            input_feats['token_gather_idx'] = token_data['token_to_rep_rigid']
            input_feats['t'] = data['t']

            if self.pairformer is not None:
                token_embed, edge_embed = self.pairformer(
                    s=input_feats['node_embed'],
                    z=input_feats['edge_embed'],
                    s_cond=input_feats['condition_embed'],
                    node_mask=input_feats['token_mask']
                )
                input_feats['node_embed'] = token_embed
                input_feats['edge_embed'] = edge_embed

            score_dict = self.ipa_score(input_feats)

        rigids_out = score_dict['final_rigids']
        # print(rigids_out.shape)

        if self.rot_preconditioning:
            t = data['t']
            def scale_rot(rot_in, rot_out):
                # print(rot_in.shape, rot_out.shape)
                rel_rot = rot_out.compose_q(rot_in.invert())
                rel_rotquat = rel_rot.get_quats()
                rel_rotvec = rotquat_to_rotvec(rel_rotquat.view(-1, 4)).view(*rel_rotquat.shape[:-1], -1)
                angle = torch.linalg.vector_norm(rel_rotvec + 1e-8, dim=-1)
                scaled_angle = angle * (1 - t)
                axis = F.normalize(rel_rotvec, dim=-1)
                scaled_rotquat = torch.cat([
                    torch.cos(scaled_angle/2)[..., None], torch.sin(scaled_angle/2)[..., None] * axis
                ], dim=-1)
                scaled_rot = ru.Rotation(quats=scaled_rotquat)
                new_rot = scaled_rot.compose_q(rot_in)
                return new_rot

            rigids_in = ru.Rigid.from_tensor_7(data['rigids']['rigids_t'])
            rots_in = rigids_in.get_rots()
            rots_out = rigids_out.get_rots()
            rigids_out = Rigid(
                rots=scale_rot(rots_in, rots_out),
                trans=rigids_out.get_trans()
            )

        seq_logits = score_dict['seq_logits']

        # print(rigids_out.shape, rigids_data['rigids_is_atomized_mask'].sum())

        # denoised_atom14_gt_seq = rigids_to_atom14(
        #     rigids_out,
        #     rigids_mask=rigids_data['rigids_mask'],
        #     rigids_is_protein_output_mask=rigids_data['rigids_is_protein_output_mask'],
        #     rigids_is_atomized_mask=rigids_data['rigids_is_atomized_mask'],
        #     token_is_atomized_mask=token_data['token_is_atomized_mask'],
        #     token_is_protein_output_mask=token_data['token_is_protein_output_mask'],
        #     seq=token_data['seq'],
        #     cg_version=self.cg_version
        # )

        pred_seq = seq_logits[..., :-1].argmax(dim=-1)
        seq_noising_mask = token_data['seq_noising_mask']
        pred_seq = pred_seq * seq_noising_mask + token_data['seq'] * (~seq_noising_mask)
        # denoised_atom14_pred_seq = rigids_to_atom14(
        #     rigids_out,
        #     rigids_mask=rigids_data['rigids_mask'],
        #     rigids_is_protein_output_mask=rigids_data['rigids_is_protein_output_mask'],
        #     rigids_is_atomized_mask=rigids_data['rigids_is_atomized_mask'],
        #     token_is_atomized_mask=token_data['token_is_atomized_mask'],
        #     token_is_protein_output_mask=token_data['token_is_protein_output_mask'],
        #     seq=pred_seq,
        #     cg_version=self.cg_version
        # )

        if rigids_out.to_tensor_7().isnan().any() or pred_seq.isnan().any():
            print("caught a nan in forward")
            exit()

        ret = {}
        ret['denoised_rigids'] = rigids_out
        # ret['denoised_atom14'] = denoised_atom14_pred_seq
        # ret['denoised_atom14_gt_seq'] = denoised_atom14_gt_seq
        ret['decoded_seq_logits'] = seq_logits
        ret['pred_seq'] = pred_seq

        if self.direct_rot_vf_output:
            pred_rot_vf = score_dict['pred_rot_vf'].float() * self.rot_vf_scaling
        else:
            pred_rot_vf = None
        ret['pred_rot_vf'] = pred_rot_vf
        # print(pred_rot_vf)

        with torch.no_grad():
            token_rigids = gather_helper(rigids_out.to_tensor_7(), token_data['token_to_rep_rigid'])
            token_rigids = ru.Rigid.from_tensor_7(token_rigids)

            motif_rigid_mask = token_data['token_is_copy_mask']
            protein_rigid_mask = ~token_data['token_is_copy_mask']
            dist_mask = motif_rigid_mask[..., None] & protein_rigid_mask[..., None, :]
            res_CA_pos = token_rigids.get_trans()
            trans_dist = torch.cdist(res_CA_pos, res_CA_pos)
            trans_dist[~dist_mask] = 1e6
            closest_neighbors = torch.argsort(trans_dist)
            motif_idx = closest_neighbors[..., 0]
            ret['motif_idx'] = motif_idx

        return ret


class MonotonicIncreasingFn(nn.Module):
    def __init__(self, c_hidden=1024):
        super().__init__()
        self.c_hidden = c_hidden
        self.l1 = nn.Linear(1, 1, bias=False)
        self.l2 = nn.Linear(1, c_hidden)
        self.l3 = nn.Linear(c_hidden, 1)

    def _forward(self, l):
        l = F.linear(l, torch.abs(self.l1.weight))
        out = F.linear(l, torch.abs(self.l2.weight), self.l2.bias)
        out = torch.sigmoid(out)
        out = F.linear(out, torch.abs(self.l3.weight))
        return l + out

    def forward(self, x, l):
        shift_scale = self._forward(l)
        ret = x / (x * (1 - shift_scale) + shift_scale)
        # print(x, l, shift_scale, ret)
        return ret


class IpaMultiRigidDenoiserV2(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_cond=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=16,
                 c_hidden=16,
                 c_hidden_trig=32,
                 c_hidden_trig_embedder=64,
                 num_heads=16,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=8,
                 trans_preconditioning=False,
                 rot_preconditioning=False,
                 block_q=16,
                 block_k=64,
                 use_conditioned_ipa=True,
                 use_conditioned_rigid_transformer=False,
                 rigid_transformer_num_blocks=1,
                 rigid_transformer_num_heads=4,
                 rigid_transformer_rigid_updates=False,
                 rigid_transformer_agg_embed=True,
                 rigid_transformer_add_vanilla_transformer=False,
                 rigid_transformer_add_second_transformer=False,
                 rigid_transformer_add_full_transformer=False,
                 use_embedder_sc_rigid_transformer=False,
                 rel_quat_pair_updates=False,
                 z_broadcast=False,
                 compile_ipa=False,
                 use_ipa_gating=False,
                 ablate_ipa_down_z=False,
                 ipa_row_dropout_r=0.,
                 tfmr_row_dropout_r=0.,
                 cg_version=1,
                 use_qk_norm=False,
                 use_amp=False,
                 predict_final_rot=False,
                 direct_rot_vf_output=False,
                 learnable_noise_schedule=False,
                 use_pairformer=False,
                 rot_vf_scaling=1,
                 self_conditioning=True
                 ):
        super().__init__()

        # some compatibility code
        self.self_conditioning = self_conditioning
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000

        self.use_amp = use_amp

        self.ipa_score = IpaScoreV2(
            c_s=c_s,
            c_cond=c_s,
            c_z=c_z,
            c_frame=c_frame,
            c_framepair=c_framepair,
            c_hidden=c_hidden // num_heads,
            c_hidden_trig=c_hidden_trig,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            num_blocks=num_blocks,
            coordinate_scaling=1 if trans_preconditioning else 0.1,
            block_q=block_q,
            block_k=block_k,
            use_conditioned_ipa=use_conditioned_ipa,
            use_conditioned_rigid_transformer=use_conditioned_rigid_transformer,
            rigid_transformer_num_blocks=rigid_transformer_num_blocks,
            rigid_transformer_num_heads=rigid_transformer_num_heads,
            rigid_transformer_rigid_updates=rigid_transformer_rigid_updates,
            rigid_transformer_agg_embed=rigid_transformer_agg_embed,
            rigid_transformer_add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
            rigid_transformer_add_second_transformer=rigid_transformer_add_second_transformer,
            rigid_transformer_add_full_transformer=rigid_transformer_add_full_transformer,
            rel_quat_pair_updates=rel_quat_pair_updates,
            z_broadcast=z_broadcast,
            compile_ipa=compile_ipa,
            use_ipa_gating=use_ipa_gating,
            ablate_ipa_down_z=ablate_ipa_down_z,
            ipa_row_dropout_r=ipa_row_dropout_r,
            tfmr_row_dropout_r=tfmr_row_dropout_r,
            use_qk_norm=use_qk_norm,
            predict_final_rot=predict_final_rot,
            direct_rot_vf_output=direct_rot_vf_output,
            detach_grad_pre_seq_pred=learnable_noise_schedule
        )

        self.embedder = EmbedderV2(
            c_s=c_s,
            c_z=c_z,
            c_frame=c_frame,
            c_framepair=c_framepair,
            c_hidden=c_hidden_trig_embedder,
            block_q=block_q,
            block_k=block_k,
            use_sc_rigid_transformer=use_embedder_sc_rigid_transformer,
            rigid_transformer_add_vanilla_transformer=rigid_transformer_add_vanilla_transformer,
            rigid_transformer_add_second_transformer=rigid_transformer_add_second_transformer,
            rigid_transformer_add_full_transformer=rigid_transformer_add_full_transformer,
            use_ipa_gating=use_ipa_gating,
            ablate_ipa_down_z=ablate_ipa_down_z,
            use_qk_norm=use_qk_norm,
        )

        if use_pairformer:
            self.pairformer = Pairformer(
                c_s=c_s,
                c_cond=c_cond,
                c_z=c_z,
            )
        else:
            self.pairformer = None

        self.c_s = c_s
        self.trans_preconditioning = trans_preconditioning
        self.rot_preconditioning = rot_preconditioning
        self.cg_version = cg_version
        self.direct_rot_vf_output = direct_rot_vf_output
        self.rot_vf_scaling = rot_vf_scaling

        if learnable_noise_schedule:
            # self.trans_gamma_t = lambda x: x # MonotonicIncreasingFn()
            self.rot_gamma_t = MonotonicIncreasingFn()
            self.trans_gamma_t = MonotonicIncreasingFn()
            # self.rot_gamma_t = lambda x: x # MonotonicIncreasingFn()

    def forward(self, data, self_condition=None):
        token_data = data['token']
        rigids_data = data['rigids']
        # print(rigids_data['rigids_t'].shape, token_data['token_seq_idx'].shape)
        # print(token_data['token_is_protein_output_mask'])
        # print(token_data['token_gather_idx'], rigids_data['rigids_token_uid'])

        if self_condition is not None:
            sc_rigids = self_condition['denoised_rigids']
        else:
            sc_rigids = None

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            input_feats = self.embedder(
                token_mask=token_data['token_mask'],
                token_seq_idx=token_data['token_seq_idx'],
                token_seq=token_data['seq'],
                token_seq_mask=token_data['seq_mask'],
                token_seq_noising_mask=token_data['seq_noising_mask'],
                token_chain_idx=token_data['chain_idx'],
                token_is_atomized_mask=token_data['token_is_atomized_mask'],
                token_is_unindexed_mask=token_data['token_is_unindexed_mask'],
                token_gather_idx=token_data['token_gather_idx'],
                t=data['t'],
                rigids=rigids_data['rigids_t'],
                rigids_token_uid=rigids_data['rigids_token_uid'],
                rigids_idx=rigids_data['rigids_idx'],
                rigids_mask=rigids_data['rigids_mask'],
                rigids_is_atomized_mask=rigids_data['rigids_is_atomized_mask'],
                rigids_is_token_rigid_mask=rigids_data['rigids_is_token_rigid_mask'],
                rigids_is_unindexed_mask=rigids_data['rigids_is_unindexed_mask'],
                rigids_noising_mask=rigids_data['rigids_noising_mask'],
                sc_rigids=sc_rigids,
            )

            # for k, v in input_feats.items():
            #     if isinstance(v, torch.Tensor) and v.isnan().any():
            #         print(k, "is nan")

            input_feats['condition_embed'] = input_feats['time_condition_embed']
            # input_feats['condition_embed'] = torch.ones_like(input_feats['node_embed'])
            input_feats['token_mask'] = token_data['token_mask']
            input_feats['token_gather_idx'] = token_data['token_gather_idx']
            input_feats['t'] = data['t']

            if self.pairformer is not None:
                token_embed, edge_embed = self.pairformer(
                    s=input_feats['node_embed'],
                    z=input_feats['edge_embed'],
                    s_cond=input_feats['condition_embed'],
                    node_mask=input_feats['token_mask']
                )
                input_feats['node_embed'] = token_embed
                input_feats['edge_embed'] = edge_embed

            score_dict = self.ipa_score(input_feats)

        rigids_out = score_dict['final_rigids']
        # print(rigids_out.shape)

        if self.rot_preconditioning:
            t = data['t']
            def scale_rot(rot_in, rot_out):
                # print(rot_in.shape, rot_out.shape)
                rel_rot = rot_out.compose_q(rot_in.invert())
                rel_rotquat = rel_rot.get_quats()
                rel_rotvec = rotquat_to_rotvec(rel_rotquat.view(-1, 4)).view(*rel_rotquat.shape[:-1], -1)
                angle = torch.linalg.vector_norm(rel_rotvec + 1e-8, dim=-1)
                scaled_angle = angle * (1 - t)
                axis = F.normalize(rel_rotvec, dim=-1)
                scaled_rotquat = torch.cat([
                    torch.cos(scaled_angle/2)[..., None], torch.sin(scaled_angle/2)[..., None] * axis
                ], dim=-1)
                scaled_rot = ru.Rotation(quats=scaled_rotquat)
                new_rot = scaled_rot.compose_q(rot_in)
                return new_rot

            rigids_in = ru.Rigid.from_tensor_7(data['rigids']['rigids_t'])
            rots_in = rigids_in.get_rots()
            rots_out = rigids_out.get_rots()
            rigids_out = Rigid(
                rots=scale_rot(rots_in, rots_out),
                trans=rigids_out.get_trans()
            )

        seq_logits = score_dict['seq_logits']

        # print(rigids_out.shape, rigids_data['rigids_is_atomized_mask'].sum())

        denoised_atom14_gt_seq = rigids_to_atom14(
            rigids_out,
            rigids_mask=rigids_data['rigids_mask'],
            rigids_is_protein_output_mask=rigids_data['rigids_is_protein_output_mask'],
            rigids_is_atomized_mask=rigids_data['rigids_is_atomized_mask'],
            token_is_atomized_mask=token_data['token_is_atomized_mask'],
            token_is_protein_output_mask=token_data['token_is_protein_output_mask'],
            seq=token_data['seq'],
            cg_version=self.cg_version
        )

        pred_seq = seq_logits[..., :-1].argmax(dim=-1)
        seq_noising_mask = token_data['seq_noising_mask']
        pred_seq = pred_seq * seq_noising_mask + token_data['seq'] * (~seq_noising_mask)
        denoised_atom14_pred_seq = rigids_to_atom14(
            rigids_out,
            rigids_mask=rigids_data['rigids_mask'],
            rigids_is_protein_output_mask=rigids_data['rigids_is_protein_output_mask'],
            rigids_is_atomized_mask=rigids_data['rigids_is_atomized_mask'],
            token_is_atomized_mask=token_data['token_is_atomized_mask'],
            token_is_protein_output_mask=token_data['token_is_protein_output_mask'],
            seq=pred_seq,
            cg_version=self.cg_version
        )

        if rigids_out.to_tensor_7().isnan().any() or pred_seq.isnan().any():
            print("caught a nan in forward")
            exit()

        ret = {}
        ret['denoised_rigids'] = rigids_out
        ret['denoised_atom14'] = denoised_atom14_pred_seq
        ret['denoised_atom14_gt_seq'] = denoised_atom14_gt_seq
        ret['decoded_seq_logits'] = seq_logits
        ret['pred_seq'] = pred_seq

        if self.direct_rot_vf_output:
            pred_rot_vf = score_dict['pred_rot_vf'].float() * self.rot_vf_scaling
        else:
            pred_rot_vf = None
        ret['pred_rot_vf'] = pred_rot_vf
        # print(pred_rot_vf)

        with torch.no_grad():
            bb_rigids = gather_helper(rigids_out.to_tensor_7(), token_data['token_gather_idx'])
            bb_rigids = ru.Rigid.from_tensor_7(bb_rigids)

            motif_bb_mask = (~token_data['token_is_protein_output_mask'] & ~token_data['token_is_ligand_mask'])
            protein_bb_mask = (token_data['token_is_protein_output_mask'] & ~token_data['token_is_ligand_mask'])
            dist_mask = motif_bb_mask[..., None] & protein_bb_mask[..., None, :]
            res_CA_pos = bb_rigids.get_trans()
            trans_dist = torch.cdist(res_CA_pos, res_CA_pos)
            trans_dist[~dist_mask] = 1e6
            closest_neighbors = torch.argsort(trans_dist)
            motif_idx = closest_neighbors[..., 0]
            ret['motif_idx'] = motif_idx

        return ret