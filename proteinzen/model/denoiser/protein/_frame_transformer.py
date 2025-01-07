from typing import Callable

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy
import functools as fn
import numpy as np

from torch_geometric.nn import knn_graph
import torch_geometric.utils as pygu
import dgl
import dgl.sparse
from dgl.ops import u_mul_e_sum, u_dot_v, u_add_v

from proteinzen.model.modules.openfold.layers_v2 import (
    Linear, swish, ipa_point_weights_init_, permute_final_dims, flatten_final_dims, BackboneUpdate, LayerNorm,
    AdaLN, ConditionedTransition)
from proteinzen.utils.openfold import rigid_utils as ru


class GatherAdaLN(nn.Module):
    def __init__(self, c_s, c_cond):
        super().__init__()
        self.ln_s = LayerNorm(c_s, elementwise_affine=False, bias=False)
        self.ln_cond = LayerNorm(c_cond, bias=False)
        self.lin_cond = Linear(c_cond, c_s)
        self.lin_cond_nobias = Linear(c_cond, c_s, bias=False)

        self.c_s = c_s
        self.c_cond = c_cond

    def forward(self, s, cond, cond_to_s_idx):
        s = self.ln_s(s)
        cond = self.ln_cond(cond)
        cond_gate = torch.gather(
            self.lin_cond(cond),
            dim=1,
            index=cond_to_s_idx[..., None].expand(-1, -1, self.c_s),
        )
        cond_bias = torch.gather(
            self.lin_cond_nobias(cond),
            dim=1,
            index=cond_to_s_idx[..., None].expand(-1, -1, self.c_s)
        )
        s = s * torch.sigmoid(cond_gate)
        s = s + cond_bias

        return s


class GatherConditionedTransition(nn.Module):
    def __init__(self, c_s, c_cond, n=2):
        super().__init__()
        self.adaln = GatherAdaLN(c_s, c_cond)
        self.lin_1 = Linear(c_s, c_s*n, bias=False)
        self.lin_2 = Linear(c_s, c_s*n, bias=False)
        self.lin_cond = Linear(c_cond, c_s)
        with torch.no_grad():
            self.lin_cond.bias.fill_(-2.0)
        self.lin_b = Linear(c_s*n, c_s, bias=False, init='final')

        self.c_s = c_s

    def forward(self, s, cond, cond_to_s_idx):
        s = self.adaln(s, cond, cond_to_s_idx)
        b = swish(self.lin_1(s)) * self.lin_2(s)
        cond_gate = torch.gather(
            self.lin_cond(cond),
            dim=1,
            index=cond_to_s_idx[..., None].expand(-1, -1, self.c_s),
        )
        s = torch.sigmoid(cond_gate) * self.lin_b(b)
        return s


# from boltz1
def get_indexing_matrix(K, W, H, device):
    assert W % 2 == 0
    assert H % (W // 2) == 0

    h = H // (W // 2)
    assert h % 2 == 0

    arange = torch.arange(2 * K, device=device)
    index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(
        min=0, max=h + 1
    )
    index = index.view(K, 2, 2 * K)[:, 0, :]
    onehot = F.one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
    return onehot.reshape(2 * K, h * K).float()

# from boltz1
def single_to_keys(single, indexing_matrix, W, H):
    B, N, D = single.shape
    K = N // W
    single = single.view(B, 2 * K, W // 2, D)
    return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
        B, K, H, D
    )

# adapted from boltz1
def pairs_to_framepairs(pairs, indexing_matrix, W, H):
    B, N, _, D = pairs.shape
    K = N // W
    pairs = pairs.view(B, K, W, 2 * K, W // 2, D)
    indexing_matrix = indexing_matrix.view(2 * K, K, -1)
    ret = torch.einsum("b x y j i d, j x h -> b x y h i d", pairs, indexing_matrix)
    ret = ret.reshape(
        B, K, W, H, D
    )
    return ret

# adapted from boltz1
def single_to_weighted_keys(single, weight, indexing_matrix, W, H):
    B, N, D = single.shape
    K = N // W
    D_out, D_in = weight.shape
    assert D == D_in
    single = single.view(B, 2 * K, W // 2, D_in)
    return torch.einsum("b j i d, j k, o d -> b k i o", single, indexing_matrix, weight).reshape(
        B, K, H, D_out
    )



class BlockInvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s,
        c_z,
        c_hidden,
        num_heads,
        num_qk_points,
        num_v_points,
        inf: float = 1e5,
        eps: float = 1e-8,
        final_init='final',
        block_Q=32,
        block_K=128,
        use_compile_path=False
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = num_heads
        self.no_qk_points = num_qk_points
        self.no_v_points = num_v_points
        self.inf = inf
        self.eps = eps
        self.block_Q = block_Q
        self.block_K = block_K
        self.use_compile_path = use_compile_path

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, bias=False)
        self.linear_k = Linear(self.c_s, hc, bias=False)
        self.linear_v = Linear(self.c_s, hc, bias=False)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq, bias=False)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv, bias=False)

        self.linear_b = Linear(self.c_z, self.no_heads, bias=False)
        self.down_z = Linear(self.c_z, self.c_z // 4, bias=False)

        self.head_weights = nn.Parameter(torch.zeros((self.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init=final_init, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.ln_s = LayerNorm(c_s)
        self.ln_z = LayerNorm(c_z)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: ru.Rigid,
        s_mask: torch.Tensor,
        to_queries: Callable,
        to_keys: Callable,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        s = self.ln_s(s)
        z = self.ln_z(z)

        q_in = to_queries(s)
        k_in = to_keys(s)

        # we have to do a bunch of shenanigans to collect
        # the proper rigids, esp since we might not know
        # if the rigids have mat or quat rots
        trans = r.get_trans()
        rots = r.get_rots().get_cur_rot()
        # [*, N, 4 or 3x3]
        rots_flat = rots.view(rots.shape[:2] + (-1,))
        # [*, N_block, block_Q]
        q_trans = to_queries(trans)
        # [*, N_block, block_Q, 4 or 3x3]
        # print("rots", rots)
        q_rots = to_queries(rots_flat)
        # print("q_rots", q_rots)
        if tuple(rots.shape[-2:]) == (3, 3):
            q_rots = q_rots.view(q_rots.shape[:3] + (3, 3))
            # [*, N_block, block_Q]
            q_rigids = ru.Rigid(
                trans=q_trans,
                rots=ru.Rotation(rot_mats=q_rots)
            )
        else:
            # [*, N_block, block_Q]
            q_rigids = ru.Rigid(
                trans=q_trans,
                rots=ru.Rotation(quats=q_rots)
            )
        # print("q_rigids", q_rigids.get_trans(), q_rigids.get_rots().get_cur_rot())
        # [*, N_block, block_K]
        k_trans = to_keys(trans)
        k_rots = to_keys(rots_flat.view(rots.shape[:2] + (-1,)))
        # print("k_rots", k_rots)
        # there's some more shenanigans that needs to happen because to_keys will
        # zero out some quats which will lead to weird problems
        if tuple(rots.shape[-2:]) == (3, 3):
            k_rots = k_rots.view(k_rots.shape[:3] + (3, 3))
            k_rots[(k_rots == 0).all(dim=(-2, -1))] = torch.eye(3, device=k_rots.device)[None, None, None]
            k_rigids = ru.Rigid(
                trans=k_trans,
                rots=ru.Rotation(rot_mats=k_rots)
            )
        else:
            mask = (k_rots == 0).all(dim=-1)
            k_rots = (
                k_rots * (~mask[..., None])
                + torch.tensor([1., 0., 0., 0.], device=k_rots.device)[None, None, None] * mask[..., None]
            )
            # print("k_rots", k_rots)
            k_rigids = ru.Rigid(
                trans=k_trans,
                rots=ru.Rotation(quats=k_rots)
            )
        # print("k_rigids", k_rigids.get_trans(), k_rigids.get_rots().get_cur_rot())

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_block, block_Q, H * C_hidden]
        q = self.linear_q(q_in)
        # [*, N_block, block_K, H * C_hidden]
        k = self.linear_k(k_in)
        # [*, N_block, block_K, H * C_hidden]
        v = self.linear_v(k_in)

        # [*, N_block, block_Q, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_block, block_K, H, C_hidden]
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        # [*, N_block, block_K, H, C_hidden]
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, N_block, block_Q, H * P_q * 3]
        q_pts = self.linear_q_points(q_in)
        # [*, N_block, block_Q, H * P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-1] + (self.no_heads * self.no_qk_points, 3))
        q_pts = q_rigids[..., None].apply(q_pts)

        # [*, N_block, block_Q, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_block, block_K, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(k_in)
        # [*, N_block, block_K, H * (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-1] + (-1, 3))
        kv_pts = k_rigids[..., None].apply(kv_pts)

        # [*, N_block, block_K, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_block, block_K, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_block, block_Q, block_K, H]
        b = self.linear_b(z)

        # [*, N_block, block_Q, block_K, H]
        a = torch.einsum("bnqhc,bnkhc->bnqkh", q, k)
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * b

        # [*, N_block, block_Q, block_K, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2

        # [*, N_block, block_Q, block_K, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_block, block_Q, block_K, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_block, block_Q, block_K]
        attn_mask = to_queries(s_mask[..., None].float()) * to_keys(s_mask[..., None].float()).transpose(-1, -2)
        attn_mask = self.inf * (attn_mask - 1)

        # [*, N_block, block_Q, block_K, H]
        a = a + pt_att
        a = a + attn_mask[..., None]
        # [*, N_block, block_Q, H, block_K]
        a = a.transpose(-1, -2)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_block, block_Q, H, C_hidden]
        o = torch.einsum("bnqhk,bnkhc->bnqhc", a, v)

        # [*, N_block, block_Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, N_block, block_Q, H, P_v, 3]
        o_pt = torch.einsum("bnqhk,bnkhvc->bnqhvc", a, v_pts)
        o_pt = q_rigids[..., None, None].invert_apply(o_pt)

        # [*, N_block, block_Q, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_block, block_Q, H * P_v * 3]
        o_pt = flatten_final_dims(o_pt, 3)

        # [*, N_block, block_Q, block_K, H, C_z // 4]
        pair_z = self.down_z(z).to(dtype=a.dtype)
        # [*, N_block, block_Q, H, C_z // 4]
        o_pair = torch.einsum("bnqhk,bnqkc->bnqhc", a, pair_z)

        # [*, N_block, block_Q, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, o_pt, o_pt_norm_feats, o_pair]

        # [*, N_block, block_Q, C_s]
        out = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            ).to(dtype=z.dtype)
        )

        return out.view(s.shape)


class ConditionedBlockInvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s,
        c_cond,
        c_z,
        c_hidden,
        num_heads,
        num_qk_points,
        num_v_points,
        inf: float = 1e5,
        eps: float = 1e-8,
        final_init='final',
        block_Q=32,
        block_K=128,
        use_compile_path=False
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super().__init__()

        self.c_s = c_s
        self.c_cond = c_cond
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = num_heads
        self.no_qk_points = num_qk_points
        self.no_v_points = num_v_points
        self.inf = inf
        self.eps = eps
        self.block_Q = block_Q
        self.block_K = block_K
        self.use_compile_path = use_compile_path

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, bias=False)
        self.linear_k = Linear(self.c_s, hc, bias=False)
        self.linear_v = Linear(self.c_s, hc, bias=False)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq, bias=False)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv, bias=False)

        self.linear_b = Linear(self.c_z, self.no_heads, bias=False)
        self.down_z = Linear(self.c_z, self.c_z // 4, bias=False)

        self.head_weights = nn.Parameter(torch.zeros((self.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init=final_init, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.adaln_s = AdaLN(c_s, c_cond)
        self.ln_z = LayerNorm(c_z)

    def forward(
        self,
        s: torch.Tensor,
        cond: torch.Tensor,
        z: torch.Tensor,
        r: ru.Rigid,
        s_mask: torch.Tensor,
        to_queries: Callable,
        to_keys: Callable
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        s = self.adaln_s(s, cond)
        z = self.ln_z(z)

        q_in = to_queries(s)
        k_in = to_keys(s)

        # we have to do a bunch of shenanigans to collect
        # the proper rigids, esp since we might not know
        # if the rigids have mat or quat rots
        trans = r.get_trans()
        rots = r.get_rots().get_cur_rot()
        # [*, N, 4 or 3x3]
        rots_flat = rots.view(rots.shape[:2] + (-1,))
        # [*, N_block, block_Q]
        q_trans = to_queries(trans)
        # [*, N_block, block_Q, 4 or 3x3]
        # print("rots", rots)
        q_rots = to_queries(rots_flat)
        # print("q_rots", q_rots)
        if tuple(rots.shape[-2:]) == (3, 3):
            q_rots = q_rots.view(q_rots.shape[:3] + (3, 3))
            # [*, N_block, block_Q]
            q_rigids = ru.Rigid(
                trans=q_trans,
                rots=ru.Rotation(rot_mats=q_rots)
            )
        else:
            # [*, N_block, block_Q]
            q_rigids = ru.Rigid(
                trans=q_trans,
                rots=ru.Rotation(quats=q_rots)
            )
        # print("q_rigids", q_rigids.get_trans(), q_rigids.get_rots().get_cur_rot())
        # [*, N_block, block_K]
        k_trans = to_keys(trans)
        k_rots = to_keys(rots_flat.view(rots.shape[:2] + (-1,)))
        # print("k_rots", k_rots)
        # there's some more shenanigans that needs to happen because to_keys will
        # zero out some quats which will lead to weird problems
        if tuple(rots.shape[-2:]) == (3, 3):
            k_rots = k_rots.view(k_rots.shape[:3] + (3, 3))
            k_rots[(k_rots == 0).all(dim=(-2, -1))] = torch.eye(3, device=k_rots.device)[None, None, None]
            k_rigids = ru.Rigid(
                trans=k_trans,
                rots=ru.Rotation(rot_mats=k_rots)
            )
        else:
            mask = (k_rots == 0).all(dim=-1)
            k_rots = (
                k_rots * (~mask[..., None])
                + torch.tensor([1., 0., 0., 0.], device=k_rots.device)[None, None, None] * mask[..., None]
            )
            # print("k_rots", k_rots)
            k_rigids = ru.Rigid(
                trans=k_trans,
                rots=ru.Rotation(quats=k_rots)
            )
        # print("k_rigids", k_rigids.get_trans(), k_rigids.get_rots().get_cur_rot())

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_block, block_Q, H * C_hidden]
        q = self.linear_q(q_in)
        # [*, N_block, block_K, H * C_hidden]
        k = self.linear_k(k_in)
        # [*, N_block, block_K, H * C_hidden]
        v = self.linear_v(k_in)

        # [*, N_block, block_Q, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_block, block_K, H, C_hidden]
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        # [*, N_block, block_K, H, C_hidden]
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, N_block, block_Q, H * P_q * 3]
        q_pts = self.linear_q_points(q_in)
        # [*, N_block, block_Q, H * P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-1] + (self.no_heads * self.no_qk_points, 3))
        q_pts = q_rigids[..., None].apply(q_pts)

        # [*, N_block, block_Q, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_block, block_K, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(k_in)
        # [*, N_block, block_K, H * (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-1] + (-1, 3))
        kv_pts = k_rigids[..., None].apply(kv_pts)

        # [*, N_block, block_K, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_block, block_K, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_block, block_Q, block_K, H]
        b = self.linear_b(z)

        # [*, N_block, block_Q, block_K, H]
        a = torch.einsum("bnqhc,bnkhc->bnqkh", q, k)
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * b

        # [*, N_block, block_Q, block_K, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2

        # [*, N_block, block_Q, block_K, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_block, block_Q, block_K, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_block, block_Q, block_K]
        attn_mask = to_queries(s_mask[..., None].float()) * to_keys(s_mask[..., None].float()).transpose(-1, -2)
        attn_mask = self.inf * (attn_mask - 1)

        # [*, N_block, block_Q, block_K, H]
        a = a + pt_att
        a = a + attn_mask[..., None]
        # [*, N_block, block_Q, H, block_K]
        a = a.transpose(-1, -2)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_block, block_Q, H, C_hidden]
        o = torch.einsum("bnqhk,bnkhc->bnqhc", a, v)

        # [*, N_block, block_Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, N_block, block_Q, H, P_v, 3]
        o_pt = torch.einsum("bnqhk,bnkhvc->bnqhvc", a, v_pts)
        o_pt = q_rigids[..., None, None].invert_apply(o_pt)

        # [*, N_block, block_Q, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_block, block_Q, H * P_v * 3]
        o_pt = flatten_final_dims(o_pt, 3)

        # [*, N_block, block_Q, block_K, H, C_z // 4]
        pair_z = self.down_z(z).to(dtype=a.dtype)
        # [*, N_block, block_Q, H, C_z // 4]
        o_pair = torch.einsum("bnqhk,bnqkc->bnqhc", a, pair_z)

        # [*, N_block, block_Q, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, o_pt, o_pt_norm_feats, o_pair]

        # [*, N_block, block_Q, C_s]
        out = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            ).to(dtype=z.dtype)
        )

        return out.view(s.shape)

class BlockAttentionPairBias(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        num_heads=4,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads

class SequenceFrameTransformerBlock(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_frame=128,
                 c_framepair=16,
                 num_heads=4,
                 num_qk_points=8,
                 num_v_points=12,
                 block_q=32,
                 block_k=128,
                 inf=1e8,
                 compile_ipa=False
                 ):
        super().__init__()

        self.c_frame = c_frame
        self.c_framepair = c_framepair
        self.c_s = c_s
        self.num_heads = num_heads
        self.c_hidden = c_frame // num_heads
        self.block_q = block_q
        self.block_k = block_k
        self.inf = inf

        self.block_ipa = BlockInvariantPointAttention(
            c_s=c_frame,
            c_z=c_framepair,
            c_hidden=self.c_hidden,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            block_Q=block_q,
            block_K=block_k,
            use_compile_path=compile_ipa
        )

        self.transition = GatherConditionedTransition(c_frame, c_s)

    def forward(
        self,
        s,
        rigids,
        rigids_embed,
        framepair_embed,
        to_queries,
        to_keys,
        rigids_mask,
        rigids_to_res_idx
    ):
        rigids_embed_update = self.block_ipa(
            s=rigids_embed,
            z=framepair_embed,
            r=rigids,
            s_mask=rigids_mask,
            to_queries=to_queries,
            to_keys=to_keys,
        )
        rigids_embed = rigids_embed + rigids_embed_update * rigids_mask[..., None]

        rigids_embed = rigids_embed + self.transition(rigids_embed, s, rigids_to_res_idx) * rigids_mask[..., None]

        return rigids_embed


class SequenceFrameTransformerUpdate(nn.Module):
    def __init__(
        self,
        c_s=256,
        c_z=128,
        c_frame=128,
        c_framepair=16,
        num_heads=4,
        num_qk_points=8,
        num_v_points=12,
        block_q=32,
        block_k=128,
        inf=1e8,
        n_blocks=2,
        do_rigid_updates=False,
        broadcast_singles=False,
        broadcast_pairs=False,
        agg_rigid_embed=True,
        compile_ipa=False,
        framepair_init=False,
        framepair_ffn=False
    ):
        super().__init__()
        self.c_s = c_s
        self.c_frame = c_frame
        self.c_framepair = c_framepair
        self.n_blocks = n_blocks
        self.block_q = block_q
        self.block_k = block_k
        self.do_rigid_updates = do_rigid_updates
        self.agg_rigid_embed = agg_rigid_embed
        self.framepair_init = framepair_init

        if framepair_init:
            self.lin_framepair_dist_vec = Linear(3, c_framepair, bias=False)
            self.lin_framepair_dist = Linear(1, c_framepair, bias=False)
            self.lin_same_res = Linear(1, c_framepair, bias=False)

        self.node_to_frame_broadcast = Linear(c_s, c_frame, bias=False)

        if broadcast_pairs:
            self.nodepair_to_framepair_broadcast = nn.Sequential(
                LayerNorm(c_z),
                Linear(c_z, c_framepair, bias=False)
            )
        else:
            self.nodepair_to_framepair_broadcast = None

        self.broadcast_singles = broadcast_singles
        if broadcast_singles:
            self.frame_ln = LayerNorm(c_frame, c_frame)
            self.frame_l_to_framepair_broadcast = Linear(c_frame, c_framepair, bias=False)
            self.frame_m_to_framepair_broadcast = Linear(c_frame, c_framepair, bias=False)

        if framepair_ffn:
            self.framepair_ffn = nn.Sequential(
                LayerNorm(c_framepair),
                Linear(c_framepair, c_framepair),
                nn.ReLU(),
                Linear(c_framepair, c_framepair),
                nn.ReLU(),
                Linear(c_framepair, c_framepair),
            )
        else:
            self.framepair_ffn = None

        self.trunk = nn.ModuleDict()
        for b in range(n_blocks):
            self.trunk[f'tfmr_{b}'] = SequenceFrameTransformerBlock(
                c_s=c_s,
                c_frame=c_frame,
                c_framepair=c_framepair,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                block_q=block_q,
                block_k=block_k,
                inf=inf,
                compile_ipa=compile_ipa
            )
            if self.do_rigid_updates:
                self.trunk[f'frame_update_{b}'] = BackboneUpdate(c_frame)

        if self.agg_rigid_embed:
            self.frame_to_node_broadcast = nn.Sequential(
                LayerNorm(c_frame),
                Linear(c_frame, c_s, bias=False, init='final')
            )

    def _broadcast_pairs(
        self,
        z,
        rigids_to_res_idx,
        rigids_mask,
        to_pairs
    ):
        n_rigids = rigids_mask.shape[-1]
        # TODO: this is a little jenk
        # with the batch dim
        with torch.no_grad():
            rigids_pair_idx = torch.stack([
                rigids_to_res_idx[0:1, ..., None].tile(1, 1, n_rigids),
                rigids_to_res_idx[0:1, ..., None, :].tile(1, n_rigids, 1)
            ], dim=-1)
            rigids_pair_mask = (
                rigids_mask[0:1, ..., None].tile(1, 1, n_rigids)
                * rigids_mask[0:1, ..., None, :].tile(1, n_rigids, 1)
            )

            framepairs_idx = to_pairs(rigids_pair_idx.float()).long()
            framepairs_mask = to_pairs(rigids_pair_mask[..., None].float()).bool()
        framepairs = z[:, framepairs_idx[0, ..., 0], framepairs_idx[0, ..., 1]]
        framepairs = framepairs * framepairs_mask
        return framepairs


    def _framepair_init(self,
                        rigids_flat,
                        rigids_to_res_idx,
                        rigids_mask,
                        to_queries,
                        to_keys):
        trans = rigids_flat.get_trans()
        rots = rigids_flat.get_rots().get_cur_rot()
        assert rots.shape[-1] == 4

        # [*, N_block, block_Q]
        q_trans = to_queries(trans)
        q_rots = to_queries(rots)
        q_rigids = ru.Rigid(
            trans=q_trans,
            rots=ru.Rotation(quats=q_rots)
        )
        # [*, N_block, block_K]
        k_trans = to_keys(trans)
        k_rots = to_keys(rots)
        # there's some more shenanigans that needs to happen because to_keys will
        # zero out some quats which will lead to weird problems
        mask = (k_rots == 0).all(dim=-1)
        k_rots = (
            k_rots * (~mask[..., None])
            + torch.tensor([1., 0., 0., 0.], device=k_rots.device)[None, None, None] * mask[..., None]
        )
        k_rigids = ru.Rigid(
            trans=k_trans,
            rots=ru.Rotation(quats=k_rots)
        )

        inv_q_rigids = q_rigids[..., None].invert()
        k_rigids = k_rigids[..., None, :]

        rel_rot = inv_q_rigids.get_rots().compose_q(k_rigids.get_rots())
        rel_trans = rel_rot.apply(k_rigids.get_trans()) + inv_q_rigids.get_trans()
        rel_dist = torch.sum(rel_trans ** 2, dim=-1, keepdim=True)

        v_lm = to_queries(rigids_to_res_idx[..., None].float())[..., None, :].long() == to_keys(rigids_to_res_idx[..., None].float())[..., None, :, :].long()
        v_lm = v_lm.float()
        framepair_embed = self.lin_framepair_dist_vec(rel_trans) * v_lm
        framepair_embed = framepair_embed + self.lin_framepair_dist(1 / (1 + rel_dist)) * v_lm
        framepair_embed = framepair_embed + self.lin_same_res(v_lm) * v_lm
        framepair_mask = to_queries(rigids_mask[..., None].float())[..., None, :].bool() & to_keys(rigids_mask[..., None].float())[..., None, :, :].bool()
        framepair_embed = framepair_embed * framepair_mask

        return framepair_embed

    def forward(
        self,
        s,
        z,
        rigids,
        rigids_embed,
        to_queries,
        to_keys,
        to_pairs,
        framepair_embed=None
    ):
        s_broadcast = self.node_to_frame_broadcast(s)
        rigids_embed = rigids_embed + s_broadcast[..., None, :]

        rigids_to_res_idx = torch.arange(rigids_embed.shape[1], device=rigids_embed.device)
        n_batch, _, rigids_per_res = rigids_embed.shape[:3]
        rigids_to_res_idx = torch.tile(rigids_to_res_idx[None, :, None], (n_batch, 1, rigids_per_res))

        # pad to proper input shape
        n_batch = rigids.shape[0]
        rigids_embed_flat = rigids_embed.view(n_batch, -1, self.c_frame)
        n_rigids = rigids_embed_flat.shape[1]
        rigids_mask = torch.ones(rigids_embed_flat.shape[:-1], device=rigids_embed_flat.device, dtype=torch.bool)
        n_padding = (self.block_q - rigids_embed_flat.shape[1] % self.block_q) % self.block_q
        rigids_embed_flat = F.pad(rigids_embed_flat, (0, 0, 0, n_padding), value=0)
        rigids_mask = F.pad(rigids_mask, (0, n_padding), value=0)
        rigids_to_res_idx = F.pad(
            rigids_to_res_idx.view(n_batch, -1),
            (0, n_padding),
            value=0
        )

        # pad the rigids
        rigids_flat = rigids.view(n_batch, -1)
        rigids_padding = ru.Rigid.identity(shape=(n_batch, n_padding), device=rigids_embed.device, fmt="quat")
        # we don't use ru.Rigid.cat cuz this automatically converts stuff to rotmats...
        rigids_flat = ru.Rigid(
            trans=torch.cat([rigids_flat.get_trans(), rigids_padding.get_trans()], dim=-2),
            rots=ru.Rotation(
                quats=torch.cat([
                    rigids_flat.get_rots().get_quats(), rigids_padding.get_rots().get_quats()
                ], dim=-2)
            )
        )

        if framepair_embed is None:
            if self.framepair_init:
                framepair_embed = self._framepair_init(
                    rigids_flat,
                    rigids_to_res_idx,
                    rigids_mask,
                    to_queries,
                    to_keys
                )
            else:
                framepair_embed = 0

        if self.broadcast_singles:
            framepair_src = self.frame_ln(rigids_embed_flat)
            framepair_l = self.frame_l_to_framepair_broadcast(framepair_src)
            framepair_m = self.frame_m_to_framepair_broadcast(framepair_src)
            framepair_l = to_queries(framepair_l).view(n_batch, -1, self.block_q, 1, self.c_framepair)
            framepair_m = to_keys(framepair_m).view(n_batch, -1, 1, self.block_k, self.c_framepair)
            framepair_embed = framepair_embed + framepair_l + framepair_m

        if self.nodepair_to_framepair_broadcast is not None:
            z_broadcast = self._broadcast_pairs(z, rigids_to_res_idx, rigids_mask, to_pairs)
            framepair_embed = framepair_embed + self.nodepair_to_framepair_broadcast(z_broadcast)

        if self.framepair_ffn is not None:
            framepair_embed = framepair_embed + self.framepair_ffn(framepair_embed)

        # rigids_padding = torch.zeros([n_batch, n_padding, 7], device=rigids_embed.device)
        # rigids_padding[:, 0] = 1
        # rigids_flat = ru.Rigid.cat([
        #     rigids_flat, ru.Rigid.from_tensor_7(rigids_padding)
        # ], dim=-1)
        for b in range(self.n_blocks):
            rigids_embed_flat = self.trunk[f'tfmr_{b}'](
                s,
                rigids_flat,
                rigids_embed_flat,
                framepair_embed,
                to_queries,
                to_keys,
                rigids_mask,
                rigids_to_res_idx
            )
            if self.do_rigid_updates:
                # print(rigids_embed_flat)
                rigids_update = self.trunk[f'frame_update_{b}'](rigids_embed_flat) * rigids_mask[..., None]
                # print(b, rigids_update)
                # print(rigids_update)
                # print(rigids_flat.get_rots().get_cur_rot().shape)
                rigids_flat = rigids_flat.compose_q_update_vec(rigids_update)
                # print(rigids_flat.get_rots().get_cur_rot().shape)
                # print(rigids_flat.get_trans(), rigids_flat.get_rots().get_cur_rot())

        rigids_embed = rigids_embed_flat[..., :n_rigids, :].view(rigids_embed.shape)

        if self.agg_rigid_embed:
            s = s + self.frame_to_node_broadcast(rigids_embed).mean(dim=-2)

        rigids_out = rigids_flat[..., :n_rigids].view(rigids.shape)
        # print(rigids_out.get_trans(), rigids_out.get_rots().get_cur_rot())
        # TODO: we only have to do this because we reshape the rigid at the output of the denoiser
        # to match with graph-based losses. if we convert to full dense we
        # should get rid of this part
        rigids_out = ru.Rigid(
            rots=rigids_out.get_rots().map_tensor_fn(lambda x: x.contiguous()),
            trans=rigids_out.get_trans()
        )

        return rigids_embed, s, framepair_embed, rigids_out


class ConditionedSequenceFrameTransformerBlock(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_frame=128,
                 c_framepair=16,
                 num_heads=4,
                 num_qk_points=8,
                 num_v_points=12,
                 block_q=32,
                 block_k=128,
                 inf=1e8,
                 compile_ipa=False
                 ):
        super().__init__()

        self.c_frame = c_frame
        self.c_framepair = c_framepair
        self.c_s = c_s
        self.num_heads = num_heads
        self.c_hidden = c_frame // num_heads
        self.block_q = block_q
        self.block_k = block_k
        self.inf = inf

        self.block_ipa = ConditionedBlockInvariantPointAttention(
            c_s=c_frame,
            c_cond=c_frame,
            c_z=c_framepair,
            c_hidden=self.c_hidden,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            block_Q=block_q,
            block_K=block_k,
            use_compile_path=compile_ipa
        )
        self.transition = ConditionedTransition(c_frame, c_frame)


    def forward(
        self,
        s,
        rigids,
        rigids_embed,
        cond_rigids_embed,
        framepair_embed,
        to_queries,
        to_keys,
        rigids_mask,
    ):
        rigids_embed_update = self.block_ipa(
            s=rigids_embed,
            cond=cond_rigids_embed,
            z=framepair_embed,
            r=rigids,
            s_mask=rigids_mask,
            to_queries=to_queries,
            to_keys=to_keys,
        )
        rigids_embed = rigids_embed + rigids_embed_update * rigids_mask[..., None]

        rigids_embed = rigids_embed + self.transition(rigids_embed, cond_rigids_embed) * rigids_mask[..., None]

        return rigids_embed


class ConditionedSequenceFrameTransformerUpdate(nn.Module):
    def __init__(
        self,
        c_s=256,
        c_z=128,
        c_frame=128,
        c_framepair=16,
        num_heads=4,
        num_qk_points=8,
        num_v_points=12,
        block_q=32,
        block_k=128,
        inf=1e8,
        n_blocks=2,
        do_rigid_updates=False,
        broadcast_pairs=False,
        agg_rigid_embed=True,
        compile_ipa=False,
        framepair_init=False
    ):
        super().__init__()
        self.c_s = c_s
        self.c_frame = c_frame
        self.c_framepair = c_framepair
        self.n_blocks = n_blocks
        self.block_q = block_q
        self.block_k = block_k
        self.do_rigid_updates = do_rigid_updates
        self.agg_rigid_embed = agg_rigid_embed
        self.framepair_init = framepair_init
        if framepair_init:
            self.lin_framepair_dist_vec = Linear(3, c_framepair, bias=False)
            self.lin_framepair_dist = Linear(1, c_framepair, bias=False)
            self.lin_same_res = Linear(1, c_framepair, bias=False)

        self.node_to_frame_broadcast = Linear(c_s, c_frame, bias=False)

        if broadcast_pairs:
            self.nodepair_to_framepair_broadcast = nn.Sequential(
                LayerNorm(c_z),
                Linear(c_z, c_framepair, bias=False)
            )
        else:
            self.nodepair_to_framepair_broadcast = None

        self.frame_ln = LayerNorm(c_frame, c_frame)
        self.frame_l_to_framepair_broadcast = Linear(c_frame, c_framepair, bias=False)
        self.frame_m_to_framepair_broadcast = Linear(c_frame, c_framepair, bias=False)
        self.framepair_ffn = nn.Sequential(
            LayerNorm(c_framepair),
            Linear(c_framepair, c_framepair),
            nn.ReLU(),
            Linear(c_framepair, c_framepair),
            nn.ReLU(),
            Linear(c_framepair, c_framepair),
        )

        self.trunk = nn.ModuleDict()
        for b in range(n_blocks):
            self.trunk[f'tfmr_{b}'] = ConditionedSequenceFrameTransformerBlock(
                c_s=c_s,
                c_frame=c_frame,
                c_framepair=c_framepair,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                block_q=block_q,
                block_k=block_k,
                inf=inf,
                compile_ipa=compile_ipa
            )
            if self.do_rigid_updates:
                self.trunk[f'frame_update_{b}'] = BackboneUpdate(c_frame)

        if self.agg_rigid_embed:
            self.frame_to_node_broadcast = nn.Sequential(
                LayerNorm(c_frame),
                Linear(c_frame, c_s, bias=False, init='final')
            )

    def _broadcast_pairs(
        self,
        z,
        rigids_to_res_idx,
        rigids_mask,
        to_pairs
    ):
        n_rigids = rigids_mask.shape[-1]
        # TODO: this is a little jenk
        # with the batch dim
        with torch.no_grad():
            rigids_pair_idx = torch.stack([
                rigids_to_res_idx[0:1, ..., None].tile(1, 1, n_rigids),
                rigids_to_res_idx[0:1, ..., None, :].tile(1, n_rigids, 1)
            ], dim=-1)
            rigids_pair_mask = (
                rigids_mask[0:1, ..., None].tile(1, 1, n_rigids)
                * rigids_mask[0:1, ..., None, :].tile(1, n_rigids, 1)
            )

            framepairs_idx = to_pairs(rigids_pair_idx.float()).long()
            framepairs_mask = to_pairs(rigids_pair_mask[..., None].float()).bool()
        framepairs = z[:, framepairs_idx[0, ..., 0], framepairs_idx[0, ..., 1]]
        framepairs = framepairs * framepairs_mask
        return framepairs


    def _framepair_init(self,
                        rigids_flat,
                        rigids_to_res_idx,
                        rigids_mask,
                        to_queries,
                        to_keys):
        trans = rigids_flat.get_trans()
        rots = rigids_flat.get_rots().get_cur_rot()
        assert rots.shape[-1] == 4

        # [*, N_block, block_Q]
        q_trans = to_queries(trans)
        q_rots = to_queries(rots)
        q_rigids = ru.Rigid(
            trans=q_trans,
            rots=ru.Rotation(quats=q_rots)
        )
        # [*, N_block, block_K]
        k_trans = to_keys(trans)
        k_rots = to_keys(rots)
        # there's some more shenanigans that needs to happen because to_keys will
        # zero out some quats which will lead to weird problems
        mask = (k_rots == 0).all(dim=-1)
        k_rots = (
            k_rots * (~mask[..., None])
            + torch.tensor([1., 0., 0., 0.], device=k_rots.device)[None, None, None] * mask[..., None]
        )
        k_rigids = ru.Rigid(
            trans=k_trans,
            rots=ru.Rotation(quats=k_rots)
        )

        inv_q_rigids = q_rigids[..., None].invert()
        k_rigids = k_rigids[..., None, :]

        rel_rot = inv_q_rigids.get_rots().compose_q(k_rigids.get_rots())
        rel_trans = rel_rot.apply(k_rigids.get_trans()) + inv_q_rigids.get_trans()
        rel_dist = torch.sum(rel_trans ** 2, dim=-1, keepdim=True)

        v_lm = to_queries(rigids_to_res_idx[..., None].float())[..., None, :].long() == to_keys(rigids_to_res_idx[..., None].float())[..., None, :, :].long()
        v_lm = v_lm.float()
        framepair_embed = self.lin_framepair_dist_vec(rel_trans) * v_lm
        framepair_embed = framepair_embed + self.lin_framepair_dist(1 / (1 + rel_dist)) * v_lm
        framepair_embed = framepair_embed + self.lin_same_res(v_lm) * v_lm
        framepair_mask = to_queries(rigids_mask[..., None].float())[..., None, :].bool() & to_keys(rigids_mask[..., None].float())[..., None, :, :].bool()
        framepair_embed = framepair_embed * framepair_mask

        return framepair_embed

    def forward(
        self,
        s,
        z,
        rigids,
        rigids_embed,
        cond_rigids_embed,
        to_queries,
        to_keys,
        to_pairs,
        framepair_embed=None
    ):
        s_broadcast = self.node_to_frame_broadcast(s)
        rigids_embed = rigids_embed + s_broadcast[..., None, :]

        rigids_to_res_idx = torch.arange(rigids_embed.shape[1], device=rigids_embed.device)
        n_batch, _, rigids_per_res = rigids_embed.shape[:3]
        rigids_to_res_idx = torch.tile(rigids_to_res_idx[None, :, None], (n_batch, 1, rigids_per_res))

        # pad to proper input shape
        n_batch = rigids.shape[0]
        rigids_embed_flat = rigids_embed.view(n_batch, -1, self.c_frame)
        cond_rigids_embed_flat = cond_rigids_embed.view(n_batch, -1, self.c_frame)
        n_rigids = rigids_embed_flat.shape[1]
        rigids_mask = torch.ones(rigids_embed_flat.shape[:-1], device=rigids_embed_flat.device, dtype=torch.bool)
        n_padding = (self.block_q - rigids_embed_flat.shape[1] % self.block_q) % self.block_q
        rigids_embed_flat = F.pad(rigids_embed_flat, (0, 0, 0, n_padding), value=0)
        cond_rigids_embed_flat = F.pad(cond_rigids_embed_flat, (0, 0, 0, n_padding), value=0)
        rigids_mask = F.pad(rigids_mask, (0, n_padding), value=0)
        rigids_to_res_idx = F.pad(
            rigids_to_res_idx.view(n_batch, -1),
            (0, n_padding),
            value=0
        )

        # pad the rigids
        rigids_flat = rigids.view(n_batch, -1)
        rigids_padding = ru.Rigid.identity(shape=(n_batch, n_padding), device=rigids_embed.device, fmt="quat")
        # we don't use ru.Rigid.cat cuz this automatically converts stuff to rotmats...
        rigids_flat = ru.Rigid(
            trans=torch.cat([rigids_flat.get_trans(), rigids_padding.get_trans()], dim=-2),
            rots=ru.Rotation(
                quats=torch.cat([
                    rigids_flat.get_rots().get_quats(), rigids_padding.get_rots().get_quats()
                ], dim=-2)
            )
        )

        framepair_src = self.frame_ln(rigids_embed_flat)
        framepair_l = self.frame_l_to_framepair_broadcast(framepair_src)
        framepair_m = self.frame_m_to_framepair_broadcast(framepair_src)
        framepair_l = to_queries(framepair_l).view(n_batch, -1, self.block_q, 1, self.c_framepair)
        framepair_m = to_keys(framepair_m).view(n_batch, -1, 1, self.block_k, self.c_framepair)
        if framepair_embed is None:
            if self.framepair_init:
                framepair_embed = self._framepair_init(
                    rigids_flat,
                    rigids_to_res_idx,
                    rigids_mask,
                    to_queries,
                    to_keys
                )
            else:
                framepair_embed = 0
        framepair_embed = framepair_embed + framepair_l + framepair_m

        if self.nodepair_to_framepair_broadcast is not None:
            z_broadcast = self._broadcast_pairs(z, rigids_to_res_idx, rigids_mask, to_pairs)
            framepair_embed = framepair_embed + self.nodepair_to_framepair_broadcast(z_broadcast)

        framepair_embed = framepair_embed + self.framepair_ffn(framepair_embed)

        # rigids_padding = torch.zeros([n_batch, n_padding, 7], device=rigids_embed.device)
        # rigids_padding[:, 0] = 1
        # rigids_flat = ru.Rigid.cat([
        #     rigids_flat, ru.Rigid.from_tensor_7(rigids_padding)
        # ], dim=-1)
        for b in range(self.n_blocks):
            rigids_embed_flat = self.trunk[f'tfmr_{b}'](
                s,
                rigids_flat,
                rigids_embed_flat,
                cond_rigids_embed_flat,
                framepair_embed,
                to_queries,
                to_keys,
                rigids_mask,
            )
            if self.do_rigid_updates:
                # print(rigids_embed_flat)
                rigids_update = self.trunk[f'frame_update_{b}'](rigids_embed_flat) * rigids_mask[..., None]
                # print(b, rigids_update)
                # print(rigids_update)
                # print(rigids_flat.get_rots().get_cur_rot().shape)
                rigids_flat = rigids_flat.compose_q_update_vec(rigids_update)
                # print(rigids_flat.get_rots().get_cur_rot().shape)
                # print(rigids_flat.get_trans(), rigids_flat.get_rots().get_cur_rot())

        rigids_embed = rigids_embed_flat[..., :n_rigids, :].view(rigids_embed.shape)

        if self.agg_rigid_embed:
            s = s + self.frame_to_node_broadcast(rigids_embed).mean(dim=-2)

        rigids_out = rigids_flat[..., :n_rigids].view(rigids.shape)
        # print(rigids_out.get_trans(), rigids_out.get_rots().get_cur_rot())
        # TODO: we only have to do this because we reshape the rigid at the output of the denoiser
        # to match with graph-based losses. if we convert to full dense we
        # should get rid of this part
        rigids_out = ru.Rigid(
            rots=rigids_out.get_rots().map_tensor_fn(lambda x: x.contiguous()),
            trans=rigids_out.get_trans()
        )

        return rigids_embed, s, framepair_embed, rigids_out

class KnnInvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s,
        c_z,
        c_hidden,
        num_heads,
        num_qk_points,
        num_v_points,
        inf: float = 1e5,
        eps: float = 1e-8,
        final_init='final',
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = num_heads
        self.no_qk_points = num_qk_points
        self.no_v_points = num_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, bias=False)
        self.linear_k = Linear(self.c_s, hc, bias=False)
        self.linear_v = Linear(self.c_s, hc, bias=False)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq, bias=False)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv, bias=False)

        self.linear_b = Linear(self.c_z, self.no_heads, bias=False)
        self.down_z = Linear(self.c_z, self.c_z // 4, bias=False)

        self.head_weights = nn.Parameter(torch.zeros((self.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init=final_init, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.ln_s = LayerNorm(c_s)
        self.ln_z = LayerNorm(c_z)

    def forward(
        self,
        s: torch.Tensor,
        edge_index: torch.Tensor,
        z: torch.Tensor,
        r: ru.Rigid,
        s_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        s = self.ln_s(s)
        z = self.ln_z(z)

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        # [*, N_res, H * C_hidden]
        k = self.linear_k(s)
        # [*, N_res, H * C_hidden]
        v = self.linear_v(s)

        # [*, N_res, K, H * C_hidden]
        k = torch.gather(
            k[..., None, :].expand(-1, -1, edge_index.shape[-1], -1),
            1,
            edge_index[..., None].tile(1, 1, 1, k.shape[-1])
        )
        # [*, N_res, K, H * C_hidden]
        v = torch.gather(
            v[..., None, :].expand(-1, -1, edge_index.shape[-1], -1),
            1,
            edge_index[..., None].tile(1, 1, 1, v.shape[-1])
        )

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        # [*, N_res, K, H, C_hidden]
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        # [*, N_res, K, H, C_hidden]
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)
        # [*, N_res, H * P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-1] + (self.no_heads * self.no_qk_points, 3))
        q_pts = r[..., None].apply(q_pts)
        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)
        # [*, N_res, K, H * (P_q + P_v) * 3]
        kv_pts = torch.gather(
            kv_pts[..., None, :].expand(-1, -1, edge_index.shape[-1], -1),
            1,
            edge_index[..., None].tile(1, 1, 1, kv_pts.shape[-1])
        )
        # [*, N_res, K, H * (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-1] + (-1, 3))
        kv_pts = r[..., None, None].apply(kv_pts)

        # [*, N_res, K, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, K, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, K, H]
        b = self.linear_b(z)

        # [*, N_res, K, H]
        a = torch.einsum("bnhc,bnkhc->bnkh", q, k)
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * b

        # [*, N_res, K, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts
        pt_att = pt_displacement ** 2

        # [*, N_res, K, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, K, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, K]
        attn_mask = s_mask[..., None].float() * torch.gather(
            s_mask[..., None].expand(-1, -1, edge_index.shape[-1]),
            1,
            edge_index
        )
        attn_mask = self.inf * (attn_mask - 1)

        # [*, N_res, K, H]
        a = a + pt_att
        a = a + attn_mask[..., None]
        # [*, N_res, H, K]
        a = a.transpose(-1, -2)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.einsum("bnhk,bnkhc->bnhc", a, v)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, N_res, H, P_v, 3]
        o_pt = torch.einsum("bnhk,bnkhvc->bnhvc", a, v_pts)
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_res, H * P_v * 3]
        o_pt = flatten_final_dims(o_pt, 3)

        # [*, N_res, K, H, C_z // 4]
        pair_z = self.down_z(z).to(dtype=a.dtype)
        # [*, N_res, H, C_z // 4]
        o_pair = torch.einsum("bnhk,bnkc->bnhc", a, pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, o_pt, o_pt_norm_feats, o_pair]

        # [*, N_res, C_s]
        out = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            ).to(dtype=z.dtype)
        )

        return out.view(s.shape)



class KnnFrameTransformerBlock(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_frame=128,
                 c_framepair=16,
                 num_heads=4,
                 num_qk_points=8,
                 num_v_points=12,
                 inf=1e8
                 ):
        super().__init__()

        self.c_frame = c_frame
        self.c_framepair = c_framepair
        self.c_s = c_s
        self.num_heads = num_heads
        self.c_hidden = c_frame // num_heads
        self.inf = inf

        self.block_ipa = KnnInvariantPointAttention(
            c_s=c_frame,
            c_z=c_framepair,
            c_hidden=self.c_hidden,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
        )

        self.transition = GatherConditionedTransition(c_frame, c_s)

    def forward(
        self,
        s,
        rigids,
        rigids_embed,
        framepair_embed,
        edge_index,
        rigids_mask,
        rigids_to_res_idx
    ):
        rigids_embed_update = self.block_ipa(
            s=rigids_embed,
            edge_index=edge_index,
            z=framepair_embed,
            r=rigids,
            s_mask=rigids_mask,
        )
        rigids_embed = rigids_embed + rigids_embed_update * rigids_mask[..., None]

        rigids_embed = rigids_embed + self.transition(rigids_embed, s, rigids_to_res_idx) * rigids_mask[..., None]

        return rigids_embed


class KnnFrameTransformerUpdate(nn.Module):
    def __init__(
        self,
        c_s=256,
        c_z=128,
        c_frame=128,
        c_framepair=16,
        num_heads=4,
        num_qk_points=8,
        num_v_points=12,
        inf=1e8,
        n_blocks=2,
        k=20
    ):
        super().__init__()
        self.c_s = c_s
        self.c_frame = c_frame
        self.c_framepair = c_framepair
        self.n_blocks = n_blocks
        self.k = k

        self.node_to_frame_broadcast = Linear(c_s, c_frame, bias=False)
        self.nodepair_ln = LayerNorm(c_z)
        self.nodepair_to_framepair_broadcast = Linear(c_z, c_framepair, bias=False)

        self.trunk = nn.ModuleDict()
        for b in range(n_blocks):
            self.trunk[f'tfmr_{b}'] = KnnFrameTransformerBlock(
                c_s=c_s,
                c_frame=c_frame,
                c_framepair=c_framepair,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                inf=inf,
            )

        self.frame_to_node_broadcast = nn.Sequential(
            LayerNorm(c_frame),
            Linear(c_frame, c_s, bias=False, init='final')
        )

    @torch.no_grad()
    def _construct_knn_graph(self, rigids):
        n_batch, n_res, n_rigids_per_res = rigids.shape
        trans = rigids.get_trans()
        bb_trans = trans[..., 0, :]
        dist_mat = torch.cdist(bb_trans, bb_trans)
        _, res_edge_index = torch.topk(dist_mat, k=self.k, dim=-1, largest=False)
        rigid_idx = torch.arange(np.prod(rigids.shape[-2:]), device=trans.device).view(rigids.shape[-2:])
        rigid_edge_index = rigid_idx[res_edge_index]
        print(rigid_edge_index.shape)
        rigid_edge_index = rigid_edge_index.view(n_batch, n_res * self.k * n_rigids_per_res)
        return res_edge_index, rigid_edge_index


    def forward(
        self,
        s,
        z,
        rigids,
        rigids_embed,
        s_mask,
    ):
        s_broadcast = self.node_to_frame_broadcast(s)
        rigids_embed = rigids_embed + s_broadcast[..., None, :]

        res_edge_index, rigid_edge_index = self._construct_knn_graph(rigids)
        framepair_embed = self.nodepair_to_framepair_broadcast(self.nodepair_ln(z))
        print(framepair_embed.shape, res_edge_index.shape)
        framepair_embed = torch.gather(
            framepair_embed,
            2,
            res_edge_index[..., None].tile(1, 1, 1, framepair_embed.shape[-1])
        )
        n_rigids_per_res = rigids.shape[-1]
        framepair_embed = torch.repeat_interleave(framepair_embed, n_rigids_per_res, dim=-2)

        rigids_to_res_idx = torch.arange(rigids_embed.shape[1], device=rigids_embed.device)
        n_batch, _, rigids_per_res = rigids_embed.shape[:3]
        rigids_to_res_idx = torch.tile(rigids_to_res_idx[None, :, None], (n_batch, 1, rigids_per_res))

        rigids_flat = rigids.view(n_batch, -1)
        rigids_embed_flat = rigids_embed.view(n_batch, -1, self.c_frame)
        rigids_mask = torch.repeat_interleave(s_mask, n_rigids_per_res, dim=-1)
        for b in range(self.n_blocks):
            rigids_embed_flat = self.trunk[f'tfmr_{b}'](
                s,
                rigids_flat,
                rigids_embed_flat,
                framepair_embed,
                rigid_edge_index,
                rigids_mask,
                rigids_to_res_idx
            )

        rigids_embed = rigids_embed + rigids_embed_flat.view(rigids_embed.shape)
        s = s + self.frame_to_node_broadcast(rigids_embed).mean(dim=-2)

        return rigids_embed, s