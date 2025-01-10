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
# import dgl
# import dgl.sparse
# from dgl.ops import u_mul_e_sum, u_dot_v, u_add_v

from proteinzen.model.modules.openfold.layers_v2 import (
    Linear, swish, ipa_point_weights_init_, permute_final_dims, flatten_final_dims, BackboneUpdate, LayerNorm,
    AdaLN, ConditionedTransition)
from proteinzen.utils.openfold import rigid_utils as ru

import taichi as ti
from taichi_kernels.torch.ipa import fused_ipa_kernel_broadcast_z_no_reshape
if torch.cuda.is_available():
    ti.init(arch=ti.cuda)
else:
    ti.init()


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


class BroadcastZInvariantPointAttention(nn.Module):
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

        hpk = self.no_heads * self.no_qk_points * 3
        hpv = self.no_heads * self.no_v_points * 3
        self.linear_k_points = Linear(self.c_s, hpk, bias=False)
        self.linear_v_points = Linear(self.c_s, hpv, bias=False)

        self.linear_b = Linear(self.c_z, self.no_heads, bias=False)
        self.down_z = Linear(self.c_z, self.c_z // 4, bias=False)

        self.head_weights = nn.Parameter(torch.zeros((self.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        self.linear_out = Linear(self.no_heads * self.c_hidden, self.c_s, init=final_init, bias=False)
        self.linear_out_bias = Linear(self.no_heads * self.c_z // 4, self.c_s, init=final_init, bias=False)
        self.linear_out_pts = Linear(self.no_heads * self.no_v_points * 4, self.c_s, init=final_init, bias=False)

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
        rigids_to_res_idx: torch.Tensor,
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
        # [*, N_block, block_Q, H * C_hidden]
        q = self.linear_q(s)
        # [*, N_block, block_K, H * C_hidden]
        k = self.linear_k(s)
        # [*, N_block, block_K, H * C_hidden]
        v = self.linear_v(s)

        # [*, N_block, block_Q, H * P_q * 3]
        q_pts = self.linear_q_points(s)
        # [*, N_block, block_Q, H * P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-1] + (self.no_heads * self.no_qk_points, 3))
        q_pts = r[..., None].apply(q_pts)

        # [*, N_block, block_K, H * (P_q + P_v) * 3]
        k_pts = self.linear_k_points(s)
        # [*, N_block, block_K, H * (P_q + P_v), 3]
        k_pts = k_pts.view(k_pts.shape[:-1] + (-1, 3))
        k_pts = r[..., None].apply(k_pts)

        # [*, N_block, block_K, H * (P_q + P_v) * 3]
        v_pts = self.linear_v_points(s)
        # [*, N_block, block_K, H * (P_q + P_v), 3]
        v_pts = v_pts.view(v_pts.shape[:-1] + (-1, 3))
        v_pts = r[..., None].apply(v_pts)

        # [*, N_block, block_Q, block_K, H]
        b = self.linear_b(z)
        # [*, N_block, block_Q, block_K, H, C_z // 4]
        pair_z = self.down_z(z)

        head_weights = self.softplus(self.head_weights)
        attn_mask = (s_mask[..., None] * s_mask[..., None]).float()
        attn_mask = self.inf * (attn_mask - 1)

        o, o_pt, o_pair = fused_ipa_kernel_broadcast_z_no_reshape(
            q=q,
            q_pts=q_pts,
            k=k,
            k_pts=k_pts,
            v=v,
            v_pts=v_pts,
            z_attn_bias=b + attn_mask[..., None],
            z_out_bias=pair_z,
            rigid_to_res_idx=rigids_to_res_idx,
            pts_bias_scale=head_weights
        )

        ################
        # Compute output
        ################
        o_pt = r[..., None].invert_apply(o_pt)
        # [*, N_block, block_Q, H * P_v]
        o_pt_norm_feats = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        # [*, N_block, block_Q, H * P_v * 3]
        o_pt = flatten_final_dims(o_pt, 2)

        o_feats = [o_pt, o_pt_norm_feats]

        out = (
            self.linear_out(o)
            + self.linear_out_bias(o_pair)
            + self.linear_out_pts(
                torch.cat(o_feats, dim=-1)
            )
        )

        return out


class SequenceFrameTransformerBlock(nn.Module):
    def __init__(self,
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

        self.ipa = BroadcastZInvariantPointAttention(
            c_s=c_frame,
            c_z=c_z,
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
        z,
        rigids,
        rigids_embed,
        s_mask,
        rigids_mask,
        rigids_to_res_idx
    ):
        rigids_embed_update = self.ipa(
            s=rigids_embed,
            z=z,
            r=rigids,
            s_mask=s_mask,
            rigids_to_res_idx=rigids_to_res_idx
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
        agg_rigid_embed=True,
        compile_ipa=False,
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

        self.node_to_frame_broadcast = Linear(c_s, c_frame, bias=False)

        self.trunk = nn.ModuleDict()
        for b in range(n_blocks):
            self.trunk[f'tfmr_{b}'] = SequenceFrameTransformerBlock(
                c_s=c_s,
                c_z=c_z,
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

        rigids_to_res_idx = torch.arange(rigids_embed.shape[1], device=rigids_embed.device)
        n_batch, _, rigids_per_res = rigids_embed.shape[:3]
        rigids_to_res_idx = torch.tile(rigids_to_res_idx[None, :, None], (n_batch, 1, rigids_per_res))

        n_batch = rigids.shape[0]
        rigids_embed_flat = rigids_embed.view(n_batch, -1, self.c_frame)
        n_rigids = rigids_embed_flat.shape[1]
        rigids_mask = torch.ones(rigids_embed_flat.shape[:-1], device=rigids_embed_flat.device, dtype=torch.bool)
        rigids_flat = rigids.view(n_batch, -1)
        rigids_flat_mask = torch.ones(rigids_flat.shape, device=rigids_embed_flat.device, dtype=torch.bool)
        rigids_to_res_idx = rigids_to_res_idx.reshape(n_batch, -1)

        for b in range(self.n_blocks):
            rigids_embed_flat = self.trunk[f'tfmr_{b}'](
                s,
                z,
                rigids_flat,
                rigids_embed_flat,
                s_mask,
                rigids_flat_mask,
                rigids_to_res_idx
            )
            if self.do_rigid_updates:
                rigids_update = self.trunk[f'frame_update_{b}'](rigids_embed_flat) * rigids_mask[..., None]
                rigids_flat = rigids_flat.compose_q_update_vec(rigids_update)

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

        return rigids_embed, s, rigids_out
