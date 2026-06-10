"""
Flash attention with factorized pair bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Sequence
import math
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_func

from proteinzen.openfold.layers.layers import Linear, LayerNorm, flatten_final_dims
from proteinzen.openfold.layers.layers_v2 import Transition

from .flash_ipa import pad_input, unpad_input

attn_dtype_dict = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

class FlashAttentionFactorizedPairBias(nn.Module):
    def __init__(
        self,
        use_packed: bool = True,
        c_s: int = 256,
        c_z: int = 128,
        c_hidden: int = 128,
        no_heads: int = 8,
        z_factor_rank: int = 2,  # 0 for no factorization
        inf: float = 1e5,
        eps: float = 1e-8,
        accumulate_pair=False,
        qk_rmsnorm = True
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
        self.use_packed = use_packed

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.z_factor_rank = z_factor_rank
        self.inf = inf
        self.eps = eps
        self.accumulate_pair = accumulate_pair
        self.qk_rmsnorm = qk_rmsnorm

        self.ln_s = LayerNorm(c_s)
        self.ln_z = LayerNorm(c_z)

        self.rms_q = nn.RMSNorm(c_hidden, dtype=torch.bfloat16)
        self.rms_k = nn.RMSNorm(c_hidden, dtype=torch.bfloat16)

        hqkv = no_heads * c_hidden
        self.lin_qkv = Linear(c_s, 3 * hqkv, bias=False)
        self.lin_b = Linear(c_z, no_heads, bias=False)

        if accumulate_pair:
            hdz = no_heads * self.c_z // 4
            self.down_z = Linear(self.c_z, hdz, bias=False)
            self.lin_out = Linear(hqkv + hdz, c_s, bias=False)
        else:
            self.lin_out = Linear(hqkv, c_s, bias=False)
            self.down_z = None

    def forward(
        self,
        s,
        z_factor_1: torch.Tensor,
        z_factor_2: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        s = self.ln_s(s)
        z_comb = torch.cat([z_factor_1.unsqueeze(1), z_factor_2.unsqueeze(1)], dim=1)
        z_comb_ln = self.ln_z(z_comb)
        z_factor_1 = z_comb_ln[:, 0, :, :, :]  # B, N_res, rank, C_z
        z_factor_2 = z_comb_ln[:, 1, :, :, :]  # B, N_res, rank, C_z

        #######################################
        # Generate scalar activations
        #######################################
        # [*, N_res, 3 * H * C_hidden]
        qkv = self.lin_qkv(s)
        # [*, N_res, 3, H, C_hidden]
        qkv = qkv.view(qkv.shape[:-1] + (3, self.no_heads, -1))
        q, k, v = torch.unbind(qkv, dim=-3)

        if self.qk_rmsnorm:
            q = self.rms_q(q)
            k = self.rms_k(k)

        ## Compute pair bias factors
        # z_factor_1 has shape [B, N_res, rank, C_z]
        b = self.lin_b(z_comb)
        b1 = b[:, 0, :, :, :].permute(0, 1, 3, 2)  # B, N_res, H, rank
        b2 = b[:, 1, :, :, :].permute(0, 1, 3, 2)  # B, N_res, H, rank

        ## Compute q_aggregated
        q_aggregated = torch.cat([q, b1], dim=-1)

        ## Compute k_aggregated
        k_scaled = k * math.sqrt(1.0 / self.c_hidden)
        k_aggregated = torch.cat([k_scaled, b2], dim=-1)

        if self.accumulate_pair:
            assert self.down_z is not None  # for mypy
            z_comb_down = self.down_z(z_comb_ln)
            z_factor_1_down = z_comb_down[:, 0, :, :, :]  # B, N_res, rank, C_z//4
            z_factor_2_down = z_comb_down[:, 1, :, :, :]  # B, N_res, rank, C_z//4

            ## Compute v_aggregated
            v_aggregated = torch.cat(
                [
                    v,
                    z_factor_2_down.view(*z_factor_2_down.shape[:2], 1, -1).expand(-1, -1, self.no_heads, -1),
                ],
                dim=-1,
            )
        else:
            v_aggregated = v
            z_factor_1_down = None
            z_factor_2_down = None


        if mask is None:
            mask = torch.ones((q.shape[0], q.shape[1]), device=q.device, dtype=torch.bool)

        # FA2 requires that QKV have same size for last dimension. So just choose the smallest possible size.
        max_dim_sz = max(q_aggregated.shape[-1], k_aggregated.shape[-1], v_aggregated.shape[-1])
        q_aggregated = F.pad(q_aggregated, (0, max_dim_sz - q_aggregated.shape[-1]), value=0.0)
        k_aggregated = F.pad(k_aggregated, (0, max_dim_sz - k_aggregated.shape[-1]), value=0.0)
        v_aggregated = F.pad(v_aggregated, (0, max_dim_sz - v_aggregated.shape[-1]), value=0.0)
        if self.use_packed:
            qkv = torch.cat([q_aggregated.unsqueeze(2), k_aggregated.unsqueeze(2), v_aggregated.unsqueeze(2)], dim=2)
            (
                qkv,
                indices,
                cu_seqlens,
                max_seqlen,
                _,
            ) = unpad_input(qkv, mask)

            attn_res = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, softmax_scale=1)

        else:
            q_aggregated, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q_aggregated, mask)
            k_aggregated, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k_aggregated, mask)
            v_aggregated, _, cu_seqlens_v, max_seqlen_v, _ = unpad_input(v_aggregated, mask)

            attn_res = flash_attn_varlen_func(
                q_aggregated,
                k_aggregated,
                v_aggregated,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale=1,
            )
        attn_res = pad_input(
            attn_res,
            indices=indices,
            batch=q.shape[0],
            seqlen=q.shape[1],
        )

        # if attn_res.dtype != torch.float32:
        #     attn_res = attn_res.float()

        if self.accumulate_pair:
            assert z_factor_1_down is not None
            assert z_factor_2_down is not None

            attn_res = attn_res[
                :, :, :, : self.c_hidden + self.z_factor_rank * z_factor_2_down.shape[-1]
            ]

            o = attn_res[:, :, :, : self.c_hidden]
            o = flatten_final_dims(o, 2)

            # calculate o_pair
            o_pair = attn_res[:, :, :, self.c_hidden:].view(
                *attn_res.shape[:3], self.z_factor_rank, -1
            )  # B, L, H, rank, C_z//4
            o_pair = torch.einsum("b n r d, b n h r d -> b n h d", z_factor_1_down, o_pair)
            o_pair = flatten_final_dims(o_pair, 2)
            o_feats = [o, o_pair]

            s = self.lin_out(torch.cat(o_feats, dim=-1))
        else:
            o = attn_res[
                :, :, :, : self.c_hidden
            ]
            o = flatten_final_dims(o, 2)
            # print(attn_res.shape, o.shape)
            s = self.lin_out(o)

        return s


class FactorizedPairBiasTransformer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        use_packed: bool = True,
        c_s: int = 256,
        c_z: int = 128,
        c_hidden: int = 128,
        no_heads: int = 8,
        z_factor_rank: int = 2,  # 0 for no factorization
        accumulate_pair=False,
        qk_rmsnorm = True
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.trunk = nn.ModuleDict()

        for i in range(num_blocks):
            self.trunk[f'attn_{i}'] = FlashAttentionFactorizedPairBias(
                use_packed=use_packed,
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                z_factor_rank=z_factor_rank,
                accumulate_pair=accumulate_pair,
                qk_rmsnorm=qk_rmsnorm
            )
            self.trunk[f'ffn_{i}'] = Transition(c_s)

    def forward(
        self,
        s,
        z_factor_1: torch.Tensor,
        z_factor_2: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        for i in range(self.num_blocks):
            s = s + self.trunk[f'attn_{i}'](
                s, z_factor_1, z_factor_2, mask
            )
            s = s + self.trunk[f'ffn_{i}'](s) * (mask[..., None] if mask is not None else 1)

        return s
