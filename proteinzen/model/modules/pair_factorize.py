"""
Modules for neural low-rank factorization of pair representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Sequence
import math
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_func

from cuequivariance_torch import attention_pair_bias

from proteinzen.openfold.layers.layers import Linear, LayerNorm, flatten_final_dims
from proteinzen.openfold.layers.layers_v2 import Transition

from proteinzen.model.modules.attention import CrossAttentionPairBiasLayer, TransformerPairBiasLayer, FlashTransformerEncoderLayer


class TransitionFactorizer(nn.Module):
    def __init__(
        self,
        c_factor: int = 256,
        c_z: int = 128,
        z_factor_rank: int = 2,
    ):
        super().__init__()

        self.c_factor = c_factor
        self.z_factor_rank = z_factor_rank

        self.ffn_z1 = Transition(c_z)
        self.out_z1 = Linear(c_z, c_factor * z_factor_rank)
        self.ffn_z2 = Transition(c_z)
        self.out_z2 = Linear(c_z, c_factor * z_factor_rank)

    def forward(self, z):
        z1_pre = z + self.ffn_z1(z)
        z1 = self.out_z1(z1_pre.mean(dim=-2))
        z1 = z1.view(*z1.shape[:-1], self.z_factor_rank, self.c_factor)

        z2_pre = z + self.ffn_z2(z)
        z2 = self.out_z2(z2_pre.mean(dim=-3))
        z2 = z2.view(*z2.shape[:-1], self.z_factor_rank, self.c_factor)
        return z1, z2


class CrossAttentionFactorizer(nn.Module):
    def __init__(
        self,
        c_s: int = 256,
        c_factor: int = 256,
        c_z: int = 128,
        z_factor_rank: int = 2,
        no_heads: int = 4,
        no_blocks: int = 4
    ):
        super().__init__()
        self.c_factor = c_factor
        self.z_factor_rank = z_factor_rank
        self.no_blocks = no_blocks

        self.lin_z1_in = Linear(c_s, c_s, bias=False)
        self.lin_z2_in = Linear(c_s, c_s, bias=False)

        self.out_z1 = Linear(c_s, c_factor * z_factor_rank)
        self.out_z2 = Linear(c_s, c_factor * z_factor_rank)

        self.trunk = nn.ModuleDict()
        for i in range(no_blocks):
            self.trunk[f'z1_attn_{i}'] = CrossAttentionPairBiasLayer(
                c_s,
                c_z,
                no_heads,
                dropout=0.0,
                use_qk_norm=True
            )
            self.trunk[f'z2_attn_{i}'] = CrossAttentionPairBiasLayer(
                c_s,
                c_z,
                no_heads,
                dropout=0.0,
                use_qk_norm=True
            )

    def forward(self, x, z, x_mask):
        z1_pre = self.lin_z1_in(x)
        z2_pre = self.lin_z2_in(x)

        for i in range(self.no_blocks):
            z1_pre = self.trunk[f'z1_attn_{i}'](
                z1_pre,
                z2_pre,
                z,
                x_mask
            )
            z2_pre = self.trunk[f'z2_attn_{i}'](
                z2_pre,
                z1_pre,
                z.transpose(-2, -3),
                x_mask
            )

        z1 = self.out_z1(z1_pre)
        z1 = z1.view(*z1.shape[:-1], self.z_factor_rank, self.c_factor)

        z2 = self.out_z2(z2_pre)
        z2 = z2.view(*z2.shape[:-1], self.z_factor_rank, self.c_factor)

        return z1, z2


class AttentionFactorizer(nn.Module):
    def __init__(
        self,
        c_s: int = 256,
        c_factor: int = 256,
        c_z: int = 128,
        z_factor_rank: int = 2,
        no_heads: int = 4,
        no_blocks: int = 4
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_factor = c_factor
        self.z_factor_rank = z_factor_rank
        self.no_blocks = no_blocks

        self.lin_z1_in = Linear(c_s, c_z * z_factor_rank, bias=False)
        self.lin_z2_in = Linear(c_s, c_z * z_factor_rank, bias=False)

        self.lin_z1_out = Linear(c_z, c_factor, bias=False)
        self.lin_z2_out = Linear(c_z, c_factor, bias=False)

        self.trunk = nn.ModuleDict()
        for i in range(no_blocks):
            self.trunk[f'z1_attn_{i}'] = FlashTransformerEncoderLayer(
                h_dim=c_z,
                no_heads=no_heads,
                h_ff=c_z,
                ln_first=True,
                bias=False,
                dropout=0.0,
            )
            self.trunk[f'z2_attn_{i}'] = FlashTransformerEncoderLayer(
                h_dim=c_z,
                no_heads=no_heads,
                h_ff=c_z,
                ln_first=True,
                bias=False,
                dropout=0.0,
            )

    def forward(self, x, z, x_mask):
        z1 = self.lin_z1_in(x)
        z1 = z1.view(*z1.shape[:-1], self.z_factor_rank, self.c_z)
        z2 = self.lin_z2_in(x)
        z2 = z2.view(*z2.shape[:-1], self.z_factor_rank, self.c_z)

        zfac_z_mask = F.pad(x_mask, (self.z_factor_rank, 0), value=True)[..., None, :]

        for i in range(self.no_blocks):
            z1_z = torch.cat([z1, z], dim=-2)
            print(z1_z.shape, zfac_z_mask.shape)

            z1_z = self.trunk[f'z1_attn_{i}'](
                z1_z,
                zfac_z_mask
            )
            z1 = z1_z[..., :self.z_factor_rank, :]
            z = z1_z[..., self.z_factor_rank:, :]

            z2_z = torch.cat([z2, z.transpose(-2, -3)], dim=-2)
            z2_z = self.trunk[f'z1_attn_{i}'](
                z2_z,
                zfac_z_mask
            )
            z2 = z2_z[..., :self.z_factor_rank, :]
            z = z2_z[..., self.z_factor_rank:, :]

        z1 = self.lin_z1_out(z1)
        z2 = self.lin_z1_out(z2)

        return z1, z2