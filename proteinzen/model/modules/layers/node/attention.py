from proteinzen.model.modules.openfold.layers import Linear, flatten_final_dims, ipa_point_weights_init_, _deepspeed_evo_attn
from proteinzen.model.denoiser.protein._attn import DropoutRowwise, evoformer_supported
from proteinzen.model.modules.openfold.layers_v2 import LayerNorm, Transition, permute_final_dims, AdaLN
from proteinzen.utils.openfold.rigid_utils import Rigid

from proteinzen.utils.openfold import rigid_utils as ru

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch_geometric.utils as pygu



import math
from typing import Optional, Sequence, Tuple, Union, Callable

import importlib
deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
ds4s_is_installed = deepspeed_is_installed and importlib.util.find_spec("deepspeed.ops.deepspeed4science") is not None
if deepspeed_is_installed:
    import deepspeed

if ds4s_is_installed:
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention


class GraphInvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s,
        c_z,
        c_hidden,
        no_heads,
        no_qk_points,
        no_v_points,
        inf: float = 1e5,
        eps: float = 1e-8,
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
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        # TODO: Remove after published checkpoint is updated without these weights.
        self.linear_rbf = Linear(20, 1)

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
        reduce_memory=True,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [N_res, C_s] single representation, scalars
            z:
                [N_edge, C_e] pair representation
            edge_index:
                [2, N_edge] edge index
            r:
                [N_res] transformation object
            mask:
                [N_res] mask
        Returns:
            [N_res, C_s] single scalar representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]
        n_nodes = s.shape[0]

        src = edge_index[1]
        dst = edge_index[0]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        # [N_edge, H, C_hidden]
        q_src = q[src]

        # [N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [N_res, k, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )
        # [N_edge, H, P_q, 3]
        q_pts_src = q_pts[src]

        # [N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [N_res, k, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_edge, H]
        b = self.linear_b(z[0])

        if(_offload_inference):
            z[0] = z[0].cpu()

        # [N_edge, H]
        k_dst = k[dst]
        a = torch.matmul(
            q_src[..., None, :],  # [N_edge, H, 1, C_hidden]
            k_dst[..., :, None],  # [N_edge, H, C_hidden, 1]
        ).squeeze(-1).squeeze(-1)
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * b)

        # [N_edge, H, P_q, 3]
        k_pts_dst = k_pts[dst]
        pt_displacement = q_pts_src - k_pts_dst
        pt_att = pt_displacement ** 2

        # [N_edge, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [N_edge, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        edge_mask = mask[dst] * mask[src]
        edge_mask = self.inf * (edge_mask - 1)
        # [N_edge, H]
        a = a + pt_att
        a = a + edge_mask[..., None]
        a = pygu.softmax(a, src, num_nodes=n_nodes)

        ################
        # Compute output
        ################
        v_dst = v[edge_index[0]]
        # [N_res, H, C_hidden]
        o = pygu.scatter(
            a[..., None] *  # [N_edge, H]
            v_dst,  # [N_edge, H, C_hidden]
            src,
            dim=0,
            dim_size=n_nodes
        )
        # [N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [N_res, H, P_v, 3]
        v_pts_dst = v_pts[edge_index[0]]
        o_pt = pygu.scatter(
            a[..., None, None]  # [N_edge, H, 1, 1]
            * v_pts_dst,  # [N_edge, H, P_v, 3]
            src,
            dim=0,
            dim_size=n_nodes
        )
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if(_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [N_res, H, C_z // 4]
        pair_z = self.down_z(z[0]).to(dtype=a.dtype)  # [N_edge, C_z // 4]
        o_pair = pygu.scatter(
            a[..., None]  # [N_edge, H, 1]
            * pair_z[..., None, :],  # [N_edge, 1, C_z // 4]
            src,
            dim=0,
            dim_size=n_nodes
        )

        # [N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            ).to(dtype=z[0].dtype)
        )

        return s


class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 h_dim,
                 no_heads,
                 h_ff=2048,
                 ln_first=False,
                 bias=False,
                 dropout=0.1,
                 dtype=None
    ):
        super().__init__()
        self.h_head = h_dim // no_heads
        self.no_heads = no_heads
        self.ln_first = ln_first
        self.dtype = dtype

        self.lin_q = Linear(h_dim, self.h_head * no_heads, bias=bias)
        self.lin_kv = Linear(h_dim, 2 * self.h_head * no_heads, bias=bias)
        self.ln1 = LayerNorm(h_dim)
        self.ln2 = LayerNorm(h_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            Linear(h_dim, h_ff, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(h_ff, h_dim, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, x_mask):
        dtype = x.dtype
        x_mask = x_mask.bool()

        if self.ln_first:
            _x = self.ln1(x)
        else:
            _x = x

        q = self.lin_q(_x)
        kv = self.lin_kv(_x)
        k, v = kv.split(self.h_head * self.no_heads, dim=-1)
        q = q.view(*q.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)
        k = k.view(*k.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)
        v = v.view(*v.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)
        if self.dtype is not None:
            q = q.type(self.dtype)
            k = k.type(self.dtype)
            v = v.type(self.dtype)
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        #     update = F.scaled_dot_product_attention(
        #         q, k, v, attn_mask=x_mask
        #     )
        print(q.is_contiguous(), k.is_contiguous(), v.is_contiguous(), x_mask.is_contiguous())
        update = F.scaled_dot_product_attention(
            q, k, v, attn_mask=x_mask[..., None, :, None]
        )
        update = update.transpose(-2, -3).flatten(-2, -1)
        if self.dtype is not None:
            update = update.to(dtype)
        x = x + self.dropout(update)
        if not self.ln_first:
            x = self.ln1(x)

        if self.ln_first:
            _x = self.ln2(x)
        else:
            _x = x
        x = x + self.ffn(_x) * x_mask[..., None]
        if not self.ln_first:
            x = self.ln2(x)

        return x


class FlashTransformerEncoder(nn.Module):
    def __init__(self,
                 h_dim,
                 no_heads,
                 n_layers,
                 h_ff=2048,
                 ln_first=False,
                 bias=False,
                 dropout=0.1,
                 dtype=None
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FlashTransformerEncoderLayer(
                    h_dim=h_dim,
                    no_heads=no_heads,
                    h_ff=h_ff,
                    ln_first=ln_first,
                    bias=bias,
                    dropout=dropout,
                    dtype=dtype
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, x_mask):
        for layer in self.layers:
            x = layer(x, x_mask)
        return x




class ConditionedTransformerPairBiasLayer(nn.Module):
    def __init__(self,
                 c_s,
                 c_cond,
                 c_z,
                 no_heads,
                 dropout=0.1,
                 inf=1e8

    ):
        super().__init__()
        self.h_head = c_s // no_heads
        self.no_heads = no_heads
        self.inf = inf

        self.lin_q = Linear(c_s, self.h_head * no_heads, bias=False)
        self.lin_kv = Linear(c_s, 2 * self.h_head * no_heads, bias=False)
        self.lin_b = Linear(c_z, no_heads, bias=False)
        self.out_gate = Linear(c_s, c_s, bias=False)
        self.lin_out = Linear(c_s, c_s, bias=False)
        self.cond_gate = Linear(c_cond, c_s)
        with torch.no_grad():
            self.cond_gate.bias.fill_(-2.0)

        self.ln_s = AdaLN(c_s, c_cond)
        self.ln_z = LayerNorm(c_z)
        self.dropout = nn.Dropout(dropout)

        self.ffn = Transition(c_s, n=2)

    def forward(self, x, cond, z, x_mask):
        x_mask = x_mask.bool()

        _x = self.ln_s(x, cond)
        _z = self.ln_z(z)

        q = self.lin_q(_x)
        kv = self.lin_kv(_x)
        b = self.lin_b(_z)
        k, v = kv.split(self.h_head * self.no_heads, dim=-1)

        if evoformer_supported:
            # take advantage of evoformer MSA row attention kernel for efficient pair bias
            # by processing an "MSA" of one sequence
            q = q.view(*q.shape[:2], self.no_heads, self.h_head)[:, None]
            k = k.view(*k.shape[:2], self.no_heads, self.h_head)[:, None]
            v = v.view(*v.shape[:2], self.no_heads, self.h_head)[:, None]
            res_mask = x_mask[..., None, None, None, :]
            pair_bias = permute_final_dims(b[:, None], (2, 0, 1))
            # print(q.shape, k.shape, v.shape, res_mask.shape, pair_bias.shape)
            # DeepSpeed attn. kernel requires inputs to be type bf16 or fp16
            # Cast to bf16 so kernel can be used during inference
            biases = [res_mask.to(torch.bfloat16), pair_bias]
            orig_dtype = q.dtype
            if orig_dtype not in [torch.bfloat16, torch.float16]:
                o = DS4Sci_EvoformerAttention(q.to(dtype=torch.bfloat16),
                                            k.to(dtype=torch.bfloat16),
                                            v.to(dtype=torch.bfloat16),
                                            [b.to(dtype=torch.bfloat16) for b in biases])
                o = o.to(dtype=orig_dtype)
            else:
                o = DS4Sci_EvoformerAttention(q, k, v, biases)
            update = o.reshape(q.shape)
            # print(update.shape)
            # update = _deepspeed_evo_attn(q, k, v, [res_mask, pair_bias])
            update = update.squeeze(1)
            update = update.flatten(-2, -1)
        else:
            b += self.inf * (x_mask[..., None, :, None].float() - 1)
            b += self.inf * (x_mask[..., None, None].float() - 1)

            q = q.view(*q.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)
            k = k.view(*k.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)
            v = v.view(*v.shape[:2], self.no_heads, self.h_head).transpose(-2, -3)
            # print(q.shape, k.shape, v.shape, b.shape)
            update = F.scaled_dot_product_attention(
                q, k, v, attn_mask=permute_final_dims(b, (2, 0, 1))
            )
            update = update.transpose(-2, -3).flatten(-2, -1)
        out_gate = self.out_gate(_x)
        cond_gate = self.cond_gate(cond)
        update = self.lin_out(update * torch.sigmoid(out_gate))
        update = update * torch.sigmoid(cond_gate)
        x = x + self.dropout(update)

        x = x + self.ffn(x) * x_mask[..., None]

        return x


class ConditionedTransformerPairBias(nn.Module):
    def __init__(self,
                 c_s,
                 c_cond,
                 c_z,
                 no_heads,
                 n_layers,
                 dropout=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConditionedTransformerPairBiasLayer(
                    c_s,
                    c_cond,
                    c_z,
                    no_heads=no_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, cond, z, x_mask):
        for layer in self.layers:
            x = layer(x, cond, z, x_mask)
        return x



