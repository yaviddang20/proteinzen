import torch
import math
from torch import nn
from typing import Optional, Callable, List, Sequence, Union
import torch.nn.functional as F
import importlib

from boltz.data import const

from proteinzen.data.datasets.featurize.common import _rbf
from proteinzen.model.modules.openfold.layers import (
    Linear, flatten_final_dims, permute_final_dims, _deepspeed_evo_attn, DropoutRowwise, DropoutColumnwise,
    TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing
)
from proteinzen.model.modules.openfold.layers_v2 import ConditionedTransition, LayerNorm
from proteinzen.utils.openfold import rigid_utils as ru

from cuequivariance_torch import triangle_attention

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed

try:
    cuda_major, cuda_minor = torch.cuda.get_device_capability(device=None)
    if cuda_major >= 7:
        evoformer_supported = False #True
    else:
        evoformer_supported = False
except RuntimeError:
    evoformer_supported = False
print(f"using deepspeed kernels: {evoformer_supported}")

try:
    cuda_major, cuda_minor = torch.cuda.get_device_capability(device=None)
    if cuda_major >= 8:
        cuet_supported = True
    else:
        cuet_supported = False
except RuntimeError:
    cuet_supported = False
print(f"using cuet kernels: {cuet_supported}")


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
        Softmax, but without automatic casting to fp32 when the input is of
        type bfloat16
    """
    d = t.dtype
    deepspeed_is_initialized = (
        deepspeed_is_installed and
        deepspeed.comm.comm.is_initialized()
    )
    if d is torch.bfloat16 and not deepspeed_is_initialized:
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s


@torch.jit.script
def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]) -> torch.Tensor:
    # print(query.shape, key.shape, value.shape, [b.shape for b in biases])
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = softmax_no_cast(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a

def swish(x):
    return x * torch.sigmoid(x)

class SwishTransition(nn.Module):
    def __init__(self, c, dilation=4):
        super().__init__()
        self.ln = LayerNorm(c)
        self.lin_a = Linear(c, c*dilation, bias=False)
        self.lin_b = Linear(c, c*dilation, bias=False)
        self.lin_out = Linear(c*dilation, c, bias=False)

    def forward(self, x):
        x = self.ln(x)
        a = self.lin_a(x)
        b = self.lin_b(x)
        out = self.lin_out(swish(a) * b)
        return out


class TriangleAttentionCore(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """
    def __init__(
        self,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        starting=True,
        use_qk_norm=False,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super().__init__()

        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating
        self.starting = starting
        self.use_qk_norm = use_qk_norm

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(
            self.c_z, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_k = Linear(
            self.c_z, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_v = Linear(
            self.c_z, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_z, init="final",
            bias=False
        )

        if self.use_qk_norm:
            self.q_norm = LayerNorm(self.c_hidden)
            self.k_norm = LayerNorm(self.c_hidden)

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_z, self.c_hidden * self.no_heads, init="gating",
                bias=False
            )

        self.sigmoid = nn.Sigmoid()

        self.ln = LayerNorm(c_z)

    def _prep_qkv(self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        apply_scale: bool = True
    ):

        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self,
        o: torch.Tensor,
        q_x: torch.Tensor
    ) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        z: torch.Tensor,
        mask_bias: torch.Tensor,
        edge_bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory-efficient attention kernel.
                If none of the "use_<...>" flags are True, a stock PyTorch
                implementation is used instead
            lma_q_chunk_size:
                Query chunk size (for LMA)
            lma_kv_chunk_size:
                Key/Value chunk size (for LMA)
        Returns
            [*, Q, C_q] attention update
        """
        z = self.ln(z)

        if not self.starting:
            z = z.transpose(-2, -3)
            mask_bias = mask_bias.transpose(-4, -1)
            edge_bias = edge_bias.transpose(-1, -2)
        biases = [mask_bias, edge_bias]

        # DeepSpeed attention kernel applies scaling internally
        q, k, v = self._prep_qkv(z, z,
                                 apply_scale=False)

        global evoformer_supported
        global cuet_supported
        if cuet_supported:
            try:
                o = triangle_attention(q, k, v, bias=edge_bias, mask=mask_bias)
                o = o.transpose(-2, -3)
            except RuntimeError:
                print("wasn't able to run triangle_attention from cuet, switching it off")
                cuet_supported = False
                attn_mask = 0
                for b in biases:
                    attn_mask = attn_mask + b
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
                # o = _attention(q, k, v, biases)
                o = o.transpose(-2, -3)
        elif evoformer_supported:
            try:
                o = _deepspeed_evo_attn(q, k, v, biases)
            except RuntimeError:
                evoformer_supported = False
                attn_mask = 0
                for b in biases:
                    attn_mask = attn_mask + b
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
                # o = _attention(q, k, v, biases)
                o = o.transpose(-2, -3)
        else:
            attn_mask = 0
            for b in biases:
                attn_mask = attn_mask + b
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            o = o.transpose(-2, -3)

        o = self._wrap_up(o, z)

        if not self.starting:
            o = o.transpose(-2, -3)

        return o


# inspired by Pallatom
class PairUpdate(nn.Module):
    def __init__(
            self,
            c_z,
            c_hidden,
            no_heads=4,
            num_rbf=39,
            dropout=0.25,
            rbf_min=3.25,
            rbf_max=50.75,
            inf=1e8,
            include_rel_quat=False
        ):
        super().__init__()
        self.no_heads = no_heads
        self.c_hidden = c_hidden
        self.num_rbf = num_rbf
        self.NM_TO_ANG_SCALE = 10.0
        self.D_min = rbf_min / self.NM_TO_ANG_SCALE
        self.D_max = rbf_max / self.NM_TO_ANG_SCALE
        self.inf = inf
        self.include_rel_quat = include_rel_quat

        if include_rel_quat:
            self.emb_rbf = nn.Sequential(
                Linear(num_rbf + 4, c_hidden, bias=False),
                LayerNorm(c_hidden),
                Linear(c_hidden, no_heads, bias=False)
            )
        else:
            self.emb_rbf = nn.Sequential(
                Linear(num_rbf, c_hidden, bias=False),
                LayerNorm(c_hidden),
                Linear(c_hidden, no_heads, bias=False)
            )
        self.ln_third_edge = LayerNorm(c_z)
        self.emb_third_edge = Linear(c_z, no_heads, bias=False)

        self.trig_attn_start = TriangleAttentionCore(c_z, self.c_hidden, self.no_heads, starting=True)
        self.trig_bias_start = Linear(c_hidden, no_heads, bias=False)
        self.trig_attn_end = TriangleAttentionCore(c_z, self.c_hidden, self.no_heads, starting=False)
        self.trig_bias_end = Linear(c_hidden, no_heads, bias=False)
        self.transition = SwishTransition(c_z)

        self.dropout_row_layer = DropoutRowwise(dropout)
        self.dropout_col_layer = DropoutColumnwise(dropout)
        self.layer_norm_start = LayerNorm(c_z)
        self.layer_norm_end = LayerNorm(c_z)

    def forward(self, edge_embed, rigids, edge_mask):
        # get pair bias from rbf of distance
        coords = rigids.get_trans()
        distances = torch.cdist(coords, coords)

        if self.include_rel_quat:
            rots = rigids.get_rots()
            rel_quats = rots[..., None, :].compose_q(rots[..., None].invert())
            # [B, I, J, H]
            dist_bias = self.emb_rbf(
                torch.cat([
                    _rbf(distances, D_min=self.D_min, D_max=self.D_max, D_count=self.num_rbf, device=distances.device),
                    rel_quats.get_quats()
                ], dim=-1)
            )

        else:
            # [B, I, J, H]
            dist_bias = self.emb_rbf(
                _rbf(distances, D_min=self.D_min, D_max=self.D_max, D_count=self.num_rbf, device=distances.device)
            )
        # [B, I, 1, 1, J, H]
        mask_bias = (edge_mask[..., :, None, None, :].float() - 1) * self.inf

        third_edge = self.emb_third_edge(self.ln_third_edge(edge_embed))
        edge_bias = dist_bias + third_edge

        edge_bias = edge_bias.unsqueeze(-4)

        z = edge_embed
        z = z + self.trig_attn_start(
            z,
            edge_bias=permute_final_dims(edge_bias, (2, 0, 1)),
            mask_bias=mask_bias
        )
        z = z + self.trig_attn_end(
            z,
            edge_bias=permute_final_dims(edge_bias, (2, 0, 1)),
            mask_bias=mask_bias
        )
        z = z + self.transition(z)

        return z


# inspired by Pallatom
class ConditionedPairUpdate(nn.Module):
    def __init__(
            self,
            c_s,
            c_z,
            c_hidden,
            c_cond=None,
            no_heads=4,
            num_rbf=39,
            dropout=0.25,
            rbf_min=3.25,
            rbf_max=50.75,
            inf=1e8,
            include_rel_quat=False,
            use_qk_norm=False
        ):
        super().__init__()
        self.no_heads = no_heads
        self.c_hidden = c_hidden
        self.num_rbf = num_rbf
        self.NM_TO_ANG_SCALE = 10.0
        self.D_min = rbf_min / self.NM_TO_ANG_SCALE
        self.D_max = rbf_max / self.NM_TO_ANG_SCALE
        self.inf = inf
        self.include_rel_quat = include_rel_quat
        if c_cond is None:
            self.c_cond = c_s // 4

        if include_rel_quat:
            self.emb_rbf = nn.Sequential(
                Linear(num_rbf + 4, c_hidden, bias=False),
                LayerNorm(c_hidden),
                Linear(c_hidden, no_heads, bias=False)
            )
        else:
            self.emb_rbf = nn.Sequential(
                Linear(num_rbf, c_hidden, bias=False),
                LayerNorm(c_hidden),
                Linear(c_hidden, no_heads, bias=False)
            )
        self.ln_third_edge = LayerNorm(c_z)
        self.emb_third_edge = Linear(c_z, no_heads, bias=False)

        self.trig_attn_start = TriangleAttentionCore(c_z, self.c_hidden, self.no_heads, starting=True, use_qk_norm=use_qk_norm)
        self.trig_bias_start = Linear(c_hidden, no_heads, bias=False)
        self.trig_attn_end = TriangleAttentionCore(c_z, self.c_hidden, self.no_heads, starting=False, use_qk_norm=use_qk_norm)
        self.trig_bias_end = Linear(c_hidden, no_heads, bias=False)
        self.cond_proj = nn.Sequential(
            LayerNorm(c_s),
            Linear(c_s, self.c_cond, bias=False)
        )
        self.transition = ConditionedTransition(c_z, c_cond=self.c_cond*2)

        self.dropout_row_layer = DropoutRowwise(dropout)
        self.dropout_col_layer = DropoutColumnwise(dropout)
        self.layer_norm_start = LayerNorm(c_z)
        self.layer_norm_end = LayerNorm(c_z)

    def forward(self, node_embed, edge_embed, rigids, edge_mask):
        # get pair bias from rbf of distance
        coords = rigids.get_trans()
        distances = torch.cdist(coords, coords)
        if self.include_rel_quat:
            rots = rigids.get_rots()
            rel_quats = rots[..., None].invert().compose_q(rots[..., None, :])
            # [B, I, J, H]
            dist_bias = self.emb_rbf(
                torch.cat([
                    _rbf(distances, D_min=self.D_min, D_max=self.D_max, D_count=self.num_rbf, device=distances.device),
                    rel_quats.get_quats()
                ], dim=-1)
            )

        else:
            # [B, I, J, H]
            dist_bias = self.emb_rbf(
                _rbf(distances, D_min=self.D_min, D_max=self.D_max, D_count=self.num_rbf, device=distances.device)
            )
        # [B, I, 1, 1, J, H]
        # mask_bias = (edge_mask[..., :, None, None, :].float() - 1) * self.inf
        global cuet_supported
        if cuet_supported:
            mask_bias = edge_mask[..., :, None, None, :]
        else:
            mask_bias = (edge_mask[..., :, None, None, :].float() - 1) * self.inf

        third_edge = self.emb_third_edge(self.ln_third_edge(edge_embed))
        edge_bias = dist_bias + third_edge

        edge_bias = edge_bias.unsqueeze(-4)

        z = edge_embed
        z = z + self.trig_attn_start(
            z,
            edge_bias=permute_final_dims(edge_bias, (2, 0, 1)),
            mask_bias=mask_bias
        )
        z = z + self.trig_attn_end(
            z,
            edge_bias=permute_final_dims(edge_bias, (2, 0, 1)),
            mask_bias=mask_bias
        )
        # B x N x c_cond
        cond = self.cond_proj(node_embed)
        num_res = node_embed.shape[-2]
        cond = torch.cat([
            torch.tile(cond[..., None, :], (1, 1, num_res, 1)),
            torch.tile(cond[..., None, :, :], (1, num_res, 1, 1))
        ], dim=-1)
        z = z + self.transition(z, cond)

        return z

# inspired by Pallatom
class ConditionedPairUpdateV2(nn.Module):
    def __init__(
            self,
            c_s,
            c_z,
            c_hidden,
            c_cond=None,
            no_heads=4,
            num_rbf=39,
            dropout=0.25,
            rbf_min=3.25,
            rbf_max=50.75,
            inf=1e8,
        ):
        super().__init__()
        self.no_heads = no_heads
        self.c_hidden = c_hidden
        self.num_rbf = num_rbf
        self.NM_TO_ANG_SCALE = 10.0
        self.D_min = rbf_min
        self.D_max = rbf_max
        self.inf = inf
        if c_cond is None:
            self.c_cond = c_s // 4

        self.emb_rbf = nn.Sequential(
            Linear(num_rbf, c_hidden, bias=False),
            LayerNorm(c_hidden),
            Linear(c_hidden, no_heads, bias=False)
        )
        self.ln_third_edge = LayerNorm(c_z)
        self.emb_third_edge = Linear(c_z, no_heads, bias=False)

        self.trig_attn_start = TriangleAttentionCore(c_z, self.c_hidden, self.no_heads, starting=True, use_qk_norm=True)
        self.trig_bias_start = Linear(c_hidden, no_heads, bias=False)
        self.trig_attn_end = TriangleAttentionCore(c_z, self.c_hidden, self.no_heads, starting=False, use_qk_norm=True)
        self.trig_bias_end = Linear(c_hidden, no_heads, bias=False)
        self.cond_proj = nn.Sequential(
            LayerNorm(c_s),
            Linear(c_s, self.c_cond, bias=False)
        )
        self.transition = ConditionedTransition(c_z, c_cond=self.c_cond*2)

        self.dropout_row_layer = DropoutRowwise(dropout)
        self.dropout_col_layer = DropoutColumnwise(dropout)
        self.layer_norm_start = LayerNorm(c_z)
        self.layer_norm_end = LayerNorm(c_z)

    def forward(self, node_embed, edge_embed, coords, edge_mask):
        # get pair bias from rbf of distance
        distances = torch.cdist(coords, coords)
        # [B, I, J, H]
        dist_bias = self.emb_rbf(
            _rbf(distances, D_min=self.D_min, D_max=self.D_max, D_count=self.num_rbf, device=distances.device)
        )
        # [B, I, 1, 1, J, H]
        mask_bias = (edge_mask[..., :, None, None, :].float() - 1) * self.inf

        third_edge = self.emb_third_edge(self.ln_third_edge(edge_embed))
        edge_bias = dist_bias + third_edge

        edge_bias = edge_bias.unsqueeze(-4)

        z = edge_embed
        z = z + self.trig_attn_start(
            z,
            edge_bias=permute_final_dims(edge_bias, (2, 0, 1)),
            mask_bias=mask_bias
        )
        z = z + self.trig_attn_end(
            z,
            edge_bias=permute_final_dims(edge_bias, (2, 0, 1)),
            mask_bias=mask_bias
        )
        # B x N x c_cond
        cond = self.cond_proj(node_embed)
        num_res = node_embed.shape[-2]
        cond = torch.cat([
            torch.tile(cond[..., None, :], (1, 1, num_res, 1)),
            torch.tile(cond[..., None, :, :], (1, num_res, 1, 1))
        ], dim=-1)
        z = z + self.transition(z, cond)

        return z


def relpos(node_idx, clip=32):
    relpos_idx = torch.clip(
        node_idx[..., None]  - node_idx[..., None, :],
        min=-clip,
        max=clip
    )
    relpos_idx = relpos_idx + clip
    return F.one_hot(relpos_idx, num_classes=2*clip+1).float()


def calc_distogram(dists_2d, min_bin, max_bin, num_bins):
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=dists_2d.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    # print(dists_2d.shape, lower.shape, upper.shape)
    dgram = ((dists_2d[..., None] > lower) * (dists_2d[..., None] < upper)).type(dists_2d.dtype)
    return dgram


class PairEmbedder(nn.Module):
    def __init__(self,
                 c_z,
                 c_latent,
                 c_hidden,
                 num_rbf=39,
                 rbf_min=3.25,
                 rbf_max=50.75,
                 no_heads=4,
                 no_blocks=2,
                 relpos_clip=32,
                 dropout=0.25,
                 inf=1e9,
                 latent_pairs=False):
        super().__init__()

        self.c_z = c_z
        self.c_hidden = c_hidden
        self.c_latent = c_latent
        self.D_min = rbf_min
        self.D_max = rbf_max
        self.num_rbf = num_rbf
        self.relpos_clip = relpos_clip
        self.no_blocks = no_blocks
        self.latent_pairs = latent_pairs
        self.inf = inf

        self.c_a = (
            num_rbf
            + 3  # unit vecs
            + 4  # rel quat
        )

        # self.ln_relpos = LayerNorm(2*relpos_clip+1)
        self.lin_z_ij = Linear(2*relpos_clip+1, c_hidden)

        self.ln_latent = LayerNorm(c_latent if latent_pairs else 2*c_latent)
        self.lin_latent = Linear(
            c_latent if latent_pairs else 2*c_latent,
            c_hidden, bias=False)

        self.lin_a_ij = Linear(2 * self.c_a, c_hidden, bias=False)

        c_head = c_hidden // no_heads

        self.trunk = nn.ModuleDict()
        self.dropout_row_layer = DropoutRowwise(dropout)
        self.dropout_col_layer = DropoutColumnwise(dropout)
        for i in range(no_blocks):
            self.trunk[f'mult_out_{i}'] = TriangleMultiplicationOutgoing(c_hidden, c_hidden)
            self.trunk[f'mult_in_{i}'] = TriangleMultiplicationIncoming(c_hidden, c_hidden)
            self.trunk[f'edge_bias_{i}'] = nn.Sequential(
                LayerNorm(c_hidden),
                Linear(c_hidden, no_heads, bias=False)
            )
            self.trunk[f'attn_start_{i}'] = TriangleAttentionCore(c_hidden, c_head, no_heads, starting=True)
            self.trunk[f'attn_end_{i}'] = TriangleAttentionCore(c_hidden, c_head, no_heads, starting=False)
            self.trunk[f'transition_{i}'] = SwishTransition(c_hidden)

        self.final_layer = nn.Sequential(
            LayerNorm(c_hidden),
            nn.ReLU(),
            Linear(c_hidden, c_z, bias=False)
        )

    def _featurize_rigids(self, rigids, distogram=False):
        X_ca = rigids.get_trans()
        quats = rigids.get_rots().get_quats()
        quat_i = quats[..., None, :]
        quat_j = quats[..., None, :, :]
        quat_rel = ru.quat_multiply(
            ru.invert_quat(quat_i),
            quat_j
        )
        edge_dist_vec = X_ca[..., None, :] - X_ca[..., None, : ,:]
        unit_edge_dist_vecs = F.normalize(edge_dist_vec, dim=-1)

        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
        if distogram:
            dist_features = calc_distogram(edge_dist, min_bin=self.D_min, max_bin=self.D_max, num_bins=self.num_rbf)
        else:
            dist_features = _rbf(edge_dist, D_min=self.D_min, D_max=self.D_max, D_count=self.num_rbf, device=edge_dist.device)

        return torch.cat([quat_rel, unit_edge_dist_vecs, dist_features], dim=-1)


    def forward(self, latent_features, rigids, node_mask, sc_rigids=None):
        edge_mask = node_mask[..., None] & node_mask[..., None, :]

        node_idx = torch.arange(rigids.shape[-1], device=rigids.device)[None].expand(
            rigids.shape[0], -1
        )
        relpos_feats = relpos(node_idx, clip=self.relpos_clip)
        z = self.lin_z_ij(relpos_feats)
        a_current = self._featurize_rigids(rigids, distogram=False)
        if sc_rigids is not None:
            a_sc = self._featurize_rigids(sc_rigids, distogram=True)
        else:
            a_sc = torch.zeros_like(a_current)

        z = z + self.lin_a_ij(torch.cat([a_current, a_sc], dim=-1))

        if self.latent_pairs:
            latent_pairs = latent_features
        else:
            n_res = latent_features.shape[-2]
            latent_pairs = torch.cat([
                torch.tile(latent_features[..., None, :], (1, 1, n_res, 1)),
                torch.tile(latent_features[..., None, :, :], (1, n_res, 1, 1)),
            ], dim=-1)
        z = z + self.lin_latent(self.ln_latent(latent_pairs))
        z = z * edge_mask[..., None]

        mask_bias = (edge_mask[..., :, None, None, :].float() - 1) * self.inf

        for i in range(self.no_blocks):
            z = z + self.dropout_row_layer(
                self.trunk[f'mult_out_{i}'](z, edge_mask)
            )
            z = z + self.dropout_row_layer(
                self.trunk[f'mult_in_{i}'](z, edge_mask)
            )
            edge_bias = self.trunk[f'edge_bias_{i}'](z)
            edge_bias = permute_final_dims(edge_bias.unsqueeze(-4), (2, 0, 1))
            z = z + self.dropout_row_layer(
                self.trunk[f'attn_start_{i}'](z, mask_bias=mask_bias, edge_bias=edge_bias)
            )
            z = z + self.dropout_col_layer(
                self.trunk[f'attn_end_{i}'](z, mask_bias=mask_bias, edge_bias=edge_bias)
            )
            z = z + self.trunk[f'transition_{i}'](z)

        return self.final_layer(z) * edge_mask[..., None]


class Atom14PairEmbedder(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_rbf=39,
                 rbf_min=3.25,
                 rbf_max=50.75,
                 no_heads=4,
                 no_blocks=2,
                 relpos_clip=32,
                 dropout=0.25,
                 inf=1e9):
        super().__init__()

        self.c_z = c_z
        self.c_hidden = c_hidden
        self.D_min = rbf_min
        self.D_max = rbf_max
        self.num_rbf = num_rbf
        self.relpos_clip = relpos_clip
        self.no_blocks = no_blocks
        self.inf = inf

        self.c_a = (
            num_rbf
            + 3  # unit vecs
            + 4  # rel quat
        )

        self.ln_node = LayerNorm(c_s*2)
        self.lin_node = Linear(
            c_s*2,
            c_hidden, bias=False)

        # self.ln_relpos = LayerNorm(2*relpos_clip+1)
        self.lin_z_ij = Linear(2*relpos_clip+1, c_hidden)

        self.lin_a_ij = Linear(2 * self.c_a, c_hidden, bias=False)

        c_head = c_hidden // no_heads

        self.trunk = nn.ModuleDict()
        self.dropout_row_layer = DropoutRowwise(dropout)
        self.dropout_col_layer = DropoutColumnwise(dropout)
        for i in range(no_blocks):
            self.trunk[f'mult_out_{i}'] = TriangleMultiplicationOutgoing(c_hidden, c_hidden)
            self.trunk[f'mult_in_{i}'] = TriangleMultiplicationIncoming(c_hidden, c_hidden)
            self.trunk[f'edge_bias_{i}'] = nn.Sequential(
                LayerNorm(c_hidden),
                Linear(c_hidden, no_heads, bias=False)
            )
            self.trunk[f'attn_start_{i}'] = TriangleAttentionCore(c_hidden, c_head, no_heads, starting=True)
            self.trunk[f'attn_end_{i}'] = TriangleAttentionCore(c_hidden, c_head, no_heads, starting=False)
            self.trunk[f'transition_{i}'] = SwishTransition(c_hidden)

        self.final_layer = nn.Sequential(
            LayerNorm(c_hidden),
            nn.ReLU(),
            Linear(c_hidden, c_z, bias=False)
        )

    def _featurize_rigids(self, rigids, distogram=False):
        X_ca = rigids.get_trans()
        quats = rigids.get_rots().get_quats()
        quat_i = quats[..., None, :]
        quat_j = quats[..., None, :, :]
        quat_rel = ru.quat_multiply(
            ru.invert_quat(quat_i),
            quat_j
        )
        edge_dist_vec = X_ca[..., None, :] - X_ca[..., None, : ,:]
        unit_edge_dist_vecs = F.normalize(edge_dist_vec, dim=-1)

        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
        if distogram:
            dist_features = calc_distogram(edge_dist, min_bin=self.D_min, max_bin=self.D_max, num_bins=self.num_rbf)
        else:
            dist_features = _rbf(edge_dist, D_min=self.D_min, D_max=self.D_max, D_count=self.num_rbf, device=edge_dist.device)

        return torch.cat([quat_rel, unit_edge_dist_vecs, dist_features], dim=-1)


    def forward(self, node_embed, rigids, node_mask, sc_rigids=None):
        edge_mask = node_mask[..., None] & node_mask[..., None, :]

        node_idx = torch.arange(rigids.shape[-1], device=rigids.device)[None].expand(
            rigids.shape[0], -1
        )
        relpos_feats = relpos(node_idx, clip=self.relpos_clip)
        z = self.lin_z_ij(relpos_feats)
        a_current = self._featurize_rigids(rigids, distogram=False)
        if sc_rigids is not None:
            a_sc = self._featurize_rigids(sc_rigids, distogram=True)
        else:
            a_sc = torch.zeros_like(a_current)

        z = z + self.lin_a_ij(torch.cat([a_current, a_sc], dim=-1))

        n_res = node_embed.shape[-2]
        node_pairs = torch.cat([
            torch.tile(node_embed[..., None, :], (1, 1, n_res, 1)),
            torch.tile(node_embed[..., None, :, :], (1, n_res, 1, 1)),
        ], dim=-1)
        z = z + self.lin_node(F.relu(self.ln_node(node_pairs)))
        z = z * edge_mask[..., None]

        mask_bias = (edge_mask[..., :, None, None, :].float() - 1) * self.inf

        for i in range(self.no_blocks):
            z = z + self.dropout_row_layer(
                self.trunk[f'mult_out_{i}'](z, edge_mask)
            )
            z = z + self.dropout_row_layer(
                self.trunk[f'mult_in_{i}'](z, edge_mask)
            )
            edge_bias = self.trunk[f'edge_bias_{i}'](z)
            edge_bias = permute_final_dims(edge_bias.unsqueeze(-4), (2, 0, 1))
            z = z + self.dropout_row_layer(
                self.trunk[f'attn_start_{i}'](z, mask_bias=mask_bias, edge_bias=edge_bias)
            )
            z = z + self.dropout_col_layer(
                self.trunk[f'attn_end_{i}'](z, mask_bias=mask_bias, edge_bias=edge_bias)
            )
            z = z + self.trunk[f'transition_{i}'](z)

        return self.final_layer(z) * edge_mask[..., None]


class MultiRigidPairEmbedder(nn.Module):
    def __init__(self,
                 c_z=128,
                 c_hidden=64,
                 num_rbf=39,
                 rbf_min=3.25,
                 rbf_max=50.75,
                 no_heads=4,
                 no_blocks=2,
                 relpos_clip=32,
                 dropout=0.25,
                 inf=1e9,
                 num_bond_types=len(const.bond_types) + 1,
                 use_qk_norm=False
    ):
        super().__init__()

        self.c_z = c_z
        self.c_hidden = c_hidden
        self.D_min = rbf_min
        self.D_max = rbf_max
        self.num_rbf = num_rbf
        self.relpos_clip = relpos_clip
        self.no_blocks = no_blocks
        self.inf = inf

        self.c_a = (
            num_rbf
            + 3  # unit vecs
            + 4  # rel quat
        )

        # self.ln_relpos = LayerNorm(2*relpos_clip+1)
        self.lin_z_ij = Linear(2*relpos_clip+1, c_hidden)
        self.lin_token_bonds = nn.Embedding(num_bond_types, c_hidden)

        self.lin_a_ij = Linear(2 * self.c_a, c_hidden, bias=False)

        c_head = c_hidden // no_heads

        self.trunk = nn.ModuleDict()
        self.dropout_row_layer = DropoutRowwise(dropout)
        self.dropout_col_layer = DropoutColumnwise(dropout)
        for i in range(no_blocks):
            self.trunk[f'mult_out_{i}'] = TriangleMultiplicationOutgoing(c_hidden, c_hidden)
            self.trunk[f'mult_in_{i}'] = TriangleMultiplicationIncoming(c_hidden, c_hidden)
            self.trunk[f'edge_bias_{i}'] = nn.Sequential(
                LayerNorm(c_hidden),
                Linear(c_hidden, no_heads, bias=False)
            )
            self.trunk[f'attn_start_{i}'] = TriangleAttentionCore(c_hidden, c_head, no_heads, starting=True, use_qk_norm=use_qk_norm)
            self.trunk[f'attn_end_{i}'] = TriangleAttentionCore(c_hidden, c_head, no_heads, starting=False, use_qk_norm=use_qk_norm)
            self.trunk[f'transition_{i}'] = SwishTransition(c_hidden)

        self.final_layer = nn.Sequential(
            LayerNorm(c_hidden),
            nn.ReLU(),
            Linear(c_hidden, c_z, bias=False)
        )

    def _featurize_rigids(self, rigids, distogram=False):
        X_ca = rigids.get_trans()
        quats = rigids.get_rots().get_quats()
        quat_i = quats[..., None, :]
        quat_j = quats[..., None, :, :]
        quat_rel = ru.quat_multiply(
            ru.invert_quat(quat_i),
            quat_j
        )
        edge_dist_vec = X_ca[..., None, :] - X_ca[..., None, : ,:]
        unit_edge_dist_vecs = F.normalize(edge_dist_vec, dim=-1)

        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
        if distogram:
            dist_features = calc_distogram(edge_dist, min_bin=self.D_min, max_bin=self.D_max, num_bins=self.num_rbf)
        else:
            dist_features = _rbf(edge_dist, D_min=self.D_min, D_max=self.D_max, D_count=self.num_rbf, device=edge_dist.device)

        return torch.cat([quat_rel, unit_edge_dist_vecs, dist_features], dim=-1)


    def forward(self,
                rigids,
                node_mask,
                seq_idx,
                chain_idx,
                token_is_unindexed_mask,
                token_bonds,
                sc_rigids=None
    ):
        edge_mask = node_mask[..., None] & node_mask[..., None, :]
        same_chain_mask = (chain_idx[..., None] == chain_idx[..., None, :])
        pair_is_unindexed_mask = token_is_unindexed_mask[..., None] | token_is_unindexed_mask[..., None, :]

        relpos_feats = relpos(seq_idx, clip=self.relpos_clip)
        relpos_feats = relpos_feats * same_chain_mask[..., None]
        relpos_feats = relpos_feats * ~pair_is_unindexed_mask[..., None]
        # relpos_feats[~same_chain_mask][..., -1] += 1

        z = self.lin_z_ij(relpos_feats)
        a_current = self._featurize_rigids(rigids, distogram=False)
        if sc_rigids is not None:
            a_sc = self._featurize_rigids(sc_rigids, distogram=True)
        else:
            a_sc = torch.zeros_like(a_current)

        z = z + self.lin_a_ij(torch.cat([a_current, a_sc], dim=-1))
        z += self.lin_token_bonds(token_bonds)
        z = z * edge_mask[..., None]

        global cuet_supported
        if cuet_supported:
            mask_bias = edge_mask[..., :, None, None, :]
        else:
            mask_bias = (edge_mask[..., :, None, None, :].float() - 1) * self.inf

        for i in range(self.no_blocks):
            z = z + self.dropout_row_layer(
                self.trunk[f'mult_out_{i}'](z, edge_mask)
            )
            z = z + self.dropout_row_layer(
                self.trunk[f'mult_in_{i}'](z, edge_mask)
            )
            edge_bias = self.trunk[f'edge_bias_{i}'](z)
            edge_bias = permute_final_dims(edge_bias.unsqueeze(-4), (2, 0, 1))
            z = z + self.dropout_row_layer(
                self.trunk[f'attn_start_{i}'](z, mask_bias=mask_bias, edge_bias=edge_bias)
            )
            z = z + self.dropout_col_layer(
                self.trunk[f'attn_end_{i}'](z, mask_bias=mask_bias, edge_bias=edge_bias)
            )
            z = z + self.trunk[f'transition_{i}'](z)

        return self.final_layer(z) * edge_mask[..., None]


class MultiRigidPairEmbedderV2(nn.Module):
    def __init__(self,
                 c_z=128,
                 c_hidden=64,
                 num_rbf=39,
                 rbf_min=3.25,
                 rbf_max=50.75,
                 no_heads=4,
                 no_blocks=2,
                 relpos_clip=32,
                 dropout=0.25,
                 inf=1e9,
                 use_qk_norm=False
    ):
        super().__init__()

        self.c_z = c_z
        self.c_hidden = c_hidden
        self.D_min = rbf_min
        self.D_max = rbf_max
        self.num_rbf = num_rbf
        self.relpos_clip = relpos_clip
        self.no_blocks = no_blocks
        self.inf = inf

        self.c_a = (
            num_rbf
            + 3  # unit vecs
            + 4  # rel quat
        )

        # self.ln_relpos = LayerNorm(2*relpos_clip+1)
        self.lin_z_ij = Linear(2*relpos_clip+1, c_hidden)

        self.lin_a_ij = Linear(2 * self.c_a, c_hidden, bias=False)

        c_head = c_hidden // no_heads

        self.trunk = nn.ModuleDict()
        self.dropout_row_layer = DropoutRowwise(dropout)
        self.dropout_col_layer = DropoutColumnwise(dropout)
        for i in range(no_blocks):
            self.trunk[f'mult_out_{i}'] = TriangleMultiplicationOutgoing(c_hidden, c_hidden)
            self.trunk[f'mult_in_{i}'] = TriangleMultiplicationIncoming(c_hidden, c_hidden)
            self.trunk[f'edge_bias_{i}'] = nn.Sequential(
                LayerNorm(c_hidden),
                Linear(c_hidden, no_heads, bias=False)
            )
            self.trunk[f'attn_start_{i}'] = TriangleAttentionCore(c_hidden, c_head, no_heads, starting=True, use_qk_norm=use_qk_norm)
            self.trunk[f'attn_end_{i}'] = TriangleAttentionCore(c_hidden, c_head, no_heads, starting=False, use_qk_norm=use_qk_norm)
            self.trunk[f'transition_{i}'] = SwishTransition(c_hidden)

        self.final_layer = nn.Sequential(
            LayerNorm(c_hidden),
            nn.ReLU(),
            Linear(c_hidden, c_z, bias=False)
        )

    def _featurize_rigids(self, rigids, distogram=False):
        X_ca = rigids.get_trans()
        quats = rigids.get_rots().get_quats()
        quat_i = quats[..., None, :]
        quat_j = quats[..., None, :, :]
        quat_rel = ru.quat_multiply(
            ru.invert_quat(quat_i),
            quat_j
        )
        edge_dist_vec = X_ca[..., None, :] - X_ca[..., None, : ,:]
        unit_edge_dist_vecs = F.normalize(edge_dist_vec, dim=-1)

        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
        if distogram:
            dist_features = calc_distogram(edge_dist, min_bin=self.D_min, max_bin=self.D_max, num_bins=self.num_rbf)
        else:
            dist_features = _rbf(edge_dist, D_min=self.D_min, D_max=self.D_max, D_count=self.num_rbf, device=edge_dist.device)

        return torch.cat([quat_rel, unit_edge_dist_vecs, dist_features], dim=-1)


    def forward(self,
                rigids,
                node_mask,
                seq_idx,
                chain_idx,
                token_is_unindexed_mask,
                sc_rigids=None
    ):
        edge_mask = node_mask[..., None] & node_mask[..., None, :]
        same_chain_mask = (chain_idx[..., None] == chain_idx[..., None, :])
        pair_is_unindexed_mask = token_is_unindexed_mask[..., None] | token_is_unindexed_mask[..., None, :]

        relpos_feats = relpos(seq_idx, clip=self.relpos_clip)
        relpos_feats = relpos_feats * same_chain_mask[..., None]
        relpos_feats = relpos_feats * ~pair_is_unindexed_mask[..., None]
        # relpos_feats[~same_chain_mask][..., -1] += 1

        z = self.lin_z_ij(relpos_feats)
        a_current = self._featurize_rigids(rigids, distogram=False)
        if sc_rigids is not None:
            a_sc = self._featurize_rigids(sc_rigids, distogram=True)
        else:
            a_sc = torch.zeros_like(a_current)

        z = z + self.lin_a_ij(torch.cat([a_current, a_sc], dim=-1))
        z = z * edge_mask[..., None]

        global cuet_supported
        if cuet_supported:
            mask_bias = edge_mask[..., :, None, None, :]
        else:
            mask_bias = (edge_mask[..., :, None, None, :].float() - 1) * self.inf

        for i in range(self.no_blocks):
            z = z + self.dropout_row_layer(
                self.trunk[f'mult_out_{i}'](z, edge_mask)
            )
            z = z + self.dropout_row_layer(
                self.trunk[f'mult_in_{i}'](z, edge_mask)
            )
            edge_bias = self.trunk[f'edge_bias_{i}'](z)
            edge_bias = permute_final_dims(edge_bias.unsqueeze(-4), (2, 0, 1))
            z = z + self.dropout_row_layer(
                self.trunk[f'attn_start_{i}'](z, mask_bias=mask_bias, edge_bias=edge_bias)
            )
            z = z + self.dropout_col_layer(
                self.trunk[f'attn_end_{i}'](z, mask_bias=mask_bias, edge_bias=edge_bias)
            )
            z = z + self.trunk[f'transition_{i}'](z)

        return self.final_layer(z) * edge_mask[..., None]


class PairEmbedderV2(nn.Module):
    def __init__(self,
                 c_z=128,
                 c_hidden=64,
                 num_rbf=39,
                 rbf_min=3.25,
                 rbf_max=50.75,
                 no_heads=4,
                 no_blocks=2,
                 relpos_clip=32,
                 dropout=0.25,
                 inf=1e9):
        super().__init__()

        self.c_z = c_z
        self.c_hidden = c_hidden
        self.D_min = rbf_min
        self.D_max = rbf_max
        self.num_rbf = num_rbf
        self.relpos_clip = relpos_clip
        self.no_blocks = no_blocks
        self.inf = inf

        self.c_a = num_rbf

        # self.ln_relpos = LayerNorm(2*relpos_clip+1)
        self.lin_z_ij = Linear(2*relpos_clip+1, c_hidden)

        self.lin_a_ij = Linear(2 * self.c_a, c_hidden, bias=False)

        c_head = c_hidden // no_heads

        self.trunk = nn.ModuleDict()
        self.dropout_row_layer = DropoutRowwise(dropout)
        self.dropout_col_layer = DropoutColumnwise(dropout)
        for i in range(no_blocks):
            self.trunk[f'mult_out_{i}'] = TriangleMultiplicationOutgoing(c_hidden, c_hidden)
            self.trunk[f'mult_in_{i}'] = TriangleMultiplicationIncoming(c_hidden, c_hidden)
            self.trunk[f'edge_bias_{i}'] = nn.Sequential(
                LayerNorm(c_hidden),
                Linear(c_hidden, no_heads, bias=False)
            )
            self.trunk[f'attn_start_{i}'] = TriangleAttentionCore(c_hidden, c_head, no_heads, starting=True, use_qk_norm=True)
            self.trunk[f'attn_end_{i}'] = TriangleAttentionCore(c_hidden, c_head, no_heads, starting=False, use_qk_norm=True)
            self.trunk[f'transition_{i}'] = SwishTransition(c_hidden)

        self.final_layer = nn.Sequential(
            LayerNorm(c_hidden),
            nn.ReLU(),
            Linear(c_hidden, c_z, bias=False)
        )

    def _featurize_edges(self, X_ca, distogram=False):
        edge_dist_vec = X_ca[..., None, :] - X_ca[..., None, : ,:]
        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
        if distogram:
            dist_features = calc_distogram(edge_dist, min_bin=self.D_min, max_bin=self.D_max, num_bins=self.num_rbf)
        else:
            dist_features = _rbf(edge_dist, D_min=self.D_min, D_max=self.D_max, D_count=self.num_rbf, device=edge_dist.device)

        return dist_features


    def forward(self, X_ca, node_mask, sc_X_ca=None):
        edge_mask = node_mask[..., None] & node_mask[..., None, :]

        node_idx = torch.arange(X_ca.shape[-2], device=X_ca.device)[None].expand(
            X_ca.shape[0], -1
        )
        relpos_feats = relpos(node_idx, clip=self.relpos_clip)
        z = self.lin_z_ij(relpos_feats)
        a_current = self._featurize_edges(X_ca, distogram=False)
        if sc_X_ca is not None:
            a_sc = self._featurize_edges(sc_X_ca, distogram=True)
        else:
            a_sc = torch.zeros_like(a_current)

        z = z + self.lin_a_ij(torch.cat([a_current, a_sc], dim=-1))
        z = z * edge_mask[..., None]

        global cuet_supported
        if cuet_supported:
            mask_bias = edge_mask[..., :, None, None, :]
        else:
            mask_bias = (edge_mask[..., :, None, None, :].float() - 1) * self.inf

        for i in range(self.no_blocks):
            z = z + self.dropout_row_layer(
                self.trunk[f'mult_out_{i}'](z, edge_mask)
            )
            z = z + self.dropout_row_layer(
                self.trunk[f'mult_in_{i}'](z, edge_mask)
            )
            edge_bias = self.trunk[f'edge_bias_{i}'](z)
            edge_bias = permute_final_dims(edge_bias.unsqueeze(-4), (2, 0, 1))
            z = z + self.dropout_row_layer(
                self.trunk[f'attn_start_{i}'](z, mask_bias=mask_bias, edge_bias=edge_bias)
            )
            z = z + self.dropout_col_layer(
                self.trunk[f'attn_end_{i}'](z, mask_bias=mask_bias, edge_bias=edge_bias)
            )
            z = z + self.trunk[f'transition_{i}'](z)

        return self.final_layer(z) * edge_mask[..., None]

