import torch
import math
from torch import nn
from typing import Optional, Callable, List, Sequence, Union
import torch.nn.functional as F

from proteinzen.data.datasets.featurize.common import _rbf
from proteinzen.model.modules.openfold.layers import (
    Linear, flatten_final_dims, permute_final_dims, _deepspeed_evo_attn, DropoutRowwise, DropoutColumnwise,
    TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing
)
from proteinzen.utils.openfold import rigid_utils as ru


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

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_z, self.c_hidden * self.no_heads, init="gating",
                bias=False
            )

        self.sigmoid = nn.Sigmoid()

        self.ln = nn.LayerNorm(c_z)

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

        o = _deepspeed_evo_attn(q, k, v, biases)

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
            inf=1e8
        ):
        super().__init__()
        self.no_heads = no_heads
        self.c_hidden = c_hidden
        self.num_rbf = num_rbf
        self.NM_TO_ANG_SCALE = 10.0
        self.D_min = rbf_min / self.NM_TO_ANG_SCALE
        self.D_max = rbf_max / self.NM_TO_ANG_SCALE
        self.inf = inf

        self.emb_rbf = nn.Sequential(
            Linear(num_rbf, c_hidden, bias=False),
            nn.LayerNorm(c_hidden),
            Linear(c_hidden, no_heads, bias=False)
        )
        self.ln_third_edge = nn.LayerNorm(c_z)
        self.emb_third_edge = Linear(c_z, no_heads, bias=False)

        self.trig_attn_start = TriangleAttentionCore(c_z, self.c_hidden, self.no_heads, starting=True)
        self.trig_bias_start = Linear(c_hidden, no_heads, bias=False)
        self.trig_attn_end = TriangleAttentionCore(c_z, self.c_hidden, self.no_heads, starting=False)
        self.trig_bias_end = Linear(c_hidden, no_heads, bias=False)

        self.dropout_row_layer = DropoutRowwise(dropout)
        self.dropout_col_layer = DropoutColumnwise(dropout)
        self.layer_norm_start = nn.LayerNorm(c_z)
        self.layer_norm_end = nn.LayerNorm(c_z)

    def forward(self, edge_embed, rigids, edge_mask):
        # get pair bias from rbf of distance
        coords = rigids.get_trans()
        distances = torch.cdist(coords, coords)

        # [B, I, J, H]
        dist_bias = self.emb_rbf(
            _rbf(distances, D_min=self.D_min, D_max=self.D_max, D_count=self.num_rbf, device=distances.device)
        )
        # [B, I, 1, 1, J, H]
        mask_bias = (edge_mask[..., :, None, None, :] - 1) * self.inf

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

        return z


def relpos(node_idx, clip=32):
    relpos_idx = torch.clip(
        node_idx[..., None]  - node_idx[..., None, :],
        min=-clip,
        max=clip
    )
    return F.one_hot(relpos_idx, num_classes=2*clip+1)


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

        self.c_a = (
            num_rbf
            + 3  # unit vecs
            + 4  # rel quat
        )

        self.ln_relpos = nn.LayerNorm(2*relpos_clip+1)
        self.lin_z_ij = Linear(2*relpos_clip+1, c_hidden)

        self.ln_latent = nn.LayerNorm(c_latent)
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
            self.trunk[f'attn_start_{i}'] = TriangleAttentionCore(c_hidden, c_head, no_heads, starting=True)
            self.trunk[f'attn_end_{i}'] = TriangleAttentionCore(c_hidden, c_head, no_heads, starting=False)

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
        z = self.lin_z_ij(self.ln_relpos(relpos_feats))
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

        for i in range(self.no_blocks):
            z = z + self.dropout_row_layer(
                self.trunk[f'mult_out_{i}'](z, edge_mask)
            )
            z = z + self.dropout_row_layer(
                self.trunk[f'mult_in_{i}'](z, edge_mask)
            )
            z = z + self.dropout_row_layer(
                self.trunk[f'trig_start_{i}'](z, edge_mask)
            )
            z = z + self.dropout_col_layer(
                self.trunk[f'trig_end_{i}'](z, edge_mask)
            )

        return z



