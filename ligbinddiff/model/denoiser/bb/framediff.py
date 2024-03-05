"""Fork of Openfold's IPA."""

import numpy as np
import torch
import math
from scipy.stats import truncnorm
import torch.nn as nn
from typing import Optional, Callable, List, Sequence

import functools as fn
from torch_geometric.utils import sort_edge_index

from ligbinddiff.data.datasets.featurize.common import _node_positional_embeddings, _edge_positional_embeddings, _rbf
from ligbinddiff.model.modules.common import GaussianRandomFourierBasis
from ligbinddiff.model.modules.openfold.frames import Linear, flatten_final_dims, ipa_point_weights_init_
from ligbinddiff.model.utils.graph import get_data_lens, batchwise_to_nodewise, gen_spatial_graph_features, sample_inv_cubic_edges, sequence_local_graph
from ligbinddiff.utils.openfold.rigid_utils import Rigid, batchwise_center
from ligbinddiff.utils.framediff.all_atom import compute_backbone


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class StructureModuleTransition(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransition, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)

        return s


class EdgeTransition(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        edge_embed = torch.cat(
            [edge_embed, edge_bias], axis=-1).reshape(
                batch_size * num_res**2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(
            batch_size, num_res, num_res, -1
        )
        return edge_embed


class InvariantPointAttention(nn.Module):
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
        super(InvariantPointAttention, self).__init__()

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
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((self.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
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
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if(_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if(_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0]).to(dtype=a.dtype)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            ).to(dtype=z[0].dtype)
        )

        return s


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


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update


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



class Embedder(nn.Module):

    def __init__(self, c_s, c_z,
                 index_embed_size=32):
        super(Embedder, self).__init__()

        # Time step embedding
        t_embed_size = index_embed_size
        node_embed_dims = t_embed_size + 1
        edge_in = (t_embed_size + 1) * 2

        # Sequence index embedding
        node_embed_dims += index_embed_size
        edge_in += index_embed_size

        node_embed_size = c_s
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        edge_embed_size = c_z
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=index_embed_size
        )

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
        prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])
        pair_feats.append(self.index_embedder(rel_seq_offset))

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        return node_embed, edge_embed


class IpaScore(nn.Module):

    def __init__(self,
                 diffuser,
                 c_s=256,
                 c_z=128,
                 c_skip=64,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1):
        super(IpaScore, self).__init__()
        self.diffuser = diffuser

        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        self.num_blocks = num_blocks

        for b in range(num_blocks):
            self.trunk[f'ipa_{b}'] = InvariantPointAttention(
                 c_s=c_s,
                 c_z=c_z,
                 c_hidden=c_hidden,
                 num_heads=num_heads,
                 num_qk_points=num_qk_points,
                 num_v_points=num_v_points,
            )
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(c_s)
            self.trunk[f'skip_embed_{b}'] = Linear(
                c_s,
                c_skip,
                init="final"
            )
            tfmr_in = c_s + c_skip
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=4,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, 2)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                tfmr_in, c_s, init="final")
            self.trunk[f'node_transition_{b}'] = StructureModuleTransition(
                c=c_s)
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(c_s)

            if b < num_blocks-1:
                # No edge update on the last block.
                edge_in = c_z
                self.trunk[f'edge_transition_{b}'] = EdgeTransition(
                    node_embed_size=c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=c_z,
                )

        self.torsion_pred = TorsionAngles(c_s, 1)

    def forward(self, init_node_embed, edge_embed, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = input_feats['rigids_t'].type(torch.float32)

        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        init_rigids = Rigid.from_tensor_7(init_frames)

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_in = torch.cat([
                node_embed, self.trunk[f'skip_embed_{b}'](init_node_embed)
            ], dim=-1)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                seq_tfmr_in, src_key_padding_mask=1 - node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * diffuse_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update * diffuse_mask[..., None])

            if b < self.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
        rot_score = self.diffuser.calc_rot_score(
            init_rigids.get_rots(), # .get_rot_mats(),
            curr_rigids.get_rots(), # .get_rot_mats(),
            input_feats['t']
        )
        rot_score = rot_score * node_mask[..., None]

        curr_rigids = self.unscale_rigids(curr_rigids)
        trans_score = self.diffuser.calc_trans_score(
            init_rigids.get_trans(),
            curr_rigids.get_trans(),
            input_feats['t'][:, None, None],
            use_torch=True
        )
        trans_score = trans_score * node_mask[..., None]
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'rot_score': rot_score,
            'trans_score': trans_score,
            'final_rigids': curr_rigids,
            'node_embed': node_embed
        }
        return model_out


class IpaScoreWrapper(nn.Module):
    def __init__(self,
                 diffuser,
                 c_s=256,
                 c_z=128,
                 c_skip=64,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 ):
        super().__init__()

        self.ipa_score = IpaScore(
            diffuser,
            c_s=c_s,
            c_z=c_z,
            c_skip=c_skip,
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            num_blocks=num_blocks
        )
        self.embedder = Embedder(c_s=c_s, c_z=c_z)

    def forward(self, data):
        device = data['x'].device

        data_list = data.to_data_list()
        seq_idx = [torch.arange(data_list[0].num_nodes, device=device) for _ in data_list]
        seq_idx = torch.stack(seq_idx)
        t = data['t']
        batch_size = t.shape[0]
        fixed_mask = data['fixed_mask'].view(batch_size, -1)

        node_embed, edge_embed = self.embedder(seq_idx=seq_idx, t=t, fixed_mask=fixed_mask)

        node_mask = ~data['x_mask'].view(batch_size, -1)

        # center the training example at the mean of the x_cas
        rigids_t = Rigid.from_tensor_7(data['rigids_t'])
        center = batchwise_center(rigids_t, data.batch, node_mask)
        rigids_t = rigids_t.translate(-center)
        rigids_t = rigids_t.view([t.shape[0], -1])

        input_feats = {
            'fixed_mask': fixed_mask,
            'res_mask': node_mask,
            'rigids_t': rigids_t.to_tensor_7(),
            't': t
        }

        score_dict = self.ipa_score(node_embed, edge_embed, input_feats)
        rigids = score_dict['final_rigids'].view(-1)
        rigids = rigids.translate(center)

        psi = score_dict['psi'].view(-1, 2)

        ret = {}
        ret['denoised_frames'] = rigids
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)[:, :5]
        ret['denoised_bb'] = denoised_bb
        ret['node_features'] = score_dict['node_embed']
        # intermediates['rot_score'] = score_dict['rot_score']
        # intermediates['trans_score'] = score_dict['trans_score']
        ret['psi'] = psi

        return ret


class KnnInvariantPointAttention(nn.Module):
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

        self.head_weights = nn.Parameter(torch.zeros((self.no_heads)))
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
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, k, C_z] pair representation
            edge_index:
                [*, N_res, k] edge index
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, k, H, C_hidden]
        edge_index_expand = edge_index[..., None, None].expand(
            -1, -1, -1, kv.shape[-2], kv.shape[-1])
        kv_expand = kv.unsqueeze(-3).expand(-1, -1, edge_index.shape[-1], -1, -1)
        kv_gather = torch.gather(kv_expand, 1, edge_index_expand)
        # kv_gather = gather(kv, edge_index)
        # print(kv_gather.shape)
        k_gather, v_gather = torch.split(kv_gather, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, k, H, P_q/P_v, 3]
        edge_index_expand = edge_index[..., None, None, None].expand(
            -1, -1, -1, kv_pts.shape[-3], kv_pts.shape[-2], kv_pts.shape[-1])
        kv_pts_expand = kv_pts.unsqueeze(-4).expand(-1, -1, edge_index.shape[-1], -1, -1, -1)
        kv_pts_gather = torch.gather(kv_pts_expand, 1, edge_index_expand)
        k_pts_gather, v_pts_gather = torch.split(
            kv_pts_gather, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, k, H]
        b = self.linear_b(z[0])

        if(_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, k]
        # q: [*, N_res, H, C_hidden]
        # k_gather: [*, N_res, k, H, C_hidden]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2))[..., None, :],  # [*, H, N_res, 1, C_hidden]
            permute_final_dims(k_gather, (2, 0, 3, 1)),  # [*, H, N_res, C_hidden, k]
        ).squeeze(-2)
        # torch.einsum("...nhc,...nkhc->...hnk", q, k_gather)
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, k, H, P_q, 3]
        # q_pts: [*, N_res, H, P_q, 3]
        # k_pts_gather: [*, N_res, k, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts_gather
        pt_att = pt_displacement ** 2

        # [*, N_res, k, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, k, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, k]
        edge_mask = torch.gather(
            mask[..., None].expand(-1, -1, edge_index.shape[-1]),
            1,
            edge_index)
        edge_mask = self.inf * (edge_mask - 1)

        # [*, H, N_res, k]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + edge_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        # a: [*, H, N_res, k], originally [*, H, N_res, N_res]
        # v: [*, N_res, H, C_hidden]
        # v_gather: [*, N_res, k, H, C_hidden]
        o = torch.matmul(
            a.transpose(-2, -3)[..., None, :], # [*, N_res, H, 1, k]
            v_gather.transpose(-2, -3).to(dtype=a.dtype)  # [*, N_res, H, k, C_hidden]
        ).squeeze(-2)
        # torch.einsum("...hnk,...nkhc->...nhc", a, v_gather)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        # a: [*, H, N_res, k], originally [*, H, N_res, N_res]
        # v_pts [*, N_res, H, P_v, 3]
        # v_pts_gather: [*, N_res, k, H, P_v, 3]
        o_pt = torch.matmul(
            a[..., None, :, None, :],  # a: [*, H, 1, N_res, 1, k]
            permute_final_dims(v_pts_gather, (2, 4, 0, 1, 3))  # [*, H, 3, N_res, k, P_v]
        ).squeeze(-2)
        # torch.einsum("...hnk,...nkhvc->...hcnv", a, v_pts_gather)

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if(_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0]).to(dtype=a.dtype)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            ).to(dtype=z[0].dtype)
        )

        return s

class KnnEdgeTransition(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super().__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed, edge_index):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape

        node_dst = torch.gather(
            node_embed[..., None, :].expand(-1, -1, edge_index.shape[-1], -1),
            1,
            edge_index[..., None].expand(-1, -1, -1, node_embed.shape[-1])
        )
        node_src = node_embed[..., None, :].expand(-1, -1, edge_index.shape[-1], -1)

        edge_bias = torch.cat([
            node_dst,
            node_src
        ], dim=-1)

        edge_embed = torch.cat([edge_embed, edge_bias], dim=-1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        return edge_embed


class KnnIpaScore(nn.Module):
    def __init__(self,
                 diffuser,
                 c_s=256,
                 c_z=128,
                 c_hidden=256,
                 c_skip=64,
                 num_heads=8,
                 num_qk_pts=8,
                 num_v_pts=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 knn_k=30, local_k=10):
        super().__init__()
        self.diffuser = diffuser
        self.num_blocks = num_blocks

        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        for b in range(self.num_blocks):
            self.trunk[f'spatial_ipa_{b}'] = KnnInvariantPointAttention(
                c_s,
                c_z,
                c_hidden,
                num_heads,
                num_qk_pts,
                num_v_pts,
            )
            self.trunk[f'spatial_ipa_ln_{b}'] = nn.LayerNorm(c_s)
            # self.trunk[f'spatial_skip_embed_{b}'] = Linear(
            #     c_s,
            #     c_skip,
            #     init="final"
            # )
            # tfmr_in = c_s + c_skip
            # self.trunk[f'post_spatial_{b}'] = Linear(
            #     tfmr_in, c_s, init="final")
            self.trunk[f'seq_ipa_{b}'] = KnnInvariantPointAttention(
                c_s,
                c_z,
                c_hidden,
                num_heads,
                num_qk_pts,
                num_v_pts,
            )
            self.trunk[f'seq_ipa_ln_{b}'] = nn.LayerNorm(c_s)
            # self.trunk[f'seq_skip_embed_{b}'] = Linear(
            #     c_s,
            #     c_skip,
            #     init="final"
            # )
            # self.trunk[f'post_seq_{b}'] = Linear(
            #     tfmr_in, c_s, init="final")
            self.trunk[f'node_transition_{b}'] = StructureModuleTransition(
                c=c_s)
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(c_s)

            if b < self.num_blocks-1:
                # No edge update on the last block.
                edge_in = c_z
                self.trunk[f'spatial_edge_transition_{b}'] = KnnEdgeTransition(
                    node_embed_size=c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=c_z,
                )
                self.trunk[f'seq_edge_transition_{b}'] = KnnEdgeTransition(
                    node_embed_size=c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=c_z,
                )

        self.torsion_pred = TorsionAngles(c_s, 1)
        self.local_k = local_k

    def forward(self,
                init_node_embed,
                spatial_edge_embed,
                seq_edge_embed,
                spatial_edge_index,
                seq_edge_index,
                input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        init_frames = input_feats['rigids_t'].type(torch.float32)

        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        init_rigids = Rigid.from_tensor_7(init_frames)

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        for b in range(self.num_blocks):
            spatial_ipa_embed = self.trunk[f'spatial_ipa_{b}'](
                s=node_embed,
                z=spatial_edge_embed,
                edge_index=spatial_edge_index,
                r=curr_rigids,
                mask=node_mask)
            spatial_ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'spatial_ipa_ln_{b}'](node_embed + spatial_ipa_embed)

            seq_ipa_embed = self.trunk[f'seq_ipa_{b}'](
                node_embed,
                seq_edge_embed,
                seq_edge_index,
                curr_rigids,
                node_mask)
            seq_ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'seq_ipa_ln_{b}'](node_embed + seq_ipa_embed)

            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * diffuse_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update * diffuse_mask[..., None])

            if b < self.num_blocks-1:
                spatial_edge_embed = self.trunk[f'spatial_edge_transition_{b}'](
                    node_embed, spatial_edge_embed, spatial_edge_index)
                seq_edge_embed = self.trunk[f'seq_edge_transition_{b}'](
                    node_embed, seq_edge_embed, seq_edge_index)
        # rot_score = self.diffuser.calc_rot_score(
        #     init_rigids.get_rots(),
        #     curr_rigids.get_rots(),
        #     input_feats['t']
        # )
        # rot_score = rot_score * node_mask[..., None]

        curr_rigids = self.unscale_rigids(curr_rigids)
        # trans_score = self.diffuser.calc_trans_score(
        #     init_rigids.get_trans(),
        #     curr_rigids.get_trans(),
        #     input_feats['t'][:, None, None],
        #     use_torch=True,
        # )
        # trans_score = trans_score * node_mask[..., None]
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            # 'rot_score': rot_score,
            # 'trans_score': trans_score,
            'final_rigids': curr_rigids,
        }
        return model_out


class KnnIpaScoreWrapper(nn.Module):
    def __init__(self,
                 diffuser,
                 c_s=256,
                 c_z=128,
                 c_skip=64,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 knn_k=20,
                 lrange_k=40,
                 h_time=64,
                 scalar_h_dim=128,
                 ):
        super().__init__()

        self.ipa_score = KnnIpaScore(
            diffuser,
            c_s=c_s,
            c_z=c_z,
            c_skip=c_skip,
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_pts=num_qk_points,
            num_v_pts=num_v_points,
            num_blocks=num_blocks
        )

        self.c_s = c_s
        self.c_z = c_z

        self.h_time = h_time
        self.time_rbf = GaussianRandomFourierBasis(n_basis=h_time//2)
        self.time_mlp = nn.Sequential(
            nn.Linear(h_time, scalar_h_dim),
            nn.ReLU(),
            nn.Linear(scalar_h_dim, h_time),
            nn.ReLU()
        )

        self.embed_node = nn.Linear(
            c_s + h_time + 1, c_s
        )


        self.knn_k = knn_k
        self.lrange_k = lrange_k

    def forward(self, data):
        ## prep features
        t = data['t']  # (B,)
        batch_size = t.shape[0]
        x_mask = data['x_mask']
        rigids_t = Rigid.from_tensor_7(data['rigids_t'])
        batch = data.batch
        device = t.device
        num_nodes = data.to_data_list()[0].num_nodes


        # center the training example at the mean of the x_cas
        center = batchwise_center(rigids_t, data.batch, x_mask)
        rigids_t = rigids_t.translate(-center)

        # generate sequence edges
        seq_local_edge_index = []
        offset = 0
        for i in range(data.batch.max().item() + 1):
            select = (data.batch == i)
            seq_local_edge_index.append(
                self.sequence_local_graph(select.sum().item())
            )
            offset += select.sum().item()
        seq_local_edge_index = torch.stack(seq_local_edge_index, dim=0).to(device)

        # generate spatial edges
        X_ca = rigids_t.get_trans()
        masked_X_ca = X_ca.clone()
        masked_X_ca[x_mask] = torch.inf
        edge_index = self.sample_inv_cubic_edges(masked_X_ca, x_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        # compute edge features
        residx = torch.arange(num_nodes, device=device)
        X_ca = X_ca.view(batch_size, num_nodes, -1)
        X_dst = torch.gather(
            X_ca[..., None, :].expand(-1, -1, edge_index.shape[-1], -1),
            1,
            edge_index[..., None].expand(-1, -1, -1, X_ca.shape[-1])
        )
        edge_dist_vec = X_dst - X_ca.unsqueeze(-2)
        edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device, D_count=self.c_z//2)  # edge_channels_list
        edge_dist_rel_pos = _node_positional_embeddings(edge_index - residx[None, :, None], num_embeddings=self.c_z//2, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        X_dst = torch.gather(
            X_ca[..., None, :].expand(-1, -1, seq_local_edge_index.shape[-1], -1),
            1,
            seq_local_edge_index[..., None].expand(-1, -1, -1, X_ca.shape[-1])
        )
        seq_edge_dist_vec = X_dst - X_ca.unsqueeze(-2)
        seq_edge_dist = torch.linalg.vector_norm(seq_edge_dist_vec, dim=-1)
        seq_edge_dist_rbf = _rbf(seq_edge_dist, device=seq_edge_dist.device, D_count=self.c_z//2)  # edge_channels_list
        seq_edge_dist_rel_pos = _node_positional_embeddings(seq_local_edge_index - residx[None, :, None], num_embeddings=self.c_z//2, device=edge_dist.device)  # edge_channels_list
        seq_edge_features = torch.cat([seq_edge_dist_rbf, seq_edge_dist_rel_pos], dim=-1)

        ## create time embedding
        fourier_time = self.time_rbf(t.unsqueeze(-1))  # (N_node x h_time,)
        embedded_time = self.time_mlp(fourier_time)  # (N_node x h_time)

        ## denoising
        fixed_mask = data['fixed_mask'].view(batch_size, -1)
        node_mask = ~data['x_mask'].view(batch_size, -1)
        rigids_t = rigids_t.view([batch_size, -1])

        # generate node features
        res_pos = _node_positional_embeddings(
            residx,
            num_embeddings=self.c_s, device=device)
        node_features = self.embed_node(
            torch.cat([
                res_pos[None].expand(batch_size, -1, -1),
                embedded_time[:, None].expand(-1, num_nodes, -1),
                data['noising_mask'].float().view(batch_size, -1)[..., None]
            ], dim=-1)
        )


        input_feats = {
            'fixed_mask': fixed_mask,
            'res_mask': node_mask,
            'rigids_t': rigids_t.to_tensor_7(),
            't': t
        }

        # print(node_features.shape, edge_features.shape, seq_edge_features.shape, edge_index.shape, seq_local_edge_index.shape, rigids_t.shape)

        score_dict = self.ipa_score(
            node_features,
            edge_features,
            seq_edge_features,
            edge_index,
            seq_local_edge_index,
            input_feats)
        rigids = score_dict['final_rigids'].view(-1)
        rigids = rigids.translate(center)

        psi = score_dict['psi'].view(-1, 2)

        ret = {}
        ret['denoised_frames'] = rigids
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)[:, :5]
        ret['denoised_bb'] = denoised_bb
        # intermediates['rot_score'] = score_dict['rot_score']
        # intermediates['trans_score'] = score_dict['trans_score']
        ret['psi'] = psi

        return ret

    def sequence_local_graph(self, num_nodes, half_local_size=5):
        local_index = torch.cat([
            -(torch.arange(half_local_size) + 1),
            (torch.arange(half_local_size) + 1)
        ], dim=-1)
        offset = torch.arange(num_nodes)[..., None]
        global_edge_index = local_index[None].expand(num_nodes, -1) + offset

        head = torch.empty(half_local_size, half_local_size*2)
        for i in range(half_local_size):
            chunk = torch.cat([
                torch.arange(0, i),
                torch.arange(i+1, half_local_size*2+1)
            ], dim=-1)
            head[i] = chunk

        tail = torch.empty(half_local_size, half_local_size*2)
        for i in range(half_local_size):
            i = i+1
            start = num_nodes - half_local_size * 2
            current = num_nodes - i
            chunk = torch.cat([
                torch.arange(start-1, current),
                torch.arange(current+1, num_nodes)
            ], dim=-1)
            tail[-i] = chunk

        global_edge_index[:half_local_size] = head
        global_edge_index[-half_local_size:] = tail
        return global_edge_index


    def sample_inv_cubic_edges(self, batched_X_ca, batched_x_mask, batch, knn_k=30, inv_cube_k=10):
        edge_indicies = []
        offset = 0
        for i in range(batch.max().item() + 1):
            X_ca = batched_X_ca[batch == i]
            x_mask = batched_x_mask[batch == i]

            X_ca[x_mask] = torch.inf
            rel_pos_CA = X_ca.unsqueeze(1) - X_ca.unsqueeze(0)  # N x N x 3
            dist_CA = torch.linalg.vector_norm(rel_pos_CA, dim=-1)  # N x N
            sorted_dist, sorted_edges = torch.sort(dist_CA, dim=-1)  # N x N
            knn_edges = sorted_edges[..., 1:knn_k+1]  # first edge will always be self

            # remove knn edges
            remaining_dist = sorted_dist[..., knn_k+1:]  # N x (N - knn_k - 1)
            remaining_edges = sorted_edges[..., knn_k+1:]  # N x (N - knn_k - 1)

            ## inv cube
            uniform = torch.distributions.Uniform(0,1)
            dist_noise = uniform.sample(remaining_dist.shape).to(batched_X_ca.device)  # N x (N - knn_k - 1)

            logprobs = -3 * torch.log(remaining_dist)  # N x (N - knn_k)
            perturbed_logprobs = logprobs - torch.log(-torch.log(dist_noise))  # N x (N - knn_k - 1)

            good_edges = torch.isfinite(perturbed_logprobs)
            perturbed_logprobs[~good_edges] = -torch.inf

            # if we don't have inv_cube_k nodes to sample, sample the max we can
            num_bad_edges = (~good_edges).sum(dim=-1)
            max_num_bad_edges = int(num_bad_edges.max())
            if inv_cube_k > perturbed_logprobs.shape[-1] - max_num_bad_edges:
                inv_cube_k = perturbed_logprobs.shape[-1] - max_num_bad_edges

            _, sampled_edges_relative_idx = torch.topk(perturbed_logprobs, k=inv_cube_k, dim=-1)
            sampled_edges = torch.gather(remaining_edges, -1, sampled_edges_relative_idx)  # N x inv_cube_k

            edge_sinks = torch.cat([knn_edges, sampled_edges], dim=-1)  # B x N x (knn_k + inv_cube_k)
            edge_indicies.append(edge_sinks)
            offset = offset + (batch == i).long().sum()

        edge_index = torch.stack(edge_indicies, dim=0)
        edge_mask = batched_x_mask[edge_index].any(dim=-1)
        # edge_mask[:, batched_x_mask] = True
        return edge_index#, edge_mask
