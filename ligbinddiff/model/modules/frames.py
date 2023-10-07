import numpy as np
import torch
import math
from scipy.stats import truncnorm
import torch.nn as nn
from typing import Optional, Callable, List, Sequence, Tuple

import torch_geometric.utils as pygu

from ...utils.openfold.rigid_utils import Rigid
from .equiformer_v2.so3 import SO3_Grid, SO3_LinearV2
from .equiformer_v2.activation import SeparableS2Activation
from .equiformer_v2.layer_norm import EquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3

def permute_final_dims(tensor: torch.Tensor, inds: Sequence[int]):
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


class PointSetAttentionWithEdgeBias(nn.Module):
    """
    Modified version of PSAEB from NeuralPLexer
    """
    def __init__(
        self,
        c_s,
        c_v,
        c_z,
        c_hidden,
        no_heads,
        no_qk_points,
        no_v_points,
        inf: float = 1e5,
        eps: float = 1e-8,
        D_points=10,
        gen_vectors=False,
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
        self.c_v = c_v
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        self.D_points = D_points
        self.gen_vectors = gen_vectors

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        if gen_vectors:
            hpq = self.no_heads * self.no_qk_points * 3
            self.linear_q_points = Linear(self.c_s, hpq)

            hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
            self.linear_kv_points = Linear(self.c_s, hpkv)
        else:
            hpq = self.no_heads * self.no_qk_points
            self.linear_t_q = Linear(self.c_v, hpq, bias=False)

            hpkv = self.no_heads * (self.no_qk_points + self.no_v_points)
            self.linear_t_kv = Linear(self.c_v, hpkv, bias=False)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        self.linear_out_s_s = Linear(self.no_heads * self.c_hidden, self.c_s, init="final")
        self.linear_out_s_v = Linear(self.no_heads * self.no_v_points, self.c_v, init="final", bias=False)

        self.softplus = nn.Softplus()

    def forward(
        self,
        node_scalars: torch.Tensor,
        rigids: Rigid,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_vectors: Optional[torch.Tensor]=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s_s:
                [N_res, C_s] single representation, scalars
            s_v:
                [N_res, C_v, 3] single representation, vectors
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
            [N_res, C_v, 3] single vector representation update
        """
        s_s = node_scalars
        s_v = node_vectors
        z = edge_features
        r = rigids
        n_nodes = s_s.shape[0]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [N_res, H * C_hidden]
        q = self.linear_q(s_s)
        kv = self.linear_kv(s_s)

        # [N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        if self.gen_vectors:
            # [N_res, H * P_q * 3]
            q_pts = self.linear_q_points(s_s)

            # This is kind of clunky, but it's how the original does it
            # [N_res, H * P_q, 3]
            q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
            q_pts = torch.stack(q_pts, dim=-1)
            q_pts = r[..., None].apply(q_pts)

            # [N_res, H, P_q, 3]
            q_pts = q_pts.view(
                q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
            )

            # [N_res, H * (P_q + P_v) * 3]
            kv_pts = self.linear_kv_points(s_s)

            # [N_res, H * (P_q + P_v), 3]
            kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
            kv_pts = torch.stack(kv_pts, dim=-1)
            kv_pts = r[..., None].apply(kv_pts)

            # [N_res, H, (P_q + P_v), 3]
            kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

            # [N_res, H, P_q/P_v, 3]
            k_pts, v_pts = torch.split(
                kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
            )

            t_q = q_pts
            t_k = k_pts
            t_v = v_pts

        else:
            assert s_v is not None
            t = r.get_trans()
            # [N_res, 3, H * P_q]
            t_q = self.linear_t_q(s_v.transpose(-1, -2)) + t[..., None]/self.D_points
            # [N_res, H, P_q, 3]
            t_q = t_q.transpose(-1, -2).view([-1, self.no_heads, self.no_qk_points, 3])

            # [N_res, 3, H * (P_q + P_v)]
            t_kv = self.linear_t_kv(s_v.transpose(-1, -2))
            # [N_res, H, (P_q + P_v), 3]
            t_kv = t_kv.transpose(-1, -2).view([-1 ,self.no_heads, (self.no_qk_points + self.no_v_points), 3])

            # [N_res, H, P_q or P_v, 3]
            t_k, t_v = t_kv.split([self.no_qk_points, self.no_v_points], dim=-2)


        ##########################
        # Compute attention scores
        ##########################
        # [N_edge, H]
        b = self.linear_b(z)

        # [N_edge, H]
        q_src = q[edge_index[1]]  # [N_edge, H, C_hidden]
        k_dst = k[edge_index[0]]  # [N_edge, H, C_hidden]
        a = torch.matmul(
            q_src[..., None, :],  # [N_edge, H, 1, C_hidden]
            k_dst[..., :, None],  # [N_edge, H, C_hidden, 1]
        )
        a = a.squeeze(-1).squeeze(-1)
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * b)

        # [N_edge, H, P_q, 3]
        t_q_src = t_q[edge_index[1]]
        t_k_dst = t_k[edge_index[0]]
        pt_displacement = t_q_src - t_k_dst
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

        # [N_edge, H]
        a = a + pt_att
        a = pygu.softmax(a, edge_index[1], num_nodes=n_nodes)

        ################
        # Compute output
        ################
        # [N_res, H, C_hidden]
        o = pygu.scatter(
            a[..., None] *  # [N_edge, H]
            v[edge_index[1]],  # [N_edge, H, C_hidden]
            edge_index[0],
            dim=0,
            dim_size=n_nodes
        )

        # [N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [N_res, H, P_v, 3]
        o_pt = pygu.scatter(
            a[..., None, None]  # [N_edge, H, 1, 1]
            * t_v[edge_index[1]],  # [N_edge, H, P_v, 3]
            edge_index[0],
            dim=0,
            dim_size=n_nodes
        )
        # [N_res, H * P_v, 3]
        o_pt = o_pt.flatten(start_dim=-3, end_dim=-2)

        # [N_res, C_s]
        out_s_s = self.linear_out_s_s(o)
        # [N_res, C_v, 3]
        out_s_v = self.linear_out_s_v(o_pt.transpose(-1, -2)).transpose(-1, -2)

        return out_s_s, out_s_v


class LocalFrameUpdate(nn.Module):
    def __init__(self, c_s, c_v, c_hidden):
        super().__init__()

        self.c_s = c_s
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.mlp = nn.Sequential(
            Linear(self.c_s + self.c_v * 4, self.c_hidden, init="relu"),
            nn.ReLU(),
            Linear(self.c_hidden, self.c_hidden, init="relu"),
            nn.ReLU(),
            Linear(self.c_hidden, self.c_s + 3 * self.c_v, init="final")
        )

    def forward(self, f_s, f_v, r):
        f_vloc = r[..., None].invert_apply(f_v)
        f_loc = torch.cat([
            f_s,
            f_vloc.flatten(start_dim=-2, end_dim=-1),
            torch.linalg.vector_norm(f_v, dim=-1)
        ], dim=-1)
        f_s_vloc = self.mlp(f_loc)
        f_s, f_vloc = f_s_vloc.split([self.c_s, 3*self.c_v], dim=-1)
        f_v = r[..., None].apply(f_vloc.view([-1, self.c_v, 3]))
        return f_s, f_v


class BackboneUpdateVectorBias(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s, c_v):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super().__init__()

        self.c_s = c_s
        self.c_v = c_v

        self.linear_s = Linear(self.c_s, 3, init="final")
        self.linear_v = Linear(self.c_v, 1, init="final")

    def forward(self, s: torch.Tensor, v):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update_s = self.linear_s(s)
        update_v = self.linear_v(v.transpose(-1, -2)).squeeze(-1)
        update = torch.cat([update_s, update_v], dim=-1)

        return update


class Lmax1_LinearV2(torch.nn.Module):
    def __init__(self, in_features, out_features):
        '''
            1. Use `torch.einsum` to prevent slicing and concatenation
            2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = 1

        self.weight = torch.nn.Parameter(torch.randn((self.lmax + 1), out_features, in_features))
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(self.lmax + 1) ** 2]).long()
        for l in range(self.lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            expand_index[start_idx : (start_idx + length)] = l
        self.register_buffer('expand_index', expand_index)


    def forward(self, input_embedding):

        weight = torch.index_select(self.weight, dim=0, index=self.expand_index) # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum('bmi, moi -> bmo', input_embedding, weight) # [N, (L_max + 1) ** 2, C_out]
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"


class VectorLayerNorm(nn.Module):
    def __init__(self, c_v):
        super().__init__()
        self.c_v = c_v
        self.ln_v = NormSO3(lmax=1, num_channels=self.c_v)

    def forward(self, v):
        in_v = v.transpose(-1, -2)
        v_dummy_scalars = torch.zeros([in_v.shape[0], 1, self.c_v], device=in_v.device)
        out_v = self.ln_v(
            torch.cat([v_dummy_scalars, in_v], dim=-2)
        )[..., 1:, :].transpose(-1, -2)
        return out_v


class NodeTransition(nn.Module):
    def __init__(self, c_s, c_v):
        super().__init__()

        self.c_s = c_s
        self.c_v = c_v
        self.c_hidden = max(c_s, c_v)

        lmax = 1

        self.embed_s = Linear(self.c_s, self.c_hidden)
        self.embed_v = Linear(self.c_v, self.c_hidden, bias=False)

        self.linear2 = Lmax1_LinearV2(self.c_hidden, self.c_hidden)

        self.act = SeparableS2Activation(1, 1)
        self.SO3_grid_list = nn.ModuleList()
        for l in range(lmax+1):
            SO3_m_grid = nn.ModuleList()
            for m in range(lmax+1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            self.SO3_grid_list.append(SO3_m_grid)


        self.out_s = Linear(self.c_hidden, self.c_s)
        self.out_v = Linear(self.c_hidden, self.c_v, bias=False)

        self.ln_s = nn.LayerNorm(self.c_s)
        self.ln_v = VectorLayerNorm(self.c_v)

    def forward(self, s, v):

        # "linear 1"
        embed_s = self.embed_s(s)
        embed_v = self.embed_v(v.transpose(-1, -2))
        embedding = torch.cat([embed_s[..., None, :], embed_v], dim=-2)

        embedding = self.act(embedding[:, 0], embedding, self.SO3_grid_list)
        embedding = self.linear2(embedding)
        embedding = self.act(embedding[:, 0], embedding, self.SO3_grid_list)

        # "linear 3"
        out_s = self.out_s(embedding[:, 0])
        out_v = self.out_v(embedding[:, 1:])

        out_s = self.ln_s(s + out_s)
        out_v = v + out_v.transpose(-1, -2)
        out_v = self.ln_v(out_v)

        return out_s, out_v


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

        node_dst = node_embed[edge_index[0]]
        node_src = node_embed[edge_index[1]]

        edge_bias = torch.cat([
            node_dst,
            node_src
        ], dim=-1)

        edge_embed = torch.cat([edge_embed, edge_bias], dim=-1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        return edge_embed


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
        super().__init__()

        self.c_s = c_s

        self.linear_s = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update_s = self.linear_s(s)

        return update_s


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

        #######################################
        # Generate scalar and point activations
        #######################################
        # [N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        # [N_edge, H, C_hidden]
        q_src = q[edge_index[1]]

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
        q_pts_src = q_pts[edge_index[1]]

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
        k_dst = k[edge_index[0]]
        a = torch.matmul(
            q_src[..., None, :],  # [N_edge, H, 1, C_hidden]
            k_dst[..., :, None],  # [N_edge, H, C_hidden, 1]
        ).squeeze(-1).squeeze(-1)
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * b)

        # [N_edge, H, P_q, 3]
        k_pts_dst = k_pts[edge_index[0]]
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

        # [N_edge, H]
        a = a + pt_att
        a = pygu.softmax(a, edge_index[1], num_nodes=n_nodes)

        ################
        # Compute output
        ################
        v_dst = v[edge_index[1]]
        # [N_res, H, C_hidden]
        o = pygu.scatter(
            a[..., None] *  # [N_edge, H]
            v_dst,  # [N_edge, H, C_hidden]
            edge_index[0],
            dim=0,
            dim_size=n_nodes
        )
        # [N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [N_res, H, P_v, 3]
        v_pts_dst = v_pts[edge_index[1]]
        o_pt = pygu.scatter(
            a[..., None, None]  # [N_edge, H, 1, 1]
            * v_pts_dst,  # [N_edge, H, P_v, 3]
            edge_index[0],
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
            edge_index[0],
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
            self.trunk[f'spatial_ipa_{b}'] = GraphInvariantPointAttention(
                c_s,
                c_z,
                c_hidden,
                num_heads,
                num_qk_pts,
                num_v_pts,
            )
            self.trunk[f'spatial_ipa_ln_{b}'] = nn.LayerNorm(c_s)
            self.trunk[f'spatial_skip_embed_{b}'] = Linear(
                c_s,
                c_skip,
                init="final"
            )
            tfmr_in = c_s + c_skip
            self.trunk[f'post_spatial_{b}'] = Linear(
                tfmr_in, c_s, init="final")
            self.trunk[f'seq_ipa_{b}'] = GraphInvariantPointAttention(
                c_s,
                c_z,
                c_hidden,
                num_heads,
                num_qk_pts,
                num_v_pts,
            )
            self.trunk[f'seq_ipa_ln_{b}'] = nn.LayerNorm(c_s)
            self.trunk[f'seq_skip_embed_{b}'] = Linear(
                c_s,
                c_skip,
                init="final"
            )
            self.trunk[f'post_seq_{b}'] = Linear(
                tfmr_in, c_s, init="final")
            self.trunk[f'node_transition_{b}'] = StructureModuleTransition(
                c=c_s)
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(c_s)

            if b < self.num_blocks-1:
                # No edge update on the last block.
                edge_in = c_z
                self.trunk[f'spatial_edge_transition_{b}'] = EdgeTransition(
                    node_embed_size=c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=c_z,
                )
                self.trunk[f'seq_edge_transition_{b}'] = EdgeTransition(
                    node_embed_size=c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=c_z,
                )

        # self.torsion_pred = TorsionAngles(ipa_conf.c_s, 1)
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
        # _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            # 'psi': psi_pred,
            # 'rot_score': rot_score,
            # 'trans_score': trans_score,
            'final_rigids': curr_rigids,
        }
        return model_out
