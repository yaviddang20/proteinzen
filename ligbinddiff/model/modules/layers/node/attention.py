from ligbinddiff.model.modules.openfold.frames import Linear, flatten_final_dims, ipa_point_weights_init_
from ligbinddiff.utils.openfold.rigid_utils import Rigid


import torch
import torch.nn as nn
import torch_geometric.utils as pygu


import math
from typing import Optional, Sequence, Tuple


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

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        h_t_q = self.no_heads * self.no_qk_points
        self.linear_t_q = Linear(self.c_v + hpq//3, h_t_q, bias=False)

        h_t_kv = self.no_heads * (self.no_qk_points + self.no_v_points)
        self.linear_t_kv = Linear(self.c_v + hpkv//3, h_t_kv, bias=False)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        self.linear_out_s_s = Linear(
            self.no_heads * self.c_hidden + self.no_heads * (self.c_z // 4),
            self.c_s,
            init="final")
        self.linear_out_s_v = Linear(
            self.no_heads * self.no_v_points,
            self.c_v,
            bias=False)

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

        t = r.get_trans()

        # [N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s_s)

        # This is kind of clunky, but it's how the original does it
        # [N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].get_rots().apply(q_pts)

        # [N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s_s)

        # [N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].get_rots().apply(kv_pts)

        if s_v is None:
            s_v = torch.zeros(
                list(s_s.shape[:-1])
                + [self.c_v, 3],
                device=s_s.device
            )

        # [N_res, 2 * H * P_q, 3]
        t_q_in = torch.cat([
            s_v, q_pts
        ], dim=-2)

        t_q = self.linear_t_q(t_q_in.transpose(-1, -2)) + t[..., None]/self.D_points
        # [N_res, H, P_q, 3]
        t_q = t_q.transpose(-1, -2).view([-1, self.no_heads, self.no_qk_points, 3])

        # [N_res, 2 * H * (P_q + P_v), 3]
        t_kv_in = torch.cat([
            s_v, kv_pts
        ], dim=-2)
        # [N_res, 3, H * (P_q + P_v)]
        t_kv = self.linear_t_kv(t_kv_in.transpose(-1, -2))
        # [N_res, H, (P_q + P_v), 3]
        t_kv = t_kv.transpose(-1, -2).view([-1 ,self.no_heads, (self.no_qk_points + self.no_v_points), 3])

        # [N_res, H, P_q or P_v, 3]
        t_k, t_v = t_kv.split([self.no_qk_points, self.no_v_points], dim=-2)
        t_k = t_k + t[..., None, None, :]/self.D_points


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
        a *= math.sqrt(1.0 / (self.c_hidden / self.no_heads))
        a += b

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
            1.0 / (18 * self.c_hidden / self.no_heads)
        )
        pt_att = pt_att * head_weights

        # [N_edge, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-1.0)

        # [N_edge, H]
        a = a + pt_att
        a = pygu.softmax(a, edge_index[1], num_nodes=n_nodes)

        ################
        # Compute output
        ################
        # [N_res, H, C_hidden]
        o = pygu.scatter(
            a[..., None] *  # [N_edge, H]
            v[edge_index[0]],  # [N_edge, H, C_hidden]
            edge_index[1],
            dim=0,
            dim_size=n_nodes
        )

        # [N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [N_res, H, P_v, 3]
        o_pt = pygu.scatter(
            a[..., None, None]  # [N_edge, H, 1, 1]
            * t_v[edge_index[0]],  # [N_edge, H, P_v, 3]
            edge_index[1],
            dim=0,
            dim_size=n_nodes
        )
        # [N_res, H * P_v, 3]
        o_pt = o_pt.flatten(start_dim=-3, end_dim=-2)

        # [N_res, H, C_z // 4]
        pair_z = self.down_z(z)
        o_pair = pygu.scatter(
            a[..., None] *  # [N_edge, H, 1]
            pair_z[..., None, :],  # [N_edge, 1, C_z // 4]
            edge_index[1],
            dim=0,
            dim_size=n_nodes
        )
        # [N_res, H * C_z // 4]
        o_pair = o_pair.flatten(-2, -1)

        # [N_res, H * C_hidden + H * C_z // 4]
        o_feats = torch.cat([o, o_pair], dim=-1)

        # [N_res, C_s]
        out_s_s = self.linear_out_s_s(o_feats)
        # [N_res, C_v, 3]
        out_s_v = self.linear_out_s_v(o_pt.transpose(-1, -2)).transpose(-1, -2)

        return out_s_s, out_s_v


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

        edge_mask = mask[edge_index[0]] * mask[edge_index[1]]
        edge_mask = self.inf * (edge_mask - 1)
        # [N_edge, H]
        a = a + pt_att
        a = a + edge_mask[..., None]
        a = pygu.softmax(a, edge_index[1], num_nodes=n_nodes)

        ################
        # Compute output
        ################
        v_dst = v[edge_index[0]]
        # [N_res, H, C_hidden]
        o = pygu.scatter(
            a[..., None] *  # [N_edge, H]
            v_dst,  # [N_edge, H, C_hidden]
            edge_index[1],
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
            edge_index[1],
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
            edge_index[1],
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

class InvariantPointMlpAttention(nn.Module):
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
        self.w_mlp = nn.Sequential(
            Linear(2*c_s + c_z, c_hidden),
            nn.ReLU(),
            Linear(c_hidden, c_hidden),
            nn.ReLU(),
            Linear(c_hidden, self.no_heads),
        )
        self.v_mlp = nn.Sequential(
            Linear(c_s + c_z, c_hidden),
            nn.ReLU(),
            Linear(c_hidden, c_hidden),
            nn.ReLU(),
            Linear(c_hidden, c_hidden),
        )

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden // self.no_heads + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
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
        n_nodes = s.shape[0]
        src = edge_index[1]
        dst = edge_index[0]
        r = r.scale_translation(1/10)

        #######################################
        # Generate point activations
        #######################################

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
        b = self.linear_b(z)

        # [N_edge, H]
        w_ins = torch.cat([
            s[src],
            s[dst],
            z
        ], dim=-1)
        a = self.w_mlp(w_ins)
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

        edge_mask = mask[edge_index[0]] * mask[edge_index[1]]
        edge_mask = self.inf * (edge_mask - 1)
        # [N_edge, H]
        a = a + pt_att
        a = a + edge_mask[..., None]
        a = pygu.softmax(a, edge_index[1], num_nodes=n_nodes)

        ################
        # Compute output
        ################
        v_dst_in = torch.cat([
            s[dst],
            z
        ], dim=-1)
        v_dst = self.v_mlp(v_dst_in).view(
            edge_index.shape[1],
            self.no_heads,
            -1
        )  # [N_res, H, C_hidden // H]
        # [N_res, H, C_hidden // H]
        o = pygu.scatter(
            a[..., None] *  # [N_edge, H]
            v_dst,  # [N_edge, H, C_hidden // H]
            edge_index[1],
            dim=0,
            dim_size=n_nodes
        )
        # [N_res,  H * (C_hidden // H)]
        o = flatten_final_dims(o, 2)

        # [N_res, H, P_v, 3]
        v_pts_dst = v_pts[edge_index[0]]
        o_pt = pygu.scatter(
            a[..., None, None]  # [N_edge, H, 1, 1]
            * v_pts_dst,  # [N_edge, H, P_v, 3]
            edge_index[1],
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

        # [N_res, H, C_z // 4]
        pair_z = self.down_z(z).to(dtype=a.dtype)  # [N_edge, C_z // 4]
        o_pair = pygu.scatter(
            a[..., None]  # [N_edge, H, 1]
            * pair_z[..., None, :],  # [N_edge, 1, C_z // 4]
            edge_index[1],
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
            ).to(dtype=z.dtype)
        )

        return s
