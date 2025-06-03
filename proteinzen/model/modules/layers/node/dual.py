from typing import Optional

import math

import torch
from torch import nn
from e3nn import o3

from proteinzen.model.modules.openfold.layers import Linear, ipa_point_weights_init_, permute_final_dims, flatten_final_dims
from proteinzen.utils.openfold.rigid_utils import Rigid


class DenseEquivariantPointAttention(nn.Module):
    """
    An attention layer which accomedates both frame-based and TFN-based features, largely based off of IPA
    """
    def __init__(
        self,
        c_s,
        c_z,
        c_hidden,
        node_irreps: o3.Irreps,
        num_heads,
        num_qk_points,
        num_v_points,
        inf: float = 1e5,
        eps: float = 1e-8,
        pre_ln=False,
        lin_bias=True,
        final_init='final'
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
        self.linear_q = Linear(self.c_s, hc, bias=lin_bias)
        self.linear_kv = Linear(self.c_s, 2 * hc, bias=lin_bias)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq, bias=lin_bias)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv, bias=lin_bias)

        self.linear_b = Linear(self.c_z, self.no_heads, bias=lin_bias)
        self.down_z = Linear(self.c_z, self.c_z // 4, bias=lin_bias)

        self.head_weights = nn.Parameter(torch.zeros((self.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init=final_init, bias=lin_bias)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.pre_ln = pre_ln
        if pre_ln:
            self.pre_ln_s = nn.LayerNorm(c_s)
            self.pre_ln_z = nn.LayerNorm(c_z)

        self.node_irreps = node_irreps
        self.q_irreps = o3.Irreps(f"{hc}x0e + {self.no_heads * self.no_qk_points}x1o")
        self.kv_irreps = o3.Irreps(f"{2*hc}x0e + {self.no_heads * (self.no_qk_points + self.no_v_points)}x1o")
        tfn_concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points
        )
        self.final_irreps = o3.Irreps(f"{self.no_heads * tfn_concat_out_dim}x0e + {self.no_heads * self.no_v_points}x1o")
        self.tfn_q = o3.Linear(node_irreps, self.q_irreps)
        self.tfn_kv = o3.Linear(node_irreps, self.kv_irreps)
        self.tfn_out = o3.Linear(self.final_irreps, node_irreps)

    def forward(
        self,
        *,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
        is_atom: torch.BoolTensor
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
        do_atom_calcs = is_atom.any()
        do_res_calcs = ~(is_atom.all())

        if self.pre_ln:
            s[is_atom] = self.pre_ln_s(s[is_atom])
            z = self.pre_ln_z(z)

        q = torch.zeros(list(s.shape[:-1]) + [self.no_heads, self.c_hidden], device=s.device)
        k = torch.zeros(list(s.shape[:-1]) + [self.no_heads, self.c_hidden], device=s.device)
        v = torch.zeros(list(s.shape[:-1]) + [self.no_heads, self.c_hidden], device=s.device)
        q_pts = torch.zeros(list(s.shape[:-1]) + [self.no_heads, self.no_qk_pts, 3], device=s.device)
        k_pts = torch.zeros(list(s.shape[:-1]) + [self.no_heads, self.no_qk_pts, 3], device=s.device)
        v_pts = torch.zeros(list(s.shape[:-1]) + [self.no_heads, self.no_v_pts, 3], device=s.device)

        if do_res_calcs:
            res_s = s[~is_atom, ..., :self.c_s]
            #######################################
            # Generate scalar and point activations
            #######################################
            # [*, N_res, H * C_hidden]
            res_q = self.linear_q(res_s)
            res_kv = self.linear_kv(res_s)

            # [*, N_res, H, C_hidden]
            res_q = res_q.view(res_q.shape[:-1] + (self.no_heads, -1))

            # [*, N_res, H, 2 * C_hidden]
            res_kv = res_kv.view(res_kv.shape[:-1] + (self.no_heads, -1))

            # [*, N_res, H, C_hidden]
            res_k, res_v = torch.split(res_kv, self.c_hidden, dim=-1)

            # [*, N_res, H * P_q * 3]
            res_q_pts = self.linear_q_points(res_s)

            # This is kind of clunky, but it's how the original does it
            # [*, N_res, H * P_q, 3]
            res_q_pts = torch.split(res_q_pts, res_q_pts.shape[-1] // 3, dim=-1)
            res_q_pts = torch.stack(res_q_pts, dim=-1)
            res_q_pts = r[..., None].apply(res_q_pts)

            # [*, N_res, H, P_q, 3]
            res_q_pts = res_q_pts.view(
                res_q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
            )

            # [*, N_res, H * (P_q + P_v) * 3]
            res_kv_pts = self.linear_kv_points(res_s)

            # [*, N_res, H * (P_q + P_v), 3]
            res_v_pts = torch.split(res_kv_pts, res_kv_pts.shape[-1] // 3, dim=-1)
            res_v_pts = torch.stack(res_kv_pts, dim=-1)
            res_v_pts = r[..., None].apply(res_kv_pts)

            # [*, N_res, H, (P_q + P_v), 3]
            res_kv_pts = res_kv_pts.view(res_kv_pts.shape[:-2] + (self.no_heads, -1, 3))

            # [*, N_res, H, P_q/P_v, 3]
            res_k_pts, res_v_pts = torch.split(
                res_kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
            )

            q[~is_atom] = res_q
            k[~is_atom] = res_k
            v[~is_atom] = res_v
            q_pts[~is_atom] = res_q_pts
            k_pts[~is_atom] = res_k_pts
            v_pts[~is_atom] = res_v_pts


        if do_atom_calcs:
            atom_s = s[is_atom, ..., :self.node_irreps.dim]
            atom_q_all = self.tfn_q(atom_s)

            atom_q, atom_q_pts = atom_q_all.split([self.q_irreps.count("0e"), self.q_irreps.count("1o")], dim=-1)
            atom_q = atom_q.view(atom_q.shape[:-1] + (self.no_heads, -1))
            atom_q_pts = atom_q_pts.view(atom_q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))
            atom_q_pts = atom_q_pts + r.get_trans()[..., None, None, :]

            atom_kv_all = self.tfn_kv(atom_s)
            atom_kv, atom_kv_pts = atom_kv_all.split([self.kv_irreps.count("0e"), self.kv_irreps.count("1o")], dim=-1)
            atom_kv = atom_kv.view(atom_kv.shape[:-1] + (self.no_heads, -1))
            atom_k, atom_v = atom_kv.split(self.c_hidden, dim=-1)
            atom_kv_pts = atom_kv_pts.view(atom_kv_pts.shape[:-2] + (self.no_heads, -1, 3))
            atom_kv_pts = atom_kv_pts + r.get_trans()[..., None, None, :]
            atom_k_pts, atom_v_pts = torch.split(
                atom_kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
            )

            q[is_atom] = atom_q
            k[is_atom] = atom_k
            v[is_atom] = atom_v
            q_pts[is_atom] = atom_q_pts
            k_pts[is_atom] = atom_k_pts
            v_pts[is_atom] = atom_v_pts


        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z)

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

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z).to(dtype=a.dtype)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        s_out = torch.zeros_like(s)

        if do_res_calcs:
            o_feats = [
                o,
                *torch.unbind(o_pt, dim=-1),
                o_pt_norm_feats,
                o_pair
            ]
            o_feats = [_item[~is_atom] for _item in o_feats]

            # [*, N_res, C_s]
            res_s_out = self.linear_out(
                torch.cat(
                    o_feats, dim=-1
                ).to(dtype=z.dtype)
            )
            s_out[~is_atom] = res_s_out

        if do_atom_calcs:
            tfn_feats = [
                o,
                o_pt_norm_feats,
                o_pair,
                flatten_final_dims(o_pt, 2)
            ]
            tfn_feats = [_item[is_atom] for _item in tfn_feats]
            tfn_s_out = self.tfn_out(
                torch.cat(
                    tfn_feats, dim=-1
                ).to(dtype=z.dtype)
            )
            s_out[is_atom] = tfn_s_out

        return s