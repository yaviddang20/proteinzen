import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_cluster import knn, knn_graph, radius_graph
from torch_scatter import scatter_mean
import torch_geometric.utils as pygu

from e3nn import o3
from e3nn.nn import NormActivation, Gate, BatchNorm

from proteinzen.model.modules.common import GaussianRandomFourierBasis
from proteinzen.model.modules.layers.edge.tfn import FasterTensorProduct
from proteinzen.data.datasets.featurize.common import _edge_positional_embeddings, _rbf
from proteinzen.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from proteinzen.model.modules.openfold.layers import Linear, flatten_final_dims, ipa_point_weights_init_
from proteinzen.model.modules.layers.node.tfn import separate_heads_dim, fuse_heads_dim
from proteinzen.utils.openfold.rigid_utils import Rigid


class TFN2FrameCrossIPA(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_hidden,
        no_heads,
        no_qk_points,
        no_v_points,
        feat_irreps,
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
        self.k_irreps = o3.Irreps(f"{hc}x0e+{no_qk_points}x1o")
        self.v_irreps = o3.Irreps(f"{hc}x0e+{no_v_points}x1o")
        self.linear_q = Linear(self.c_s, hc)
        self.linear_k = o3.Linear(feat_irreps, self.k_irreps)
        self.linear_v = o3.Linear(feat_irreps, self.v_irreps)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

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

    def forward(
        self,
        frame_s: torch.Tensor,
        tfn_t: torch.Tensor,
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
        n_nodes = frame_s.shape[0]

        src = edge_index[1]
        dst = edge_index[0]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [N_res, H * C_hidden]
        q = self.linear_q(frame_s)
        tfn_t_k = self.linear_k(tfn_t)
        tfn_t_v = self.linear_k(tfn_t)

        # [N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        # [N_edge, H, C_hidden]
        q_src = q[src]

        # [N_res, H, 2 * C_hidden]
        k = tfn_t_k[..., :self.c_hidden * self.no_heads].view(-1, self.no_heads, self.c_hidden)
        v = tfn_t_v[..., :self.c_hidden * self.no_heads].view(-1, self.no_heads, self.c_hidden)

        # [N_res, H * P_q * 3]
        q_pts = self.linear_q_points(frame_s)

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
        k_pts = tfn_t_k[..., self.c_hidden * self.no_heads:].split([3 for _ in range(self.no_k_points)], dim=-1)
        k_pts = torch.stack(k_pts, dim=-1)
        k_pts = k_pts.view(k_pts.shape[:-2] + (self.no_heads, -1, 3))
        v_pts = tfn_t_v[..., self.c_hidden * self.no_heads:].split([3 for _ in range(self.no_v_points)], dim=-1)
        v_pts = torch.stack(v_pts, dim=-1)
        v_pts = v_pts.view(v_pts.shape[:-2] + (self.no_heads, -1, 3))

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


class Frame2TFNCrossAttentionUpdate(nn.Module):
    def __init__(self,
                 c_s,
                 feat_irreps,
                 sh_irreps,
                 h_edge,
                 qk_irreps,
                 v_irreps,
                 num_heads=4):
        super().__init__()
        self.c_s = c_s
        self.feat_irreps = feat_irreps
        self.sh_irreps = sh_irreps
        self.num_heads = num_heads

        self.qk_irreps = qk_irreps
        self.v_irreps = v_irreps

        # TODO
        # this is a real weird way of doing this
        # but im trying to avoid some of the weird quirks for when we have irreps
        # which are neither scalars nor vectors
        self.frame_0e_lin = nn.Linear(c_s, self.feat_irreps.count("0e") * num_heads)
        self.frame_1o_lin = nn.Linear(c_s, self.feat_irreps.count("1o") * num_heads * 3)
        frame_irreps = o3.Irreps(
            f"{self.feat_irreps.count('0e')*num_heads}x0e+{self.feat_irreps.count('1o')*num_heads}x1o"
        )
        self.gen_frame_irreps = o3.Linear(frame_irreps, feat_irreps)

        self.q_lin = o3.Linear(feat_irreps, qk_irreps)

        self.k_tp = FasterTensorProduct(
            feat_irreps,
            sh_irreps,
            self.qk_irreps * num_heads,
            shared_weights=False
        )
        self.v_tp = FasterTensorProduct(
            feat_irreps,
            sh_irreps,
            self.v_irreps * num_heads,
            shared_weights=False
        )

        self.k_fc = nn.Sequential(
            nn.Linear(h_edge, h_edge),
            nn.ReLU(),
            nn.Linear(h_edge, self.kv_tp.weight_numel)
        )
        self.k_fc = nn.Sequential(
            nn.Linear(h_edge, h_edge),
            nn.ReLU(),
            nn.Linear(h_edge, self.kv_tp.weight_numel)
        )
        self.lin_self = o3.Linear(feat_irreps, feat_irreps)
        self.lin_out = o3.Linear(v_irreps * num_heads, feat_irreps)
        self.norm = BatchNorm(feat_irreps)

    def forward(self,
                frame_features: torch.Tensor,
                rigids: Rigid,
                tfn_features: torch.Tensor,
                edge_features: torch.Tensor,
                edge_sh,
                edge_index: torch.Tensor):
        edge_dst, edge_src = edge_index

        frame_0e = self.q_0e_lin(frame_features)
        # produce vectors, rotate them, then flatten again
        frame_1o = self.q_1o_lin(frame_features)
        frame_1o = frame_1o.view(frame_1o.shape[:-1] + [-1, 3])
        frame_1o = rigids[..., None].apply(frame_1o)
        frame_1o = frame_1o.view(frame_1o.shape[:-2] + [-1])
        frame_irrep_features = torch.cat([frame_0e, frame_1o], dim=-1)
        frame_irrep_features = self.gen_frame_irreps(frame_irrep_features)

        q = self.q_lin(tfn_features)
        q = separate_heads_dim(self.qk_irreps, q, self.num_heads)
        k = self.tp(
            frame_irrep_features[edge_dst],
            edge_sh,
            self.k_fc(edge_features)
        )
        k = separate_heads_dim(self.qk_irreps, k, self.num_heads)
        v = self.tp(
            frame_irrep_features[edge_dst],
            edge_sh,
            self.v_fc(edge_features)
        )
        v = separate_heads_dim(self.v_irreps, v, self.num_heads)

        attn = torch.einsum("...j,...j->...", q, k)
        attn = pygu.softmax(attn, edge_src, dim=0)

        update = pygu.scatter(
            v * attn[..., None],
            edge_src,
            dim=0,
            dim_size=tfn_features.shape[0]
        )
        update = fuse_heads_dim(self.v_irreps, update, self.num_heads)
        update = self.lin_out(update)
        self_update = self.lin_self(tfn_features)

        return self.norm(self_update + update)