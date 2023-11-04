from ligbinddiff.model.modules.openfold.frames import Linear, flatten_final_dims, ipa_point_weights_init_
from ligbinddiff.utils.openfold.rigid_utils import Rigid


import torch
import torch.nn as nn
import torch_geometric.utils as pygu


import math
from typing import Optional, Sequence, Tuple


class IPMP(nn.Module):
    """
    A (close) implementation of IPMP from PIPPack
    """
    def __init__(
        self,
        c_s,
        c_z,
        c_hidden,
        no_points=8,
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
        self.no_points = no_points
        self.inf = inf
        self.eps = eps

        self.linear_pts = Linear(self.c_s, self.no_points * 3)
        self.ln = nn.LayerNorm(c_s)
        self.mlp = nn.Sequential(
            Linear(2 * c_s + c_z + 8 * no_points + no_points ** 2, c_hidden),
            nn.ReLU(),
            Linear(c_hidden, c_hidden),
            nn.ReLU(),
            Linear(c_hidden, c_s, init='final'),
        )


    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
        eps=1e-8
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
        n_edges = z.shape[0]
        src = edge_index[1]
        dst = edge_index[0]

        s_src = s[src]
        s_dst = s[dst]
        r = r.scale_translation(1/10)
        src_rigids = r[src]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [N_res, C_pts, 3]
        # 1. Node i's local points
        pts_local = self.linear_pts(s)
        pts_local = pts_local.view(n_nodes, -1, 3)
        pts_local_src = pts_local[src]

        # 2. Distance between node i's local points and its CA
        pts_norm = torch.linalg.vector_norm(pts_local_src + eps, dim=-1)

        # 3. Node j's points in i's local frame
        pts_global = r[..., None].apply(pts_local)
        pts_global_dst = pts_global[dst]
        pts_dst_in_src_local = src_rigids[..., None].invert_apply(pts_global_dst)

        # 4. Distance between node j's points in i's local frame and i's CA
        pts_dst_in_src_local_norm = torch.linalg.vector_norm(pts_dst_in_src_local + eps, dim=-1)

        # 5. Distance between node i's global points and node j's global points
        pts_global_src = pts_global[src]
        pts_rel_dist = torch.linalg.norm(
            pts_global_src[..., None, :] - pts_global_dst[..., None, :, :] + eps,
            dim=-1
        ).view(n_edges, -1)
        # their code suggests this but this doesn't make sense to me
        # pts_rel_dist = torch.linalg.norm(pts_global_src - pts_global_dst + eps, dim=-1)

        premessage = torch.cat([
            s_src,
            s_dst,
            z,
            pts_local_src.view(n_edges, -1),
            pts_norm,
            pts_dst_in_src_local.view(n_edges, -1),
            pts_dst_in_src_local_norm,
            pts_rel_dist
        ], dim=-1)

        msg = self.mlp(premessage)
        update = pygu.scatter(
            msg,
            edge_index[1],
            dim=0,
            dim_size=n_nodes,
            reduce='mean'
        )

        return update


class VectorBiasIPMP(nn.Module):
    """
    Modified version of IPMP from PIPPack
    """
    def __init__(
        self,
        c_s,
        c_v,
        c_z,
        c_hidden,
        no_s_points = 8,
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
        self.no_s_points = no_s_points
        self.inf = inf
        self.eps = eps
        self.D_points = D_points
        self.gen_vectors = gen_vectors

        self.gen_s_points = Linear(self.c_s, self.no_s_points * 3)
        self.linear_s_pts = Linear(self.c_v + self.no_s_points, self.c_v, bias=False)
        self.linear_c_v = Linear(self.c_v + self.no_s_points, self.c_v, bias=False)

        self.s_mlp = nn.Sequential(
            Linear(2 * c_s + c_z + 9 * c_v, )
        )

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
        t = r.get_trans()

        # [N_res, H * P_q * 3]
        s_pts = self.linear_v_points(s_s)

        # This is kind of clunky, but it's how the original does it
        # [N_res, H * P_q, 3]
        s_pts = torch.split(s_pts, s_pts.shape[-1] // 3, dim=-1)
        s_pts = torch.stack(s_pts, dim=-1)
        s_pts = r[..., None].get_rots().apply(s_pts)

        if s_v is None:
            s_v = torch.zeros(
                list(s_s.shape[:-1])
                + [self.c_v, 3],
                device=s_s.device
            )

        # [N_res, P_s + C_v, 3]
        vecs = torch.cat([
            s_v, s_pts
        ], dim=-2)

        vecs = self.linear_comb_vecs(vecs.transpose(-1, -2)) + t[..., None]/self.D_points
        # [N_res, C_v, 3]
        # vecs in global frame
        vecs = vecs.transpose(-1, -2)


        return out_s_s, out_s_v
