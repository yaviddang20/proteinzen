from proteinzen.model.modules.openfold.layers import Linear, flatten_final_dims, ipa_point_weights_init_
from proteinzen.utils.openfold.rigid_utils import Rigid
from proteinzen.utils.framediff.all_atom import torsion_angles_to_frames, rotate_torsion_angles_on_frames


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pygu


import math
from typing import Optional, Sequence, Tuple


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff, final_init="default"):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = Linear(num_ff, num_hidden, bias=True, init=final_init)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


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
        dropout=0.1,
        edge_dropout=0.2,
        inf: float = 1e5,
        eps: float = 1e-8,
        final_init="default",
        update_edge=True,
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

        self.dropout = nn.Dropout(dropout)
        self.p_dropout = dropout
        self.edge_dropout = edge_dropout

        premsg_dim = (
            2 * c_s  # each node
            + c_z  # edge
            + 8 * no_points  # (1)-(4) point features
            + no_points ** 2 # (5) point features
        )

        self.node_pts = Linear(self.c_s, self.no_points * 3)
        self.node_msg_mlp = nn.Sequential(
            Linear(
                premsg_dim,
                c_hidden
            ),
            nn.ReLU(),
            Linear(c_hidden, c_hidden),
            nn.ReLU(),
            Linear(c_hidden, c_s, init='final'),
        )
        self.node_ln1 = nn.LayerNorm(c_s)
        self.node_ffn = PositionWiseFeedForward(c_s, c_s*4, final_init=final_init)
        self.node_ln2 = nn.LayerNorm(c_s)

        self.update_edge = update_edge

        if update_edge:
            self.edge_pts = Linear(self.c_s, self.no_points * 3)
            self.edge_msg_mlp = nn.Sequential(
                Linear(
                    premsg_dim,
                    c_hidden
                ),
                nn.ReLU(),
                Linear(c_hidden, c_hidden),
                nn.ReLU(),
                Linear(c_hidden, c_z, init='final'),
            )
            self.edge_ln = nn.LayerNorm(c_z)

    def _gen_premessage(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        r: Rigid,
        eps=1e-8,
        edge=False,
    ):
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
        if edge:
            pts_local = self.edge_pts(s)
        else:
            pts_local = self.node_pts(s)
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
        return premessage

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
        eps=1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        dropout_edge_index, dropout_edge_mask = pygu.dropout_edge(edge_index, p=self.edge_dropout, training=self.training)
        dropout_z = z[dropout_edge_mask]

        node_premessage = self._gen_premessage(s, dropout_z, dropout_edge_index, r, edge=False)
        node_msg = self.node_msg_mlp(node_premessage)
        node_update = pygu.scatter(
            node_msg,
            dropout_edge_index[1],
            dim=0,
            dim_size=n_nodes,
            reduce='mean'
        )
        s = self.node_ln1(s + self.dropout(node_update) * mask[..., None])
        s = self.node_ln2(s + self.node_ffn(s) * mask[..., None])

        if self.update_edge:
            edge_premessage = self._gen_premessage(s, z, edge_index, r, edge=True)
            edge_message = self.edge_msg_mlp(edge_premessage)
            z = self.edge_ln(z + self.dropout(edge_message))

        return s, z


class BilevelIPMP(nn.Module):
    """
    An extension of IPMP from PIPPack to multiple frames per residue
    """
    def __init__(
        self,
        c_s,
        c_z,
        c_hidden,
        no_points=8,
        dropout=0.1,
        edge_dropout=0.2,
        inf: float = 1e5,
        eps: float = 1e-8,
        final_init="default",
        update_edge=True,
        design_frames_transition="project",
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
        assert design_frames_transition in ["update", "project"]
        self.design_frames_transition = design_frames_transition
        self.update_torsions = nn.Linear(c_s, 2)

        self.dropout = nn.Dropout(dropout)
        self.p_dropout = dropout
        self.edge_dropout = edge_dropout

        premsg_dim = (
            2 * c_s  # each node
            + c_z  # edge
            + 8 * no_points  # (1)-(4) point features
            + no_points ** 2 # (5) point features
        )

        self.node_pts = Linear(self.c_s, self.no_points * 3)
        self.node_msg_mlp = nn.Sequential(
            Linear(
                premsg_dim,
                c_hidden
            ),
            nn.ReLU(),
            Linear(c_hidden, c_hidden),
            nn.ReLU(),
            Linear(c_hidden, c_s, init='final'),
        )
        self.node_ln1 = nn.LayerNorm(c_s)
        self.node_ffn = PositionWiseFeedForward(c_s*5, c_s, final_init=final_init)
        self.node_ln2 = nn.LayerNorm(c_s)

        self.update_edge = update_edge

        if update_edge:
            self.edge_pts = Linear(self.c_s, self.no_points * 3)
            premsg_dim = (
                2 * c_s  # each node
                + 8 * no_points  # (1)-(4) point features
                + no_points ** 2 # (5) point features
            )
            self.edge_msg_mlp = nn.Sequential(
                Linear(
                    premsg_dim * 25 + c_z,
                    c_hidden
                ),
                nn.ReLU(),
                Linear(c_hidden, c_hidden),
                nn.ReLU(),
                Linear(c_hidden, c_z, init='final'),
            )
            self.edge_ln = nn.LayerNorm(c_z)

    def _update_sidechain_frames(
        self,
        s: torch.Tensor,
        r: Rigid,
        design_mask: torch.Tensor
    ):
        """ Update the design frames based on the update mode of the module

        Args:
            s (torch.Tensor): [N_res, 5, c_s]
            r (Rigid): [N_res, 5]
            design_mask (torch.Tensor): [N_res, 5]
        """
        if self.design_frames_transition == "update":
            raise NotImplementedError
            new_torsions = self.update_torsions(s)
            new_torsions = F.normalize(new_torsions.view(-1, 5, 2), dim=-1)
            new_rigids = rotate_torsion_angles_on_frames(
                r,
                new_torsions[:, 1:],
            )

            updated_rigids = (
                r.to_tensor_4x4() * (~design_mask)[..., None, None, None]
                + new_rigids.to_tensor_4x4() * design_mask[..., None, None, None]
            )
            updated_rigids = Rigid.from_tensor_4x4(updated_rigids)
            return updated_rigids

        elif self.design_frames_transition == "project":
            new_torsions = self.update_torsions(s)
            new_torsions = F.normalize(new_torsions.view(-1, 5, 2), dim=-1)
            bb_rigid = r[..., 0]
            # lys
            fake_seq = torch.ones(bb_rigid.shape).long() * 11
            fake_torsions = torch.tensor([[[0., 1.]]], device=new_torsions.device).expand(
                new_torsions.shape[0],
                2,
                -1)
            new_rigids = torsion_angles_to_frames(
                bb_rigid,
                torch.cat([fake_torsions, new_torsions], dim=-2),
                fake_seq
            )
            new_rigids = Rigid.cat(
                [bb_rigid[:, None], new_rigids[:, 4:]],
                dim=-1
            )

            updated_rigids = (
                r.to_tensor_4x4() * (~design_mask)[..., None, None, None]
                + new_rigids.to_tensor_4x4() * design_mask[..., None, None, None]
            )
            updated_rigids = Rigid.from_tensor_4x4(updated_rigids)
            return updated_rigids


    def _gen_premessage(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        r: Rigid,
        rigid_mask: torch.Tensor,
        eps=1e-8,
        edge=False,
    ):
        """_summary_

        Args:
            s (torch.Tensor): [N_res x 5 x c_s]
            z (torch.Tensor): [N_edge x c_z]
            edge_index (torch.Tensor): [2 x N_edge]
            r (Rigid): [N_res x 5]
            eps (_type_, optional): _description_. Defaults to 1e-8.
            edge (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        n_nodes = s.shape[0]
        n_edges = z.shape[0]
        src = edge_index[1]
        dst = edge_index[0]

        s_src = s[src]
        s_dst = s[dst]
        s_src_expand = s_src[:, :, None].expand(-1, -1, 5, -1)
        s_dst_expand = s_dst[:, None].expand(-1, 5, -1, -1)
        z_expand = z[:, None, None].expand(-1, 5, 5, -1)
        cross_rigid_mask = rigid_mask[src, :, None] & rigid_mask[dst, None]


        r = r.scale_translation(1/10)
        src_rigids = r[src]  # N_edge x 4

        #######################################
        # Generate scalar and point activations
        #######################################
        # [N_edge, 4, C_pts, 3]
        # 1. Node i's local points
        if edge:
            pts_local = self.edge_pts(s)
        else:
            pts_local = self.node_pts(s)
        pts_local = pts_local.view(n_nodes, 5, self.no_points, 3)
        pts_local_src = pts_local[src]
        pts_local_src_expand = pts_local_src.view(-1, 5, 1, self.no_points * 3).expand(
            -1, -1, 5, -1
        )

        # [N_edge, 4, C_pts]
        # 2. Distance between node i's local points and its CA
        pts_norm = torch.linalg.vector_norm(pts_local_src + eps, dim=-1)
        pts_norm_expand = pts_norm[:, :, None].expand(-1, -1, 5, -1)

        # [N_edge, 4, 4, c_pts, 3]
        # 3. Node j's points in i's local frame
        pts_global = r[..., None].apply(pts_local)  # N_res x 4 x c_pts x 3
        pts_global_dst = pts_global[dst]  # N_edge x 4 x c_pts x 3
        pts_dst_in_src_local = src_rigids[..., None, None].invert_apply(pts_global_dst[:, None])

        # [N_edge, 4, 4, c_pts]
        # 4. Distance between node j's points in i's local frame and i's CA
        pts_dst_in_src_local_norm = torch.linalg.vector_norm(pts_dst_in_src_local + eps, dim=-1)

        # [N_edge, 4, 4, c_pts ** 2]
        # 5. Distance between node i's global points and node j's global points
        pts_global_src = pts_global[src]
        pts_rel_dist = torch.linalg.norm(
            pts_global_src[..., None, :, None, :] - pts_global_dst[..., None, :, None, :, :] + eps,
            dim=-1
        ).view(n_edges, 5, 5, -1)
        # their code suggests this but this doesn't make sense to me
        # pts_rel_dist = torch.linalg.norm(pts_global_src - pts_global_dst + eps, dim=-1)

        # [N_edge, 4, 4, -1]

        if edge:
            premessage = torch.cat([
                s_src_expand,
                s_dst_expand,
                pts_local_src_expand,
                pts_norm_expand,
                pts_dst_in_src_local.view(-1, 5, 5, self.no_points*3),
                pts_dst_in_src_local_norm,
                pts_rel_dist
            ], dim=-1)
            premessage = premessage * cross_rigid_mask[..., None]
            premessage = premessage.view(n_edges, -1)
            premessage = torch.cat([premessage, z], dim=-1)
        else:
            premessage = torch.cat([
                s_src_expand,
                s_dst_expand,
                z_expand,
                pts_local_src_expand,
                pts_norm_expand,
                pts_dst_in_src_local.view(-1, 5, 5, self.no_points*3),
                pts_dst_in_src_local_norm,
                pts_rel_dist
            ], dim=-1)
            premessage = premessage * cross_rigid_mask[..., None]

        return premessage


    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
        rigid_mask: torch.Tensor,
        design_targets: torch.Tensor,
        eps=1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [N_res, 5, C_s] single representation, scalars per frame
            z:
                [N_edge, C_e] pair representation, 4x4 for frames
            edge_index:
                [2, N_edge] edge index
            r:
                [N_res, 5] sidechain frames
            mask:
                [N_res, 5] frame mask
            design_targets:
                [N_res] design mask
        Returns:
            [N_res, 5, C_s] single scalar representation update
        """
        n_nodes = s.shape[0]
        r = self._update_sidechain_frames(s, r, design_mask=design_targets)

        dropout_edge_index, dropout_edge_mask = pygu.dropout_edge(edge_index, p=self.edge_dropout, training=self.training)
        dropout_z = z[dropout_edge_mask]

        node_premessage = self._gen_premessage(s, dropout_z, dropout_edge_index, r, edge=False, rigid_mask=rigid_mask)
        node_msg = self.node_msg_mlp(node_premessage)
        node_msg = node_msg.sum(dim=2)
        node_update = pygu.scatter(
            node_msg,
            dropout_edge_index[1],
            dim=0,
            dim_size=n_nodes,
            reduce='mean'
        )
        s = self.node_ln1(s + self.dropout(node_update) * mask[..., None, None])
        node_update = self.node_ffn(s.view(n_nodes, -1))
        node_update = node_update.view(n_nodes, 5, -1)
        s = self.node_ln2(s + node_update * mask[..., None, None])

        if self.update_edge:
            edge_premessage = self._gen_premessage(s, z, edge_index, r, edge=True, rigid_mask=rigid_mask)
            edge_message = self.edge_msg_mlp(edge_premessage)
            z = self.edge_ln(z + self.dropout(edge_message))


        return s, z, r

