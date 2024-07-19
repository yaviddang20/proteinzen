import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric.utils as pygu
from torch_geometric.nn import radius_graph

from proteinzen.data.datasets.featurize.common import _rbf
from proteinzen.data.datasets.featurize.sidechain import _ideal_virtual_Cb
from proteinzen.model.modules.openfold.layers import Linear


# def cross_sidechain_graph(atom14_mask,
#                           res_edge_features,
#                           res_edge_index,
#                           edge_subset_select):
#     device = atom14_mask.device
#     # reduce the number of edges
#     res_edge_subset_features = res_edge_features[edge_subset_select]
#     res_edge_subset_index = res_edge_index[:, edge_subset_select]
#     # add self-edges to allow for attention within blocks
#     res_edge_subset_index, self_edge_mask = pygu.add_remaining_self_loops(
#         res_edge_subset_index,
#         torch.ones_like(res_edge_subset_index[1]),
#         fill_value=0.
#     )
#     res_edge_subset_self_edge_mask = self_edge_mask.bool()
#
#     store = torch.zeros(self_edge_mask.shape[0], res_edge_subset_features.shape[-1], device=device)
#     store[res_edge_subset_self_edge_mask] = res_edge_subset_features
#     res_edge_subset_features = store
#
#     num_atoms = torch.sum(atom14_mask)
#     atom_idx = torch.arange(num_atoms, device=device)
#     atom14_idx = torch.zeros_like(atom14_mask).long()
#     atom14_idx[atom14_mask] = atom_idx
#     # res_idx = torch.arange(atom14_mask.shape[0], device=device)[..., None].expand(-1, 14)
#     # atom_res_idx = res_idx[atom14_mask]
#
#     src = res_edge_subset_index[1]
#     dst = res_edge_subset_index[0]
#
#     atom14_src = atom14_idx[src][..., None].expand(-1, -1, 14)
#     atom14_dst = atom14_idx[dst][..., None, :].expand(-1, 14, -1)
#     cross_atom14 = torch.stack([
#         atom14_dst, atom14_src
#     ], dim=-1)
#     atom14_src_mask = atom14_mask[src][..., None].expand(-1, -1, 14)
#     atom14_dst_mask = atom14_mask[dst][..., None, :].expand(-1, 14, -1)
#     cross_atom14_mask = atom14_src_mask & atom14_dst_mask
#     cross_atom14_edges = cross_atom14[cross_atom14_mask].transpose(-1, -2)
#
#     atom14_ca_idx = torch.zeros_like(atom14_mask).bool()
#     atom14_ca_idx[:, 1] = True
#     ca_select = atom14_ca_idx[atom14_mask]
#     # print(ca_select.sum())
#     # print(atom14_mask[:, 1].sum())
#
#     res_edge_subset_id = torch.arange(res_edge_subset_index.shape[1], device=device)
#     atom_edge_to_res_edge_subset = res_edge_subset_id[..., None, None].expand(-1, 14, 14)[cross_atom14_mask]
#     res_edge_id = torch.arange(res_edge_index.shape[1], device=device)
#     res_edge_subset_to_res_edge = res_edge_id[edge_subset_select]
#
#     return (
#         ca_select,
#         cross_atom14_edges,
#         atom_edge_to_res_edge_subset,
#         res_edge_subset_to_res_edge,
#         res_edge_subset_features,
#         res_edge_subset_index,
#         res_edge_subset_self_edge_mask
#     )


def full_cross_sidechain_graph(atom14_mask,
                          res_edge_features,
                          res_edge_index):
                          #edge_subset_select):
    device = atom14_mask.device
    # add self-edges to allow for attention within blocks
    res_edge_index, res_edge_features = pygu.add_remaining_self_loops(
        res_edge_index,
        res_edge_features,
        fill_value=0.
    )

    num_atoms = torch.sum(atom14_mask)
    atom_idx = torch.arange(num_atoms, device=device)
    atom14_idx = torch.zeros_like(atom14_mask).long()
    atom14_idx[atom14_mask] = atom_idx
    # res_idx = torch.arange(atom14_mask.shape[0], device=device)[..., None].expand(-1, 14)
    # atom_res_idx = res_idx[atom14_mask]

    src = res_edge_index[1]
    dst = res_edge_index[0]

    atom14_src = atom14_idx[src][..., None].expand(-1, -1, 14)
    atom14_dst = atom14_idx[dst][..., None, :].expand(-1, 14, -1)
    cross_atom14 = torch.stack([
        atom14_dst, atom14_src
    ], dim=-1)
    atom14_src_mask = atom14_mask[src][..., None].expand(-1, -1, 14)
    atom14_dst_mask = atom14_mask[dst][..., None, :].expand(-1, 14, -1)
    cross_atom14_mask = atom14_src_mask & atom14_dst_mask
    cross_atom14_edges = cross_atom14[cross_atom14_mask].transpose(-1, -2)

    atom14_ca_idx = torch.zeros_like(atom14_mask).bool()
    atom14_ca_idx[:, 1] = True
    atom14_ca_idx = atom14_ca_idx & atom14_mask
    ca_select = atom14_ca_idx[atom14_mask]
    # print(ca_select.sum())
    # print(atom14_mask[:, 1].sum())

    res_edge_id = torch.arange(res_edge_index.shape[1], device=device)
    atom_edge_to_res_edge = res_edge_id[..., None, None].expand(-1, 14, 14)[cross_atom14_mask]

    return (
        ca_select,
        cross_atom14_edges,
        atom_edge_to_res_edge,
        res_edge_features,
        res_edge_index
    )


class BilevelGraphAttention(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_atom,
                 num_heads=4,
                 num_rbf_atom=16,
                 inf=1e5):
        super().__init__()
        self.num_heads = num_heads
        self.num_rbf_atom = num_rbf_atom
        self.atoms_per_res=14
        self.c_s = c_s
        self.c_z = c_z
        self.c_atom = c_atom
        self.inf = inf

        self.atom_dist_embed = nn.Sequential(
            nn.Linear(num_rbf_atom, num_rbf_atom),
            nn.ReLU(),
            nn.Linear(num_rbf_atom, num_rbf_atom),
            nn.ReLU(),
            nn.Linear(num_rbf_atom, num_heads),
        )
        self.res_level_bias = nn.Sequential(
            nn.Linear(2*c_s + c_z, c_atom),
            nn.ReLU(),
            nn.Linear(c_atom, num_heads),
        )
        self.atom_q = nn.Linear(c_atom, c_atom * num_heads)
        self.atom_kv = nn.Linear(c_atom, 2 * c_atom * num_heads)
        # self.atom_out = Linear(c_atom * num_heads, c_atom, init='final')
        self.atom_out = Linear(c_atom * num_heads, c_atom)

        # self.res_q = nn.Linear(c_s, c_s * num_heads)
        # self.res_kv = nn.Linear(c_s, 2 * c_s * num_heads)
        # self.res_out = Linear(c_s * num_heads, c_s, init='final')
        # self.res_out = Linear(c_s * num_heads, c_s)
        # self.edge_bias = nn.Linear(c_z, num_heads)

    def forward(self,
                node_features,
                res_edge_features,
                res_edge_index,
                atom_features,
                atom_coords,
                cross_atom_edge_index,
                atom_edge_to_res_edge,
                eps=1e-8):
        # atom attn
        src = cross_atom_edge_index[1]
        dst = cross_atom_edge_index[0]

        # dist bias
        atom_dist_vecs = atom_coords[dst] - atom_coords[src]
        atom_dists = torch.linalg.vector_norm(atom_dist_vecs + eps, dim=-1)
        atom_dist_rbf = _rbf(atom_dists, D_count=self.num_rbf_atom, device=atom_dists.device)
        atom_dist_bias = self.atom_dist_embed(atom_dist_rbf)
        # res level bias
        res_level_bias = self.res_level_bias(
            torch.cat([
                res_edge_features,
                node_features[res_edge_index[0]],
                node_features[res_edge_index[1]]
            ], dim=-1)
        )

        atom_q = self.atom_q(atom_features)[src].view(-1, self.num_heads, self.c_atom)
        atom_kv = self.atom_kv(atom_features)[dst].view(-1, self.num_heads, 2*self.c_atom)
        atom_k, atom_v = atom_kv.split([self.c_atom, self.c_atom], dim=-1)

        # Eu x h x 14 x 14
        atom_R = torch.einsum("...j,...j->...", atom_q, atom_k)
        atom_R = 1/np.sqrt(self.c_atom) * atom_R
        # atom_R = atom_R + res_level_bias[atom_edge_to_res_edge]
        # TODO: num_points=8, this is hardcoded
        # atom_R = atom_R - 1/np.sqrt(2 * 9 * 8) * atom_dists[..., None]
        atom_R = atom_R + res_level_bias[atom_edge_to_res_edge] + atom_dist_bias

        alpha = pygu.softmax(
            atom_R,
            src,
            dim=0
        )

        block_r = pygu.scatter(
            atom_R,
            atom_edge_to_res_edge,
            dim=0,
            dim_size=res_edge_features.shape[0],
            reduce="mean"
        )

        beta = pygu.softmax(
            block_r,
            res_edge_index[1],
            dim=0,
        )

        beta_per_atom_edge = beta[atom_edge_to_res_edge]

        atom_update = pygu.scatter(
            alpha[..., None] * beta_per_atom_edge[..., None] * atom_v,
            src,
            dim=0,
            dim_size=atom_features.shape[0]
        )  # Eu x h x c_atom
        atom_update = self.atom_out(atom_update.view(-1, self.c_atom * self.num_heads))

        return atom_update#, res_update

    # def forward(self,
    #             node_features,
    #             res_edge_features,
    #             res_edge_index,
    #             res_edge_subset_features,
    #             res_edge_subset_index,
    #             atom_features,
    #             atom_coords,
    #             cross_atom_edge_index,
    #             atom_edge_to_res_edge_subset,
    #             res_edge_subset_to_res_edge,
    #             res_edge_subset_self_edge_mask,
    #             eps=1e-8):
    #     # atom attn
    #     src = cross_atom_edge_index[1]
    #     dst = cross_atom_edge_index[0]

    #     # dist bias
    #     atom_dist_vecs = atom_coords[dst] - atom_coords[src]
    #     atom_dists = torch.linalg.vector_norm(atom_dist_vecs + eps, dim=-1)
    #     atom_dist_rbf = _rbf(atom_dists, D_count=self.num_rbf_atom, device=atom_dists.device)
    #     atom_dist_bias = self.atom_dist_embed(atom_dist_rbf)
    #     # res level bias
    #     res_level_bias = self.res_level_bias(
    #         torch.cat([
    #             res_edge_subset_features,
    #             node_features[res_edge_subset_index[0]],
    #             node_features[res_edge_subset_index[1]]
    #         ], dim=-1)
    #     )

    #     atom_q = self.atom_q(atom_features)[src].view(-1, self.num_heads, self.c_atom)
    #     atom_kv = self.atom_kv(atom_features)[dst].view(-1, self.num_heads, 2*self.c_atom)
    #     atom_k, atom_v = atom_kv.split([self.c_atom, self.c_atom], dim=-1)

    #     # Eu x h x 14 x 14
    #     atom_R = torch.einsum("...j,...j->...", atom_q, atom_k)
    #     atom_R = 1/np.sqrt(self.c_atom) * atom_R
    #     atom_R = atom_R + atom_dist_bias + res_level_bias[atom_edge_to_res_edge_subset]
    #     alpha = pygu.softmax(
    #         atom_R,
    #         src,
    #         dim=0
    #     )

    #     block_r = pygu.scatter(
    #         atom_R,
    #         atom_edge_to_res_edge_subset,
    #         dim=0,
    #         dim_size=res_edge_subset_features.shape[0],
    #         reduce="mean"
    #     )

    #     beta = pygu.softmax(
    #         block_r,
    #         res_edge_subset_index[1],
    #         dim=0,
    #         # num_nodes=node_features.shape[0]
    #     )

    #     beta_per_atom_edge = beta[atom_edge_to_res_edge_subset]

    #     atom_update = pygu.scatter(
    #         alpha[..., None] * beta_per_atom_edge[..., None] * atom_v,
    #         src,
    #         dim=0,
    #         dim_size=atom_features.shape[0]
    #     )  # Eu x h x c_atom
    #     atom_update = self.atom_out(atom_update.view(-1, self.c_atom * self.num_heads))

    #     # # res attn
    #     # res_src = res_edge_index[1]
    #     # res_dst = res_edge_index[0]

    #     # block_r_no_self_edge = block_r[res_edge_subset_self_edge_mask]
    #     # res_level_block_r = pygu.scatter(
    #     #     block_r_no_self_edge,
    #     #     res_edge_subset_to_res_edge,
    #     #     dim=0,
    #     #     dim_size=res_edge_index.shape[-1]
    #     # )

    #     # res_q = self.res_q(node_features).view(-1, self.num_heads, self.c_s)
    #     # res_kv = self.res_kv(node_features).view(-1, self.num_heads, 2*self.c_s)
    #     # res_k, res_v = res_kv.split([self.c_s, self.c_s], dim=-1)
    #     # edge_bias = self.edge_bias(res_edge_features)

    #     # res_attn = torch.einsum("...j,...j->...", res_q[res_src], res_k[res_dst])
    #     # res_attn = 1/np.sqrt(self.c_s) * res_attn
    #     # res_attn = res_attn + res_level_block_r + edge_bias
    #     # res_attn = pygu.softmax(
    #     #     res_attn,
    #     #     res_src,
    #     #     dim=0
    #     # )

    #     # res_update = pygu.scatter(
    #     #     res_attn[..., None] * res_v[res_dst],
    #     #     res_src,
    #     #     dim=0,
    #     #     dim_size=node_features.shape[0]
    #     # )
    #     # res_update = self.res_out(res_update.view(-1, self.c_s * self.num_heads))

    #     return atom_update#, res_update


class BilevelInvariantPointGraphAttention(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_atom,
                 num_points=4,
                 num_heads=4,
                 num_rbf_atom=16,
                ):
        super().__init__()
        self.num_points = num_points
        self.c_atom = c_atom

        self.gen_point_features = nn.Linear(
            c_s, c_atom * num_points
        )
        # self.gen_point_features = nn.Sequential(
        #     nn.Linear(c_s + c_atom * num_points, c_s),
        #     nn.ReLU(),
        #     nn.Linear(c_s, c_s),
        #     nn.ReLU(),
        #     nn.Linear(c_s, c_atom * num_points)
        # )
        # self.gen_point_coords = nn.Linear(c_s + c_atom * num_points, num_points*3)
        self.gen_point_coords = nn.Linear(c_s, num_points*3)

        # self.update_atoms = nn.Linear(
        #     c_s+c_atom, c_atom
        # )
        # self.atom_update_ln = nn.LayerNorm(c_atom)

        self.bilevel_attn = BilevelGraphAttention(
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            num_heads=num_heads,
            num_rbf_atom=num_rbf_atom)
        self.atom_ln = nn.LayerNorm(c_atom)

        # self.collect_invariant_points_info = nn.Linear(
        #     c_atom * num_points  # features
        #     + num_points * 3   # vectors
        #     + num_points  # vector mags
        #     ,
        #     c_s
        # )
        # self.fixed_res_lin = nn.Linear(c_s, c_s)

        self.atom_to_node = nn.Linear(c_atom, c_s)
        self.node_ln = nn.LayerNorm(c_s)


    def forward(self,
                node_features,
                rigids,
                edge_features,
                edge_index,
                atom14_features,
                atom14_coords,
                atom14_mask,
                mgm_mask,
                res_mask,
                eps=1e-8):

        # masked_res_point_inputs = torch.cat(
        #     [node_features, atom14_features[:, 5:5+self.num_points].view(node_features.shape[0], -1)],
        #     dim=-1
        # )
        # masked_res_point_features = self.gen_point_features(masked_res_point_inputs).view(
        #     -1, self.num_points, self.c_atom
        # )
        # masked_res_point_coords = self.gen_point_coords(masked_res_point_inputs).view(
        #     -1, self.num_points, 3
        # )
        masked_res_point_features = self.gen_point_features(node_features).view(
            -1, self.num_points, self.c_atom
        )
        masked_res_point_coords = self.gen_point_coords(node_features).view(
            -1, self.num_points, 3
        )
        masked_res_point_coords = rigids[..., None].apply(masked_res_point_coords)

        # atom14_update = self.update_atoms(
        #     torch.cat([
        #         atom14_features,
        #         node_features[..., None, :].expand(-1, 14, -1)
        #     ], dim=-1)
        # )
        # atom14_mgm_features = self.atom_update_ln(atom14_features + atom14_update * atom14_mask[..., None])
        atom14_mgm_features = atom14_features.clone()
        atom14_mgm_coords = atom14_coords.clone()

        # X_cb = _ideal_virtual_Cb(atom14_mgm_coords)
        # atom14_mgm_coords[mgm_mask, 4] = X_cb

        # clear any non-backbone points info
        atom14_mgm_features[mgm_mask, 4:] = 0.
        atom14_mgm_coords[mgm_mask, 4:] = 0.
        atom14_mgm_features[mgm_mask, 4:4+self.num_points] = masked_res_point_features[mgm_mask]
        atom14_mgm_coords[mgm_mask, 4:4+self.num_points] = masked_res_point_coords[mgm_mask]

        atom14_mgm_mask = atom14_mask.clone()
        atom14_mgm_mask[mgm_mask, 4:4+self.num_points] = True
        atom14_mgm_mask[mgm_mask, 4+self.num_points:] = False
        atom14_mgm_mask = atom14_mgm_mask * res_mask[..., None]

        # (
        #     ca_select,
        #     cross_atom14_edge_index,
        #     atom_edge_to_res_edge_subset,
        #     res_edge_subset_to_res_edge,
        #     res_edge_subset_features,
        #     res_edge_subset_index,
        #     res_edge_subset_self_edge_mask
        # ) = cross_sidechain_graph(
        #     atom14_mgm_mask,
        #     edge_features,
        #     edge_index,
        #     edge_subset_select
        # )
        (
            ca_select,
            cross_atom14_edge_index,
            atom_edge_to_res_edge,
            res_edge_features,
            res_edge_index
        ) = full_cross_sidechain_graph(
            atom14_mgm_mask,
            edge_features,
            edge_index,
        )
        atom_features = atom14_mgm_features[atom14_mgm_mask]
        atom_coords = atom14_mgm_coords[atom14_mgm_mask]
        # atom_update, res_update = self.bilevel_attn(
        #     node_features,
        #     edge_features,
        #     edge_index,
        #     res_edge_subset_features,
        #     res_edge_subset_index,
        #     atom_features,
        #     atom_coords,
        #     cross_atom14_edge_index,
        #     atom_edge_to_res_edge_subset,
        #     res_edge_subset_to_res_edge,
        #     res_edge_subset_self_edge_mask
        # )
        # atom_update = torch.utils.checkpoint.checkpoint(
        #     self.bilevel_attn,
        #     node_features,
        #     edge_features,
        #     edge_index,
        #     res_edge_features,
        #     res_edge_index,
        #     atom_features,
        #     atom_coords,
        #     cross_atom14_edge_index,
        #     atom_edge_to_res_edge,
        #     res_edge_subset_to_res_edge,
        #     res_edge_subset_self_edge_mask,
        #     use_reentrant=False
        # )
        atom_update = torch.utils.checkpoint.checkpoint(
            self.bilevel_attn,
            node_features,
            res_edge_features,
            res_edge_index,
            atom_features,
            atom_coords,
            cross_atom14_edge_index,
            atom_edge_to_res_edge,
            use_reentrant=False
        )

        atom14_new_features = atom14_mgm_features.clone()
        atom14_new_features[atom14_mgm_mask] += atom_update
        atom14_new_features = self.atom_ln(atom14_new_features)

        ca_update = atom_update[ca_select]
        node_update = self.atom_to_node(ca_update)
        # # TODO: smths not right here, w the res mask not matching up with the atom mask
        # print(res_mask.sum(), node_features.shape, ca_update.shape, res_mask.shape)
        # print(atom14_mask[:, 1].sum(), atom14_mgm_mask[:, 1].sum())
        update_store = torch.zeros_like(node_features)
        update_store[atom14_mask[:, 1]] = node_update

        # updated_point_features = atom14_new_features[mgm_mask, 5:5+self.num_points]
        # point_vecs = rigids[..., None].invert_apply(masked_res_point_coords[mgm_mask])
        # point_vec_lens = torch.linalg.vector_norm(point_vecs + eps, dim=-1)
        # masked_nodes_update = torch.cat([
        #     updated_point_features.view(-1, self.num_points * self.c_atom),
        #     point_vecs.view(-1, self.num_points*3),
        #     point_vec_lens.view(-1, self.num_points)
        # ], dim=-1)
        # masked_nodes_update = self.collect_invariant_points_info(masked_nodes_update)
        # masked_node_update_store = torch.zeros_like(node_features)
        # masked_node_update_store[mgm_mask] = masked_nodes_update

        total_node_update = update_store #+ masked_node_update_store # + res_update
        node_features = self.node_ln(node_features + total_node_update * res_mask[..., None])

        return node_features, atom14_new_features