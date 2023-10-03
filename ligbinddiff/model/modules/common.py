import torch
from torch import nn
from torch_geometric.utils import sort_edge_index

from ligbinddiff.model.modules.equiformer_v2.so2_ops import Nodewise_SO3_Convolution

class ProjectLayer(nn.Module):
    """ Layer to interface between different lmax features """
    def __init__(self,
                 in_lmax_list,
                 in_channels,
                 out_lmax_list,
                 out_channels,
                 edge_channels_list,
                 mappingReduced_super,
                 super_SO3_rotation,
                 super_SO3_grid):
        super().__init__()

        self.in_lmax_list = in_lmax_list
        self.out_lmax_list = out_lmax_list
        self.super_lmax_list = [max(l1, l2) for l1, l2 in zip(in_lmax_list, out_lmax_list)]

        self.super_SO3_rotation = super_SO3_rotation
        self.super_SO3_grid = super_SO3_grid

        self.conv = Nodewise_SO3_Convolution(
            sphere_channels=in_channels,
            m_output_channels=out_channels,
            lmax_list=self.super_lmax_list,
            mmax_list=self.super_lmax_list,
            mappingReduced=mappingReduced_super,
            SO3_rotation=super_SO3_rotation,
            edge_channels_list=edge_channels_list
        )

    def forward(self, node_features, edge_features, edge_index):
        node_features = node_features.to_resolutions(self.super_lmax_list, self.super_lmax_list)
        node_features = self.conv(node_features, edge_features, edge_index)
        return node_features.to_resolutions(self.out_lmax_list, self.out_lmax_list)


class EdgeUpdate(nn.Module):
    def __init__(self,
                 node_lmax_list,
                 edge_channels_list,
                 h_channels=32):
        super().__init__()
        h_dim = edge_channels_list[0]
        num_l0 = len(node_lmax_list)
        self.ff = nn.Sequential(
            nn.Linear(h_dim + h_channels * num_l0 * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)
        )
        self.norm = nn.LayerNorm(h_dim)

    def forward(self, node_features, edge_features, edge_index):
        node_src = node_features.expand_edge(edge_index[0])
        node_src_invariant = node_src.get_invariant_features(flat=True)
        node_dst = node_features.expand_edge(edge_index[1])
        node_dst_invariant = node_dst.get_invariant_features(flat=True)
        in_features = torch.cat([node_src_invariant, node_dst_invariant, edge_features], dim=-1)
        update = self.ff(in_features)

        return edge_features + self.norm(update)


def sample_inv_cubic_edges(batched_X_ca, batched_x_mask, batch, knn_k=30, inv_cube_k=10):
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
        edge_sources = torch.arange(X_ca.shape[0]).repeat_interleave(knn_k + inv_cube_k).to(edge_sinks.device)
        edge_index = torch.stack([edge_sinks.flatten(), edge_sources], dim=0)
        # sorting not strictly necessarily but it help if we ever hack things to dense knn graphs
        edge_indicies.append(sort_edge_index(edge_index, sort_by_row=False) + offset)
        offset = offset + (batch == i).long().sum()

    edge_index = torch.cat(edge_indicies, dim=-1)
    edge_mask = batched_x_mask[edge_index].any(dim=0)
    return edge_index[:, ~edge_mask]
