import numpy as np
import torch
from torch_geometric.utils import sort_edge_index
from ligbinddiff.data.datasets.featurize.common import _rbf

from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings


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


def dense_to_sparse_graph(node_features, edge_features, edge_index, node_index=None):
    batch_size, batch_n_nodes, node_n_edge = edge_index.shape
    flat_edge_index = edge_index.view(batch_size * batch_n_nodes, -1)
    if node_index is None:
        node_index = torch.arange(batch_size * batch_n_nodes, device=edge_index.device)

    sparse_node_features = {
        k: f_V.view([batch_size * batch_n_nodes] + list(f_V.shape[2:]))
        for k, f_V in node_features.items()
    }

    sparse_edge_features = {
        k: f_E.view([batch_size * batch_n_nodes] + list(f_E.shape[2:]))
        for k, f_E in edge_features.items()
    }

    node_src = node_index[:, None].expand(-1, node_n_edge).view(-1)
    node_dst = flat_edge_index.view(-1)
    sparse_edge_index = torch.stack([node_dst, node_src], dim=0)

    return sparse_node_features, sparse_edge_features, sparse_edge_index


def sparse_to_dense_graph(node_features, edge_features, edge_index, batch_size, edges_per_node=None):
    assert edge_index.shape[-1] // batch_size == 0, "we need the batch to be divisible into same-sized chunks"

    node_dst = edge_index[0]

    if edges_per_node is None:
        edges_per_node = np.sqrt(edge_index.shape[-1] / batch_size)
        assert edges_per_node == int(edges_per_node), f"we couldn't infer the number of edges per node, got {edges_per_node}"

    dense_edge_index = node_dst.view(batch_size, -1, edges_per_node)
    dense_node_features = {
        k: f_V.view([batch_size, -1] + list(f_V.shape[1:]))
        for k, f_V in node_features.items()
    }

    dense_edge_features = {
        k: f_E.view([batch_size, -1, edges_per_node] + list(f_E.shape[2:]))
        for k, f_E in edge_features.items()
    }

    return dense_node_features, dense_edge_features, dense_edge_index


def sequence_local_graph(num_nodes, x_mask, half_local_size=10):
    device = x_mask.device
    local_index = torch.cat([
        -(torch.arange(half_local_size) + 1),
        (torch.arange(half_local_size) + 1)
    ], dim=-1).to(device)
    offset = torch.arange(num_nodes)[..., None].to(device)
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

    global_edge_index[:half_local_size] = head.to(device)
    global_edge_index[-half_local_size:] = tail.to(device)

    node_src = offset.expand(-1, half_local_size*2)
    edge_index = torch.stack(
        [global_edge_index, node_src], dim=0
    )
    edge_index = edge_index.reshape(2, -1)

    masked_nodes = offset[x_mask].view(1, 1, -1)
    edge_mask = (edge_index[..., None] == masked_nodes).any(dim=-1).any(dim=0)

    edge_index = edge_index[:, ~edge_mask]
    # edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return edge_index


def nodewise_to_batchwise(t_per_node, data_lens):
    assert t_per_node.numel() == sum(data_lens)
    partition_points = np.cumsum(data_lens)[:-1]
    return torch.cat([
        t_per_node[l] for l in partition_points
    ])


def batchwise_to_nodewise(t_per_graph, data_lens):
    assert t_per_graph.numel() == len(data_lens)
    return torch.cat([
        t_per_graph[i] * torch.ones(l, device=t_per_graph.device) for i, l in enumerate(data_lens)
    ])


def gen_spatial_graph_features(X_ca, edge_index, num_rbf_embed, num_pos_embed):
    edge_dist_vec = X_ca[edge_index[0]] - X_ca[edge_index[1]]
    edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
    edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device, D_count=num_rbf_embed)  # edge_channels_list
    edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=num_pos_embed, device=edge_dist.device)  # edge_channels_list
    edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)
    return edge_features, edge_dist_vec

def get_data_lens(pyg_graph, key=None):
    if key is None:
        batch = pyg_graph.batch
        num_nodes = []
        for i in range(batch.max().item() + 1):
            select = (batch == i)
            num_nodes.append(select.long().sum().item())
        return num_nodes
    else:
        data_splits = pyg_graph._slice_dict[key]
        data_lens = data_splits[1:] - data_splits[:-1]
        return data_lens
