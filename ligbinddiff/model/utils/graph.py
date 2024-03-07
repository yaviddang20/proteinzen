import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import sort_edge_index, scatter
from ligbinddiff.data.datasets.featurize.common import _rbf

from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings
import ligbinddiff.utils.openfold.rigid_utils as ru


def sample_inv_cubic_edges(batched_X_ca, batched_x_mask, batch, knn_k=30, inv_cube_k=10, self_edge=False):
    edge_indicies = []
    knn_edge_select = []
    lrange_edge_select = []
    offset = 0
    for i in range(batch.max().item() + 1):
        X_ca = batched_X_ca[batch == i]
        x_mask = batched_x_mask[batch == i]

        X_ca[x_mask] = torch.inf
        rel_pos_CA = X_ca.unsqueeze(1) - X_ca.unsqueeze(0)  # N x N x 3
        dist_CA = torch.linalg.vector_norm(rel_pos_CA, dim=-1)  # N x N
        sorted_dist, sorted_edges = torch.sort(dist_CA, dim=-1)  # N x N
        if self_edge:
            knn_edges = sorted_edges[..., :knn_k]  # first edge will always be self
            # remove knn edges
            remaining_dist = sorted_dist[..., knn_k:]  # N x (N - knn_k - 1)
            remaining_edges = sorted_edges[..., knn_k:]  # N x (N - knn_k - 1)

        else:
            # remove self edge
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
            print(f"trimming down num edges from {inv_cube_k} to {perturbed_logprobs.shape[-1] - max_num_bad_edges}, {max_num_bad_edges} bad edges")
            inv_cube_k = perturbed_logprobs.shape[-1] - max_num_bad_edges

        _, sampled_edges_relative_idx = torch.topk(perturbed_logprobs, k=inv_cube_k, dim=-1)
        sampled_edges = torch.gather(remaining_edges, -1, sampled_edges_relative_idx)  # N x inv_cube_k

        edge_sinks = torch.cat([knn_edges, sampled_edges], dim=-1)  # B x N x (knn_k + inv_cube_k)
        edge_sources = torch.arange(X_ca.shape[0]).repeat_interleave(knn_k + inv_cube_k).to(edge_sinks.device)
        edge_index = torch.stack([edge_sinks.flatten(), edge_sources], dim=0)
        # sorting not strictly necessarily but it help if we ever hack things to dense knn graphs
        # edge_indicies.append(sort_edge_index(edge_index, sort_by_row=False) + offset)
        edge_indicies.append(edge_index + offset)
        offset = offset + (batch == i).long().sum()
        num_nodes = X_ca.shape[0]
        knn_edge_select.append(torch.cat(
            [torch.ones(knn_k), torch.zeros(inv_cube_k)]
        ).repeat(num_nodes).bool().to(X_ca.device))
        lrange_edge_select.append(torch.cat(
            [torch.zeros(knn_k), torch.ones(inv_cube_k)]
        ).repeat(num_nodes).bool().to(X_ca.device))

    edge_index = torch.cat(edge_indicies, dim=-1)
    knn_edge_select = torch.cat(knn_edge_select, dim=0)
    lrange_edge_select = torch.cat(lrange_edge_select, dim=0)
    edge_mask = batched_x_mask[edge_index].any(dim=0)
    return edge_index[:, ~edge_mask], knn_edge_select[~edge_mask], lrange_edge_select[~edge_mask]


def sparse_to_knn_graph(edge_features, edge_index):
    src = edge_index[1]
    assert (torch.sort(src)[0] == src).all(), "edge index must have monotonic increasing node index"

    num_nodes = int(edge_index.max() + 1)
    num_edges_per_node = scatter(
        torch.ones_like(edge_index[1]),
        edge_index[1],
        dim_size=num_nodes,
    )
    # max_k = int(num_edges_per_node.max())

    edges_per_node = edge_features.split(num_edges_per_node.tolist(), dim=0)
    knn_edge_features = pad_sequence(
        edges_per_node,
        batch_first=True,
        padding_value=0.
    )
    edge_index_per_node = edge_index[0].split(num_edges_per_node.tolist(), dim=0)
    knn_edge_index = pad_sequence(
        edge_index_per_node,
        batch_first=True,
        padding_value=-1
    )

    edge_mask = (knn_edge_index != -1)
    knn_edge_index[~edge_mask] = 0

    return knn_edge_features, knn_edge_index, edge_mask


def knn_to_sparse_graph(edge_features, edge_index, edge_mask):
    sparse_edge_features = edge_features[edge_mask]
    dst_edge_index = edge_index[edge_mask]
    num_nodes = edge_features.shape[0]
    knn_src_edge_index = torch.arange(
        num_nodes, device=edge_index.device
    )[..., None].expand(-1, edge_index.shape[-1])
    src_edge_index = knn_src_edge_index[edge_mask]
    return sparse_edge_features, torch.stack([dst_edge_index, src_edge_index])



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


def sequence_local_graph(num_nodes, x_mask, half_local_size=5):
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

    edge_mask = x_mask[edge_index[0]] | x_mask[edge_index[1]]

    edge_index = edge_index[:, ~edge_mask]
    # edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return edge_index



def batchwise_to_nodewise(t_per_graph, batch):
    return t_per_graph[batch]


def gen_spatial_graph_features(rigids: ru.Rigid, edge_index, num_rbf_embed, num_pos_embed):
    X_ca = rigids.get_trans()
    quats = rigids.get_rots().get_quats()
    src_quat = quats[edge_index[1]]
    dst_quat = quats[edge_index[0]]
    quat_rel = ru.quat_multiply(
        ru.invert_quat(src_quat),
        dst_quat
    )
    edge_features = [quat_rel]

    edge_dist_vec = X_ca[edge_index[0]] - X_ca[edge_index[1]]
    edge_dist = torch.linalg.vector_norm(edge_dist_vec, dim=-1)
    if num_rbf_embed > 0:
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device, D_count=num_rbf_embed)  # edge_channels_list
        edge_features.append(edge_dist_rbf)

    if num_pos_embed > 0:
        edge_features.append(
            _edge_positional_embeddings(edge_index, num_embeddings=num_pos_embed, device=edge_dist.device)  # edge_channels_list
        )
    edge_features = torch.cat(edge_features, dim=-1)

    return edge_features, edge_dist_vec

def get_data_lens(pyg_graph, key=None, hetero_key=None):
    if key is None:
        if hetero_key is not None:
            batch = pyg_graph[hetero_key].batch
        else:
            batch = pyg_graph.batch
        num_nodes = []
        for i in range(batch.max().item() + 1):
            select = (batch == i)
            num_nodes.append(select.long().sum().item())
        return num_nodes
    else:
        if hetero_key is not None:
            data_splits = pyg_graph._slice_dict[hetero_key][key]
        else:
            data_splits = pyg_graph._slice_dict[key]
        data_lens = data_splits[1:] - data_splits[:-1]
        return data_lens
