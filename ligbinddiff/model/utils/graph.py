import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import sort_edge_index, scatter, coalesce
from ligbinddiff.data.datasets.featurize.common import _rbf

from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings
import ligbinddiff.utils.openfold.rigid_utils as ru


def sample_inv_cubic_edges(
    batched_X_ca,
    batched_x_mask,
    batch,
    knn_k=30,
    inv_cube_k=10,
    self_edge=False
):
    # TODO: change x_mask to be True for retained positions
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
        # we set "num bad edges" for masked residues to 0, otherwise that would skew our count
        num_bad_edges[x_mask] = 0
        max_num_bad_edges = int(num_bad_edges.max())
        if inv_cube_k > perturbed_logprobs.shape[-1] - max_num_bad_edges:
            print(f"trimming down num edges from {inv_cube_k} to {perturbed_logprobs.shape[-1] - max_num_bad_edges}, {max_num_bad_edges} bad edges")
            if perturbed_logprobs.shape[-1] - max_num_bad_edges == 0:
                print(f"we shouldn't get rid of all edges... right now we see num_bad_edges as ", num_bad_edges)
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


def sample_logn_inv_cubic_edges(
    batched_X_ca,
    batched_x_mask,
    batch,
    knn_k=20,
    min_inv_cube_edges=40,
    logn_scale=8,
    logn_offset=-12,
    self_edge=False,
    gen_triangle_edges=False,
    gen_knn_triangle_edges=False,
    gen_lrange_triangle_edges=False,
    gen_cross_range_triangle_edges=False,
):
    # TODO: change x_mask to be True for retained positions
    edge_indicies = []
    knn_edge_select = []
    lrange_edge_select = []
    edge_colors = []
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

        # add this because sometimes we actually have fewer edges than knn_k
        knn_k = knn_edges.shape[-1]

        ## inv cube
        uniform = torch.distributions.Uniform(0,1)
        dist_noise = uniform.sample(remaining_dist.shape).to(batched_X_ca.device)  # N x (N - knn_k - 1)

        logprobs = -3 * torch.log(remaining_dist)  # N x (N - knn_k)
        perturbed_logprobs = logprobs - torch.log(-torch.log(dist_noise))  # N x (N - knn_k - 1)

        good_edges = torch.isfinite(perturbed_logprobs)
        perturbed_logprobs[~good_edges] = -torch.inf

        # if we don't have inv_cube_k nodes to sample, sample the max we can
        num_bad_edges = (~good_edges).sum(dim=-1)
        # we set "num bad edges" for masked residues to 0, otherwise that would skew our count
        num_bad_edges[x_mask] = 0
        max_num_bad_edges = int(num_bad_edges.max())
        num_nodes = (~x_mask).sum().item()
        # TODO: which log should this be...
        inv_cube_k = round(
            max(min_inv_cube_edges, logn_scale * np.log2(num_nodes) + logn_offset)
        )
        if inv_cube_k > perturbed_logprobs.shape[-1] - max_num_bad_edges:
            print(f"trimming down random k edges from {inv_cube_k} to {perturbed_logprobs.shape[-1] - max_num_bad_edges}, {max_num_bad_edges} bad edges")
            if perturbed_logprobs.shape[-1] - max_num_bad_edges == 0:
                print(f"we shouldn't get rid of all edges... right now we see num_bad_edges as ", num_bad_edges)
            inv_cube_k = perturbed_logprobs.shape[-1] - max_num_bad_edges

        _, sampled_edges_relative_idx = torch.topk(perturbed_logprobs, k=inv_cube_k, dim=-1)
        sampled_edges = torch.gather(remaining_edges, -1, sampled_edges_relative_idx)  # N x inv_cube_k

        edge_sinks = torch.cat([knn_edges, sampled_edges], dim=-1)  # N x (knn_k + inv_cube_k)
        edge_sources = torch.arange(X_ca.shape[0]).repeat_interleave(knn_k + inv_cube_k).to(edge_sinks.device)
        edge_index = torch.stack([edge_sinks.flatten(), edge_sources], dim=0)

        # sorting not strictly necessarily but it help if we ever hack things to dense knn graphs
        # edge_indicies.append(sort_edge_index(edge_index, sort_by_row=False) + offset)
        edge_indicies.append(edge_index + offset)
        num_nodes = X_ca.shape[0]
        knn_edge_select.append(torch.cat(
            [torch.ones(knn_k), torch.zeros(inv_cube_k)]
        ).repeat(num_nodes).bool().to(X_ca.device))
        lrange_edge_select.append(torch.cat(
            [torch.zeros(knn_k), torch.ones(inv_cube_k)]
        ).repeat(num_nodes).bool().to(X_ca.device))
        edge_colors.append(torch.cat(
            [torch.zeros(knn_k), torch.ones(inv_cube_k)]
        ).repeat(num_nodes).to(X_ca.device))

        if gen_knn_triangle_edges or gen_lrange_triangle_edges or gen_cross_range_triangle_edges:
            trig_edge_src = edge_sinks[..., None, :].expand(-1, knn_k + inv_cube_k, -1)
            trig_edge_dst = edge_sinks[..., None].expand(-1, -1, knn_k + inv_cube_k)
            trig_edge_index = torch.stack([trig_edge_dst, trig_edge_src])
            select_triangle_edges = torch.zeros((knn_k + inv_cube_k, knn_k + inv_cube_k), device=trig_edge_src.device).bool()
            trig_edge_colors = torch.ones((knn_k + inv_cube_k, knn_k + inv_cube_k), device=trig_edge_src.device) * 4
            if gen_knn_triangle_edges:
                select_triangle_edges[:knn_k, :knn_k] = True
                trig_edge_colors[:knn_k, :knn_k] = 2
            if gen_lrange_triangle_edges:
                select_triangle_edges[knn_k:, knn_k:] = True
                trig_edge_colors[knn_k:, knn_k:] = 3
            if gen_cross_range_triangle_edges:
                select_triangle_edges[knn_k:, :knn_k] = True
                select_triangle_edges[:knn_k, knn_k:] = True

            trig_edge_index = trig_edge_index[select_triangle_edges[None, None].expand(2, X_ca.shape[0], -1, -1)].view(2, -1)
            print(trig_edge_index.shape)
            trig_edge_colors = trig_edge_colors[None].expand(X_ca.shape[0], -1, -1)[select_triangle_edges[None].expand(X_ca.shape[0], -1, -1)]
            edge_indicies.append(trig_edge_index + offset)
            trig_offset = trig_edge_index.shape[-1]
            knn_edge_select.append(torch.zeros(trig_offset, device=X_ca.device))
            lrange_edge_select.append(torch.zeros(trig_offset, device=X_ca.device))
            edge_colors.append(trig_edge_colors)

        offset = offset + (batch == i).long().sum()

    edge_index = torch.cat(edge_indicies, dim=-1)
    knn_edge_select = torch.cat(knn_edge_select, dim=0)
    lrange_edge_select = torch.cat(lrange_edge_select, dim=0)
    edge_colors = torch.cat(edge_colors, dim=0)
    edge_mask = batched_x_mask[edge_index].any(dim=0)
    edge_index, knn_edge_select, lrange_edge_select = edge_index[:, ~edge_mask], knn_edge_select[~edge_mask], lrange_edge_select[~edge_mask]
    edge_colors = edge_colors[~edge_mask]
    if gen_triangle_edges:
        edge_index, (knn_edge_select, lrange_edge_select, edge_colors) = coalesce(
            edge_index, [knn_edge_select.long(), lrange_edge_select.long(), edge_colors.long()],
            reduce="mean"
        )
        knn_edge_select = knn_edge_select.bool()
        lrange_edge_select = lrange_edge_select.bool()
    edge_colors = edge_colors.long()

    return edge_index, knn_edge_select, lrange_edge_select, edge_colors


def sample_all_edges(
    batched_x_mask,
    batch,
):
    # TODO: change x_mask to be True for retained positions
    edge_indicies = []
    offset = 0
    for i in range(batch.max().item() + 1):
        x_mask = batched_x_mask[batch == i]
        edge_sources = torch.arange(x_mask.shape[0]).repeat_interleave(x_mask.shape[0]).to(x_mask.device)
        edge_sinks = torch.arange(x_mask.shape[0]).repeat(x_mask.shape[0]).to(x_mask.device)
        edge_index = torch.stack([edge_sinks.flatten(), edge_sources], dim=0)
        edge_indicies.append(edge_index + offset)
        offset = offset + (batch == i).long().sum()

    edge_index = torch.cat(edge_indicies, dim=-1)
    edge_mask = batched_x_mask[edge_index].any(dim=0)
    return edge_index[:, ~edge_mask], None, None




def sparse_to_knn_graph(edge_features, edge_index, num_nodes=None, batch=None, res_mask=None):
    src = edge_index[1]
    assert (torch.sort(src)[0] == src).all(), "edge index must have monotonic increasing node index"

    if num_nodes is None:
        num_nodes = int(edge_index.max() + 1)
    num_edges_per_node = scatter(
        torch.ones_like(edge_index[1]),
        edge_index[1],
        dim_size=num_nodes,
    )
    # max_k = int(num_edges_per_node.max())

    if (num_edges_per_node == num_edges_per_node[0]).all():
        # we can just reshape which saves us a lot of time and memory
        num_edges = num_edges_per_node[0].item()
        knn_edge_features = edge_features.view(num_nodes, num_edges, -1)
        knn_edge_index = edge_index[0].view(num_nodes, num_edges)
        edge_mask = (knn_edge_index != -1)
    else:
        # edges_per_node = edge_features.split(num_edges_per_node.tolist(), dim=0)
        # knn_edge_features = pad_sequence(
        #     edges_per_node,
        #     batch_first=True,
        #     padding_value=0.
        # )
        knn_edge_features = torch.zeros(
            (num_nodes, num_edges_per_node.max().item(), edge_features.shape[-1]),
            device=edge_features.device,
            dtype=edge_features.dtype
        )

        # there are a couple of assumptions here which could be broken without proper usage
        # but it should be much faster
        if batch is not None:
            # print("slower knn gen", num_edges_per_node)
            assert res_mask is not None
            offset = 0
            edge_batch = batch[edge_index[1]]
            for i in range(edge_batch.max()+1):
                select_edge = (edge_batch == i)
                select_node = (batch == i)
                subset_res_mask = res_mask[select_node].bool()
                subset_num_nodes = select_node.sum()
                num_edges = int(num_edges_per_node[select_node].max().item())
                subset_edge_features = edge_features[select_edge].view(-1, num_edges, edge_features.shape[-1])
                subset_edge_features_store = torch.zeros(
                    (subset_num_nodes, num_edges, edge_features.shape[-1]),
                    device=edge_features.device,
                    dtype=edge_features.dtype
                )
                subset_edge_features_store[subset_res_mask, :num_edges] = subset_edge_features
                knn_edge_features[offset:offset+subset_num_nodes, :num_edges] = subset_edge_features_store
                offset += subset_num_nodes

        else:
            start = 0
            for i, num_edges in enumerate(num_edges_per_node):
                knn_edge_features[i, :num_edges] = edge_features[start:start+num_edges]
                start += num_edges

        edge_index_per_node = edge_index[0].split(num_edges_per_node.tolist(), dim=0)
        knn_edge_index = pad_sequence(
            edge_index_per_node,
            batch_first=True,
            padding_value=-1
        )

        edge_mask = (knn_edge_index != -1)
        knn_edge_index[~edge_mask] = 0

    return knn_edge_features, knn_edge_index, edge_mask

def sparse_to_nested(edge_features, edge_index, num_nodes=None):
    src = edge_index[1]
    assert (torch.sort(src)[0] == src).all(), "edge index must have monotonic increasing node index"

    if num_nodes is None:
        num_nodes = int(edge_index.max() + 1)
    num_edges_per_node = scatter(
        torch.ones_like(edge_index[1]),
        edge_index[1],
        dim_size=num_nodes,
    )
    # max_k = int(num_edges_per_node.max())

    if (num_edges_per_node == num_edges_per_node[0]).all():
        # we can just reshape which saves us a lot of time and memory
        num_edges = num_edges_per_node[0].item()
        nested_edge_features = edge_features.view(num_nodes, num_edges, -1)
        nested_edge_index = edge_index[0].view(num_nodes, num_edges)
    else:
        # edges_per_node = edge_features.split(num_edges_per_node.tolist(), dim=0)
        # knn_edge_features = pad_sequence(
        #     edges_per_node,
        #     batch_first=True,
        #     padding_value=0.
        # )
        edge_features_split = edge_features.split(num_edges_per_node.tolist(), dim=0)
        nested_edge_features = torch.nested.as_nested_tensor(list(edge_features_split))
        edge_index_split = edge_index[0].split(num_edges_per_node.tolist(), dim=0)
        nested_edge_index = torch.nested.as_nested_tensor(list(edge_index_split))

    return nested_edge_features, nested_edge_index


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


def gen_spatial_graph_features(rigids: ru.Rigid, edge_index, num_rbf_embed, num_pos_embed, use_unit_vec=False):
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

    if use_unit_vec:
        edge_unit_vec = F.normalize(edge_dist_vec + 1e-8, dim=-1)
        edge_unit_vec_in_src_frame = rigids[edge_index[1]].get_rots().invert_apply(edge_unit_vec)
        edge_features.append(edge_unit_vec_in_src_frame)

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
