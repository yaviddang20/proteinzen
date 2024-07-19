import torch
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import sort_edge_index, scatter

from proteinzen.runtime.loss.atomic.common import atom91_to_atom14
from proteinzen.runtime.loss.utils import _nodewise_to_graphwise, vec_norm

def local_atomic_context_loss(
    pred_atom_pos,
    gt_atom_pos,
    batch,
    r=6,
    eps=1e-6,
    max_num_neighbors=100,
):
    edge_index = radius_graph(gt_atom_pos, r, batch, max_num_neighbors=max_num_neighbors)

    pred_dist_vec = pred_atom_pos[edge_index[0]] - pred_atom_pos[edge_index[1]]
    pred_dists = torch.linalg.vector_norm(pred_dist_vec + eps, dim=-1)

    ref_dist_vec = gt_atom_pos[edge_index[0]] - gt_atom_pos[edge_index[1]]
    ref_dists = torch.linalg.vector_norm(ref_dist_vec + eps, dim=-1)

    dist_se = torch.square(pred_dists - ref_dists)

    edge_batch = batch[edge_index[1]]
    num_graph = batch.max().item() + 1
    graphwise_dist_se = scatter(
        dist_se,
        edge_batch,
        dim=0,
        dim_size=num_graph
    )
    graphwise_num_edges = scatter(
        torch.ones_like(edge_batch),
        edge_batch,
        dim=0,
        dim_size=num_graph
    )
    return graphwise_dist_se / graphwise_num_edges
