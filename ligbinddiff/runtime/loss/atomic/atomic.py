import torch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import sort_edge_index, scatter

from ligbinddiff.runtime.loss.atomic.common import atom91_to_atom14
from ligbinddiff.runtime.loss.utils import _nodewise_to_graphwise, vec_norm


def framediff_local_atomic_context_loss(
    pred_bb,
    ref_bb,
    batch,
    res_mask,
    r=6,
    eps=1e-6,
    max_num_neighbors=32,
):
    flat_ref_bb = ref_bb[res_mask].reshape(-1, 3)
    flat_pred_bb = pred_bb[res_mask].reshape(-1, 3)
    batch_expand = batch[res_mask].repeat_interleave(5, dim=0)
    edge_index = radius_graph(flat_ref_bb, r, batch_expand, max_num_neighbors=max_num_neighbors)
    edge_index = sort_edge_index(edge_index, sort_by_row=False)

    pred_dist_vec = flat_pred_bb[edge_index[0]] - flat_pred_bb[edge_index[1]]
    pred_dists = torch.linalg.vector_norm(pred_dist_vec + eps, dim=-1)

    ref_dist_vec = flat_ref_bb[edge_index[0]] - flat_ref_bb[edge_index[1]]
    ref_dists = torch.linalg.vector_norm(ref_dist_vec + eps, dim=-1)

    dist_se = torch.square(pred_dists - ref_dists)

    edge_batch = batch_expand[edge_index[1]]
    num_graph = batch_expand.max().item() + 1
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
    graphwise_num_res = scatter(
        res_mask.long(),
        batch,
        dim=0,
        dim_size=num_graph
    )
    return graphwise_dist_se / (graphwise_num_edges - graphwise_num_res)




def atomic_neighborhood_dist_loss(ref_atom91,
                                  pred_atom91,
                                  seq,
                                  edge_index,
                                  num_edges,
                                  x_mask,
                                  radius_cutoff=6):
    """ Additional loss to push non-bonded atoms
        out of the van der waals radius of other atoms  """
    pred_atom14, atom14_mask = atom91_to_atom14(pred_atom91, seq)
    ref_atom14, _ = atom91_to_atom14(ref_atom91, seq)
    atom14_mask = atom14_mask.any(dim=-1)
    atom14_mask[x_mask] = True

    ref_res_src = ref_atom14[edge_index[0]].unsqueeze(-2)  # n_edge x n_atoms x 1 x 3
    ref_res_dst = ref_atom14[edge_index[1]].unsqueeze(-3)  # n_edge x 1 x n_atoms x 3

    pred_res_src = pred_atom14[edge_index[0]].unsqueeze(-2)  # n_edge x n_atoms x 1 x 3
    pred_res_dst = pred_atom14[edge_index[1]].unsqueeze(-3)  # n_edge x 1 x n_atoms x 3

    atom14_src_mask = atom14_mask[edge_index[0]].unsqueeze(-1)  # n_edge x n_atom x 1
    atom14_dst_mask = atom14_mask[edge_index[1]].unsqueeze(-2)  # n_edge x 1 x n_atom
    selection_mask = atom14_src_mask | atom14_dst_mask

    ref_interres_dists = vec_norm(ref_res_src - ref_res_dst, dim=-1)  # n_edge x n_atoms x n_atoms
    cutoff_mask = (ref_interres_dists < radius_cutoff)
    total_mask = selection_mask | cutoff_mask
    ref_interres_dists = ref_interres_dists[~total_mask]

    pred_interres_dists = vec_norm(pred_res_src - pred_res_dst, dim=-1)  # n_edge x n_atoms x n_atoms
    pred_interres_dists = pred_interres_dists[~total_mask]

    dist_mse = (ref_interres_dists - pred_interres_dists).square()

    return _nodewise_to_graphwise(dist_mse, num_edges, total_mask.flatten(-2, -1))
