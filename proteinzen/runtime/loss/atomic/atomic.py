import torch
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import sort_edge_index, scatter

from proteinzen.runtime.loss.atomic.common import atom91_to_atom14
from proteinzen.runtime.loss.utils import _nodewise_to_graphwise, vec_norm


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


def local_atomic_context_loss(
    pred_atom14,
    gt_atom14,
    alt_atom14,
    batch,
    atom14_mask,
    r=6,
    eps=1e-6,
    max_num_neighbors=100,
):
    res_mask = atom14_mask[:, 1]
    # re: ambiguous atom14 naming
    # im gonna use a heuristic for the ref atom14
    # where we'll just take the ref residue atom ordering
    # as the one which is lowest in rmsd to the predicted structure
    # in a lot of cases this probably won't hold but it will at least work
    # in low rmsd regimes i think
    with torch.no_grad():
        pred_gt_diff = torch.square(pred_atom14 - gt_atom14).sum(dim=-1)
        pred_alt_diff = torch.square(pred_atom14 - alt_atom14).sum(dim=-1)
        pred_gt_mse = torch.sum(pred_gt_diff * atom14_mask, dim=-1)
        pred_alt_mse = torch.sum(pred_alt_diff * atom14_mask, dim=-1)

        gt_over_alt = pred_gt_mse < pred_alt_mse
        ref_atom14 = gt_atom14 * gt_over_alt[..., None, None] + alt_atom14 * ~gt_over_alt[..., None, None]

    flat_ref_atom14 = ref_atom14[atom14_mask]
    flat_pred_atom14 = pred_atom14[atom14_mask]
    batch_expand = batch[..., None].expand(-1, atom14_mask.shape[-1])[atom14_mask]
    edge_index = radius_graph(flat_ref_atom14, r, batch_expand, max_num_neighbors=max_num_neighbors)

    pred_dist_vec = flat_pred_atom14[edge_index[0]] - flat_pred_atom14[edge_index[1]]
    pred_dists = torch.linalg.vector_norm(pred_dist_vec + eps, dim=-1)

    ref_dist_vec = flat_ref_atom14[edge_index[0]] - flat_ref_atom14[edge_index[1]]
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


def residue_knn_neighborhood_atomic_dist_loss(gt_ref_atom14,
                                  alt_ref_atom14,
                                  pred_atom14,
                                  gt_atom14_mask,
                                  alt_atom14_mask,
                                  batch,
                                  k=30,
                                  radius_cutoff=6):
    """ Additional loss to push non-bonded atoms
        out of the van der waals radius of other atoms  """
    res_mask = gt_atom14_mask[:, 1]
    ref_X_ca = gt_ref_atom14[:, 1].clone()
    ref_X_ca[~res_mask] = torch.inf
    edge_index = knn_graph(ref_X_ca, k, batch)
    num_edges = edge_index.shape[-1]
    edge_batch = batch[edge_index[1]]

    gt_ref_res_src = gt_ref_atom14[edge_index[0]].unsqueeze(-2)  # n_edge x n_atoms x 1 x 3
    gt_ref_res_dst = gt_ref_atom14[edge_index[1]].unsqueeze(-3)  # n_edge x 1 x n_atoms x 3

    alt_ref_res_src = alt_ref_atom14[edge_index[0]].unsqueeze(-2)  # n_edge x n_atoms x 1 x 3
    alt_ref_res_dst = alt_ref_atom14[edge_index[1]].unsqueeze(-3)  # n_edge x 1 x n_atoms x 3

    pred_res_src = pred_atom14[edge_index[0]].unsqueeze(-2)  # n_edge x n_atoms x 1 x 3
    pred_res_dst = pred_atom14[edge_index[1]].unsqueeze(-3)  # n_edge x 1 x n_atoms x 3

    gt_src_mask = gt_atom14_mask[edge_index[0]].unsqueeze(-1)  # n_edge x n_atom x 1
    gt_dst_mask = gt_atom14_mask[edge_index[1]].unsqueeze(-2)  # n_edge x 1 x n_atom

    alt_src_mask = alt_atom14_mask[edge_index[0]].unsqueeze(-1)  # n_edge x n_atom x 1
    alt_dst_mask = alt_atom14_mask[edge_index[1]].unsqueeze(-2)  # n_edge x 1 x n_atom

    ref_interres_dists = []  # n_edge x n_atoms x n_atoms
    total_masks = []
    for ref_res_src, ref_src_mask in zip([gt_ref_res_src, alt_ref_res_src], [gt_src_mask, alt_src_mask]):
        for ref_res_dst, ref_dst_mask in zip([gt_ref_res_dst, alt_ref_res_dst], [gt_dst_mask, alt_dst_mask]):
            ref_dists = vec_norm(ref_res_src - ref_res_dst, dim=-1)
            ref_interres_dists.append(ref_dists)
            cutoff_mask = (ref_dists < radius_cutoff)
            total_masks.append((ref_src_mask | ref_dst_mask) | cutoff_mask)

    pred_interres_dists = vec_norm(pred_res_src - pred_res_dst, dim=-1)  # n_edge x n_atoms x n_atoms

    dist_ses = [
        (ref_dists - pred_interres_dists).square() * total_mask
        for ref_dists, total_mask in zip(ref_interres_dists, total_masks)
    ]

    min_dist_ses, _ = torch.min(
        torch.stack(dist_ses).sum(dim=(-1, -2)),
        dim=0
    )  # num_edges

    batchwise_dist_ses = scatter(
        min_dist_ses,
        edge_batch,
        dim=0,
        dim_size=(batch.max().item()+1)
    )
    norm = scatter(
        total_masks[0].sum(dim=(-1, -2)),
        edge_batch,
        dim=0,
        dim_size=(batch.max().item()+1)
    )

    return batchwise_dist_ses / norm
