from proteinzen.data.datasets.featurize.sidechain import _dihedrals
from proteinzen.runtime.loss.atomic.common import atom91_to_atom14, atoms_to_angles, atoms_to_torsions
from proteinzen.utils.atom_reps import atom14_clash_radius, atom14_interact_mask


import torch
from torch_geometric.nn import radius_graph, knn_graph
import torch_geometric.utils as pygu

from proteinzen.runtime.loss.utils import _nodewise_to_graphwise, vec_norm


def compute_backbone_connections(bb, batch, eps=1e-8):
    ret_bond_lens = []
    ret_bond_angles = []
    for i in range(batch.max().item()+1):
        select = (batch == i)
        subset_bb = bb[select]

        N, CA, C = subset_bb[:, 0], subset_bb[:, 1], subset_bb[:, 2]
        C_to_N_bond = N[1:] - C[:-1]
        C_to_N_length = torch.linalg.vector_norm(C_to_N_bond + eps, dim=-1)
        ret_bond_lens.append(C_to_N_length)

        CA_C_N = torch.stack([
            CA[:-1], C[:-1], N[1:]
        ], dim=-2)
        CA_C_N_angle = atoms_to_angles(CA_C_N)
        C_N_CA = torch.stack([
            C[:-1], N[1:], CA[1:]
        ], dim=-2)
        C_N_CA_angle = atoms_to_angles(C_N_CA)
        angles = torch.stack([CA_C_N_angle, C_N_CA_angle], dim=-2)
        ret_bond_angles.append(angles)

    return torch.cat(ret_bond_lens, dim=0), torch.cat(ret_bond_angles, dim=0)


def chain_constraints_loss(pred_bb, ref_bb, num_nodes, batch_mask, x_mask, eps=1e-8):
    pred_conn_lens, pred_conn_angles = compute_backbone_connections(pred_bb, batch_mask)
    ref_conn_lens, ref_conn_angles = compute_backbone_connections(ref_bb, batch_mask)

    num_conn = [l-1 for l in num_nodes]
    conn_mask = []
    for i in range(batch_mask.max().item()+1):
        select = (batch_mask == i)
        subset_x_mask = x_mask[select]
        conn_mask.append(subset_x_mask[:-1] | subset_x_mask[1:])
    conn_mask = torch.cat(conn_mask, dim=-1)

    lens_mse = torch.square(pred_conn_lens - ref_conn_lens + eps)
    lens_mse = lens_mse[~conn_mask]
    angle_diff = torch.linalg.vector_norm(pred_conn_angles - ref_conn_angles + eps, dim=-1)
    angle_diff = angle_diff[~conn_mask]

    lens_loss = _nodewise_to_graphwise(lens_mse, num_conn, conn_mask)
    conn_mask_expand = conn_mask[:, None].expand(-1, 2)
    angle_loss = _nodewise_to_graphwise(angle_diff.flatten(), num_conn, conn_mask_expand)
    return lens_loss, angle_loss


def backbone_dihedrals_loss(pred_bb, ref_bb, num_nodes, x_mask, eps=1e-8):
    num_res = pred_bb.shape[0]
    split_pred_bb = pred_bb.split(num_nodes)
    split_ref_bb = ref_bb.split(num_nodes)
    pred_dihedrals = []
    ref_dihedrals = []
    for i, _ in enumerate(num_nodes):
        pred_dihedrals.append(_dihedrals(split_pred_bb[i]))
        ref_dihedrals.append(_dihedrals(split_ref_bb[i]))
    pred_dihedrals = torch.cat(pred_dihedrals, dim=0)
    ref_dihedrals = torch.cat(ref_dihedrals, dim=0)
    # dihedrals are featurized as unit vectors: we can use this to our advantage
    # and compute the vector norms of the diffs
    dihedral_diff = (pred_dihedrals - ref_dihedrals).view([num_res, 3, 2])
    dihedral_diff = torch.linalg.vector_norm(dihedral_diff + eps, dim=-1)
    dihedral_diff = dihedral_diff[~x_mask]
    x_mask_expand = x_mask[:, None].expand(-1, 3)
    return _nodewise_to_graphwise(dihedral_diff.flatten(), num_nodes, x_mask_expand)


def bb_clash_loss(pred_bb,
                  res_mask,
                  batch,
                  interact_cutoff=10,
                  include_chain_edges=False,
                  eps=1e-8):
    pred_bb = pred_bb[:, :4]
    bb_interact_radius = torch.as_tensor(atom14_clash_radius[0, 0, :4, :4], device=pred_bb.device)
    X_ca = pred_bb[:, 1].clone()
    X_ca[~res_mask] = torch.inf
    edge_index = radius_graph(X_ca, r=interact_cutoff, batch=batch, max_num_neighbors=10000)

    select_chain_edges = (torch.abs(edge_index[0] - edge_index[1]) == 1)
    lrange_edge_index = edge_index[:, ~select_chain_edges]

    src = lrange_edge_index[1]
    dst = lrange_edge_index[0]
    bb_src = pred_bb[src]
    bb_dst = pred_bb[dst]

    bb_dist_mat = torch.linalg.vector_norm(
        bb_src[..., None, :] - bb_dst[..., None, :, :] + eps,
        dim=-1
    )  # n_edge x 4 x 4
    clash_dists = torch.where(
        bb_dist_mat <= bb_interact_radius[None],
        bb_interact_radius[None] - bb_dist_mat,
        torch.zeros(1, device=src.device)
    )

    clash_loss = pygu.scatter(
        clash_dists.view(-1, 16),
        src,
        dim=0,
        dim_size=pred_bb.shape[0]
    ).sum(dim=-1)

    clash_loss = _nodewise_to_graphwise(clash_loss, batch, res_mask, reduction='sum')

    if include_chain_edges:
        chain_edges = edge_index[:, select_chain_edges]
        raise NotImplementedError("If you wanna test this you should implement this lmfao")

    return clash_loss * 0.1  # nm


def intersidechain_clash_loss(pred_atom14,
                              atom14_mask,
                              seq,
                              batch,
                              clash_tolerance=1.5,
                              k=30):
    """ Additional loss to push non-bonded atoms
        out of the van der waals radius of other atoms  """
    atom14_clash_radius_torch = torch.as_tensor(atom14_clash_radius, device=pred_atom14.device)

    pred_X_ca = pred_atom14[:, 1].clone()
    res_mask = atom14_mask.any(dim=-1)
    pred_X_ca[~res_mask] = torch.inf
    edge_index = knn_graph(pred_X_ca, k, batch)

    # we need to treat adjacent residues differently
    # we'll just zero out the backbone loss components, since this is generally handled by other losses
    chain_edge = (torch.abs(edge_index[0] - edge_index[1]) < 2)
    edge_batch = batch[edge_index[1]]

    src = edge_index[1]
    dst = edge_index[0]

    res_src = pred_atom14[src].unsqueeze(-2)  # n_edge x n_atoms x 1 x 3
    res_dst = pred_atom14[dst].unsqueeze(-3)  # n_edge x 1 x n_atoms x 3

    atom14_src_mask = atom14_mask[src].unsqueeze(-1)  # n_edge x n_atom x 1
    atom14_dst_mask = atom14_mask[dst].unsqueeze(-2)  # n_edge x 1 x n_atom
    total_mask = atom14_src_mask & atom14_dst_mask
    # total_mask[chain_edge, :4, :4] = False
    total_mask[:, :4, :4] = False

    interres_dists = vec_norm(res_src - res_dst, dim=-1)  # n_edge x n_atoms x n_atoms
    seq_src = seq[src]
    seq_dst = seq[dst]
    clash_dists = atom14_clash_radius_torch[(seq_src, seq_dst)]
    # TODO: allow for di-sulfide bonds?

    # clash_loss_no_tol = torch.where(
    #     interres_dists < clash_dists,
    #     clash_dists - interres_dists,
    #     0
    # )
    # print(clash_loss_no_tol.max())

    clash_loss = torch.where(
        interres_dists < clash_dists - clash_tolerance,
        clash_dists - clash_tolerance - interres_dists,
        0
    )
    clash_loss = clash_loss * total_mask

    # clash = clash_loss.sum(dim=(-1, -2)) > 0
    # print(clash_loss.max())
    # print(clash_loss[clash][:2])
    # print(edge_index[:, clash][:, :2])
    # print(clash_loss.sum(dim=(-1, -2))[clash])

    # print(pred_atom14)
    # print(edge_index)
    # print(interres_dists)
    # print(clash_dists)
    # print(clash_loss)
    # torch.set_printoptions(threshold=100000000)
    # print(clash_loss[torch.nonzero(clash_loss, as_tuple=True)])

    return _nodewise_to_graphwise(clash_loss, edge_batch, total_mask, reduction='sum')
