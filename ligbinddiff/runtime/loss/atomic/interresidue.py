from ligbinddiff.data.datasets.featurize.sidechain import _dihedrals
from ligbinddiff.runtime.loss.atomic.common import atom91_to_atom14, atoms_to_angles, atoms_to_torsions


import torch

from ligbinddiff.runtime.loss.utils import _nodewise_to_graphwise, vec_norm


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


def intersidechain_clash_loss(pred_atom91,
                              seq,
                              edge_index,
                              num_edges,
                              x_mask,
                              clash_dist=2):
    """ Additional loss to push non-bonded atoms
        out of the van der waals radius of other atoms  """

    pred_atom14, atom14_mask = atom91_to_atom14(pred_atom91, seq)
    atom14_mask = atom14_mask.any(dim=-1)
    atom14_mask[x_mask] = True

    res_src = pred_atom14[edge_index[0]].unsqueeze(-2)  # n_edge x n_atoms x 1 x 3
    res_dst = pred_atom14[edge_index[1]].unsqueeze(-3)  # n_edge x 1 x n_atoms x 3

    atom14_src_mask = atom14_mask[edge_index[0]].unsqueeze(-1)  # n_edge x n_atom x 1
    atom14_dst_mask = atom14_mask[edge_index[1]].unsqueeze(-2)  # n_edge x 1 x n_atom
    total_mask = atom14_src_mask | atom14_dst_mask

    interres_dists = vec_norm(res_src - res_dst, dim=-1)  # n_edge x n_atoms x n_atoms
    interres_dists = torch.clip(clash_dist - interres_dists[~total_mask], min=0)
    return _nodewise_to_graphwise(interres_dists, num_edges, total_mask.flatten(-2, -1), reduction='sum')
