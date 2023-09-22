from ligbinddiff.data.datasets.featurize.sidechain import _dihedrals
from ligbinddiff.runtime.loss.utils import _elemwise_to_graphwise, _nodewise_to_graphwise, vec_norm


import torch

from ligbinddiff.utils.atom_reps import alphabet, atom91_residue_angles, atom91_residue_bonds, atom91_sidechain_angles, atom91_sidechain_bonds, atom91_start_end, chi_atom_idxs, chi_pi_periodic, nonbonded_sidechain_atom_pairs, restype_1to3


def atom91_rmsd_loss(ref_atom91,
                     pred_atom91,
                     num_nodes,
                     atom91_mask,
                     no_bb=True,
                     eps=1e-4):
    """ RMSD on relevant atom91 atoms """
    if no_bb:
        ref_atom91 = ref_atom91[..., 4:, :]
        pred_atom91 = pred_atom91[..., 4:, :]
        atom91_mask = atom91_mask[..., 4:, :]

    select_sidechain_atom = (~atom91_mask).any(dim=-1)
    diff = ref_atom91[select_sidechain_atom] - pred_atom91[select_sidechain_atom]
    # eps for numerical stability
    diff = diff + eps
    sd = torch.square(diff).sum(dim=-1)
    msd = _elemwise_to_graphwise(sd, num_nodes, ~select_sidechain_atom)
    rmsd = torch.sqrt(msd + eps)
    return rmsd


def atom91_mse_loss(ref_atom91,
                     pred_atom91,
                     num_nodes,
                     atom91_mask,
                     no_bb=True,
                     eps=1e-4):
    """ RMSD on relevant atom91 atoms """
    if no_bb:
        ref_atom91 = ref_atom91[..., 4:, :]
        pred_atom91 = pred_atom91[..., 4:, :]
        atom91_mask = atom91_mask[..., 4:, :]

    select_sidechain_atom = (~atom91_mask).any(dim=-1)
    diff = ref_atom91[select_sidechain_atom] - pred_atom91[select_sidechain_atom]
    # eps for numerical stability
    diff = diff + eps
    se = torch.square(diff).sum(dim=-1)
    mse = _elemwise_to_graphwise(se, num_nodes, ~select_sidechain_atom)
    return mse


def bond_length_loss(ref_atom91,
                     pred_atom91,
                     num_nodes,
                     atom91_mask,
                     no_bb=True,
                     eps=1e-6):
    bond_store = atom91_sidechain_bonds if no_bb else atom91_residue_bonds
    bonds = []
    for b_list in bond_store.values():
        bonds += b_list
    bonds = torch.as_tensor(bonds).T  # 2 x n_bond
    bonds = bonds.to(ref_atom91.device)

    src_mask = atom91_mask[:, bonds[0]]  # n_res x n_bond
    dst_mask = atom91_mask[:, bonds[1]]
    bond_mask = src_mask | dst_mask

    ref_src = ref_atom91[:, bonds[0]][~bond_mask]  # (n_res x n_bond) x 3
    ref_dst = ref_atom91[:, bonds[1]][~bond_mask]
    ref_dist = vec_norm(ref_dst - ref_src, dim=-1)  # (n_bond x n_res)

    pred_src = pred_atom91[:, bonds[0]][~bond_mask]  # (n_res x n_bond) x 3
    pred_dst = pred_atom91[:, bonds[1]][~bond_mask]
    pred_dist = vec_norm(pred_dst - pred_src, dim=-1)  # (n_bond x n_res)

    bond_length_diff = ref_dist - pred_dist
    bond_length_mse = _elemwise_to_graphwise(bond_length_diff.square(), num_nodes, bond_mask)

    return bond_length_mse


def atoms_to_torsions(chi_atom_coords, eps=1e-8):
    # Based on https://en.wikipedia.org/wiki/Dihedral_angle

    # absolute atom positions
    c1 = chi_atom_coords[..., 0, :]
    c2 = chi_atom_coords[..., 1, :]
    c3 = chi_atom_coords[..., 2, :]
    c4 = chi_atom_coords[..., 3, :]

    # relative atom positions
    a1 = c2 - c1
    a2 = c3 - c2
    a3 = c4 - c3

    # backbone normals
    v1 = torch.cross(a1, a2)
    v2 = torch.cross(a2, a3)

    # Angle between normals
    x = torch.sum(v1 * v2, -1)
    a2_norm = vec_norm(a2, mask_nans=False)
    y = torch.sum(a1 * v2, dim=-1) * a2_norm

    angle_vec = torch.stack([x, y], dim=-1)
    norm = vec_norm(angle_vec, mask_nans=False).unsqueeze(-1)

    if (angle_vec / norm).isnan().any():
        print("angle vec", angle_vec)
        print("norm", norm)

    angle_vec = angle_vec / norm
    return angle_vec


def torsion_loss(ref_atom91,   # n_res x n_atom x 3
                 pred_atom91,  # n_res x n_atom x 3
                 num_nodes,
                 atom91_mask,  # n_res x n_atom
                 eps=1e-6):
    chi_atom_idx_select = []
    for atom_list in chi_atom_idxs.values():
        chi_atom_idx_select += atom_list
    chi_atom_idx_select = torch.as_tensor(chi_atom_idx_select, device=ref_atom91.device).long() # n_chi x 4
    # print("chi_atom_idx_select", chi_atom_idx_select.shape)

    chi_periodic = []
    for mask in chi_pi_periodic:
        chi_periodic += mask
    chi_periodic = torch.as_tensor(chi_periodic, device=ref_atom91.device).float()  # n_chi
    chi_periodic[chi_periodic == 0] = -1
    chi_periodic = chi_periodic * -1
    # print("chi_periodic", chi_periodic.shape)

    chi_mask = atom91_mask[:, chi_atom_idx_select].any(dim=-1)  # n_res x n_chi
    # print("chi_mask", chi_mask.shape)
    # print(chi_mask)

    ref_chi_atoms = ref_atom91[:, chi_atom_idx_select]  # n_res x n_chi x 4 x 3
    ref_chi_angles = atoms_to_torsions(ref_chi_atoms[~chi_mask])
    # print("ref_chi_atoms", ref_chi_atoms.shape)
    # print("ref_chi_angles", ref_chi_angles.shape)
    pred_chi_atoms = pred_atom91[:, chi_atom_idx_select]  # n_res x n_chi x 4 x 3
    pred_chi_angles = atoms_to_torsions(pred_chi_atoms[~chi_mask])
    # print("pred_chi_atoms", pred_chi_atoms.shape)
    # print("pred_chi_angles", pred_chi_angles.shape)
    pred_chi_angles_p_pi = atoms_to_torsions(pred_chi_atoms)
    # print(pred_chi_angles_p_pi.shape, chi_periodic.shape, chi_mask.shape)
    pred_chi_angles_p_pi = (pred_chi_angles_p_pi * chi_periodic.unsqueeze(0).unsqueeze(-1))[~chi_mask]

    diff = vec_norm(ref_chi_angles - pred_chi_angles, dim=-1)
    pi_diff = vec_norm(ref_chi_angles - pred_chi_angles_p_pi, dim=-1)
    # print(diff, pi_diff)
    # print("diff", diff)
    # print("pi diff", pi_diff)

    loss = torch.minimum(diff, pi_diff)
    graphwise_loss = _elemwise_to_graphwise(loss, num_nodes, chi_mask)
    # reswise_num_elem = (~chi_mask).long().sum(dim=-1)
    # num_elem_per_graph = [t.sum().item() for t in torch.split(reswise_num_elem, num_nodes)]
    # print("chi loss", loss.shape, num_elem_per_graph, num_nodes, graphwise_loss.shape)
    # if graphwise_loss.isnan().any():
    #     torch.set_printoptions(threshold=1000000)
    #     print("loss", loss)
    #     print("num_nodes", num_nodes)
    #     print("graphwise_loss", graphwise_loss)
    #     raise Exception
    #     # print("chi_mask", chi_mask)

    return graphwise_loss


def atoms_to_angles(bond_atom_coords, eps=1e-8):
    # Based on https://en.wikipedia.org/wiki/Dihedral_angle

    # absolute atom positions
    c1 = bond_atom_coords[..., 0, :]
    c2 = bond_atom_coords[..., 1, :]
    c3 = bond_atom_coords[..., 2, :]
    # print("c1", c1)
    # print("c2", c2)
    # print("c3", c3)

    # relative atom positions
    n1 = (c2 - c1) / vec_norm(c2-c1).unsqueeze(-1)
    n2 = (c3 - c2) / vec_norm(c3-c2).unsqueeze(-1)
    # print("n1", n1)
    # print("n2", n2)

    cosX = torch.sum(n1 * n2, dim=-1)
    # print("cosX", cosX)
    sinX_vec = torch.cross(n1, n2)
    # print("sinX_vec", sinX_vec)
    y_axis = torch.cross(sinX_vec, n1)
    # print("y_axis", y_axis)
    sinX_sign = torch.sign(torch.sum(y_axis * n2, dim=-1))
    sinX = vec_norm(sinX_vec, dim=-1) * sinX_sign
    # print("sinX", sinX)

    return torch.stack([cosX, sinX], dim=-1)


def angle_loss(ref_atom91,   # n_res x n_atom x 3
               pred_atom91,  # n_res x n_atom x 3
               num_nodes,
               atom91_mask,  # n_res x n_atom
               no_bb=True,
               eps=1e-6):
    angle_atom_idx_select = []
    angle_store = atom91_sidechain_angles if no_bb else atom91_residue_angles
    for atom_list in angle_store.values():
        angle_atom_idx_select += atom_list
    angle_atom_idx_select = torch.as_tensor(angle_atom_idx_select, device=ref_atom91.device).long() # n_chi x 4

    angle_mask = atom91_mask[:, angle_atom_idx_select].any(dim=-1)  # n_res x n_chi

    ref_angle_atoms = ref_atom91[:, angle_atom_idx_select]  # n_res x n_chi x 4 x 3
    ref_angles = atoms_to_angles(ref_angle_atoms[~angle_mask])
    pred_angle_atoms = pred_atom91[:, angle_atom_idx_select]  # n_res x n_chi x 4 x 3
    pred_angles = atoms_to_angles(pred_angle_atoms[~angle_mask])

    diff = vec_norm(ref_angles - pred_angles, dim=-1)
    graphwise_loss = _elemwise_to_graphwise(diff, num_nodes, angle_mask)

    return graphwise_loss


def distance_loss(ref_atom91,
                  pred_atom91,
                  num_nodes,
                  atom91_mask,
                  no_bb=True,
                  eps=1e-6):
    dist_idxs = []
    for start, end in atom91_start_end.values():
        l = end - start
        idxs = torch.arange(start, end)
        if not no_bb:
            idxs = torch.cat([torch.arange(4), idxs], dim=-1)
        src = idxs.unsqueeze(0).expand(l, -1).reshape(-1)
        dst = idxs.repeat_interleave(l)
        dist_idxs.append(torch.stack([src, dst], dim=0))

    dist_idxs = torch.cat(dist_idxs, dim=-1)
    src = dist_idxs[0]
    dst = dist_idxs[1]

    atom91_src_mask = atom91_mask[:, src]
    atom91_dst_mask = atom91_mask[:, dst]
    total_mask = atom91_src_mask | atom91_dst_mask

    ref_src = ref_atom91[:, src][~total_mask]
    ref_dst = ref_atom91[:, dst][~total_mask]
    ref_dist = vec_norm(ref_src - ref_dst, dim=-1)

    pred_src = pred_atom91[:, src][~total_mask]
    pred_dst = pred_atom91[:, dst][~total_mask]
    pred_dist = vec_norm(pred_src - pred_dst, dim=-1)

    dist_diff = (ref_dist - pred_dist).square() / 2  # we double-count bonds here
    graphwise_loss = _elemwise_to_graphwise(dist_diff, num_nodes, total_mask)

    return graphwise_loss


def intrasidechain_clash_loss(pred_atom91,
                              num_nodes,
                              atom91_mask,
                              clash_dist=2):
    """ Additional loss to push non-bonded atoms
        out of the van der waals radius of other atoms  """

    nonbonded_atom_pairs = torch.as_tensor(nonbonded_sidechain_atom_pairs, device=pred_atom91.device).T  # 2 x n_pairs
    src = nonbonded_atom_pairs[0]
    dst = nonbonded_atom_pairs[1]

    atom91_src_mask = atom91_mask[:, src]
    atom91_dst_mask = atom91_mask[:, dst]
    total_mask = atom91_src_mask | atom91_dst_mask

    pred_src = pred_atom91[:, src][~total_mask]
    pred_dst = pred_atom91[:, dst][~total_mask]
    pred_dist = vec_norm(pred_src - pred_dst, dim=-1)

    dist_diff = torch.clip(clash_dist - pred_dist, min=0)
    return _elemwise_to_graphwise(dist_diff, num_nodes, total_mask, reduction='sum')


def _atom91_to_atom14(atom91, seq):
    num_nodes = atom91.shape[0]
    atom91 = atom91.float()
    atom14 = torch.empty((num_nodes, 14, 3), device=atom91.device) * torch.nan
    atom14[:, :4] = atom91[:, :4]

    for i, aa_1lt in enumerate(alphabet):
        select_aa = (seq == i)
        start, end = atom91_start_end[restype_1to3[aa_1lt]]
        l = end - start
        atom14[select_aa, 4:4+l] = atom91[select_aa, start:end]

    atom14_mask = torch.isnan(atom14)
    atom14[atom14_mask] = 0

    return atom14, atom14_mask


def intersidechain_clash_loss(pred_atom91,
                              seq,
                              edge_index,
                              num_edges,
                              x_mask,
                              clash_dist=2):
    """ Additional loss to push non-bonded atoms
        out of the van der waals radius of other atoms  """

    pred_atom14, atom14_mask = _atom91_to_atom14(pred_atom91, seq)
    atom14_mask = atom14_mask.any(dim=-1)
    atom14_mask[x_mask] = True

    res_src = pred_atom14[edge_index[0]].unsqueeze(-2)  # n_edge x n_atoms x 1 x 3
    res_dst = pred_atom14[edge_index[1]].unsqueeze(-3)  # n_edge x 1 x n_atoms x 3

    atom14_src_mask = atom14_mask[edge_index[0]].unsqueeze(-1)  # n_edge x n_atom x 1
    atom14_dst_mask = atom14_mask[edge_index[1]].unsqueeze(-2)  # n_edge x 1 x n_atom
    total_mask = atom14_src_mask | atom14_dst_mask

    interres_dists = vec_norm(res_src - res_dst, dim=-1)  # n_edge x n_atoms x n_atoms
    interres_dists = torch.clip(clash_dist - interres_dists[~total_mask], min=0)
    return _elemwise_to_graphwise(interres_dists, num_edges, total_mask.flatten(-2, -1), reduction='sum')


def atomic_neighborhood_dist_loss(ref_atom91,
                                  pred_atom91,
                                  seq,
                                  edge_index,
                                  num_edges,
                                  x_mask,
                                  radius_cutoff=6):
    """ Additional loss to push non-bonded atoms
        out of the van der waals radius of other atoms  """
    pred_atom14, atom14_mask = _atom91_to_atom14(pred_atom91, seq)
    ref_atom14, _ = _atom91_to_atom14(ref_atom91, seq)
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

    return _elemwise_to_graphwise(dist_mse, num_edges, total_mask.flatten(-2, -1))


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
    return _elemwise_to_graphwise(dihedral_diff.flatten(), num_nodes, x_mask_expand)


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
    angle_loss = _elemwise_to_graphwise(angle_diff.flatten(), num_conn, conn_mask_expand)
    return lens_loss, angle_loss
