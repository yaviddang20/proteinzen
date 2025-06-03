import numpy as np
import torch
from proteinzen.runtime.loss.atomic.common import atoms_to_angles, atoms_to_torsions


from proteinzen.runtime.loss.utils import _nodewise_to_graphwise, vec_norm

from proteinzen.utils.atom_reps import atom91_residue_angles, atom91_residue_bonds, atom91_sidechain_angles, atom91_sidechain_bonds, atom91_start_end, chi_atom_idxs, chi_pi_periodic, nonbonded_sidechain_atom_pairs


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
    msd = _nodewise_to_graphwise(sd, num_nodes, ~select_sidechain_atom)
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
    mse = _nodewise_to_graphwise(se, num_nodes, ~select_sidechain_atom)
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
    bond_length_mse = _nodewise_to_graphwise(bond_length_diff.square(), num_nodes, bond_mask)

    return bond_length_mse


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
    graphwise_loss = _nodewise_to_graphwise(loss, num_nodes, chi_mask)
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
    graphwise_loss = _nodewise_to_graphwise(diff, num_nodes, angle_mask)

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
    graphwise_loss = _nodewise_to_graphwise(dist_diff, num_nodes, total_mask)

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
    return _nodewise_to_graphwise(dist_diff, num_nodes, total_mask, reduction='sum')
