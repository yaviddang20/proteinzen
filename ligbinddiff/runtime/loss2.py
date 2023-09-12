""" Utils for loss functions """

import torch
import torch.nn.functional as F

from ligbinddiff.utils.atom_reps import (
    alphabet, atom91_atom_masks, atom91_start_end, atom91_sidechain_bonds, atom91_residue_bonds,
    nonbonded_sidechain_atom_pairs, atom91_sidechain_angles, atom91_residue_angles, chi_atom_idxs, chi_pi_periodic, restype_1to3)
from .loss import zernike_coeff_loss
from ligbinddiff.data.datasets.featurize.sidechain import _dihedrals


def vec_norm(tensor, mask_nans=True, eps=1e-6, dim=-1):
    if mask_nans:
        tensor = torch.nan_to_num(tensor)
    norm = torch.sum(tensor * tensor, dim=dim)
    norm = (norm + eps).sqrt()
    return norm
    # return torch.nan_to_num(norm)


def _elemwise_to_graphwise(elemwise_tensor, nodes_per_graph, node_elem_mask, reduction='mean'):
    if len(nodes_per_graph) == 1:
        return elemwise_tensor.mean().unsqueeze(0)

    reswise_num_elem = (~node_elem_mask).long().sum(dim=-1)
    num_elem_per_graph = [t.sum().item() for t in torch.split(reswise_num_elem, nodes_per_graph)]
    if reduction == 'mean':
        graphwise_tensor = torch.cat([t.mean().unsqueeze(0) for t in elemwise_tensor.split(num_elem_per_graph)])
    elif reduction == 'sum':
        graphwise_tensor = torch.cat([t.sum().unsqueeze(0) for t in elemwise_tensor.split(num_elem_per_graph)])
    else:
        raise ValueError('reduction must be mean or sum')
    return graphwise_tensor


def _nodewise_to_graphwise(nodewise_tensor, nodes_per_graph, node_mask, reduction='mean'):
    if len(nodes_per_graph) == 1:
        return nodewise_tensor.mean().unsqueeze(0)
    reswise_num = (~node_mask).long()
    num_node_per_graph = [t.sum().item() for t in torch.split(reswise_num, nodes_per_graph)]
    if reduction == 'mean':
        graphwise_tensor = torch.cat([t.mean().unsqueeze(0) for t in nodewise_tensor.split(num_node_per_graph)])
    elif reduction == 'sum':
        graphwise_tensor = torch.cat([t.sum().unsqueeze(0) for t in nodewise_tensor.split(num_node_per_graph)])
    else:
        raise ValueError('reduction must be mean or sum')
    return graphwise_tensor


def seq_cce_loss(ref_seq,
                 seq_logits,
                 num_nodes,
                 mask):
    """ CCE loss on logits """
    ref_seq = ref_seq[~mask]
    seq_logits = seq_logits[~mask]
    cce = F.cross_entropy(seq_logits, ref_seq, reduction='none')
    return _nodewise_to_graphwise(cce, num_nodes, mask)


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


### TODO: move this somewhere better
def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


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


## TODO: use crystallographic numbers instead of this arbitrary constant clip_min
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


## TODO: use crystallographic numbers instead of this arbitrary constant clip_min
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


def cath_density_loss_fn(noised_batch, model_outputs, n_channels=4, use_channel_weights=False):
    density_dict = noised_batch['density']
    noised_density_dict = noised_batch['noised_density']

    seq = noised_batch['seq']
    x_mask = noised_batch['x_mask']
    atom91_centered = noised_batch['atom91_centered']
    atom91_mask = noised_batch['atom91_mask']

    denoised_density = model_outputs['denoised_density']
    seq_logits = model_outputs['seq_logits']
    pred_atom91 = model_outputs['decoded_atom91']

    if use_channel_weights:
        # compute channel weights for zernike loss
        channel_values = torch.as_tensor([
            atom91_atom_masks[atom][4:] for atom in ['C', 'N', 'O', 'S']
        ], dtype=torch.bool).T  # n_atom x 4
        atom_mask = ~atom91_mask.any(dim=-1, keepdim=True)  # n_res x n_atom x 1

        channel_atom_mask = channel_values.unsqueeze(0).to(atom_mask.device) * atom_mask[..., 4:, :]
        channel_atom_count = channel_atom_mask.float().sum(dim=-2)

        # normalize channel contribution across atom type
        channel_atom_count_per_channel = channel_atom_count.sum(dim=-2)
        total_atoms = channel_atom_count.sum()
        # to avoid divide-by-zero
        channel_atom_count_per_channel[channel_atom_count_per_channel == 0] = 1
        # normalize channel contribution across atom type
        channel_weights = channel_atom_count / channel_atom_count_per_channel * total_atoms
        channel_weights[channel_atom_count == 0] = 1  # retain pushing weights towards 0 for no density
    else:
        channel_weights = None

    data_splits = noised_batch._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    denoising_loss = zernike_coeff_loss(density_dict, denoised_density, data_lens, x_mask, n_channels=n_channels, channel_weights=channel_weights)
    ref_noise = zernike_coeff_loss(density_dict, noised_density_dict, data_lens, x_mask, n_channels=n_channels, channel_weights=channel_weights)
    seq_loss = seq_cce_loss(seq, seq_logits, data_lens, x_mask)
    atom91_mse = atom91_mse_loss(atom91_centered, pred_atom91, data_lens, atom91_mask)
    atom91_rmsd = atom91_rmsd_loss(atom91_centered, pred_atom91, data_lens, atom91_mask)
    bond_length_mse = bond_length_loss(atom91_centered, pred_atom91, data_lens, atom91_mask.any(dim=-1))
    sidechain_dists_mse = distance_loss(atom91_centered, pred_atom91, data_lens, atom91_mask.any(dim=-1))
    bond_angle_loss = angle_loss(atom91_centered, pred_atom91, data_lens, atom91_mask.any(dim=-1))
    chi_loss = torsion_loss(atom91_centered, pred_atom91, data_lens, atom91_mask.any(dim=-1))
    # try:
    # except Exception as e:
    #     import ligbinddiff.utils.atom_reps as atom_reps
    #     import numpy as np
    #     chi_atom_idx_select = []
    #     for atom_list in chi_atom_idxs.values():
    #         chi_atom_idx_select += atom_list
    #     chi_atom_idx_select = torch.as_tensor(chi_atom_idx_select).long() # n_chi x 4

    #     seq = "".join([atom_reps.num_to_letter[i] for i in seq.tolist()])
    #     seq_3lt = [atom_reps.restype_1to3[c] for c in seq]
    #     num_chi = torch.tensor([len(atom_reps.chi_angles_atoms[aa]) for aa in seq_3lt])
    #     num_chi[x_mask] = 0
    #     chi_mask = atom91_mask.any(dim=-1)[:, chi_atom_idx_select].any(dim=-1)  # n_res x n_chi
    #     chi_mask_num = (~chi_mask).sum(dim=-1)
    #     print()
    #     chi_match = (num_chi == chi_mask_num.to(num_chi.device))
    #     print("expected num chi?", chi_match.all())
    #     if not chi_match.all():
    #         print(noised_batch.name)
    #         # print("coords", atom91_centered[~chi_match.cuda()])
    #         # print("coords mask", atom91_mask[~chi_match.cuda()])
    #         print(np.array([c for c in seq])[(~chi_match).numpy(force=True)])
    #         print(torch.arange(len(chi_match))[~chi_match.cpu()])
    #         print(torch.stack([num_chi, chi_mask_num.cpu()])[:, ~chi_match.cpu()])
    #         print(chi_mask[~chi_match])

    #     raise e

    with torch.no_grad():
        correct_label_atom91_mask = atom91_mask.clone()
        pred_seq = seq_logits.argmax(dim=-1)
        same = (pred_seq == seq)
        correct_label_atom91_mask[~same] = True

        correct_label_x_mask = x_mask.clone()
        correct_label_x_mask[~same] = True

        cl_denoising_loss = zernike_coeff_loss(density_dict, denoised_density, data_lens, correct_label_x_mask, n_channels=n_channels, channel_weights=channel_weights)
        cl_atom91_rmsd = atom91_rmsd_loss(atom91_centered, pred_atom91, data_lens, correct_label_atom91_mask)
        cl_bond_length_mse = bond_length_loss(atom91_centered, pred_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
        cl_bond_angle_loss = angle_loss(atom91_centered, pred_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
        cl_chi_loss = torsion_loss(atom91_centered, pred_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
        cl_sidechain_dists_mse = distance_loss(atom91_centered, pred_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))

    unscaled_loss = (
        denoising_loss + seq_loss + atom91_mse +
        bond_length_mse + sidechain_dists_mse +
        bond_angle_loss + chi_loss
    )
    loss = (noised_batch['loss_weight'] * unscaled_loss).mean()
    return {
        "loss": loss,
        "denoising_loss": denoising_loss,
        "ref_noise": ref_noise,
        "seq_loss": seq_loss,
        "atom91_rmsd": atom91_rmsd,
        "bond_length_mse": bond_length_mse,
        "sidechain_dists_mse": sidechain_dists_mse,
        "bond_angle_loss": bond_angle_loss,
        "chi_loss": chi_loss,
        "cl_denoising_loss": cl_denoising_loss,
        "cl_atom91_rmsd": cl_atom91_rmsd,
        "cl_sidechain_dists_mse": cl_sidechain_dists_mse,
        "cl_bond_length_mse": cl_bond_length_mse,
        "cl_bond_angle_loss": cl_bond_angle_loss,
        "cl_chi_loss": cl_chi_loss
    }


def cath_superposition_loss_fn(noised_batch, model_outputs, use_channel_weights=None):
    atom91_centered = noised_batch['atom91_centered']
    noised_atom91 = noised_batch['noised_atom91']

    seq = noised_batch['seq']
    x_mask = noised_batch['x_mask']
    atom91_mask = noised_batch['atom91_mask']

    denoised_atom91 = model_outputs['denoised_atom91']
    seq_logits = model_outputs['seq_logits']

    data_splits = noised_batch._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    denoising_loss = atom91_mse_loss(atom91_centered, denoised_atom91, data_lens, atom91_mask)
    ref_noise = atom91_rmsd_loss(atom91_centered, noised_atom91, data_lens, atom91_mask)
    seq_loss = seq_cce_loss(seq, seq_logits, data_lens, x_mask)
    atom91_rmsd = atom91_rmsd_loss(atom91_centered, denoised_atom91, data_lens, atom91_mask)
    bond_length_mse = bond_length_loss(atom91_centered, denoised_atom91, data_lens, atom91_mask.any(dim=-1))
    sidechain_dists_mse = distance_loss(atom91_centered, denoised_atom91, data_lens, atom91_mask.any(dim=-1))
    bond_angle_loss = angle_loss(atom91_centered, denoised_atom91, data_lens, atom91_mask.any(dim=-1))
    chi_loss = torsion_loss(atom91_centered, denoised_atom91, data_lens, atom91_mask.any(dim=-1))

    with torch.no_grad():
        correct_label_atom91_mask = atom91_mask.clone()
        pred_seq = seq_logits.argmax(dim=-1)
        same = (pred_seq == seq)
        correct_label_atom91_mask[~same] = True

        cl_denoising_loss = atom91_mse_loss(atom91_centered, denoised_atom91, data_lens, correct_label_atom91_mask)
        cl_atom91_rmsd = atom91_rmsd_loss(atom91_centered, denoised_atom91, data_lens, correct_label_atom91_mask)
        cl_bond_length_mse = bond_length_loss(atom91_centered, denoised_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
        cl_bond_angle_loss = angle_loss(atom91_centered, denoised_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
        cl_chi_loss = torsion_loss(atom91_centered, denoised_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
        cl_sidechain_dists_mse = distance_loss(atom91_centered, denoised_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))

    unscaled_loss = (
        denoising_loss + seq_loss + # atom91_rmsd +
        bond_length_mse + sidechain_dists_mse +
        bond_angle_loss + chi_loss
    )
    loss = (noised_batch['loss_weight'] * unscaled_loss).mean()
    return {
        "loss": loss,
        "denoising_loss": denoising_loss,
        "ref_noise": ref_noise,
        "seq_loss": seq_loss,
        "atom91_rmsd": atom91_rmsd,
        "bond_length_mse": bond_length_mse,
        "sidechain_dists_mse": sidechain_dists_mse,
        "bond_angle_loss": bond_angle_loss,
        "chi_loss": chi_loss,
        "cl_denoising_loss": cl_denoising_loss,
        "cl_atom91_rmsd": cl_atom91_rmsd,
        "cl_sidechain_dists_mse": cl_sidechain_dists_mse,
        "cl_bond_length_mse": cl_bond_length_mse,
        "cl_bond_angle_loss": cl_bond_angle_loss,
        "cl_chi_loss": cl_chi_loss
    }


def so3_embedding_mse(ref_so3, pred_so3, num_nodes, x_mask):
    vec_diff = ref_so3.embedding[~x_mask] - pred_so3.embedding[~x_mask]
    splits = []
    for lmax in ref_so3.lmax_list:
        for l in range(lmax+1):
            splits.append(2*l+1)
    vec_diffs = vec_diff.split(splits, dim=1)
    vec_diff_norms = [vec_norm(v.transpose(-1, -2)) for v in vec_diffs]  # vector dim as final dim
    nodewise_loss = torch.cat(vec_diff_norms, dim=-1).sum(dim=-1)

    return _nodewise_to_graphwise(nodewise_loss, num_nodes, x_mask)


def so3_embedding_kl(so3_mu, so3_logvar, num_nodes, x_mask):
    splits = []
    for lmax in so3_mu.lmax_list:
        for l in range(lmax+1):
            splits.append(2*l+1)
    split_mu = so3_mu.embedding[~x_mask].split(splits, dim=1)
    split_logvar = so3_logvar.embedding[~x_mask].split(splits, dim=1)
    kl_div = 0
    for m, mu, logvar in zip(splits, split_mu, split_logvar):
        comp_kl_div = -0.5 * (logvar.sum(dim=-2) - mu.square().sum(dim=-2) - logvar.exp().sum(dim=-2) + m)
        kl_div = kl_div + comp_kl_div.mean(dim=-1)

    return _nodewise_to_graphwise(kl_div, num_nodes, x_mask)


def generator_loss(model_outputs, num_nodes, x_mask):
    discrim_logits_real = model_outputs['discrim_logits_real'][~x_mask]
    discrim_logits_fake = model_outputs['discrim_logits_fake'][~x_mask]
    logprobs_real_incorrect = discrim_logits_real[:, 0]
    logprobs_fake_incorrect = discrim_logits_fake[:, 1]
    total_logprobs = logprobs_real_incorrect + logprobs_fake_incorrect
    return -_nodewise_to_graphwise(total_logprobs, num_nodes, x_mask)


def cath_latent_loss_fn(noised_batch, model_outputs, use_channel_weights=None, warmup=False, ae_loss_weight=1):
    atom91_centered = noised_batch['atom91_centered']
    seq = noised_batch['seq']
    X_ca = noised_batch['x']
    bb = noised_batch['bb']
    bb_rel = bb - X_ca.unsqueeze(-2)
    x_mask = noised_batch['x_mask']
    atom91_mask = noised_batch['atom91_mask']

    latent = model_outputs['latent']
    latent_mu = model_outputs['latent_mu']
    latent_logvar = model_outputs['latent_logvar']

    decoded_atom91 = model_outputs['decoded_latent']
    decoded_atom91[..., :4, :] = bb_rel
    seq_logits = model_outputs['decoded_seq_logits']

    data_splits = noised_batch._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    autoencoder_loss = atom91_mse_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask, no_bb=False)
    seq_loss = seq_cce_loss(seq, seq_logits, data_lens, x_mask)
    atom91_rmsd = autoencoder_loss.sqrt()
    sidechain_dists_mse = distance_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1), no_bb=False)
    bond_length_mse = bond_length_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1), no_bb=False)
    bond_angle_loss = angle_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1), no_bb=False)
    chi_loss = torsion_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1))
    kl_div = so3_embedding_kl(latent_mu, latent_logvar, data_lens, x_mask)

    edge_splits = noised_batch._slice_dict['edge_index']
    num_edges = (edge_splits[1:] - edge_splits[:-1]).tolist()

    intrares_clash_loss = intrasidechain_clash_loss(decoded_atom91, data_lens, atom91_mask.any(dim=-1))
    interres_clash_loss = intersidechain_clash_loss(
        decoded_atom91 + X_ca.unsqueeze(-2),
        seq,
        noised_batch.edge_index,
        num_edges,
        x_mask
    )
    local_atomic_dist_loss = atomic_neighborhood_dist_loss(
        atom91_centered + X_ca.unsqueeze(-2),
        decoded_atom91 + X_ca.unsqueeze(-2),
        seq,
        noised_batch.edge_index,
        num_edges,
        x_mask
    )
    # print(kl_div)

    if 'discrim_logits_real' in model_outputs:
        gen_loss = generator_loss(model_outputs, data_lens, x_mask)
    else:
        gen_loss = torch.zeros(1, device=chi_loss.device)

    correct_label_atom91_mask = atom91_mask.clone()
    pred_seq = seq_logits.argmax(dim=-1)
    same = (pred_seq == seq)
    correct_label_atom91_mask[~same] = True

    with torch.no_grad():
        cl_atom91_rmsd = atom91_rmsd_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask, no_bb=False)
        cl_bond_length_mse = bond_length_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1), no_bb=False)
        cl_sidechain_dists_mse = distance_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1), no_bb=False)
        cl_bond_angle_loss = angle_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1), no_bb=False)
        cl_chi_loss = torsion_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
        cl_intrares_clash_loss = intrasidechain_clash_loss(decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
        cl_interres_clash_loss = intersidechain_clash_loss(
            decoded_atom91 + X_ca.unsqueeze(-2),
            seq,
            noised_batch.edge_index,
            num_edges,
            x_mask | (~same)
        )
        cl_local_atomic_dist_loss = atomic_neighborhood_dist_loss(
            atom91_centered + X_ca.unsqueeze(-2),
            decoded_atom91 + X_ca.unsqueeze(-2),
            seq,
            noised_batch.edge_index,
            num_edges,
            x_mask | (~same)
        )

    if not warmup:
        noised_latent = model_outputs['noised_latent']
        denoised_latent = model_outputs['denoised_latent']

        denoising_loss = so3_embedding_mse(latent, denoised_latent, data_lens, x_mask)
        ref_noise = so3_embedding_mse(latent, noised_latent, data_lens, x_mask)

        correct_label_x_mask = x_mask.clone()
        correct_label_x_mask[~same] = True
        with torch.no_grad():
            cl_denoising_loss = so3_embedding_mse(latent, denoised_latent, data_lens, correct_label_x_mask)

    else:
        denoising_loss = torch.zeros(1, device=chi_loss.device)
        cl_denoising_loss = torch.zeros(1, device=chi_loss.device)
        ref_noise = torch.zeros(1, device=chi_loss.device)

    vae_loss = (
        autoencoder_loss + 1e-4 * kl_div +
        seq_loss +  # atom91_rmsd +
        bond_length_mse + sidechain_dists_mse +
        bond_angle_loss + chi_loss +
        gen_loss # +
        # intrares_clash_loss + interres_clash_loss + local_atomic_dist_loss
    )
    if not warmup:
        loss = (noised_batch['loss_weight'] * denoising_loss + vae_loss * ae_loss_weight).mean()
    else:
        noised_batch['t'] = torch.zeros(1)
        loss = vae_loss.mean()

    return {
        "loss": loss,
        "denoising_loss": denoising_loss,
        "ref_noise": ref_noise,
        "seq_loss": seq_loss,
        "atom91_mse": autoencoder_loss,
        "atom91_rmsd": atom91_rmsd,
        "sidechain_dists_mse": sidechain_dists_mse,
        "bond_length_mse": bond_length_mse,
        "bond_angle_loss": bond_angle_loss,
        "gen_loss": gen_loss,
        "chi_loss": chi_loss,
        "intrares_clash_loss": intrares_clash_loss,
        "interres_clash_loss": interres_clash_loss,
        "local_atomic_dist_loss": local_atomic_dist_loss,
        "cl_denoising_loss": cl_denoising_loss,
        "cl_atom91_rmsd": cl_atom91_rmsd,
        "cl_sidechain_dists_mse": cl_sidechain_dists_mse,
        "cl_bond_length_mse": cl_bond_length_mse,
        "cl_bond_angle_loss": cl_bond_angle_loss,
        "cl_chi_loss": cl_chi_loss,
        "cl_intrares_clash_loss": cl_intrares_clash_loss,
        "cl_interres_clash_loss": cl_interres_clash_loss,
        "cl_local_atomic_dist_loss": cl_local_atomic_dist_loss,
    }


def discriminator_loss(model_outputs):
    data_splits = model_outputs._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()
    x_mask = model_outputs['x_mask']

    discrim_logits_real = model_outputs['discrim_logits_real'][~x_mask]
    discrim_logits_fake = model_outputs['discrim_logits_fake'][~x_mask]
    logprobs_real_correct = discrim_logits_real[:, 1]
    logprobs_fake_correct = discrim_logits_fake[:, 0]
    total_logprobs = logprobs_real_correct + logprobs_fake_correct
    return -_nodewise_to_graphwise(total_logprobs, data_lens, x_mask)


def backbone_dihedrals_loss(pred_bb, ref_bb, num_nodes, x_mask, eps=1e-8):
    num_res = pred_bb.shape[0]
    pred_dihedrals = _dihedrals(pred_bb)
    ref_dihedrals = _dihedrals(ref_bb)
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


def backbone_connection_loss(pred_bb, ref_bb, num_nodes, batch_mask, x_mask, eps=1e-8):
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


def inpaint_latent_loss_fn(batch,
                           latent_outputs,
                           decoder_outputs,
                           use_channel_weights=None,
                           warmup=False,
                           ae_loss_weight=1,
                           inpaint_k=30,
                           absolute_error=False):
    ### extract important values

    residx = latent_outputs['noised_residx']

    atom91_centered = batch['atom91_centered']
    bb = batch['bb']
    seq = batch['seq']
    X_ca = batch['x']
    x_mask = batch['x_mask']
    atom91_mask = batch['atom91_mask']

    latent = latent_outputs['latent_sidechain']
    latent_mu = latent_outputs['latent_mu']
    latent_logvar = latent_outputs['latent_logvar']
    denoised_bb = latent_outputs['denoised_bb']


    ### autoencoder loss

    if absolute_error:
        atom91 = atom91_centered + X_ca.unsqueeze(-2)
        denoised_x_ca = denoised_bb[..., 1, :]
        decoded_atom91 = decoder_outputs['decoded_latent'] + denoised_x_ca.unsqueeze(-2)
        decoded_atom91[..., :4, :] = denoised_bb
        seq_logits = decoder_outputs['decoded_seq_logits']
    else:
        atom91 = atom91_centered
        decoded_atom91 = decoder_outputs['decoded_latent']
        bb_rel = bb - X_ca.unsqueeze(-2)
        decoded_atom91[..., :4, :] = bb_rel
        seq_logits = decoder_outputs['decoded_seq_logits']

    data_splits = batch._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    autoencoder_loss = atom91_mse_loss(atom91, decoded_atom91, data_lens, atom91_mask, no_bb=False)
    seq_loss = seq_cce_loss(seq, seq_logits, data_lens, x_mask)
    atom91_rmsd = autoencoder_loss.sqrt()
    sidechain_dists_mse = distance_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1), no_bb=False)
    bond_length_mse = bond_length_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1), no_bb=False)
    bond_angle_loss = angle_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1), no_bb=False)
    chi_loss = torsion_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1))
    kl_div = so3_embedding_kl(latent_mu, latent_logvar, data_lens, x_mask)

    edge_splits = batch._slice_dict['edge_index']
    num_edges = (edge_splits[1:] - edge_splits[:-1]).tolist()

    intrares_clash_loss = intrasidechain_clash_loss(decoded_atom91, data_lens, atom91_mask.any(dim=-1))
    interres_clash_loss = intersidechain_clash_loss(
        decoded_atom91,
        seq,
        batch.edge_index,
        num_edges,
        x_mask
    )
    local_atomic_dist_loss = atomic_neighborhood_dist_loss(
        atom91,
        decoded_atom91,
        seq,
        batch.edge_index,
        num_edges,
        x_mask
    )
    # print(kl_div)

    correct_label_atom91_mask = atom91_mask.clone()
    pred_seq = seq_logits.argmax(dim=-1)
    same = (pred_seq == seq)
    correct_label_atom91_mask[~same] = True

    with torch.no_grad():
        cl_atom91_rmsd = atom91_rmsd_loss(atom91, decoded_atom91, data_lens, correct_label_atom91_mask, no_bb=False)
        cl_bond_length_mse = bond_length_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1), no_bb=False)
        cl_sidechain_dists_mse = distance_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1), no_bb=False)
        cl_bond_angle_loss = angle_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
        cl_chi_loss = torsion_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
        cl_intrares_clash_loss = intrasidechain_clash_loss(decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
        cl_interres_clash_loss = intersidechain_clash_loss(
            decoded_atom91 + X_ca.unsqueeze(-2),
            seq,
            batch.edge_index,
            num_edges,
            x_mask | (~same)
        )
        cl_local_atomic_dist_loss = atomic_neighborhood_dist_loss(
            atom91,
            decoded_atom91,
            seq,
            batch.edge_index,
            num_edges,
            x_mask | (~same)
        )


    ### diffusion losses

    noised_latent = latent_outputs['noised_latent_sidechain']
    denoised_latent = latent_outputs['denoised_latent_sidechain']

    noised_bb = latent_outputs['noised_bb'][residx]
    denoised_bb = latent_outputs['denoised_bb'][residx]

    latent_denoising_loss = so3_embedding_mse(latent, denoised_latent, data_lens, x_mask)
    latent_ref_noise = so3_embedding_mse(latent, noised_latent, data_lens, x_mask)

    residx_mask = x_mask[residx]
    residx_num_nodes = [inpaint_k + 1 for _ in range(len(residx)//(inpaint_k+1))]
    bb_denoising_loss = _nodewise_to_graphwise(
        (denoised_bb - bb[residx]).square().mean(dim=-1),
        residx_num_nodes,
        residx_mask
    )
    bb_ref_noise = _nodewise_to_graphwise(
        (noised_bb - bb[residx]).square().mean(dim=-1),
        residx_num_nodes,
        residx_mask
    )

    residx_x_mask = torch.ones_like(x_mask).bool()
    residx_x_mask[residx] = False
    # we wanna unmask the +1 and -1 residues so we can get the relative positionings
    # in context with the un-noised structure
    residx_p1 = residx + 1
    residx_p1 = residx_p1[residx_p1 < len(x_mask)]  # prevent selecting a non-existant residue
    residx_x_mask[residx_p1] = False
    residx_m1 = residx - 1
    residx_m1 = residx_m1[residx_m1 > -1]  # prevent selecting a non-existant residue
    residx_x_mask[residx_m1] = False
    # remasked residues
    residx_x_mask[x_mask] = True

    bb_dihedrals_loss = backbone_dihedrals_loss(
        latent_outputs['denoised_bb'],
        bb,
        data_lens,
        residx_x_mask)

    bb_conn_lens, bb_conn_angles = backbone_connection_loss(
        latent_outputs['denoised_bb'],
        bb,
        data_lens,
        batch.batch,
        residx_x_mask)

    bb_denoising_loss = bb_denoising_loss + bb_dihedrals_loss + bb_conn_lens + bb_conn_angles

    correct_label_x_mask = x_mask.clone()
    correct_label_x_mask[~same] = True
    correct_label_residx_mask = correct_label_x_mask[residx]
    with torch.no_grad():
        cl_latent_denoising_loss = so3_embedding_mse(latent, denoised_latent, data_lens, correct_label_x_mask)
        cl_latent_ref_noise = so3_embedding_mse(latent, noised_latent, data_lens, correct_label_x_mask)

        # cl_x_denoising_loss = _nodewise_to_graphwise(
        #     (denoised_x - X_ca[residx]).square().mean(dim=-1),
        #     [30 + 1 for _ in range(len(residx)//31)],
        #     correct_label_residx_mask
        # )
        # cl_x_ref_noise = _nodewise_to_graphwise(
        #     (noised_x - X_ca[residx]).square().mean(dim=-1),
        #     [30 + 1 for _ in range(len(residx)//31)],
        #     correct_label_residx_mask
        # )
        cl_denoising_loss = cl_latent_denoising_loss + 0 #cl_x_denoising_loss


    ### construct losses

    vae_loss = (
        autoencoder_loss + 1e-6 * kl_div +
        seq_loss +  # atom91_rmsd +
        bond_length_mse + sidechain_dists_mse +
        bond_angle_loss + chi_loss  #+
        # intrares_clash_loss + interres_clash_loss + local_atomic_dist_loss
    )
    # if 'sidechain_loss_weight' not in latent_outputs:
    #     latent_outputs['sidechain_loss_weight'] = 1
    # if 'bb_loss_weight' not in latent_outputs:
    #     latent_outputs['bb_loss_weight'] = 1

    denoising_loss = latent_outputs['sidechain_loss_weight'] * latent_denoising_loss + latent_outputs['bb_loss_weight'] * bb_denoising_loss
    if not warmup:
        loss = (denoising_loss + vae_loss * ae_loss_weight).mean()
    else:
        loss = vae_loss.mean()

    return {
        "loss": loss,
        "denoising_loss": denoising_loss,
        "bb_denoising_loss": bb_denoising_loss,
        "latent_denoising_loss": latent_denoising_loss,
        "bb_ref_noise": bb_ref_noise,
        "latent_ref_noise": latent_ref_noise,
        "bb_dihedrals_loss": bb_dihedrals_loss,
        "bb_connection_dist_mse": bb_conn_lens,
        "bb_connection_angle_loss": bb_conn_angles,
        "seq_loss": seq_loss,
        "atom91_mse": autoencoder_loss,
        "atom91_rmsd": atom91_rmsd,
        "sidechain_dists_mse": sidechain_dists_mse,
        "bond_length_mse": bond_length_mse,
        "bond_angle_loss": bond_angle_loss,
        "chi_loss": chi_loss,
        "intrares_clash_loss": intrares_clash_loss,
        "interres_clash_loss": interres_clash_loss,
        "local_atomic_dist_loss": local_atomic_dist_loss,
        "cl_denoising_loss": cl_denoising_loss,
        "cl_atom91_rmsd": cl_atom91_rmsd,
        "cl_sidechain_dists_mse": cl_sidechain_dists_mse,
        "cl_bond_length_mse": cl_bond_length_mse,
        "cl_bond_angle_loss": cl_bond_angle_loss,
        "cl_chi_loss": cl_chi_loss,
        "cl_intrares_clash_loss": cl_intrares_clash_loss,
        "cl_interres_clash_loss": cl_interres_clash_loss,
        "cl_local_atomic_dist_loss": cl_local_atomic_dist_loss,
    }
