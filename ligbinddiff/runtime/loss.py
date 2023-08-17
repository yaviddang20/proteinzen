""" Utils for loss functions """

import torch
import torch.nn.functional as F

from ligbinddiff.utils.type_l import type_l_add, type_l_sub, type_l_apply
from ligbinddiff.utils.fiber import compact_fiber_to_nl
from ligbinddiff.utils.atom_reps import atom91_atom_masks, atom91_start_end, atom91_bonds, atom91_angles, chi_atom_idxs, chi_pi_periodic


def zernike_coeff_loss(ref_density, pred_density, mask, n_channels=1, channel_weights=None, eps=1e-6):
    """ Invariant MSE loss on Zernike coeffs
    Average of norms of the difference between all type-l vectors """
    ref_density = compact_fiber_to_nl(ref_density, n_channels=n_channels)
    pred_density = compact_fiber_to_nl(pred_density, n_channels=n_channels)

    # only compute residues where all atoms are present
    ref_density = type_l_apply(lambda x: x[~mask], ref_density)
    pred_density = type_l_apply(lambda x: x[~mask], pred_density)

    # print({k: v.abs().max() for k, v in ref_density.items()})
    # print({k: v.abs().max() for k, v in pred_density.items()})
    diff = type_l_sub(ref_density, pred_density)

    # eps for numerical stability
    diff = type_l_apply(lambda t: t + eps, diff)

    square_diff = type_l_apply(torch.square, diff)
    loss = 0
    numel = 0
    for (n, l), elems in square_diff.items():
        if (n - l) % 2 == 1:
            continue
        mags = elems.sum(dim=-1).sqrt()
        if channel_weights is not None:
            # print(mags.shape, channel_weights[~mask].shape)
            mags = mags * channel_weights[~mask]
        loss = loss + mags.sum()
        numel += mags.numel()

    return loss / numel


def seq_cce_loss(ref_seq, seq_logits, mask):
    """ CCE loss on logits """
    ref_seq = ref_seq[~mask]
    seq_logits = seq_logits[~mask]
    return F.cross_entropy(seq_logits, ref_seq)


def atom91_rmsd_loss(ref_atom91, pred_atom91, atom91_mask, eps=1e-4):
    """ RMSD on relevant atom91 atoms """
    diff = ref_atom91[..., 4:, :][~atom91_mask[..., 4:, :]] - pred_atom91[..., 4:, :][~atom91_mask[..., 4:, :]]
    # eps for numerical stability
    diff = diff + eps
    sd = torch.square(diff)
    msd = torch.mean(sd)
    rmsd = torch.sqrt(msd + eps)
    return rmsd


def density_consistency(pred_density, atoms, atom_mask, channel_mask, zt):
    """ Compute the value of the density at the given atomic coordinates

    NOTE: density cannot be in batched form, since otherwise you might get
    competing contributions from residues in other batches """

    pred_rho = zt.reswise_back_transform(pred_density, device=atoms.device)
    print("density consistency ins", atoms.shape, atom_mask.shape, channel_mask.shape)
    poswise_rho = pred_rho(atoms, atom_mask, channel_mask)
    print("density consistency out", poswise_rho.shape)
    return poswise_rho.sum()


def bond_length_loss(ref_atom91, pred_atom91, atom91_mask, eps=1e-6):
    bonds = []
    for b_list in atom91_bonds.values():
        bonds += b_list
    bonds = torch.as_tensor(bonds).T  # 2 x n_bond
    bonds = bonds.to(ref_atom91.device)
    # torch.set_printoptions(threshold=100000)
    # print("bonds", bonds)
    # print("atom91_mask", atom91_mask)

    src_mask = atom91_mask[:, bonds[0]]  # n_res x n_bond
    dst_mask = atom91_mask[:, bonds[1]]
    bond_mask = src_mask | dst_mask
    # print("bond_mask", bond_mask)

    ref_src = ref_atom91[:, bonds[0]]  # n_res x n_bond x 3
    ref_dst = ref_atom91[:, bonds[1]]
    ref_dist = torch.linalg.vector_norm(ref_dst - ref_src, dim=-1)  # n_bond x n_res
    # print("ref_dist no mask", ref_dist)
    # print("ref_dist mask", ref_dist)
    # print(ref_dist.shape)

    pred_src = pred_atom91[:, bonds[0]]  # n_res x n_bond x 3
    pred_dst = pred_atom91[:, bonds[1]]
    pred_dist = torch.linalg.vector_norm(pred_dst - pred_src, dim=-1)  # n_bond x n_res
    # print("pred_dist no mask", pred_dist)
    # print(pred_dist.shape)
    # print("pred_dist mask", pred_dist)

    bond_length_diff = ref_dist - pred_dist
    bond_length_mse = torch.sum((bond_length_diff[~bond_mask]) ** 2) / (~bond_mask).long().sum()
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
    y = torch.sum(a1 * v2, dim=-1) * torch.norm(a2, dim=-1)
    angle_vec = torch.stack([x, y], dim=-1)

    norm = (torch.norm(angle_vec, dim=-1, keepdim=True) + eps)
    if (angle_vec / norm).isnan().any():
        print("angle vec", angle_vec)
        print("norm", norm)

    angle_vec = angle_vec / norm
    return angle_vec

def torsion_loss(ref_atom91,   # n_res x n_atom x 3
                 pred_atom91,  # n_res x n_atom x 3
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

    diff = torch.linalg.vector_norm(ref_chi_angles - pred_chi_angles, dim=-1)
    pi_diff = torch.linalg.vector_norm(ref_chi_angles - pred_chi_angles_p_pi, dim=-1)
    # print(diff, pi_diff)
    # print("diff", diff)
    # print("pi diff", pi_diff)

    loss = torch.minimum(diff, pi_diff)
    # print(loss)

    return loss.mean()


def atoms_to_angles(bond_atom_coords, eps=1e-8):
    # Based on https://en.wikipedia.org/wiki/Dihedral_angle

    # absolute atom positions
    c1 = bond_atom_coords[..., 0, :]
    c2 = bond_atom_coords[..., 1, :]
    c3 = bond_atom_coords[..., 2, :]

    # relative atom positions
    n1 = _normalize(c2 - c1)
    n2 = _normalize(c3 - c2)

    cosX = torch.sum(n1 * n2, dim=-1)
    sinX_vec = torch.cross(n1, n2)
    y_axis = torch.cross(sinX_vec, n1)
    sinX_sign = torch.sign(torch.sum(y_axis * n2, dim=-1))
    sinX = torch.norm(sinX_vec, dim=-1) * sinX_sign

    return torch.stack([cosX, sinX], dim=-1)


def angle_loss(ref_atom91,   # n_res x n_atom x 3
               pred_atom91,  # n_res x n_atom x 3
               atom91_mask,  # n_res x n_atom
               eps=1e-6):
    angle_atom_idx_select = []
    for atom_list in atom91_angles.values():
        angle_atom_idx_select += atom_list
    angle_atom_idx_select = torch.as_tensor(angle_atom_idx_select, device=ref_atom91.device).long() # n_chi x 4

    angle_mask = atom91_mask[:, angle_atom_idx_select].any(dim=-1)  # n_res x n_chi

    ref_angle_atoms = ref_atom91[:, angle_atom_idx_select]  # n_res x n_chi x 4 x 3
    ref_angles = atoms_to_angles(ref_angle_atoms[~angle_mask])
    pred_angle_atoms = pred_atom91[:, angle_atom_idx_select]  # n_res x n_chi x 4 x 3
    pred_angles = atoms_to_angles(pred_angle_atoms[~angle_mask])

    diff = torch.linalg.vector_norm(ref_angles - pred_angles, dim=-1)

    return diff.mean()

def distance_loss(ref_atom91, pred_atom91, atom91_mask, eps=1e-6):
    dist_idxs = []
    for start, end in atom91_start_end.values():
        l = end - start
        idxs = torch.arange(start, end)
        src = idxs.unsqueeze(0).expand(l, -1).reshape(-1)
        dst = idxs.repeat_interleave(l)
        dist_idxs.append(torch.stack([src, dst], dim=0))

    dist_idxs = torch.cat(dist_idxs, dim=-1)
    src = dist_idxs[0]
    dst = dist_idxs[1]

    ref_src = ref_atom91[:, src]
    ref_dst = ref_atom91[:, dst]
    ref_dist = torch.linalg.vector_norm(ref_src - ref_dst, dim=-1)

    pred_src = pred_atom91[:, src]
    pred_dst = pred_atom91[:, dst]
    pred_dist = torch.linalg.vector_norm(pred_src - pred_dst, dim=-1)

    atom91_src_mask = atom91_mask[:, src]
    atom91_dst_mask = atom91_mask[:, dst]
    total_mask = atom91_src_mask | atom91_dst_mask

    dist_diff = ref_dist - pred_dist
    dist_mse = dist_diff[~total_mask].square().sum() / (~total_mask).long().sum()

    return dist_mse


def cath_loss_fn(noised_batch, model_outputs, n_channels=4, use_channel_weights=False):
    density_dict = noised_batch['density']
    noised_density_dict = noised_batch['noised_density']

    seq = noised_batch['seq']
    x_mask = noised_batch['x_mask']
    atom91_centered = noised_batch['atom91_centered']
    atom91_mask = noised_batch['atom91_mask']

    denoised_density = model_outputs['density']
    seq_logits = model_outputs['seq_logits']
    pred_atom91 = model_outputs['atom91']

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

    denoising_loss = zernike_coeff_loss(density_dict, denoised_density, x_mask, n_channels=n_channels, channel_weights=channel_weights)
    ref_noise = zernike_coeff_loss(density_dict, noised_density_dict, x_mask, n_channels=n_channels, channel_weights=channel_weights)
    seq_loss = seq_cce_loss(seq, seq_logits, x_mask)
    atom91_rmsd = atom91_rmsd_loss(atom91_centered, pred_atom91, atom91_mask)
    bond_length_mse = bond_length_loss(atom91_centered, pred_atom91, atom91_mask.any(dim=-1))
    bond_angle_loss = angle_loss(atom91_centered, pred_atom91, atom91_mask.any(dim=-1))
    chi_loss = torsion_loss(atom91_centered, pred_atom91, atom91_mask.any(dim=-1))

    unscaled_loss = (
        denoising_loss + seq_loss + atom91_rmsd +
        bond_length_mse + bond_angle_loss + chi_loss
    )
    loss = noised_batch['loss_weight'] * unscaled_loss
    return {
        "loss": loss,
        "denoising_loss": denoising_loss,
        "ref_noise": ref_noise,
        "seq_loss": seq_loss,
        "atom91_rmsd": atom91_rmsd,
        "bond_length_mse": bond_length_mse,
        "bond_angle_loss": bond_angle_loss,
        "chi_loss": chi_loss
    }
