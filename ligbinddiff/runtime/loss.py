""" Utils for loss functions """

import torch
import torch.nn.functional as F

from ligbinddiff.utils.type_l import type_l_add, type_l_sub, type_l_apply
from ligbinddiff.utils.fiber import compact_fiber_to_nl
from ligbinddiff.utils.atom_reps import atom91_bonds


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
    bonds = torch.as_tensor(bonds, device=ref_atom91.device).T  # 2 x n_bond
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
