""" Utils for loss functions """

import torch
import torch.nn.functional as F

from ligbinddiff.utils.type_l import type_l_add, type_l_sub, type_l_apply
# from ligbinddiff.utils.fiber import compact_fiber_to_nl
from ligbinddiff.utils.atom_reps import atom91_atom_masks, atom91_start_end, atom91_sidechain_bonds, atom91_sidechain_angles, chi_atom_idxs, chi_pi_periodic


# TODO: brought this in to avoid importing dgl in the utils, do this in a less hacky way
def compact_fiber_to_nl(fiber_dict, n_channels=1):
    """ Convert from Fiber dict (no n) to Zernike coeffs.
        n_max is inferred from the fiber structure """
    n_max = max([int(l) for l in fiber_dict.keys()])

    nl_dict = {}
    for l, coeffs in fiber_dict.items():
        i = 0
        for n in range(n_max, l-1, -1):
            if (n - l) % 2 == 0:
                nl_dict[(n,l)] = coeffs[..., i:i+n_channels, :]
                i += n_channels

    return nl_dict

def _mask(tensor, mask):
    ret = tensor.clone()
    ret[mask] = 0
    return ret

def vec_norm(tensor, mask_nans=True, eps=1e-6, dim=-1):
    if mask_nans:
        tensor = torch.nan_to_num(tensor)
    norm = torch.sum(tensor * tensor, dim=dim)
    norm = (norm + eps).sqrt()
    return norm
    # return torch.nan_to_num(norm)


def _nodewise_to_graphwise(tensor, num_nodes, nodewise_numel):
    per_graph_t = tensor.split(num_nodes, dim=0)
    per_graph_t = torch.cat([t.sum().unsqueeze(-1) for t in per_graph_t])
    per_graph_numel = nodewise_numel.split(num_nodes)
    per_graph_numel = torch.cat([t.sum().unsqueeze(-1) for t in per_graph_numel])

    return per_graph_t / per_graph_numel


def zernike_coeff_loss(ref_density,
                       pred_density,
                       num_nodes,
                       mask,
                       n_channels=1,
                       channel_weights=None,
                       eps=1e-6):
    """ Invariant MSE loss on Zernike coeffs
    Average of norms of the difference between all type-l vectors """
    ref_density = compact_fiber_to_nl(ref_density, n_channels=n_channels)
    pred_density = compact_fiber_to_nl(pred_density, n_channels=n_channels)

    # only compute residues where all atoms are present
    ref_density = type_l_apply(lambda x: _mask(x, mask), ref_density)
    pred_density = type_l_apply(lambda x: _mask(x, mask), pred_density)

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
            mags = mags * _mask(channel_weights, mask)
        loss = loss + mags#.sum()
        numel = numel + (~mask).long()

    per_graph_loss = _nodewise_to_graphwise(loss, num_nodes, numel)
    return per_graph_loss


def seq_cce_loss(ref_seq,
                 seq_logits,
                 num_nodes,
                 mask):
    """ CCE loss on logits """
    # ref_seq = _mask(ref_seq, mask)
    # seq_logits = _mask(seq_logits, mask)

    cce = F.cross_entropy(seq_logits, ref_seq, reduction='none')
    cce = _mask(cce, mask)

    return _nodewise_to_graphwise(cce, num_nodes, (~mask).long())


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

    ref_atom91 = _mask(ref_atom91, atom91_mask)
    pred_atom91 = _mask(pred_atom91, atom91_mask)
    diff = ref_atom91 - pred_atom91
    # eps for numerical stability
    diff = diff + eps
    sd = torch.square(diff)
    sum_sd = torch.sum(sd, dim=(-2, -1))
    numel = (~atom91_mask.any(dim=-1)).long()
    msd = _nodewise_to_graphwise(sum_sd, num_nodes, numel)
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


def bond_length_loss(ref_atom91,
                     pred_atom91,
                     num_nodes,
                     atom91_mask,
                     eps=1e-6):
    bonds = []
    for b_list in atom91_sidechain_bonds.values():
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
    ref_dist = vec_norm(ref_dst - ref_src, dim=-1)  # n_bond x n_res
    # print("ref_dist no mask", ref_dist)
    # print("ref_dist mask", ref_dist)
    # print(ref_dist.shape)

    pred_src = pred_atom91[:, bonds[0]]  # n_res x n_bond x 3
    pred_dst = pred_atom91[:, bonds[1]]
    pred_dist = vec_norm(pred_dst - pred_src, dim=-1)  # n_bond x n_res
    # print("pred_dist no mask", pred_dist)
    # print(pred_dist.shape)
    # print("pred_dist mask", pred_dist)

    bond_length_diff = ref_dist - pred_dist
    bond_length_diff = _mask(bond_length_diff, bond_mask)
    bond_length_se = bond_length_diff.square()
    bond_numel = (~bond_mask).long().sum(dim=-1)
    return _nodewise_to_graphwise(bond_length_se, num_nodes, bond_numel)


### TODO: move this somewhere better
def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def atoms_to_torsions(chi_atom_coords, chi_mask, eps=1e-8):
    # Based on https://en.wikipedia.org/wiki/Dihedral_angle
    # torch.set_printoptions(threshold=10000000)
    # print("chi_atom_coords", chi_atom_coords)

    # absolute atom positions
    c1 = chi_atom_coords[..., 0, :]
    c2 = chi_atom_coords[..., 1, :]
    c3 = chi_atom_coords[..., 2, :]
    c4 = chi_atom_coords[..., 3, :]
    # torch.save(chi_atom_coords, "tmp/chi_atom_coords.pt")

    # relative atom positions
    a1 = c2 - c1
    a2 = c3 - c2
    a3 = c4 - c3
    # print("a1", a1)
    # print("a2", a2)
    # print("a3", a3)
    # torch.save(a1, "tmp/a1.pt")
    # torch.save(a2, "tmp/a2.pt")
    # torch.save(a3, "tmp/a3.pt")

    # backbone normals
    v1 = torch.cross(a1, a2)
    v2 = torch.cross(a2, a3)
    # torch.save(v1, "tmp/v1.pt")
    # torch.save(v2, "tmp/v2.pt")
    # print("v1", v1)
    # print("v2", v2)

    # Angle between normals
    x = torch.sum(v1 * v2, -1)
    a2_norm = vec_norm(a2, mask_nans=False)
    y = torch.sum(a1 * v2, dim=-1) * a2_norm
    # torch.save(x, "tmp/x.pt")
    # torch.save(a2_norm, "tmp/a2_norm.pt")
    # torch.save(y, "tmp/y.pt")
    # print("x", x)
    # print("y", y)

    angle_vec = torch.stack([x, y], dim=-1)
    norm = vec_norm(angle_vec, mask_nans=False).unsqueeze(-1)
    # torch.save(angle_vec, "tmp/angle_vec.pt")
    # print("angle vec", angle_vec)
    # print("norm", norm)

    angle_vec = angle_vec.clone()
    angle_vec[chi_mask] = eps  # not sure if necessary but trying to avoid in-place gradient cutoffs
    norm = norm.clone()
    norm[chi_mask] = 1

    if (angle_vec / norm).isnan().any():
        torch.set_printoptions(threshold=10000000)
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

    ref_chi_atoms = ref_atom91[:, chi_atom_idx_select]  # n_res x n_chi x 4 x 3
    ref_chi_angles = atoms_to_torsions(ref_chi_atoms, chi_mask)
    pred_chi_atoms = pred_atom91[:, chi_atom_idx_select]  # n_res x n_chi x 4 x 3
    pred_chi_angles = atoms_to_torsions(pred_chi_atoms, chi_mask)  # n_res x n_chi x 2
    pred_chi_angles_p_pi = atoms_to_torsions(pred_chi_atoms, chi_mask)  # n_res x n_chi x 2
    pred_chi_angles_p_pi = pred_chi_angles_p_pi * chi_periodic.unsqueeze(0).unsqueeze(-1)

    # print(ref_chi_angles - pred_chi_angles + eps)
    diff = vec_norm(ref_chi_angles - pred_chi_angles)
    pi_diff = vec_norm(ref_chi_angles - pred_chi_angles_p_pi)
    loss = torch.minimum(diff, pi_diff)  # n_res x n_chi

    loss = _mask(loss, chi_mask)
    num_chi = (~chi_mask).long().sum(dim=-1)
    num_chi[num_chi == 0] = 1
    # print("loss", loss)
    loss = _nodewise_to_graphwise(loss, num_nodes, num_chi)

    return loss


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
               eps=1e-6):
    angle_atom_idx_select = []
    for atom_list in atom91_sidechain_angles.values():
        angle_atom_idx_select += atom_list
    angle_atom_idx_select = torch.as_tensor(angle_atom_idx_select, device=ref_atom91.device).long() # n_chi x 4

    angle_mask = atom91_mask[:, angle_atom_idx_select].any(dim=-1)  # n_res x n_angle

    ref_angle_atoms = ref_atom91[:, angle_atom_idx_select]  # n_res x n_angle x 3 x 3
    ref_angles = atoms_to_angles(ref_angle_atoms)
    pred_angle_atoms = pred_atom91[:, angle_atom_idx_select]  # n_res x n_angle x 3 x 3
    pred_angles = atoms_to_angles(pred_angle_atoms)
    diff = vec_norm(ref_angles - pred_angles, dim=-1)

    diff = _mask(diff, angle_mask)
    num_bonds = (~angle_mask).long().sum(dim=-1)
    num_bonds[num_bonds == 0] = 1
    diff = _nodewise_to_graphwise(diff, num_nodes, num_bonds)
    return diff


# def distance_loss(ref_atom91,
#                   pred_atom91,
#                   atom91_mask,
#                   num_nodes,
#                   eps=1e-6):
#     dist_idxs = []
#     for start, end in atom91_start_end.values():
#         l = end - start
#         idxs = torch.arange(start, end)
#         src = idxs.unsqueeze(0).expand(l, -1).reshape(-1)
#         dst = idxs.repeat_interleave(l)
#         dist_idxs.append(torch.stack([src, dst], dim=0))
#
#     dist_idxs = torch.cat(dist_idxs, dim=-1)
#     src = dist_idxs[0]
#     dst = dist_idxs[1]
#
#     ref_src = ref_atom91[:, src]
#     ref_dst = ref_atom91[:, dst]
#     ref_dist = torch.linalg.vector_norm(ref_src - ref_dst, dim=-1)
#
#     pred_src = pred_atom91[:, src]
#     pred_dst = pred_atom91[:, dst]
#     pred_dist = torch.linalg.vector_norm(pred_src - pred_dst, dim=-1)
#
#     atom91_src_mask = atom91_mask[:, src]
#     atom91_dst_mask = atom91_mask[:, dst]
#     total_mask = atom91_src_mask | atom91_dst_mask
#
#     dist_diff = ref_dist - pred_dist
#     dist_se = dist_diff.square()
#     dist_mse = dist_diff[~total_mask].square().sum() / (~total_mask).long().sum()
#
#     dist_mse = _nodewise_to_graphwise(dist_se, num_nodes, (~total_mask).long().sum(dim=-1))
#     return dist_mse


def cath_density_loss_fn(noised_batch, model_outputs, n_channels=4, use_channel_weights=False):
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

    data_splits = noised_batch._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    denoising_loss = zernike_coeff_loss(density_dict, denoised_density, data_lens, x_mask, n_channels=n_channels, channel_weights=channel_weights)
    ref_noise = zernike_coeff_loss(density_dict, noised_density_dict, data_lens, x_mask, n_channels=n_channels, channel_weights=channel_weights)
    seq_loss = seq_cce_loss(seq, seq_logits, data_lens, x_mask)
    atom91_rmsd = atom91_rmsd_loss(atom91_centered, pred_atom91, data_lens, atom91_mask)
    bond_length_mse = bond_length_loss(atom91_centered, pred_atom91, data_lens, atom91_mask.any(dim=-1))
    bond_angle_loss = angle_loss(atom91_centered, pred_atom91, data_lens, atom91_mask.any(dim=-1))
    chi_loss = torsion_loss(atom91_centered, pred_atom91, data_lens, atom91_mask.any(dim=-1))

    unscaled_loss = (
        denoising_loss + seq_loss + atom91_rmsd +
        bond_length_mse + bond_angle_loss + chi_loss
    )
    loss = (noised_batch['loss_weight'] * unscaled_loss).mean()
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


def cath_superposition_loss_fn(noised_batch, model_outputs, use_channel_weights=None):
    atom91_centered = noised_batch['atom91_centered']
    noised_atom91 = noised_batch['noised_atom91']

    seq = noised_batch['seq']
    x_mask = noised_batch['x_mask']
    atom91_mask = noised_batch['atom91_mask']

    denoised_atom91 = model_outputs['atom91_centered']
    seq_logits = model_outputs['seq_logits']

    data_splits = noised_batch._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    denoising_loss = atom91_rmsd_loss(atom91_centered, denoised_atom91, data_lens, atom91_mask)
    ref_noise = atom91_rmsd_loss(atom91_centered, noised_atom91, data_lens, atom91_mask)
    seq_loss = seq_cce_loss(seq, seq_logits, data_lens, x_mask)
    atom91_rmsd = denoising_loss
    # correct_label_rmsd(seq,
    #                                 seq_logits,
    #                                 atom91_centered,
    #                                 denoised_atom91,
    #                                 data_lens,
    #                                 x_mask,
    #                                 atom91_mask.any(dim=-1)).detach()
    bond_length_mse = bond_length_loss(atom91_centered, denoised_atom91, data_lens, atom91_mask.any(dim=-1))
    bond_angle_loss = angle_loss(atom91_centered, denoised_atom91, data_lens, atom91_mask.any(dim=-1))
    chi_loss = torsion_loss(atom91_centered, denoised_atom91, data_lens, atom91_mask.any(dim=-1))

    unscaled_loss = (
        denoising_loss + seq_loss + atom91_rmsd +
        bond_length_mse + bond_angle_loss + chi_loss
    )
    loss = (noised_batch['loss_weight'] * unscaled_loss).mean()
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


def so3_embedding_mse(ref_so3, pred_so3, num_nodes, x_mask):
    vec_diff = ref_so3.embedding - pred_so3.embedding
    splits = []
    for lmax in ref_so3.lmax_list:
        for l in range(lmax+1):
            splits.append(2*l+1)
    vec_diffs = vec_diff.split(splits, dim=1)
    vec_diff_norms = [vec_norm(v.transpose(-1, -2)) for v in vec_diffs]  # vector dim as final dim
    nodewise_loss = torch.cat(vec_diff_norms, dim=-1).sum(dim=-1)
    nodewise_loss = _mask(nodewise_loss, x_mask)
    numel = (~x_mask).long()

    return _nodewise_to_graphwise(nodewise_loss, num_nodes, numel)


def so3_embedding_kl(so3_mu, so3_logvar, num_nodes, x_mask):
    splits = []
    for lmax in so3_mu.lmax_list:
        for l in range(lmax+1):
            splits.append(2*l+1)
    split_mu = so3_mu.embedding.split(splits, dim=1)
    split_logvar = so3_logvar.embedding.split(splits, dim=1)
    kl_div = 0
    for m, mu, logvar in zip(splits, split_mu, split_logvar):
        comp_kl_div = -0.5 * (logvar.sum(dim=-2) - mu.square().sum(dim=-2) - logvar.exp().sum(dim=-2) + m)
        # print(comp_kl_div, comp_kl_div.max())
        kl_div = kl_div + comp_kl_div.mean(dim=-1)

    numel = (~x_mask).long()
    return _nodewise_to_graphwise(kl_div, num_nodes, numel)


def cath_latent_loss_fn(noised_batch, model_outputs, use_channel_weights=None, warmup=False, ae_loss_weight=1):
    atom91_centered = noised_batch['atom91_centered']
    seq = noised_batch['seq']
    x_mask = noised_batch['x_mask']
    atom91_mask = noised_batch['atom91_mask']

    latent = model_outputs['latent']
    latent_mu = model_outputs['latent_mu']
    latent_logvar = model_outputs['latent_logvar']

    decoded_atom91 = model_outputs['decoded_latent']
    seq_logits = model_outputs['decoded_seq_logits']

    data_splits = noised_batch._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    autoencoder_loss = atom91_rmsd_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask)
    seq_loss = seq_cce_loss(seq, seq_logits, data_lens, x_mask)
    atom91_rmsd = autoencoder_loss
    # correct_label_rmsd(seq,
    #                                 seq_logits,
    #                                 atom91_centered,
    #                                 denoised_atom91,
    #                                 data_lens,
    #                                 x_mask,
    #                                 atom91_mask.any(dim=-1)).detach()
    bond_length_mse = bond_length_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1))
    bond_angle_loss = angle_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1))
    chi_loss = torsion_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1))
    kl_div = so3_embedding_kl(latent_mu, latent_logvar, data_lens, x_mask)
    # print(kl_div)

    correct_label_atom91_mask = atom91_mask.clone()
    pred_seq = seq_logits.argmax(dim=-1)
    same = (pred_seq == seq)
    correct_label_atom91_mask[~same] = True

    with torch.no_grad():
        cl_atom91_rmsd = atom91_rmsd_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask)
        cl_bond_length_mse = bond_length_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
        cl_bond_angle_loss = angle_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
        cl_chi_loss = torsion_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))

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

    vae_loss = (autoencoder_loss + 1 * kl_div +
        seq_loss +  # atom91_rmsd +
        bond_length_mse + bond_angle_loss + chi_loss
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
        "atom91_rmsd": atom91_rmsd,
        "bond_length_mse": bond_length_mse,
        "bond_angle_loss": bond_angle_loss,
        "chi_loss": chi_loss,
        "cl_denoising_loss": cl_denoising_loss,
        "cl_atom91_rmsd": cl_atom91_rmsd,
        "cl_bond_length_mse": cl_bond_length_mse,
        "cl_bond_angle_loss": cl_bond_angle_loss,
        "cl_chi_loss": cl_chi_loss
    }
