""" Utils for loss functions """

import torch
import torch.nn.functional as F

from ligbinddiff.utils.atom_reps import atom91_atom_masks, atom91_start_end, atom91_bonds, atom91_angles, chi_atom_idxs, chi_pi_periodic
from .loss import zernike_coeff_loss


def vec_norm(tensor, mask_nans=True, eps=1e-6, dim=-1):
    if mask_nans:
        tensor = torch.nan_to_num(tensor)
    norm = torch.sum(tensor * tensor, dim=dim)
    norm = (norm + eps).sqrt()
    return norm
    # return torch.nan_to_num(norm)


def _elemwise_to_graphwise(elemwise_tensor, nodes_per_graph, node_elem_mask):
    if len(nodes_per_graph) == 1:
        return elemwise_tensor.mean().unsqueeze(0)

    reswise_num_elem = (~node_elem_mask).long().sum(dim=-1)
    num_elem_per_graph = [t.sum().item() for t in torch.split(reswise_num_elem, nodes_per_graph)]
    graphwise_tensor = torch.cat([t.mean().unsqueeze(0) for t in elemwise_tensor.split(num_elem_per_graph)])
    return graphwise_tensor


def _nodewise_to_graphwise(nodewise_tensor, nodes_per_graph, node_mask):
    if len(nodes_per_graph) == 1:
        return nodewise_tensor.mean().unsqueeze(0)
    reswise_num = (~node_mask).long()
    num_node_per_graph = [t.sum().item() for t in torch.split(reswise_num, nodes_per_graph)]
    graphwise_tensor = torch.cat([t.mean().unsqueeze(0) for t in nodewise_tensor.split(num_node_per_graph)])
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
                     eps=1e-6):
    bonds = []
    for b_list in atom91_bonds.values():
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

    diff = vec_norm(ref_angles - pred_angles, dim=-1)
    graphwise_loss = _elemwise_to_graphwise(diff, num_nodes, angle_mask)

    return graphwise_loss


def distance_loss(ref_atom91,
                  pred_atom91,
                  num_nodes,
                  atom91_mask,
                  eps=1e-6):
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

    atom91_src_mask = atom91_mask[:, src]
    atom91_dst_mask = atom91_mask[:, dst]
    total_mask = atom91_src_mask | atom91_dst_mask

    ref_src = ref_atom91[:, src][~total_mask]
    ref_dst = ref_atom91[:, dst][~total_mask]
    ref_dist = vec_norm(ref_src - ref_dst, dim=-1)

    pred_src = pred_atom91[:, src][~total_mask]
    pred_dst = pred_atom91[:, dst][~total_mask]
    pred_dist = vec_norm(pred_src - pred_dst, dim=-1)

    dist_diff = (ref_dist - pred_dist).square()
    graphwise_loss = _elemwise_to_graphwise(dist_diff, num_nodes, total_mask)

    return graphwise_loss


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
    x_mask = noised_batch['x_mask']
    atom91_mask = noised_batch['atom91_mask']

    latent = model_outputs['latent']
    latent_mu = model_outputs['latent_mu']
    latent_logvar = model_outputs['latent_logvar']

    decoded_atom91 = model_outputs['decoded_latent']
    seq_logits = model_outputs['decoded_seq_logits']

    data_splits = noised_batch._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    autoencoder_loss = atom91_mse_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask)
    seq_loss = seq_cce_loss(seq, seq_logits, data_lens, x_mask)
    atom91_rmsd = autoencoder_loss.sqrt()
    sidechain_dists_mse = distance_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1))
    bond_length_mse = bond_length_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1))
    bond_angle_loss = angle_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1))
    chi_loss = torsion_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1))
    kl_div = so3_embedding_kl(latent_mu, latent_logvar, data_lens, x_mask)
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
        cl_atom91_rmsd = atom91_rmsd_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask)
        cl_bond_length_mse = bond_length_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
        cl_sidechain_dists_mse = distance_loss(atom91_centered, decoded_atom91, data_lens, correct_label_atom91_mask.any(dim=-1))
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

    vae_loss = (
        autoencoder_loss + 1e-6 * kl_div +
        seq_loss +  # atom91_rmsd +
        bond_length_mse + sidechain_dists_mse +
        bond_angle_loss + chi_loss + gen_loss #+ discrim_loss
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
        "cl_denoising_loss": cl_denoising_loss,
        "cl_atom91_rmsd": cl_atom91_rmsd,
        "cl_sidechain_dists_mse": cl_sidechain_dists_mse,
        "cl_bond_length_mse": cl_bond_length_mse,
        "cl_bond_angle_loss": cl_bond_angle_loss,
        "cl_chi_loss": cl_chi_loss
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
