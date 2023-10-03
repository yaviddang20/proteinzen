""" Utils for loss functions """
import torch
import torch.nn.functional as F

from ligbinddiff.runtime.loss.atomic import (angle_loss, atom91_mse_loss,
                                             atom91_rmsd_loss,
                                             atomic_neighborhood_dist_loss, chain_constraints_loss,
                                             backbone_dihedrals_loss, bond_length_loss, distance_loss,
                                             intersidechain_clash_loss,
                                             intrasidechain_clash_loss,
                                             torsion_loss)
from ligbinddiff.runtime.loss.density import zernike_coeff_loss
from ligbinddiff.runtime.loss.latent import so3_embedding_kl, so3_embedding_mse
from ligbinddiff.runtime.loss.utils import (_nodewise_to_graphwise, _elemwise_to_graphwise)
from ligbinddiff.utils.atom_reps import atom91_atom_masks

from .frames import all_atom_fape_loss
from .openfold import compute_fape


def seq_cce_loss(ref_seq,
                 seq_logits,
                 num_nodes,
                 mask):
    """ CCE loss on logits """
    ref_seq = ref_seq[~mask]
    seq_logits = seq_logits[~mask]
    cce = F.cross_entropy(seq_logits, ref_seq, reduction='none')
    return _nodewise_to_graphwise(cce, num_nodes, mask)


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


def generator_loss(model_outputs, num_nodes, x_mask):
    discrim_logits_real = model_outputs['discrim_logits_real'][~x_mask]
    discrim_logits_fake = model_outputs['discrim_logits_fake'][~x_mask]
    logprobs_real_incorrect = discrim_logits_real[:, 0]
    logprobs_fake_incorrect = discrim_logits_fake[:, 1]
    total_logprobs = logprobs_real_incorrect + logprobs_fake_incorrect
    return -_nodewise_to_graphwise(total_logprobs, num_nodes, x_mask)

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

def cath_latent_loss_fn(batch, latent_outputs, decoder_outputs, use_channel_weights=None, warmup=False, ae_loss_weight=1):
    atom91_centered = batch['residue']['atom91_centered']
    seq = batch['residue']['seq']
    X_ca = batch['residue']['x']
    bb = batch['residue']['bb']
    bb_rel = bb - X_ca.unsqueeze(-2)
    x_mask = batch['residue']['x_mask']
    atom91_mask = batch['residue']['atom91_mask']

    latent = latent_outputs['latent_sidechain']
    latent_mu = latent_outputs['latent_mu']
    latent_logvar = latent_outputs['latent_logvar']

    decoded_atom91 = decoder_outputs['decoded_latent']
    decoded_atom91[..., :4, :] = bb_rel
    seq_logits = decoder_outputs['decoded_seq_logits']

    data_splits = batch._slice_dict['residue']['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    autoencoder_loss = atom91_mse_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask, no_bb=False)
    seq_loss = seq_cce_loss(seq, seq_logits, data_lens, x_mask)
    atom91_rmsd = autoencoder_loss.sqrt()
    sidechain_dists_mse = distance_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1), no_bb=False)
    bond_length_mse = bond_length_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1), no_bb=False)
    bond_angle_loss = angle_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1), no_bb=False)
    chi_loss = torsion_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1))
    kl_div = so3_embedding_kl(latent_mu, latent_logvar, data_lens, x_mask)

    edge_splits = batch._slice_dict['residue', 'knn', 'residue']['edge_index']
    num_edges = (edge_splits[1:] - edge_splits[:-1]).tolist()

    intrares_clash_loss = intrasidechain_clash_loss(decoded_atom91, data_lens, atom91_mask.any(dim=-1))
    interres_clash_loss = intersidechain_clash_loss(
        decoded_atom91 + X_ca.unsqueeze(-2),
        seq,
        batch['residue', 'knn', 'residue'].edge_index,
        num_edges,
        x_mask
    )
    local_atomic_dist_loss = atomic_neighborhood_dist_loss(
        atom91_centered + X_ca.unsqueeze(-2),
        decoded_atom91 + X_ca.unsqueeze(-2),
        seq,
        batch['residue', 'knn', 'residue'].edge_index,
        num_edges,
        x_mask
    )
    # print(kl_div)

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
            batch['residue', 'knn', 'residue'].edge_index,
            num_edges,
            x_mask | (~same)
        )
        cl_local_atomic_dist_loss = atomic_neighborhood_dist_loss(
            atom91_centered + X_ca.unsqueeze(-2),
            decoded_atom91 + X_ca.unsqueeze(-2),
            seq,
            batch['residue', 'knn', 'residue'].edge_index,
            num_edges,
            x_mask | (~same)
        )

    if not warmup:
        noised_latent = latent_outputs['noised_latent_sidechain']
        denoised_latent = latent_outputs['denoised_latent_sidechain']

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
        bond_angle_loss + chi_loss
        # + intrares_clash_loss + interres_clash_loss + local_atomic_dist_loss
    )
    if not warmup:
        loss = (latent_outputs['loss_weight'] * denoising_loss + vae_loss * ae_loss_weight).mean()
    else:
        latent_outputs['t'] = torch.zeros(1)
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

    bb_conn_lens, bb_conn_angles = chain_constraints_loss(
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
        autoencoder_loss + 1e-4 * kl_div +
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


def autoencoder_losses(batch,
                       latent_outputs,
                       decoder_outputs,
                       absolute_error=False,
                       use_fape=False,
                       use_noise_mask=False):
    atom91_centered = batch['atom91_centered']
    bb = batch['bb'][:, :3]  # we're moving O to the sidechain
    seq = batch['seq']
    X_ca = batch['x']
    x_mask = batch['x_mask']
    atom91_mask = batch['atom91_mask']

    if absolute_error:
        atom91 = atom91_centered + X_ca.unsqueeze(-2)
        denoised_bb = latent_outputs['denoised_bb']
        denoised_x_ca = denoised_bb[..., 1, :]
        decoded_atom91 = decoder_outputs['decoded_latent'] + denoised_x_ca.unsqueeze(-2)
        decoded_atom91[..., :3, :] = denoised_bb
    else:
        atom91 = atom91_centered
        decoded_atom91 = decoder_outputs['decoded_latent']
        bb_rel = bb - X_ca.unsqueeze(-2)
        decoded_atom91[..., :3, :] = bb_rel
    seq_logits = decoder_outputs['decoded_seq_logits']

    if use_noise_mask:
        noising_mask = latent_outputs['noising_mask']
        # if we don't noise anything, just eval on everything as a sanity check
        if (~noising_mask).all():
            noising_mask = ~noising_mask
        x_mask[~noising_mask] = True
        atom91_mask[~noising_mask] = True

    data_splits = batch._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    autoencoder_loss = atom91_mse_loss(atom91, decoded_atom91, data_lens, atom91_mask, no_bb=False)
    seq_loss = seq_cce_loss(seq, seq_logits, data_lens, x_mask)
    atom91_rmsd = autoencoder_loss.sqrt()
    sidechain_dists_mse = distance_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1), no_bb=False)
    bond_length_mse = bond_length_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1), no_bb=False)
    bond_angle_loss = angle_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1), no_bb=False)
    chi_loss = torsion_loss(atom91_centered, decoded_atom91, data_lens, atom91_mask.any(dim=-1))

    latent_mu = latent_outputs['latent_mu']
    latent_logvar = latent_outputs['latent_logvar']
    kl_div = so3_embedding_kl(latent_mu, latent_logvar, data_lens, x_mask)

    out_dict = {
        "seq_loss": seq_loss,
        "atom91_mse": autoencoder_loss,
        "atom91_rmsd": atom91_rmsd,
        "sidechain_dists_mse": sidechain_dists_mse,
        "bond_length_mse": bond_length_mse,
        "bond_angle_loss": bond_angle_loss,
        "chi_loss": chi_loss,
        "kl_div": kl_div,
    }

    if use_fape:
        fape = all_atom_fape_loss(
            pred_atom91=decoded_atom91,
            ref_atom91=atom91,
            seq=seq,
            data_lens=data_lens,
            x_mask=x_mask,
            no_bb=True
        )
        out_dict.update({"all_atom_fape": fape})

    return out_dict


def clash_losses(batch,
                 latent_outputs,
                 decoder_outputs,
                 absolute_error=False):
    atom91_centered = batch['atom91_centered']
    bb = batch['bb'][:, :3]  # we're moving O to the sidechain
    seq = batch['seq']
    X_ca = batch['x']
    x_mask = batch['x_mask']
    atom91_mask = batch['atom91_mask']

    if absolute_error:
        atom91 = atom91_centered + X_ca.unsqueeze(-2)
        denoised_bb = latent_outputs['denoised_bb']
        denoised_x_ca = denoised_bb[..., 1, :]
        decoded_atom91 = decoder_outputs['decoded_latent'] + denoised_x_ca.unsqueeze(-2)
        decoded_atom91[..., :3, :] = denoised_bb
    else:
        atom91 = atom91_centered
        decoded_atom91 = decoder_outputs['decoded_latent']
        bb_rel = bb - X_ca.unsqueeze(-2)
        decoded_atom91[..., :3, :] = bb_rel
    edge_splits = batch._slice_dict['edge_index']
    num_edges = (edge_splits[1:] - edge_splits[:-1]).tolist()

    data_splits = batch._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()
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
    return {
        "intraresidue_clash_loss": intrares_clash_loss,
        "interresidue_clash_loss": interres_clash_loss,
        "local_atomic_dist_loss": local_atomic_dist_loss
    }


def backbone_frame_diffusion_loss(batch,
                                  latent_outputs,
                                  decoder_outputs,
                                  time_threshold=0.25):
    x_mask = batch['x_mask']
    noising_mask = latent_outputs['noising_mask']
    # if we don't noise anything, just eval on everything as a sanity check
    if (~noising_mask).all():
        noising_mask = ~noising_mask
    total_mask = ~x_mask & noising_mask

    data_splits = batch._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    bb_frames = batch['frames']
    noised_bb_frames = latent_outputs['noised_frames']
    # denoised_bb_frames = latent_outputs['denoised_frames']
    denoised_bb_frames = decoder_outputs['denoised_frames']

    X_ca = batch['bb'][:, 1]
    pred_frame_X_ca = denoised_bb_frames.get_trans()
    ref_frame_X_ca = noised_bb_frames.get_trans()
    pred_X_ca_se = torch.square(X_ca - pred_frame_X_ca).sum(dim=-1)
    pred_X_ca_mse = _nodewise_to_graphwise(pred_X_ca_se[total_mask], data_lens, ~total_mask)
    ref_X_ca_se = torch.square(X_ca - ref_frame_X_ca).sum(dim=-1)
    ref_X_ca_mse = _nodewise_to_graphwise(ref_X_ca_se[total_mask], data_lens, ~total_mask)

    bb = batch['bb'][:, :3]  # we're moving O to the sidechain
    denoised_bb = latent_outputs['denoised_bb']
    backbone_mse = torch.square(denoised_bb - bb).sum(dim=-1)
    backbone_mse = backbone_mse[total_mask].view(-1)
    total_mask_expand = total_mask[:, None].expand(-1, 3)
    backbone_mse = _elemwise_to_graphwise(backbone_mse, data_lens, ~total_mask_expand)

    bb_fape = compute_fape(
        denoised_bb_frames,
        bb_frames,
        (~x_mask).long(),
        denoised_bb,
        bb,
        (~x_mask).long()[:, None].expand(-1, 3),#noising_mask.long()[:, None].expand(-1, 3),
        l1_clamp_distance=10
    )
    fape_mask = torch.isclose(bb_fape, torch.zeros_like(bb_fape))
    fape_mask[x_mask] = True
    bb_fape = _nodewise_to_graphwise(bb_fape[~fape_mask], data_lens, fape_mask)

    pred_rot_score, pred_trans_score = latent_outputs['pred_bb_score']
    ref_rot_score = latent_outputs['rot_score']
    ref_trans_score = latent_outputs['trans_score']

    rot_score_scaling = latent_outputs['rot_score_scaling']
    trans_score_scaling = latent_outputs['trans_score_scaling']

    # Translation score loss
    # print(ref_trans_score.shape, pred_trans_score.shape)
    trans_score_se = (ref_trans_score - pred_trans_score)**2 * total_mask[..., None]
    trans_score_loss = (trans_score_se / trans_score_scaling[:, None]**2).sum(dim=-1)
    trans_score_loss = _nodewise_to_graphwise(trans_score_loss[total_mask], data_lens, ~total_mask)

    # Rotation score loss
    # print(ref_rot_score.shape, pred_rot_score.shape)
    rot_se = (ref_rot_score - pred_rot_score)**2 * total_mask[..., None, None]
    rot_score_loss = (rot_se / rot_score_scaling[:, None, None]**2).sum(dim=(-1, -2))
    rot_score_loss = _nodewise_to_graphwise(rot_score_loss[total_mask], data_lens, ~total_mask)

    residx = latent_outputs['noised_residx']
    if residx.numel() > 0:
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
    else:
        residx_x_mask = torch.zeros_like(x_mask).bool()

    t = latent_outputs['t']
    apply_chain_loss = t < time_threshold
    residx_x_mask = residx_x_mask & apply_chain_loss

    bb_dihedrals_loss = backbone_dihedrals_loss(
        denoised_bb,
        bb,
        data_lens,
        residx_x_mask)

    bb_conn_lens, bb_conn_angles = chain_constraints_loss(
        denoised_bb,
        bb,
        data_lens,
        batch.batch,
        residx_x_mask)

    return {
        "rot_score_loss": rot_score_loss,
        "trans_score_loss": trans_score_loss,
        "pred_x_ca_mse": pred_X_ca_mse,
        "pred_bb_mse": backbone_mse,
        "ref_x_ca_mse": ref_X_ca_mse,
        "bb_fape": bb_fape,
        "bb_dihedrals_loss": bb_dihedrals_loss,
        "bb_conn_lens": bb_conn_lens,
        "bb_conn_angles": bb_conn_angles
    }


def latent_sidechain_diffusion_loss(batch,
                                    latent_outputs,
                                    decoder_outputs=None):
    x_mask = batch['x_mask']
    noising_mask = latent_outputs['noising_mask']
    # if we don't noise anything, just eval on everything as a sanity check
    if (~noising_mask).all():
        noising_mask = ~noising_mask
    total_mask = ~x_mask & noising_mask

    latent = latent_outputs['latent_sidechain']
    noised_latent = latent_outputs['noised_latent_sidechain']
    denoised_latent = latent_outputs['denoised_latent_sidechain']
    latent_score_scaling = latent_outputs['latent_sidechain_score_scaling']

    data_splits = batch._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    latent_denoising_loss = so3_embedding_mse(
        latent,
        denoised_latent,
        data_lens,
        ~total_mask,
        scaling=latent_score_scaling
    )
    latent_ref_noise = so3_embedding_mse(
        latent,
        noised_latent,
        data_lens,
        ~total_mask,
        scaling=latent_score_scaling)
    return {
        "latent_denoising_loss": latent_denoising_loss,
        "latent_ref_noise": latent_ref_noise,
    }



def inpaint_frame_latent_loss_fn(batch,
                                 latent_outputs,
                                 decoder_outputs,
                                 warmup=False,
                                 ae_loss_weight=1,
                                 absolute_error=False):
    autoencoder_loss_dict = autoencoder_losses(batch,
                                               latent_outputs,
                                               decoder_outputs,
                                               absolute_error,
                                               use_fape=True)
    clash_loss_dict = clash_losses(batch,
                                   latent_outputs,
                                   decoder_outputs,
                                   absolute_error)
    bb_frame_diffusion_loss_dict = backbone_frame_diffusion_loss(batch,
                                                                 latent_outputs,
                                                                 decoder_outputs)
    sidechain_diffusion_loss_dict = latent_sidechain_diffusion_loss(batch,
                                                                    latent_outputs,
                                                                    decoder_outputs)
    vae_loss = (
        autoencoder_loss_dict["seq_loss"] +
        # autoencoder_loss_dict["atom91_mse"] +
        # autoencoder_loss_dict["sidechain_dists_mse"] +
        # autoencoder_loss_dict["bond_length_mse"] +
        # autoencoder_loss_dict["bond_angle_loss"] +
        autoencoder_loss_dict["chi_loss"] +
        1e-6 * autoencoder_loss_dict["kl_div"]
        + autoencoder_loss_dict["all_atom_fape"]
    )
    bb_denoising_loss = (
        bb_frame_diffusion_loss_dict["rot_score_loss"] +
        # bb_frame_diffusion_loss_dict["trans_score_loss"] +
        bb_frame_diffusion_loss_dict["pred_x_ca_mse"] +
        bb_frame_diffusion_loss_dict["bb_fape"] +
        bb_frame_diffusion_loss_dict["bb_dihedrals_loss"] +
        bb_frame_diffusion_loss_dict["bb_conn_lens"] +
        bb_frame_diffusion_loss_dict["bb_conn_angles"]
    )
    latent_denoising_loss = sidechain_diffusion_loss_dict['latent_denoising_loss']

    denoising_loss = latent_denoising_loss + bb_denoising_loss
    if not warmup:
        loss = (denoising_loss + vae_loss * ae_loss_weight).mean()
    else:
        loss = vae_loss.mean()

    out_dict = {"loss": loss}
    out_dict.update(autoencoder_loss_dict)
    out_dict.update(bb_frame_diffusion_loss_dict)
    out_dict.update(sidechain_diffusion_loss_dict)
    return out_dict

def sidechain_chemical_validity_loss(batch,
                       latent_outputs,
                       decoder_outputs,
                       ):
    atom91_centered = batch['atom91_centered']
    X_ca = batch['x']
    atom91_mask = batch['atom91_mask']

    atom91 = atom91_centered
    decoded_atom91 = decoder_outputs['decoded_latent']
    bb = latent_outputs['denoised_bb'][..., :3, :]
    bb_rel = bb - X_ca.unsqueeze(-2)
    decoded_atom91[..., :3, :] = bb_rel

    data_splits = batch._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    bond_length_mse = bond_length_loss(atom91, decoded_atom91, data_lens, atom91_mask.any(dim=-1), no_bb=False)
    bond_angle_loss = angle_loss(atom91, decoded_atom91, data_lens, atom91_mask.any(dim=-1), no_bb=False)

    out_dict = {
        "pt_bond_length_mse": bond_length_mse,
        "pt_bond_angle_loss": bond_angle_loss,
    }

    return out_dict

def inpaint_frame_latent_loss_fn_2(batch,
                                 latent_outputs,
                                 decoder_outputs,
                                 passthrough_outputs,
                                 warmup=False,
                                 ae_loss_weight=1,
                                 absolute_error=False):
    autoencoder_loss_dict = autoencoder_losses(batch,
                                               latent_outputs,
                                               decoder_outputs,
                                               absolute_error,
                                               use_fape=True)
    # passthrough_loss_dict = sidechain_chemical_validity_loss(batch,
    #                                            latent_outputs,
    #                                            passthrough_outputs)
    # passthrough_decode_loss_dict = autoencoder_losses(batch,
    #                                            latent_outputs,
    #                                            passthrough_outputs,
    #                                            absolute_error=True,
    #                                            use_fape=True)
    # passthrough_denoise_loss_dict = backbone_frame_diffusion_loss(batch,
    #                                            latent_outputs,
    #                                            passthrough_outputs)
    # clash_loss_dict = clash_losses(batch,
    #                                latent_outputs,
    #                                decoder_outputs,
    #                                absolute_error)
    bb_frame_diffusion_loss_dict = backbone_frame_diffusion_loss(batch,
                                                                 latent_outputs,
                                                                 decoder_outputs)
    sidechain_diffusion_loss_dict = latent_sidechain_diffusion_loss(batch,
                                                                    latent_outputs,
                                                                    decoder_outputs)
    vae_loss = (
        autoencoder_loss_dict["seq_loss"] +
        autoencoder_loss_dict["atom91_mse"] +
        autoencoder_loss_dict["sidechain_dists_mse"] +
        autoencoder_loss_dict["bond_length_mse"] +
        autoencoder_loss_dict["bond_angle_loss"] +
        autoencoder_loss_dict["chi_loss"] +
        1e-6 * autoencoder_loss_dict["kl_div"]
        + autoencoder_loss_dict["all_atom_fape"]
    )
    # + (
    #     passthrough_decode_loss_dict["seq_loss"] +
    #     + passthrough_decode_loss_dict["all_atom_fape"]
    # ) * (latent_outputs['t_per_graph'] < 0.25)

    bb_denoising_loss = (
        bb_frame_diffusion_loss_dict["rot_score_loss"] +
        # bb_frame_diffusion_loss_dict["trans_score_loss"] +
        bb_frame_diffusion_loss_dict["pred_x_ca_mse"] +
        bb_frame_diffusion_loss_dict["bb_fape"] +
        bb_frame_diffusion_loss_dict["bb_dihedrals_loss"] +
        bb_frame_diffusion_loss_dict["bb_conn_lens"] +
        bb_frame_diffusion_loss_dict["bb_conn_angles"]
    )
    # + (
    #     passthrough_loss_dict['pt_bond_length_mse'] +
    #     passthrough_loss_dict['pt_bond_angle_loss']
    # ) + (
    #     passthrough_denoise_loss_dict["rot_score_loss"] +
    #     passthrough_denoise_loss_dict["trans_score_loss"] +
    #     passthrough_denoise_loss_dict["bb_fape"] +
    #     passthrough_denoise_loss_dict["bb_dihedrals_loss"] +
    #     passthrough_denoise_loss_dict["bb_conn_lens"] +
    #     passthrough_denoise_loss_dict["bb_conn_angles"]
    # ) * (latent_outputs['t_per_graph'] < 0.25)
    latent_denoising_loss = sidechain_diffusion_loss_dict['latent_denoising_loss']

    denoising_loss = latent_denoising_loss + bb_denoising_loss
    if not warmup:
        loss = (denoising_loss + vae_loss * ae_loss_weight).mean()
    else:
        loss = vae_loss.mean()

    out_dict = {"loss": loss}
    out_dict.update(autoencoder_loss_dict)
    out_dict.update(bb_frame_diffusion_loss_dict)
    out_dict.update(sidechain_diffusion_loss_dict)
    # out_dict.update(passthrough_loss_dict)
    # out_dict.update({"pt_" + key: value for key, value in passthrough_denoise_loss_dict.items()})
    # out_dict.update({"pt_" + key: value for key, value in passthrough_decode_loss_dict.items()})
    return out_dict

def debug_inpaint_frame_latent_loss_fn(batch,
                                 latent_outputs,
                                 decoder_outputs,
                                 warmup=False,
                                 ae_loss_weight=1,
                                 absolute_error=False,
                                 time_threshold=0.25):
    bb_frame_diffusion_loss_dict = frame_diffusion_loss(batch,
                                                        latent_outputs,
                                                        decoder_outputs)
    bb_denoising_loss = (
        bb_frame_diffusion_loss_dict["pred_x_ca_mse"] +
        bb_frame_diffusion_loss_dict["rot_score_loss"]
    )
    bb_denoising_finegrain_loss = (
        bb_frame_diffusion_loss_dict["pred_bb_mse"] +
        bb_frame_diffusion_loss_dict["bb_dihedrals_loss"] +
        bb_frame_diffusion_loss_dict["bb_conn_lens"] +
        bb_frame_diffusion_loss_dict["bb_conn_angles"]
    ) * (latent_outputs['t_per_graph'] < time_threshold)

    # loss = (bb_denoising_loss + bb_denoising_finegrain_loss).mean()
    loss = bb_denoising_loss.mean()
    # loss = bb_frame_diffusion_loss_dict["pred_x_ca_mse"].mean()

    out_dict = {"loss": loss}
    out_dict.update(bb_frame_diffusion_loss_dict)
    return out_dict

def debug_latent_loss_fn(batch,
                         latent_outputs,
                         decoder_outputs,
                         passthrough_outputs,
                         warmup=False,
                         ae_loss_weight=1,
                         absolute_error=False):
    autoencoder_loss_dict = autoencoder_losses(batch,
                                               latent_outputs,
                                               decoder_outputs,
                                               absolute_error,
                                               use_fape=True)
    # passthrough_loss_dict = sidechain_chemical_validity_loss(batch,
    #                                            latent_outputs,
    #                                            passthrough_outputs)
    # passthrough_decode_loss_dict = autoencoder_losses(batch,
    #                                            latent_outputs,
    #                                            passthrough_outputs,
    #                                            absolute_error=True,
    #                                            use_fape=True)
    # passthrough_denoise_loss_dict = backbone_frame_diffusion_loss(batch,
    #                                            latent_outputs,
    #                                            passthrough_outputs)
    # clash_loss_dict = clash_losses(batch,
    #                                latent_outputs,
    #                                decoder_outputs,
    #                                absolute_error)
    sidechain_diffusion_loss_dict = latent_sidechain_diffusion_loss(batch,
                                                                    latent_outputs,
                                                                    decoder_outputs)
    vae_loss = (
        autoencoder_loss_dict["seq_loss"] +
        autoencoder_loss_dict["atom91_mse"] +
        autoencoder_loss_dict["sidechain_dists_mse"] +
        autoencoder_loss_dict["bond_length_mse"] +
        autoencoder_loss_dict["bond_angle_loss"] +
        autoencoder_loss_dict["chi_loss"] +
        1e-6 * autoencoder_loss_dict["kl_div"]
        + autoencoder_loss_dict["all_atom_fape"]
    )
    # + (
    #     passthrough_decode_loss_dict["seq_loss"] +
    #     + passthrough_decode_loss_dict["all_atom_fape"]
    # ) * (latent_outputs['t_per_graph'] < 0.25)

    # + (
    #     passthrough_loss_dict['pt_bond_length_mse'] +
    #     passthrough_loss_dict['pt_bond_angle_loss']
    # ) + (
    #     passthrough_denoise_loss_dict["rot_score_loss"] +
    #     passthrough_denoise_loss_dict["trans_score_loss"] +
    #     passthrough_denoise_loss_dict["bb_fape"] +
    #     passthrough_denoise_loss_dict["bb_dihedrals_loss"] +
    #     passthrough_denoise_loss_dict["bb_conn_lens"] +
    #     passthrough_denoise_loss_dict["bb_conn_angles"]
    # ) * (latent_outputs['t_per_graph'] < 0.25)
    latent_denoising_loss = sidechain_diffusion_loss_dict['latent_denoising_loss']

    denoising_loss = latent_denoising_loss
    if not warmup:
        loss = (denoising_loss + vae_loss * ae_loss_weight).mean()
    else:
        loss = vae_loss.mean()

    out_dict = {"loss": loss}
    out_dict.update(autoencoder_loss_dict)
    out_dict.update(sidechain_diffusion_loss_dict)
    # out_dict.update(passthrough_loss_dict)
    # out_dict.update({"pt_" + key: value for key, value in passthrough_denoise_loss_dict.items()})
    # out_dict.update({"pt_" + key: value for key, value in passthrough_decode_loss_dict.items()})
    return out_dict

def backbone_r3_diffusion_loss(batch,
                               diffusion_outputs,
                               time_threshold=0.25):
    x_mask = batch['residue']['x_mask']
    noising_mask = diffusion_outputs['noising_select']
    # if we don't noise anything, just eval on everything as a sanity check
    if (~noising_mask).all():
        noising_mask = ~noising_mask
    total_mask = ~x_mask & noising_mask

    data_splits = batch._slice_dict['residue']['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    X_ca = batch['residue']['bb'][:, 1]
    pred_X_ca =  diffusion_outputs['denoised_bb'][:, 1]
    ref_X_ca = diffusion_outputs['noised_bb'][:, 1]
    pred_X_ca_se = torch.square(X_ca - pred_X_ca).sum(dim=-1)
    pred_X_ca_mse = _nodewise_to_graphwise(pred_X_ca_se[total_mask], data_lens, ~total_mask)
    ref_X_ca_se = torch.square(X_ca - ref_X_ca).sum(dim=-1)
    ref_X_ca_mse = _nodewise_to_graphwise(ref_X_ca_se[total_mask], data_lens, ~total_mask)

    bb = batch['residue']["bb"]
    denoised_bb = diffusion_outputs['denoised_bb']
    noised_bb = diffusion_outputs['noised_bb']

    pred_backbone_mse = torch.square(denoised_bb - bb).sum(dim=-1)
    pred_backbone_mse = pred_backbone_mse[total_mask].view(-1)
    total_mask_expand = total_mask[:, None].expand(-1, 4)
    pred_backbone_mse = _elemwise_to_graphwise(pred_backbone_mse, data_lens, ~total_mask_expand)

    ref_backbone_mse = torch.square(noised_bb - bb).sum(dim=-1)
    ref_backbone_mse = ref_backbone_mse[total_mask].view(-1)
    total_mask_expand = total_mask[:, None].expand(-1, 4)
    ref_backbone_mse = _elemwise_to_graphwise(ref_backbone_mse, data_lens, ~total_mask_expand)

    bb_rel = bb - bb[:, 1].unsqueeze(-2)
    denoised_bb_rel = denoised_bb - denoised_bb[:, 1].unsqueeze(-2)
    noised_bb_rel = noised_bb - noised_bb[:, 1].unsqueeze(-2)

    pred_bb_rel_mse = torch.square(denoised_bb_rel - bb_rel)[:, (0, 2, 3)].sum(dim=-1)
    pred_bb_rel_mse = pred_bb_rel_mse[total_mask].view(-1)
    total_mask_expand = total_mask[:, None].expand(-1, 3)
    pred_bb_rel_mse = _elemwise_to_graphwise(pred_bb_rel_mse, data_lens, ~total_mask_expand)

    ref_bb_rel_mse = torch.square(noised_bb_rel - bb_rel)[:, (0, 2, 3)].sum(dim=-1)
    ref_bb_rel_mse = ref_bb_rel_mse[total_mask].view(-1)
    total_mask_expand = total_mask[:, None].expand(-1, 3)
    ref_bb_rel_mse = _elemwise_to_graphwise(ref_bb_rel_mse, data_lens, ~total_mask_expand)

    residx = diffusion_outputs['noised_residx']
    if residx.numel() > 0:
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
    else:
        residx_x_mask = torch.zeros_like(x_mask).bool()

    t = diffusion_outputs['t']
    apply_chain_loss = t < time_threshold
    apply_chain_loss = torch.cat([
        torch.ones(data_len, device=t.device) * apply_chain_loss[i]
        for i, data_len in enumerate(data_lens)
    ], dim=0).bool()
    residx_x_mask = residx_x_mask & apply_chain_loss

    bb_dihedrals_loss = backbone_dihedrals_loss(
        denoised_bb,
        bb,
        data_lens,
        residx_x_mask)

    bb_conn_lens, bb_conn_angles = chain_constraints_loss(
        denoised_bb,
        bb,
        data_lens,
        batch['residue'].batch,
        residx_x_mask)

    return {
        "pred_x_ca_mse": pred_X_ca_mse,
        "ref_x_ca_mse": ref_X_ca_mse,
        "pred_bb_mse": pred_backbone_mse,
        "ref_bb_mse": ref_backbone_mse,
        "pred_bb_rel_mse": pred_bb_rel_mse,
        "ref_bb_rel_mse": ref_bb_rel_mse,
        "bb_dihedrals_loss": bb_dihedrals_loss,
        "bb_conn_lens": bb_conn_lens,
        "bb_conn_angles": bb_conn_angles
    }

def bb_inpaint_r3_loss_fn(batch,
                          denoiser_outputs,
                          absolute_error=False,
                          time_threshold=0.25):

    bb_denoising_dict = backbone_r3_diffusion_loss(
        batch,
        denoiser_outputs
    )

    bb_denoising_loss = (
        bb_denoising_dict["pred_x_ca_mse"] +
        (bb_denoising_dict["pred_bb_rel_mse"] if not absolute_error else bb_denoising_dict["pred_bb_mse"]) #  * denoiser_outputs["bb_loss_weight"]
    )
    bb_denoising_finegrain_loss = (
        bb_denoising_dict["pred_bb_mse"] +
        bb_denoising_dict["bb_dihedrals_loss"] +
        bb_denoising_dict["bb_conn_lens"] +
        bb_denoising_dict["bb_conn_angles"]
    ) * (denoiser_outputs['t_per_graph'] < time_threshold) # * denoiser_outputs["bb_loss_weight"]

    loss = (bb_denoising_loss + bb_denoising_finegrain_loss).mean()

    out_dict = {"loss": loss}
    out_dict.update(bb_denoising_dict)
    return out_dict


def frame_diffusion_loss(batch,
                         noised_data,
                         denoiser_outputs,
                         time_threshold=0.25):
    x_mask = batch['x_mask']
    noising_mask = noised_data['noising_mask']
    # if we don't noise anything, just eval on everything as a sanity check
    if (~noising_mask).all():
        noising_mask = ~noising_mask
    total_mask = ~x_mask & noising_mask

    data_splits = batch._slice_dict['x']
    data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

    bb_frames = batch['frames']
    noised_bb_frames = noised_data['noised_frames']
    # denoised_bb_frames = latent_outputs['denoised_frames']
    denoised_bb_frames = denoiser_outputs['denoised_frames']

    X_ca = batch['bb'][:, 1]
    pred_frame_X_ca = denoised_bb_frames.get_trans()
    ref_frame_X_ca = noised_bb_frames.get_trans()
    pred_X_ca_se = torch.square(X_ca - pred_frame_X_ca).sum(dim=-1)
    pred_X_ca_mse = _nodewise_to_graphwise(pred_X_ca_se[total_mask], data_lens, ~total_mask)
    ref_X_ca_se = torch.square(X_ca - ref_frame_X_ca).sum(dim=-1)
    ref_X_ca_mse = _nodewise_to_graphwise(ref_X_ca_se[total_mask], data_lens, ~total_mask)

    bb = batch['bb'][:, :3]  # we're moving O to the sidechain
    denoised_bb = denoiser_outputs['denoised_bb']
    backbone_mse = torch.square(denoised_bb - bb).sum(dim=-1)
    backbone_mse = backbone_mse[total_mask].view(-1)
    total_mask_expand = total_mask[:, None].expand(-1, 3)
    backbone_mse = _elemwise_to_graphwise(backbone_mse, data_lens, ~total_mask_expand)

    bb_fape = compute_fape(
        denoised_bb_frames,
        bb_frames,
        (~x_mask).long(),
        denoised_bb,
        bb,
        (~x_mask).long()[:, None].expand(-1, 3),#noising_mask.long()[:, None].expand(-1, 3),
        l1_clamp_distance=10
    )
    fape_mask = torch.isclose(bb_fape, torch.zeros_like(bb_fape))
    fape_mask[x_mask] = True
    bb_fape = _nodewise_to_graphwise(bb_fape[~fape_mask], data_lens, fape_mask)

    pred_rot_score, pred_trans_score = denoiser_outputs['pred_bb_score']
    ref_rot_score = noised_data['rot_score']
    ref_trans_score = noised_data['trans_score']

    rot_score_scaling = noised_data['rot_score_scaling']
    trans_score_scaling = noised_data['trans_score_scaling']

    # Translation score loss
    trans_score_se = (ref_trans_score - pred_trans_score)**2 * total_mask[..., None]
    trans_score_loss = (trans_score_se / trans_score_scaling[:, None]**2).sum(dim=-1)
    trans_score_loss = _nodewise_to_graphwise(trans_score_loss[total_mask], data_lens, ~total_mask)

    # Rotation score loss
    rot_se = (ref_rot_score - pred_rot_score)**2 * total_mask[..., None, None]
    rot_score_loss = (rot_se / rot_score_scaling[:, None, None]**2).sum(dim=(-1, -2))
    rot_score_loss = _nodewise_to_graphwise(rot_score_loss[total_mask], data_lens, ~total_mask)

    residx = noised_data['noised_residx']
    if residx.numel() > 0:
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
    else:
        residx_x_mask = torch.zeros_like(x_mask).bool()

    t = noised_data['t']
    apply_chain_loss = t < time_threshold
    residx_x_mask = residx_x_mask & apply_chain_loss

    bb_dihedrals_loss = backbone_dihedrals_loss(
        denoised_bb,
        bb,
        data_lens,
        residx_x_mask)

    bb_conn_lens, bb_conn_angles = chain_constraints_loss(
        denoised_bb,
        bb,
        data_lens,
        batch.batch,
        residx_x_mask)

    return {
        "rot_score_loss": rot_score_loss,
        "trans_score_loss": trans_score_loss,
        "pred_x_ca_mse": pred_X_ca_mse,
        "pred_bb_mse": backbone_mse,
        "ref_x_ca_mse": ref_X_ca_mse,
        "bb_fape": bb_fape,
        "bb_dihedrals_loss": bb_dihedrals_loss,
        "bb_conn_lens": bb_conn_lens,
        "bb_conn_angles": bb_conn_angles
    }
