import numpy as np
import torch
import torch.nn.functional as F

from ligbinddiff.data.openfold.residue_constants import restype_order_with_x

from .utils import _nodewise_to_graphwise
from .atomic.atom14 import atom14_mse_loss, chi_loss
from .atomic.atomic import atomic_neighborhood_dist_loss
from .atomic.interresidue import intersidechain_clash_loss
from .latent import so3_embedding_kl, scalars_kl_div


def seq_cce_loss(ref_seq,
                 seq_logits,
                 batch,
                 mask):
    """ CCE loss on logits """
    cce = F.cross_entropy(seq_logits, ref_seq * mask, reduction='none')
    return _nodewise_to_graphwise(cce, batch, mask)

def seq_recov(ref_seq,
              seq_logits,
              batch,
              mask):
    """ CCE loss on logits """
    pred_seq = torch.argmax(seq_logits, dim=-1)
    res_correct = (ref_seq == pred_seq).float()
    return _nodewise_to_graphwise(res_correct, batch, mask)


def _collect_from_seq(tensor, seq, seq_mask):
    seq_expand = seq.view(
        [-1] + [1 for _ in range(tensor.dim()-1)]
    ).expand([-1, 1] + list(tensor.shape[2:]))
    seq_mask_expand = seq_mask.view(
        [-1] + [1 for _ in range(seq_expand.dim()-1)])

    seq_expand = seq_expand * seq_mask_expand

    ret = torch.gather(tensor, 1, seq_expand)
    ret = ret * seq_mask_expand
    return ret.squeeze(1)


def autoencoder_losses(batch,
                       model_outputs):
    res_data = batch['residue']

    gt_atom14 = res_data['atom14_gt_positions']
    alt_atom14 = res_data['atom14_alt_gt_positions']
    gt_torsions = res_data['torsion_angles_sin_cos']
    alt_torsions = res_data['alt_torsion_angles_sin_cos']
    torsions_mask = res_data['torsion_angles_mask']

    seq = res_data['seq']
    seq_mask = res_data['seq_mask']

    res_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    mlm_mask = res_data['mlm_mask']
    atom14_gt_mask = res_data['atom14_gt_exists']
    atom14_alt_gt_mask = res_data['atom14_alt_gt_exists']

    minimal_mask = res_mask & seq_mask
    ae_mask = res_mask & mlm_mask & seq_mask
    denoiser_mask = res_mask & noising_mask & seq_mask
    # decoded_all_atom14 = model_outputs['decoded_all_atom14']
    # decoded_all_chis = model_outputs['decoded_all_chis']
    pred_atom14 = model_outputs['decoded_atom14']
    pred_atom14_gt_seq = model_outputs['decoded_atom14_gt_seq']
    pred_chis = model_outputs['decoded_chis']
    seq_logits = model_outputs['decoded_seq_logits']

    # pred_atom14 = _collect_from_seq(decoded_all_atom14, seq, seq_mask)
    # pred_chis = _collect_from_seq(decoded_all_chis, seq, seq_mask)

    atom14_mse = atom14_mse_loss(
        gt_atom14,
        alt_atom14,
        pred_atom14_gt_seq,
        res_data.batch,
        atom14_gt_mask,
        atom14_alt_gt_mask,
        minimal_mask,
        no_bb=False)
    seq_loss = seq_cce_loss(
        seq,
        seq_logits,
        res_data.batch,
        denoiser_mask)
    per_seq_recov = seq_recov(
        seq,
        seq_logits,
        res_data.batch,
        denoiser_mask)
    atom14_rmsd = atom14_mse.sqrt()
    sidechain_dists_mse = atomic_neighborhood_dist_loss(
        gt_ref_atom14=gt_atom14,
        alt_ref_atom14=alt_atom14,
        pred_atom14=pred_atom14_gt_seq,
        gt_atom14_mask=atom14_gt_mask.bool(),
        alt_atom14_mask=atom14_alt_gt_mask.bool(),
        batch=res_data.batch
    )
    sidechain_chi_loss = chi_loss(
        gt_torsions[..., 3:, :],
        alt_torsions[..., 3:, :],
        model_outputs['decoded_chis_gt_seq'],
        res_data.batch,
        torsions_mask[..., 3:]
    )
    # pred_atom14_clash_loss = intersidechain_clash_loss(
    #     pred_atom14=pred_atom14,
    #     atom14_mask=model_outputs['decoded_atom14_mask'].bool(),
    #     seq=seq_logits.argmax(dim=-1),
    #     batch=res_data.batch
    # )


    latent_mu = model_outputs['latent_mu']
    latent_logvar = model_outputs['latent_logvar']
    kl_div = scalars_kl_div(latent_mu, latent_logvar, res_data.batch, minimal_mask)

    out_dict = {
        "seq_loss": seq_loss,
        "per_seq_recov": per_seq_recov,
        "atom14_mse": atom14_mse,
        "atom14_rmsd": atom14_rmsd,
        "sidechain_dists_mse": sidechain_dists_mse,
        # "pred_sidechain_clash_loss": pred_atom14_clash_loss,
        # "bond_length_mse": bond_length_mse,
        # "bond_angle_loss": bond_angle_loss,
        "chi_loss": sidechain_chi_loss,
        "kl_div": kl_div,
        "percent_masked": _nodewise_to_graphwise(
            denoiser_mask.float(),
            res_data.batch,
            torch.ones_like(mlm_mask))
        #  "kl_div_l1": kl_div[1],
    }

    return out_dict


def pt_autoencoder_losses(batch,
                          model_outputs):
    res_data = batch['residue']

    gt_atom14 = res_data['atom14_gt_positions']
    alt_atom14 = res_data['atom14_alt_gt_positions']
    gt_torsions = res_data['torsion_angles_sin_cos']
    alt_torsions = res_data['alt_torsion_angles_sin_cos']
    torsions_mask = res_data['torsion_angles_mask']

    seq = res_data['seq']
    seq_mask = res_data['seq_mask']

    res_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    mlm_mask = res_data['mlm_mask']
    atom14_gt_mask = res_data['atom14_gt_exists']
    atom14_alt_gt_mask = res_data['atom14_alt_gt_exists']

    minimal_mask = res_mask & seq_mask
    ae_mask = res_mask & mlm_mask & seq_mask
    denoiser_mask = res_mask & noising_mask & seq_mask
    # decoded_all_atom14 = model_outputs['decoded_all_atom14']
    # decoded_all_chis = model_outputs['decoded_all_chis']
    pred_atom14 = model_outputs['decoded_atom14']
    pred_chis = model_outputs['decoded_chis']
    seq_logits = model_outputs['decoded_seq_logits']

    # pred_atom14 = _collect_from_seq(decoded_all_atom14, seq, seq_mask)
    # pred_chis = _collect_from_seq(decoded_all_chis, seq, seq_mask)

    atom14_mse = atom14_mse_loss(
        gt_atom14,
        alt_atom14,
        pred_atom14,
        res_data.batch,
        atom14_gt_mask,
        atom14_alt_gt_mask,
        minimal_mask,
        no_bb=False)
    seq_loss = seq_cce_loss(
        seq,
        seq_logits,
        res_data.batch,
        denoiser_mask)
    per_seq_recov = seq_recov(
        seq,
        seq_logits,
        res_data.batch,
        denoiser_mask)
    atom14_rmsd = atom14_mse.sqrt()
    sidechain_dists_mse = atomic_neighborhood_dist_loss(
        gt_ref_atom14=gt_atom14,
        alt_ref_atom14=alt_atom14,
        pred_atom14=pred_atom14,
        gt_atom14_mask=atom14_gt_mask.bool(),
        alt_atom14_mask=atom14_alt_gt_mask.bool(),
        batch=res_data.batch
    )
    sidechain_chi_loss = chi_loss(
        gt_torsions[..., 3:, :],
        alt_torsions[..., 3:, :],
        pred_chis,
        res_data.batch,
        torsions_mask[..., 3:]
    )

    pred_atom14_clash_loss = intersidechain_clash_loss(
        pred_atom14=pred_atom14,
        atom14_mask=model_outputs['decoded_atom14_mask'],
        seq=seq_logits.argmax(dim=-1),
        batch=res_data.batch
    )

    latent_mu = model_outputs['latent_mu']
    latent_logvar = model_outputs['latent_logvar']
    kl_div = scalars_kl_div(latent_mu, latent_logvar, res_data.batch, minimal_mask)

    out_dict = {
        "seq_loss": seq_loss,
        "per_seq_recov": per_seq_recov,
        "atom14_mse": atom14_mse,
        "atom14_rmsd": atom14_rmsd,
        "sidechain_dists_mse": sidechain_dists_mse,
        "pred_sidechain_clash_loss": pred_atom14_clash_loss,
        # "bond_length_mse": bond_length_mse,
        # "bond_angle_loss": bond_angle_loss,
        "chi_loss": sidechain_chi_loss,
        "kl_div": kl_div,
        "percent_masked": _nodewise_to_graphwise(
            denoiser_mask.float(),
            res_data.batch,
            torch.ones_like(mlm_mask))
    }

    return out_dict


def _nll(data, mu, logvar):
    ll = -0.5 * (logvar + torch.square(data - mu) / torch.exp(logvar) + np.log(np.pi * 2))
    return -ll.sum(dim=-1)


def latent_scalar_sidechain_diffusion_loss(batch,
                                           latent_outputs):
    res_data = batch['residue']
    x_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    total_mask = x_mask #& noising_mask

    latent = latent_outputs['latent_sidechain']
    # latent = latent_outputs['latent_mu']
    noised_latent = latent_outputs['noised_latent_sidechain']
    denoised_latent = latent_outputs['pred_latent_sidechain']

    latent_denoising_loss = torch.square(denoised_latent - latent).sum(dim=-1) * total_mask
    latent_denoising_loss = _nodewise_to_graphwise(latent_denoising_loss, res_data.batch, total_mask)

    latent_ref_noise = torch.square(noised_latent - latent).sum(dim=-1) * total_mask
    latent_ref_noise = _nodewise_to_graphwise(latent_ref_noise, res_data.batch, total_mask)

    latent_mu = latent_outputs['latent_mu']
    latent_logvar = latent_outputs['latent_logvar']
    latent_denoising_nll = _nll(denoised_latent, latent_mu, latent_logvar)
    latent_denoising_nll = _nodewise_to_graphwise(latent_denoising_nll, res_data.batch, total_mask)

    latent_ref_noise_nll = _nll(latent, latent_mu, latent_logvar)
    latent_ref_noise_nll = _nodewise_to_graphwise(latent_ref_noise_nll, res_data.batch, total_mask)

    return {
        "latent_denoising_loss": latent_denoising_loss,
        "latent_ref_noise": latent_ref_noise,
        "latent_denoising_nll": latent_denoising_nll,
        "latent_ref_noise_nll": latent_ref_noise_nll,
    }


def latent_scalar_sidechain_fm_loss(batch,
                                    latent_outputs,
                                    t_norm_clip=0.9):
    res_data = batch['residue']
    x_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    total_mask = x_mask & noising_mask

    latent = latent_outputs['latent_sidechain']
    # latent = latent_outputs['latent_mu']
    noised_latent = latent_outputs['noised_latent_sidechain']
    denoised_latent = latent_outputs['pred_latent_sidechain']

    latent_denoising_loss = torch.square(denoised_latent - latent).sum(dim=-1) * total_mask
    latent_denoising_loss = _nodewise_to_graphwise(latent_denoising_loss, res_data.batch, total_mask)

    t = batch['t']
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )
    latent_fm_loss = latent_denoising_loss / (norm_scale ** 2) * 0.01

    latent_ref_noise = torch.square(noised_latent - latent).sum(dim=-1) * total_mask
    latent_ref_noise = _nodewise_to_graphwise(latent_ref_noise, res_data.batch, total_mask)

    latent_mu = latent_outputs['latent_mu']
    latent_logvar = latent_outputs['latent_logvar']
    latent_denoising_nll = _nll(denoised_latent, latent_mu, latent_logvar)
    latent_denoising_nll = _nodewise_to_graphwise(latent_denoising_nll, res_data.batch, total_mask)

    latent_ref_noise_nll = _nll(latent, latent_mu, latent_logvar)
    latent_ref_noise_nll = _nodewise_to_graphwise(latent_ref_noise_nll, res_data.batch, total_mask)

    return {
        "latent_denoising_loss": latent_denoising_loss,
        "latent_fm_loss": latent_fm_loss,
        "latent_ref_noise": latent_ref_noise,
        "latent_denoising_nll": latent_denoising_nll,
        "latent_ref_noise_nll": latent_ref_noise_nll,
    }


def dirichlet_fm_losses(batch,
                       model_outputs):
    res_data = batch['residue']

    gt_atom14 = res_data['atom14_gt_positions']
    alt_atom14 = res_data['atom14_alt_gt_positions']
    gt_torsions = res_data['torsion_angles_sin_cos']
    alt_torsions = res_data['alt_torsion_angles_sin_cos']
    torsions_mask = res_data['torsion_angles_mask']

    seq = res_data['seq']
    seq_mask = res_data['seq_mask']

    res_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    mlm_mask = res_data['mlm_mask']
    atom14_gt_mask = res_data['atom14_gt_exists']
    atom14_alt_gt_mask = res_data['atom14_alt_gt_exists']

    minimal_mask = res_mask & seq_mask
    ae_mask = res_mask & mlm_mask & seq_mask
    denoiser_mask = res_mask & noising_mask & seq_mask
    # decoded_all_atom14 = model_outputs['decoded_all_atom14']
    # decoded_all_chis = model_outputs['decoded_all_chis']
    pred_atom14 = model_outputs['decoded_atom14']
    pred_atom14_gt_seq = model_outputs['decoded_atom14_gt_seq']
    pred_chis = model_outputs['decoded_chis']
    seq_logits = model_outputs['decoded_seq_logits']

    # pred_atom14 = _collect_from_seq(decoded_all_atom14, seq, seq_mask)
    # pred_chis = _collect_from_seq(decoded_all_chis, seq, seq_mask)

    atom14_mse = atom14_mse_loss(
        gt_atom14,
        alt_atom14,
        pred_atom14_gt_seq,
        res_data.batch,
        atom14_gt_mask,
        atom14_alt_gt_mask,
        minimal_mask,
        no_bb=False)
    seq_loss = seq_cce_loss(
        seq,
        seq_logits,
        res_data.batch,
        denoiser_mask)
    per_seq_recov = seq_recov(
        seq,
        seq_logits,
        res_data.batch,
        denoiser_mask)
    atom14_rmsd = atom14_mse.sqrt()
    sidechain_dists_mse = atomic_neighborhood_dist_loss(
        gt_ref_atom14=gt_atom14,
        alt_ref_atom14=alt_atom14,
        pred_atom14=pred_atom14_gt_seq,
        gt_atom14_mask=atom14_gt_mask.bool(),
        alt_atom14_mask=atom14_alt_gt_mask.bool(),
        batch=res_data.batch
    )
    sidechain_chi_loss = chi_loss(
        gt_torsions[..., 3:, :],
        alt_torsions[..., 3:, :],
        model_outputs['decoded_chis_gt_seq'],
        res_data.batch,
        torsions_mask[..., 3:]
    )
    # pred_atom14_clash_loss = intersidechain_clash_loss(
    #     pred_atom14=pred_atom14,
    #     atom14_mask=model_outputs['decoded_atom14_mask'].bool(),
    #     seq=seq_logits.argmax(dim=-1),
    #     batch=res_data.batch
    # )


    latent_mu = model_outputs['latent_mu']
    latent_logvar = model_outputs['latent_logvar']
    kl_div = scalars_kl_div(latent_mu, latent_logvar, res_data.batch, minimal_mask)

    out_dict = {
        "seq_loss": seq_loss,
        "per_seq_recov": per_seq_recov,
        "atom14_mse": atom14_mse,
        "atom14_rmsd": atom14_rmsd,
        "sidechain_dists_mse": sidechain_dists_mse,
        # "pred_sidechain_clash_loss": pred_atom14_clash_loss,
        # "bond_length_mse": bond_length_mse,
        # "bond_angle_loss": bond_angle_loss,
        "chi_loss": sidechain_chi_loss,
        "kl_div": kl_div,
        "percent_masked": _nodewise_to_graphwise(
            denoiser_mask.float(),
            res_data.batch,
            torch.ones_like(mlm_mask))
        #  "kl_div_l1": kl_div[1],
    }

    return out_dict