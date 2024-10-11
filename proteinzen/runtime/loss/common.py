import numpy as np
import torch
import torch.nn.functional as F

from proteinzen.data.openfold.residue_constants import restype_order_with_x

from .utils import _nodewise_to_graphwise
from .atomic.atom14 import atom14_mse_loss, chi_loss
from .atomic.atomic import residue_knn_neighborhood_atomic_dist_loss, local_atomic_context_loss, smooth_lddt_loss, sparse_smooth_lddt_loss
from .atomic.interresidue import intersidechain_clash_loss
from .latent import so3_embedding_kl, scalars_kl_div, gaussian_nll
from .frames import all_atom_fape_loss

from proteinzen.utils.openfold.rigid_utils import Rigid


def seq_cce_loss(ref_seq,
                 seq_logits,
                 batch,
                 mask,
                 label_smoothing=0.0,
                 logits_as_probs=False):
    """ CCE loss on logits """
    if logits_as_probs:
        assert label_smoothing == 0, "Label smoothing not implemented for CCE with probs"
        seq_probs = torch.gather(seq_logits, 1, (ref_seq * mask)[..., None])
        cce = -torch.log(seq_probs.clip(min=1e-6, max=1-1e-6))
        cce = cce.squeeze(-1)
    else:
        cce = F.cross_entropy(seq_logits, ref_seq * mask, reduction='none', label_smoothing=label_smoothing)
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
                       model_outputs,
                       label_smoothing=0.0,
                       logit_norm_loss=0.0,
                       use_smooth_lddt=False,
                       use_fape=False,
                       fape_length_scale=1.,
                       t_norm_clip=0.9,
                       apply_seq_noising_mask=False,
                       use_sidechain_dists_mse_loss=True,
                       use_local_atomic_dist_loss=True,
                       use_sidechain_clash_loss=True,
                       kl_loss=True,
):
    res_data = batch['residue']

    t = batch['t']
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )

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
    if apply_seq_noising_mask:
        seq_noising_mask = res_data['seq_noising_mask']
    else:
        seq_noising_mask = ~res_data['seq_mask']

    if "seq_probs" in model_outputs and model_outputs["seq_probs"] is not None:
        seq_loss = seq_cce_loss(
            seq,
            model_outputs["seq_probs"],
            res_data.batch,
            ~seq_noising_mask & seq_mask,
            label_smoothing=label_smoothing,
            logits_as_probs=True)
    else:
        seq_loss = seq_cce_loss(
            seq,
            seq_logits,
            res_data.batch,
            ~seq_noising_mask & seq_mask,
            label_smoothing=label_smoothing)
    per_seq_recov = seq_recov(
        seq,
        seq_logits,
        res_data.batch,
        denoiser_mask)
    atom14_rmsd = atom14_mse.sqrt()
    if use_sidechain_dists_mse_loss:
        sidechain_dists_mse = residue_knn_neighborhood_atomic_dist_loss(
            gt_ref_atom14=gt_atom14,
            alt_ref_atom14=alt_atom14,
            pred_atom14=pred_atom14_gt_seq,
            gt_atom14_mask=atom14_gt_mask.bool(),
            alt_atom14_mask=atom14_alt_gt_mask.bool(),
            batch=res_data.batch
        )
    else:
        sidechain_dists_mse = torch.zeros_like(t)

    if use_local_atomic_dist_loss:
        local_atomic_dist_loss = local_atomic_context_loss(
            pred_atom14=pred_atom14_gt_seq,
            gt_atom14=gt_atom14,
            alt_atom14=alt_atom14,
            batch=res_data.batch,
            atom14_mask=atom14_gt_mask.bool()
        )
    else:
        local_atomic_dist_loss = torch.zeros_like(t)

    scaled_local_atomic_dist_loss = local_atomic_dist_loss / (norm_scale**2) * 0.01
    scaled_atom14_mse = atom14_mse / (norm_scale**2) * 0.01

    if use_smooth_lddt:
        smooth_lddt = sparse_smooth_lddt_loss(
            pred_atom14=pred_atom14_gt_seq,
            gt_atom14=gt_atom14,
            alt_atom14=alt_atom14,
            batch=res_data.batch,
            atom14_mask=atom14_gt_mask.bool()
        )
    else:
        smooth_lddt = torch.zeros_like(t)

    if use_fape:
        fape = all_atom_fape_loss(
            pred_atom14=pred_atom14,
            gt_atom14=gt_atom14,
            pred_rigids=model_outputs['final_rigids'],
            gt_rigids=Rigid.from_tensor_7(batch['residue']['rigids_1']),
            batch=res_data.batch,
            atom14_mask=atom14_gt_mask,
            length_scale=fape_length_scale
        )
    else:
        fape = torch.zeros_like(t)


    sidechain_chi_loss = chi_loss(
        gt_torsions[..., 3:, :],
        alt_torsions[..., 3:, :],
        model_outputs['decoded_chis_gt_seq'],
        res_data.batch,
        torsions_mask[..., 3:]
    )
    if use_sidechain_clash_loss:
        pred_atom14_clash_loss = intersidechain_clash_loss(
            pred_atom14=pred_atom14,
            atom14_mask=model_outputs['decoded_atom14_mask'].bool(),
            seq=seq_logits.argmax(dim=-1),
            batch=res_data.batch
        )
    else:
        pred_atom14_clash_loss = torch.zeros_like(t)

    if kl_loss:
        latent_mu = model_outputs['latent_mu']
        latent_logvar = model_outputs['latent_logvar']
        res_data_batch = res_data.batch
        _mask = minimal_mask

        # TODO: this is really hacky, we're assuming we're doing convolutional
        # compression if we have shape mismatch
        if latent_mu.shape[0] < minimal_mask.shape[0]:
            res_data_batch = torch.arange(
                batch.num_graphs, device=latent_mu.device
            ).repeat_interleave(latent_mu.shape[0] // batch.num_graphs)
            _mask = torch.ones_like(res_data_batch, dtype=torch.bool)

        kl_div = scalars_kl_div(latent_mu, latent_logvar, res_data_batch, _mask)
    else:
        kl_div = torch.zeros_like(t)

    logit_norm_loss = logit_norm_loss * _nodewise_to_graphwise(
        torch.sum(seq_logits**2, dim=-1),
        res_data.batch,
        denoiser_mask
    )

    out_dict = {
        "seq_loss": seq_loss,
        "per_seq_recov": per_seq_recov,
        "atom14_mse": atom14_mse,
        "atom14_rmsd": atom14_rmsd,
        "sidechain_dists_mse": sidechain_dists_mse,
        "local_atomic_dist_loss": local_atomic_dist_loss,
        "scaled_local_atomic_dist_loss": scaled_local_atomic_dist_loss,
        "scaled_atom14_mse": scaled_atom14_mse,
        "pred_sidechain_clash_loss": pred_atom14_clash_loss,
        "smooth_lddt": smooth_lddt,
        "fape": fape,
        # "bond_length_mse": bond_length_mse,
        # "bond_angle_loss": bond_angle_loss,
        "chi_loss": sidechain_chi_loss,
        "kl_div": kl_div,
        "logit_norm_loss": logit_norm_loss,
        "percent_masked": _nodewise_to_graphwise(
            (~seq_noising_mask).float(),
            res_data.batch,
            torch.ones_like(mlm_mask)),
        #  "kl_div_l1": kl_div[1],
    }

    return out_dict


def kl_losses(batch,
              model_outputs):

    res_data = batch['residue']
    res_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    mask = res_mask & noising_mask
    mask = mask.view(batch.num_graphs, -1)

    def kl(mu, logvar):
        kl_div = -0.5 * (logvar - mu.square() - logvar.exp() + 1)
        kl_div = kl_div.sum(dim=-1)
        return kl_div

    latent_node_mu = model_outputs['latent_node_mu']
    latent_node_logvar = model_outputs['latent_node_logvar']
    node_kl_div = kl(latent_node_mu, latent_node_logvar) * mask
    node_kl_div = node_kl_div.sum(dim=-1) / mask.sum(dim=-1)

    edge_mask = mask[..., None] & mask[..., None, :]

    latent_edge_mu = model_outputs['latent_edge_mu']
    latent_edge_logvar = model_outputs['latent_edge_logvar']
    edge_kl_div = kl(latent_edge_mu, latent_edge_logvar) * edge_mask
    edge_kl_div = edge_kl_div.sum(dim=(-1, -2)) / edge_mask.sum(dim=(-1, -2))

    return {
        'node_kl_div': node_kl_div,
        'edge_kl_div': edge_kl_div
    }



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
    sidechain_dists_mse = residue_knn_neighborhood_atomic_dist_loss(
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


def latent_scalar_sidechain_fm_loss(
        batch,
        latent_outputs,
        t_norm_clip=0.9,
        scale=0.01,
        pointwise=False,
        detach_gt_latent_grad=False,
    ):
    res_data = batch['residue']
    x_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    total_mask = x_mask & noising_mask

    latent = latent_outputs['latent_sidechain']
    if detach_gt_latent_grad:
        latent = latent.detach()
    # latent = latent_outputs['latent_mu']
    noised_latent = latent_outputs['noised_latent_sidechain']
    denoised_latent = latent_outputs['pred_latent_sidechain']

    res_data_batch = res_data.batch

    # TODO: this is super hacky, we're assuming we did convolutional
    # compression from shape mismatch
    if noised_latent.shape[0] < total_mask.shape[0]:
        total_mask = torch.ones(latent.shape[:-1], device=latent.device, dtype=torch.bool)
        res_data_batch = torch.arange(
            batch.num_graphs, device=latent.device
        ).repeat_interleave(latent.shape[0] // batch.num_graphs)
        # print(latent.shape, res_data_batch.shape, total_mask.shape)

    latent_denoising_loss = torch.square(denoised_latent - latent).sum(dim=-1) * total_mask
    latent_denoising_loss = _nodewise_to_graphwise(latent_denoising_loss, res_data_batch, total_mask)

    t = batch['t']
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )
    latent_fm_loss = latent_denoising_loss / (norm_scale ** 2) * scale

    latent_ref_noise = torch.square(noised_latent - latent).sum(dim=-1) * total_mask
    latent_ref_noise = _nodewise_to_graphwise(latent_ref_noise, res_data_batch, total_mask)

    if pointwise:
        dim_size = latent.shape[-1]
        latent_denoising_loss = latent_denoising_loss / dim_size
        latent_fm_loss = latent_fm_loss / dim_size
        latent_ref_noise = latent_ref_noise / dim_size


    # latent_mu = latent_outputs['latent_mu']
    # latent_logvar = latent_outputs['latent_logvar']
    # latent_denoising_nll = _nll(denoised_latent, latent_mu, latent_logvar)
    # latent_denoising_nll = _nodewise_to_graphwise(latent_denoising_nll, res_data.batch, total_mask)

    # latent_ref_noise_nll = _nll(latent, latent_mu, latent_logvar)
    # latent_ref_noise_nll = _nodewise_to_graphwise(latent_ref_noise_nll, res_data.batch, total_mask)

    return {
        "latent_denoising_loss": latent_denoising_loss,
        "latent_fm_loss": latent_fm_loss,
        "latent_ref_noise": latent_ref_noise,
        # "latent_denoising_nll": latent_denoising_nll,
        # "latent_ref_noise_nll": latent_ref_noise_nll,
    }

def latent_scalar_dense_sidechain_fm_loss(
        batch,
        latent_outputs,
        t_norm_clip=0.9,
        scale=0.01,
        pointwise=False,
        detach_gt_latent_grad=False
    ):
    res_data = batch['residue']
    x_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    total_mask = x_mask & noising_mask
    total_mask = total_mask.view(batch.num_graphs, -1)

    latent = latent_outputs['latent_sidechain']
    if detach_gt_latent_grad:
        latent = latent.detach()
    # latent = latent_outputs['latent_mu']
    noised_latent = latent_outputs['noised_latent_sidechain']
    denoised_latent = latent_outputs['pred_latent_sidechain']

    latent_denoising_loss = torch.square(denoised_latent - latent).sum(dim=-1) * total_mask
    latent_denoising_loss = latent_denoising_loss.sum(dim=-1) / total_mask.sum(dim=-1)

    t = batch['t']
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )
    latent_fm_loss = latent_denoising_loss / (norm_scale ** 2) * scale

    latent_ref_noise = torch.square(noised_latent - latent).sum(dim=-1) * total_mask
    latent_ref_noise = latent_ref_noise.sum(dim=-1) / total_mask.sum(dim=-1)


    if pointwise:
        dim_size = latent.shape[-1]
        latent_denoising_loss = latent_denoising_loss / dim_size
        latent_fm_loss = latent_fm_loss / dim_size
        latent_ref_noise = latent_ref_noise / dim_size


    # latent_mu = latent_outputs['latent_mu']
    # latent_logvar = latent_outputs['latent_logvar']
    # latent_denoising_nll = _nll(denoised_latent, latent_mu, latent_logvar)
    # latent_denoising_nll = _nodewise_to_graphwise(latent_denoising_nll, res_data.batch, total_mask)

    # latent_ref_noise_nll = _nll(latent, latent_mu, latent_logvar)
    # latent_ref_noise_nll = _nodewise_to_graphwise(latent_ref_noise_nll, res_data.batch, total_mask)

    return {
        "latent_denoising_loss": latent_denoising_loss,
        "latent_fm_loss": latent_fm_loss,
        "latent_ref_noise": latent_ref_noise,
        # "latent_denoising_nll": latent_denoising_nll,
        # "latent_ref_noise_nll": latent_ref_noise_nll,
    }

def latent_scalar_edge_fm_loss(
        batch,
        latent_outputs,
        t_norm_clip=0.9,
        scale=0.01,
        pointwise=False,
        detach_gt_latent_grad=False,
    ):
    res_data = batch['residue']
    x_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    total_mask = x_mask & noising_mask
    total_mask = total_mask.view(batch.num_graphs, -1)
    total_mask = total_mask[..., None] & total_mask[..., None, :]

    latent = latent_outputs['latent_edge']
    if detach_gt_latent_grad:
        latent = latent.detach()
    # latent = latent_outputs['latent_mu']
    noised_latent = latent_outputs['noised_latent_edge']
    denoised_latent = latent_outputs['pred_latent_edge']

    latent_denoising_loss = torch.square(denoised_latent - latent).sum(dim=-1) * total_mask
    latent_denoising_loss = latent_denoising_loss.sum(dim=(-1, -2)) / total_mask.sum(dim=(-1, -2))

    t = batch['t']
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )
    latent_fm_loss = latent_denoising_loss / (norm_scale ** 2) * scale

    latent_ref_noise = torch.square(noised_latent - latent).sum(dim=-1) * total_mask
    latent_ref_noise = latent_ref_noise.sum(dim=(-1, -2)) / total_mask.sum(dim=(-1, -2))

    if pointwise:
        dim_size = latent.shape[-1]
        latent_denoising_loss = latent_denoising_loss / dim_size
        latent_fm_loss = latent_fm_loss / dim_size
        latent_ref_noise = latent_ref_noise / dim_size

    return {
        "latent_denoising_loss": latent_denoising_loss,
        "latent_fm_loss": latent_fm_loss,
        "latent_ref_noise": latent_ref_noise,
        # "latent_denoising_nll": latent_denoising_nll,
        # "latent_ref_noise_nll": latent_ref_noise_nll,
    }


def latent_encoder_consistency_loss(
    batch,
    latent_outputs,
    detach_gt_latent_grad=True,
    t_norm_clip=0.9,
    scale=0.01,
    reduction='mean'
):
    res_data = batch['residue']
    x_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    total_mask = x_mask & noising_mask

    latent = latent_outputs['latent_sidechain']
    denoised_latent = latent_outputs['pred_latent_sidechain']
    if detach_gt_latent_grad:
        latent = latent.detach()

    consistency_data = latent_outputs['consistency_data']
    consistent_mu = consistency_data['latent_mu']
    consistent_logvar = consistency_data['latent_logvar']
    gt_consistency_nll = gaussian_nll(
        latent,
        consistent_mu,
        consistent_logvar,
        batch=res_data.batch,
        mask=total_mask,
        reduction=reduction
    )
    self_consistency_nll = gaussian_nll(
        denoised_latent,
        consistent_mu,
        consistent_logvar,
        batch=res_data.batch,
        mask=total_mask,
        reduction=reduction
    )

    t = batch['t']
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )
    scaled_gt_nll = gt_consistency_nll * scale / (norm_scale ** 2)
    scaled_self_nll = self_consistency_nll * scale / (norm_scale ** 2)

    return {
        "consistency_nll_gt": gt_consistency_nll,
        "consistency_nll_gt_scaled": scaled_gt_nll,
        "consistency_nll_self": self_consistency_nll,
        "consistency_nll_self_scaled": scaled_self_nll
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
    sidechain_dists_mse = residue_knn_neighborhood_atomic_dist_loss(
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