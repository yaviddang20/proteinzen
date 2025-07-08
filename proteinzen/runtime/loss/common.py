import numpy as np
import torch
import torch.nn.functional as F

from proteinzen.data.openfold.residue_constants import restype_order_with_x
from proteinzen.stoch_interp import utils as du

from .utils import _nodewise_to_graphwise
from .atomic.atom14 import atom14_mse_loss, chi_loss
from .atomic.atom10 import atom10_local_mse_loss
from .atomic.atomic import residue_knn_neighborhood_atomic_dist_loss, local_atomic_context_loss, smooth_lddt_loss, sparse_smooth_lddt_loss
from .atomic.interresidue import intersidechain_clash_loss
from .latent import so3_embedding_kl, scalars_kl_div, nodewise_kl_div, gaussian_nll
from .frames import all_atom_fape_loss

from proteinzen.utils.openfold.rigid_utils import Rigid


def seq_cce_loss(ref_seq,
                 seq_logits,
                 batch,
                 mask,
                 label_smoothing=0.0,
                 seqwise_weight=None,
                 logits_as_probs=False):
    """ CCE loss on logits """
    if logits_as_probs:
        assert label_smoothing == 0, "Label smoothing not implemented for CCE with probs"
        seq_probs = torch.gather(seq_logits, 1, (ref_seq * mask)[..., None])
        cce = -torch.log(seq_probs.clip(min=1e-6, max=1-1e-6))
        cce = cce.squeeze(-1)
    else:
        cce = F.cross_entropy(seq_logits, ref_seq * mask, reduction='none', label_smoothing=label_smoothing)

    if seqwise_weight is not None:
        cce = cce * seqwise_weight
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


def atomic_losses_reduced(batch,
                  model_outputs,
                  polar_upweight=False,
                  sidechain_upweight=False,
):
    res_data = batch['residue']

    gt_atom14 = res_data['atom14_gt_positions']
    alt_atom14 = res_data['atom14_alt_gt_positions']

    seq = res_data['seq']
    seq_mask = res_data['seq_mask']

    res_mask = res_data['res_mask']
    seq_noising_mask = res_data['seq_noising_mask']
    atom14_gt_mask = res_data['atom14_gt_exists']
    atom14_alt_gt_mask = res_data['atom14_alt_gt_exists']

    minimal_mask = res_mask & seq_mask
    seq_loss_mask = res_mask & seq_mask & seq_noising_mask
    pred_atom14_gt_seq = model_outputs['denoised_atom14_gt_seq']
    seq_logits = model_outputs['decoded_seq_logits']
    # print("loss", batch.name, gt_atom14)
    # print("pred", batch.name, model_outputs['denoised_atom14_gt_seq'])

    seqwise_weight = 1
    if polar_upweight:
        seqwise_weight = seqwise_weight * (res_data['polar_mask'].float()+1)[..., None]
    if sidechain_upweight:
        sidechain_weight = torch.ones(14, device=gt_atom14.device)
        sidechain_weight[5:] += 1
        seqwise_weight = seqwise_weight * sidechain_weight[None]


    atom14_mse = atom14_mse_loss(
        gt_atom14,
        alt_atom14,
        pred_atom14_gt_seq,
        res_data.batch,
        atom14_gt_mask,
        atom14_alt_gt_mask,
        res_data['rigids_noising_mask'][..., 0], #minimal_mask,
        no_bb=False,
        seqwise_weight=seqwise_weight,
        ignore_symmetry=True
    )
    seq_loss = seq_cce_loss(
        seq,
        seq_logits,
        res_data.batch,
        seq_loss_mask,
        seqwise_weight=(res_data['polar_mask'].float()+1 if polar_upweight else None)
    )
    per_seq_recov = seq_recov(
        seq,
        seq_logits,
        res_data.batch,
        seq_loss_mask)
    atom14_rmsd = atom14_mse.sqrt()

    smooth_lddt = smooth_lddt_loss(
        pred_atom14=pred_atom14_gt_seq,
        gt_atom14=gt_atom14,
        alt_atom14=alt_atom14,
        batch=res_data.batch,
        atom14_mask=atom14_gt_mask.bool()
    )

    out_dict = {
        "seq_loss": seq_loss,
        "per_seq_recov": per_seq_recov,
        "atom14_mse": atom14_mse,
        "atom14_rmsd": atom14_rmsd,
        "smooth_lddt": smooth_lddt,
    }

    return out_dict

def dense_smooth_lddt_loss(
    pred_atom14,
    gt_atom14,
    alt_atom14,
    atom14_mask,
    eps=1e-6,
):
    """ smooth_lddt_loss from AlphaFold3

    Args:
        pred_atom14 (_type_): _description_
        gt_atom14 (_type_): _description_
        alt_atom14 (_type_): _description_
        batch (_type_): _description_
        atom14_mask (_type_): _description_
        r (int, optional): _description_. Defaults to 6.
        eps (_type_, optional): _description_. Defaults to 1e-6.
        max_num_neighbors (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    # with torch.no_grad():
    #     pred_gt_diff = torch.square(pred_atom14 - gt_atom14).sum(dim=-1)
    #     pred_alt_diff = torch.square(pred_atom14 - alt_atom14).sum(dim=-1)
    #     pred_gt_mse = torch.sum(pred_gt_diff * atom14_mask, dim=-1)
    #     pred_alt_mse = torch.sum(pred_alt_diff * atom14_mask, dim=-1)

    #     gt_over_alt = pred_gt_mse < pred_alt_mse
    #     ref_atom14 = gt_atom14 * gt_over_alt[..., None, None] + alt_atom14 * ~gt_over_alt[..., None, None]
    ref_atom14 = gt_atom14
    # print(pred_atom14.shape, ref_atom14.shape)

    flat_ref_atom14 = ref_atom14.flatten(-3, -2)
    flat_pred_atom14 = pred_atom14.flatten(-3, -2)
    pred_atom14_dists = torch.cdist(flat_pred_atom14, flat_pred_atom14, p=2)
    ref_atom14_dists = torch.cdist(flat_ref_atom14, flat_ref_atom14, p=2).squeeze(0)
    abs_dev = torch.abs(pred_atom14_dists - ref_atom14_dists + eps)
    lddt = 0.25 * (
        torch.sigmoid(0.5 - abs_dev)
        + torch.sigmoid(1 - abs_dev)
        + torch.sigmoid(2 - abs_dev)
        + torch.sigmoid(4 - abs_dev)
    )
    mask = 1 - torch.eye(lddt.shape[1], device=lddt.device)[None]
    # print(atom14_mask.shape, mask.shape)
    mask = mask * atom14_mask.flatten(-2, -1)[..., None] * atom14_mask.flatten(-2, -1)[..., None, :]
    radius_mask = (ref_atom14_dists < 15)
    smooth_lddt = torch.sum(lddt * mask * radius_mask) / torch.sum(mask)
    return 1 - smooth_lddt

def atomic_losses_dense_batch(
    batch,
    model_outputs,
    seqwise_weight=1,
):
    atom_data = batch['atom']
    token_data = batch['token']

    gt_atom14 = atom_data['atom14_gt_positions']
    alt_atom14 = atom_data['atom14_alt_gt_positions']
    atom14_gt_mask = atom_data['atom14_gt_exists']

    seq = token_data['seq']
    seq_mask = token_data['seq_mask']
    token_mask = token_data['token_mask']
    seq_noising_mask = token_data['seq_noising_mask']

    seq_loss_mask = token_mask & seq_mask & seq_noising_mask
    pred_atom14_gt_seq = model_outputs['denoised_atom14_gt_seq']
    seq_logits = model_outputs['decoded_seq_logits']

    flat_masked_seq = (seq * seq_loss_mask).flatten(0, 1)
    flat_seq_logits = seq_logits.flatten(0, 1)
    seq_loss = F.cross_entropy(
        flat_seq_logits,
        flat_masked_seq,
        reduction='none'
    )
    seq_loss = seq_loss.view(seq.shape)
    seq_loss = seq_loss * seq_loss_mask

    if (~seq_loss_mask).all(dim=-1).any():
        print("seq_loss_mask all False", batch['name'], (~seq_loss_mask).all(dim=-1))

    seq_loss = seq_loss * seqwise_weight
    seq_loss = seq_loss.sum(dim=-1) / seq_loss_mask.sum(dim=-1).clip(min=1)

    per_seq_recov = (seq == seq_logits[..., :-1].argmax(dim=-1))
    per_seq_recov = (per_seq_recov * seq_loss_mask).sum(dim=-1) / seq_loss_mask.sum(dim=-1).clip(min=1)

    smooth_lddt = dense_smooth_lddt_loss(
        pred_atom14=pred_atom14_gt_seq,
        gt_atom14=gt_atom14,
        alt_atom14=alt_atom14,
        atom14_mask=atom14_gt_mask.bool()
    )

    out_dict = {
        "seq_loss": seq_loss,
        "per_seq_recov": per_seq_recov,
        "smooth_lddt": smooth_lddt,
    }

    return out_dict



def atom14_fm_losses(batch,
                  model_outputs,
                  label_smoothing=0.0,
                  logit_norm_loss=0.0,
                  use_smooth_lddt=False,
                  use_fape=False,
                  fape_length_scale=10.,
                  t_norm_clip=0.9,
                  use_sidechain_dists_mse_loss=True,
                  use_local_atomic_dist_loss=True,
                  use_sidechain_clash_loss=True,
                  polar_upweight=False,
                  sidechain_upweight=False,
                  preconditioning=False,
                  rigid_align=False
):
    res_data = batch['residue']

    t = batch['t']
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )

    gt_atom14 = res_data['atom14']
    alt_atom14 = res_data['atom14']
    ref_atom14 = res_data['noised_atom14']

    seq = res_data['seq']
    seq_mask = res_data['seq_mask']

    res_mask = res_data['res_mask']
    noising_mask = res_data['res_noising_mask']
    # mlm_mask = res_data['mlm_mask']
    atom14_gt_mask = res_data['atom14_mask']
    atom14_alt_gt_mask = res_data['atom14_mask']

    minimal_mask = res_mask & seq_mask
    # ae_mask = res_mask & mlm_mask & seq_mask
    denoiser_mask = res_mask & noising_mask & seq_mask
    # decoded_all_atom14 = model_outputs['decoded_all_atom14']
    # decoded_all_chis = model_outputs['decoded_all_chis']
    pred_atom14 = model_outputs['denoised_atom14']
    pred_atom14_gt_seq = model_outputs['denoised_atom14_gt_seq']
    seq_logits = model_outputs['decoded_seq_logits']

    # pred_atom14 = _collect_from_seq(decoded_all_atom14, seq, seq_mask)
    # pred_chis = _collect_from_seq(decoded_all_chis, seq, seq_mask)
    seqwise_weight = 1
    if polar_upweight:
        seqwise_weight = seqwise_weight * (res_data['polar_mask'].float()+1)[..., None]
    if sidechain_upweight:
        sidechain_weight = torch.ones(14, device=gt_atom14.device)
        sidechain_weight[5:] += 1
        seqwise_weight = seqwise_weight * sidechain_weight[None]


    if rigid_align:
        flat_gt_atom14 = gt_atom14.flatten(0, 1)
        flat_pred_atom14 = pred_atom14_gt_seq.flatten(0, 1)
        atom_batch = res_data.batch[..., None].expand(-1, 14).reshape(-1)
        flat_pred_atom14, _, _ = du.align_structures(
            flat_pred_atom14,
            atom_batch,
            flat_gt_atom14,
        )
        pred_atom14_gt_seq = flat_pred_atom14.unflatten(0, (-1, 14))

    atom14_mse = atom14_mse_loss(
        gt_atom14,
        alt_atom14,
        pred_atom14_gt_seq,
        res_data.batch,
        atom14_gt_mask,
        atom14_alt_gt_mask,
        minimal_mask,
        no_bb=False,
        seqwise_weight=seqwise_weight,
        ignore_symmetry=True
    )
    with torch.no_grad():
        unscaled_atom14_mse = atom14_mse_loss(
            gt_atom14,
            alt_atom14,
            pred_atom14_gt_seq,
            res_data.batch,
            atom14_gt_mask,
            atom14_alt_gt_mask,
            minimal_mask,
            no_bb=False,
            ignore_symmetry=True
        )
        ref_atom14_mse = atom14_mse_loss(
            gt_atom14,
            alt_atom14,
            ref_atom14,
            res_data.batch,
            atom14_gt_mask,
            atom14_alt_gt_mask,
            minimal_mask,
            no_bb=False,
            seqwise_weight=seqwise_weight,
            ignore_symmetry=True
        )

    # print(unscaled_atom14_mse, ref_atom14_mse)
    # print(batch.name, 'loss pred', pred_atom14_gt_seq)
    # print(batch.name, 'loss ref', ref_atom14)
    # print(batch.name, 'loss gt', gt_atom14)
    # exit()

    seq_loss = seq_cce_loss(
        seq,
        seq_logits,
        res_data.batch,
        seq_mask,
        label_smoothing=label_smoothing,
        seqwise_weight=(res_data['polar_mask'].float()+1 if polar_upweight else None)
    )
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

    if preconditioning:
        scaled_atom14_mse = atom14_mse * batch['loss_weighting']
    else:
        scaled_atom14_mse = atom14_mse / (norm_scale**2) * 0.01

    if use_smooth_lddt:
        smooth_lddt = smooth_lddt_loss(
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
            pred_atom14=pred_atom14_gt_seq,
            gt_atom14=gt_atom14,
            pred_rigids=model_outputs['final_rigids'][..., 0],
            gt_rigids=Rigid.from_tensor_7(batch['residue']['rigids_1'])[..., 0],
            batch=res_data.batch,
            atom14_mask=atom14_gt_mask,
            length_scale=fape_length_scale
        )
    else:
        fape = torch.zeros_like(t)
    scaled_fape = fape / norm_scale


    if use_sidechain_clash_loss:
        pred_atom14_clash_loss = intersidechain_clash_loss(
            pred_atom14=pred_atom14,
            atom14_mask=model_outputs['decoded_atom14_mask'].bool(),
            seq=seq_logits.argmax(dim=-1),
            batch=res_data.batch
        )
    else:
        pred_atom14_clash_loss = torch.zeros_like(t)


    logit_norm_loss = logit_norm_loss * _nodewise_to_graphwise(
        torch.sum(seq_logits**2, dim=-1),
        res_data.batch,
        denoiser_mask
    )

    out_dict = {
        "seq_loss": seq_loss,
        "per_seq_recov": per_seq_recov,
        "atom14_mse": atom14_mse,
        "unscaled_atom14_mse": unscaled_atom14_mse,
        "atom14_rmsd": atom14_rmsd,
        "unscaled_atom14_rmsd": unscaled_atom14_mse.sqrt(),
        "ref_atom14_mse": ref_atom14_mse,
        "ref_atom14_rmsd": ref_atom14_mse.sqrt(),
        "sidechain_dists_mse": sidechain_dists_mse,
        "local_atomic_dist_loss": local_atomic_dist_loss,
        "scaled_local_atomic_dist_loss": scaled_local_atomic_dist_loss,
        "scaled_atom14_mse": scaled_atom14_mse,
        "pred_sidechain_clash_loss": pred_atom14_clash_loss,
        "smooth_lddt": smooth_lddt,
        "fape": fape,
        "scaled_fape": scaled_fape,
        # "bond_length_mse": bond_length_mse,
        # "bond_angle_loss": bond_angle_loss,
        "logit_norm_loss": logit_norm_loss,
    }

    return out_dict


def seq_losses_dense_batch(
    batch,
    model_outputs,
    seqwise_weight=1,
):
    token_data = batch['token']

    seq = token_data['seq']
    # seq_mask = token_data['seq_mask']
    seq_mask = token_data['token_mask']
    token_mask = token_data['token_mask']
    seq_noising_mask = token_data['seq_noising_mask']

    seq_loss_mask = token_mask & seq_mask & seq_noising_mask
    seq_logits = model_outputs['decoded_seq_logits']

    flat_masked_seq = (seq * seq_loss_mask).flatten(0, 1)
    flat_seq_logits = seq_logits.flatten(0, 1)
    seq_loss = F.cross_entropy(
        flat_seq_logits,
        flat_masked_seq,
        reduction='none'
    )
    seq_loss = seq_loss.view(seq.shape)
    seq_loss = seq_loss * seq_loss_mask

    if (~seq_loss_mask).all(dim=-1).any():
        # print("seq_loss_mask all False", batch['name'], (~seq_loss_mask).all(dim=-1))
        print("seq_loss_mask all False", (~seq_loss_mask).all(dim=-1))

    seq_loss = seq_loss * seqwise_weight
    seq_loss = seq_loss.sum(dim=-1) / seq_loss_mask.sum(dim=-1).clip(min=1)

    per_seq_recov = (seq == seq_logits[..., :-1].argmax(dim=-1))
    per_seq_recov = (per_seq_recov * seq_loss_mask).sum(dim=-1) / seq_loss_mask.sum(dim=-1).clip(min=1)

    out_dict = {
        "seq_loss": seq_loss,
        "per_seq_recov": per_seq_recov,
    }

    return out_dict