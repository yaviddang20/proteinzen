import torch
import numpy as np
import torch.nn.functional as F

from proteinzen.data.openfold.residue_constants import restypes
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.model.utils.graph import batchwise_to_nodewise, get_data_lens

from .utils import _nodewise_to_graphwise
from .atomic.atomic import framediff_local_atomic_context_loss
from .frames import angle_axis_rot_vf_loss

from proteinzen.stoch_interp.interpolate import so3_utils as so3_fm_utils


def multiframe_fm_loss(
    batch,
    denoiser_outputs,
    t_norm_clip=0.9,
    sep_rot_loss=True,
    local_atomic_dist_r=6,
    square_aux_loss_time_factor=False,
    trans_preconditioning=False,
    rot_preconditioning=False,
    use_fafe=False,
    use_fafe_l2=False,
    polar_upweight=False,
    sidechain_upweight=False,
    rot_exp_weighting=False,
    rot_vf_angle_loss_weight=0.5,
    ignore_rigid_2_vf_loss=False,
    disable_rot_vf_time_scaling=False,
    rot_cap_loss_weight=0.
):
    res_data = batch['residue']
    res_mask = res_data['res_mask']
    noising_mask = res_data['res_noising_mask']
    mask = res_mask & noising_mask
    res_batch = res_data.batch
    device = res_mask.device
    total_mask = res_mask & noising_mask

    if isinstance(res_data['rigids_t'], torch.Tensor):
        noised_frames = ru.Rigid.from_tensor_7(res_data['rigids_t'].to(device))
    else:
        noised_frames = res_data['rigids_t']
        dtype = noised_frames._trans.dtype
        noised_frames = ru.Rigid(
            rots=noised_frames._rots.to(device=device, dtype=dtype),
            trans=noised_frames._trans.to(device)
        )
    total_mask = total_mask[..., None].expand([-1, noised_frames.shape[-1]])
    if ignore_rigid_2_vf_loss:
        total_mask = total_mask.contiguous()
        total_mask[..., -1] = False

    denoised_frames = denoiser_outputs['final_rigids']
    gt_frames = ru.Rigid.from_tensor_7(res_data['rigids_1'].to(device))

    gt_frame_trans = gt_frames.get_trans()
    pred_frame_trans = denoised_frames.get_trans()
    ref_frame_trans = noised_frames.get_trans()
    pred_frame_trans_se = torch.square(gt_frame_trans - pred_frame_trans).sum(dim=-1)
    pred_frame_trans_mse = _nodewise_to_graphwise(pred_frame_trans_se, res_batch, total_mask)
    ref_frame_trans_se = torch.square(gt_frame_trans - ref_frame_trans).sum(dim=-1)
    ref_frame_trans_mse = _nodewise_to_graphwise(ref_frame_trans_se, res_batch, total_mask)

    bb = res_data['atom37'][:, (0, 1, 2, 4, 3)]
    bb_mask = res_data['atom37_mask'][:, (0, 1, 2, 4, 3)].bool()
    denoised_bb = denoiser_outputs['denoised_bb']
    backbone_mse = torch.square(denoised_bb - bb).sum(dim=-1)
    backbone_mse = _nodewise_to_graphwise(backbone_mse, res_batch, bb_mask & mask[..., None])

    t = batch['t']
    batch_size = t.shape[0]
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )
    t = batchwise_to_nodewise(t, res_data.batch)
    nodewise_norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )
    rots_t = noised_frames.get_rots().get_rot_mats()
    rots_1_pred = denoised_frames.get_rots().get_rot_mats()
    rots_1 = gt_frames.get_rots().get_rot_mats()
    pred_rot_vf = so3_fm_utils.calc_rot_vf(rots_t, rots_1_pred)
    gt_rot_vf = so3_fm_utils.calc_rot_vf(rots_t, rots_1)

    seqwise_weight = 1
    if polar_upweight:
        seqwise_weight = seqwise_weight * (res_data["polar_mask"].float() + 1)[..., None]
    if sidechain_upweight:
        sidechain_weight = torch.tensor([1] + [2 for _ in range(noised_frames.shape[-1]-1)], device=t.device)
        seqwise_weight = seqwise_weight * sidechain_weight[None]

    seqwise_loss = {}

    if sep_rot_loss:
        rot_vf_loss = angle_axis_rot_vf_loss(
            pred_rot_vf,
            gt_rot_vf,
            res_data.batch,
            total_mask,
            (norm_scale[..., None] if not disable_rot_vf_time_scaling else torch.ones_like(norm_scale[..., None])),
            seqwise_weight=seqwise_weight,
            angle_loss_weight=rot_vf_angle_loss_weight,
            rot_cap_loss_weight=rot_cap_loss_weight
        )
        with torch.no_grad():
            unscaled_rot_vf_loss = angle_axis_rot_vf_loss(
                pred_rot_vf,
                gt_rot_vf,
                res_data.batch,
                total_mask,
                norm_scale[..., None],
            )
            for i in range(19):
                seqwise_loss[f"zzz_unscaled_rot_vf_loss_{restypes[i]}"] = angle_axis_rot_vf_loss(
                    pred_rot_vf,
                    gt_rot_vf,
                    res_data.batch,
                    total_mask * (res_data['seq'] == i)[..., None],
                    norm_scale[..., None],
                )
    else:
        rot_vf_loss = torch.square(pred_rot_vf - gt_rot_vf).sum(dim=-1)
        unscaled_rot_vf_loss = rot_vf_loss
        rot_vf_loss = rot_vf_loss * seqwise_weight
        rot_vf_loss = _nodewise_to_graphwise(rot_vf_loss, res_data.batch, total_mask)
        rot_vf_loss = rot_vf_loss / ((norm_scale ** 2) if not disable_rot_vf_time_scaling else 1)
        unscaled_rot_vf_loss = _nodewise_to_graphwise(unscaled_rot_vf_loss, res_data.batch, total_mask)
        unscaled_rot_vf_loss = unscaled_rot_vf_loss / (norm_scale ** 2)
        for i in range(19):
            _unscaled_rot_vf_loss = _nodewise_to_graphwise(unscaled_rot_vf_loss, res_data.batch, total_mask * (res_data['seq'] == i)[..., None])
            _unscaled_rot_vf_loss = _unscaled_rot_vf_loss / (norm_scale ** 2)
            seqwise_loss[f"zzz_unscaled_rot_vf_loss_{restypes[i]}"] = _unscaled_rot_vf_loss

    if rot_exp_weighting:
        rot_vf_loss = 10 * rot_vf_loss * (norm_scale ** 2)

    trans_1_pred = denoised_frames.get_trans()
    trans_1 = gt_frames.get_trans()
    if trans_preconditioning:
        trans_vf_loss = torch.square(trans_1_pred - trans_1).sum(dim=-1) / (nodewise_norm_scale[..., None] ** 2)
        unscaled_trans_vf_loss = trans_vf_loss * 0.01
        trans_vf_loss = trans_vf_loss * seqwise_weight
        trans_vf_loss = _nodewise_to_graphwise(trans_vf_loss, res_data.batch, total_mask) * batch['trans_loss_weighting']

        with torch.no_grad():
            for i in range(19):
                seqwise_loss[f"zzz_unscaled_trans_vf_loss_{restypes[i]}"] = _nodewise_to_graphwise(
                    unscaled_trans_vf_loss, res_data.batch, total_mask * (res_data['seq'] == i)[..., None])

        unscaled_trans_vf_loss = _nodewise_to_graphwise(unscaled_trans_vf_loss, res_data.batch, total_mask)
    else:
        trans_vf_loss = torch.square(trans_1_pred - trans_1).sum(dim=-1) / (nodewise_norm_scale[..., None] ** 2)
        unscaled_trans_vf_loss = trans_vf_loss
        trans_vf_loss = trans_vf_loss * seqwise_weight
        trans_vf_loss = _nodewise_to_graphwise(trans_vf_loss, res_data.batch, total_mask)
        trans_vf_loss *= 0.01  # Angstroms to nm

        with torch.no_grad():
            for i in range(19):
                seqwise_loss[f"zzz_unscaled_trans_vf_loss_{restypes[i]}"] = _nodewise_to_graphwise(
                    unscaled_trans_vf_loss, res_data.batch, total_mask * (res_data['seq'] == i)[..., None]) * 0.01

        unscaled_trans_vf_loss = _nodewise_to_graphwise(unscaled_trans_vf_loss, res_data.batch, total_mask)
        unscaled_trans_vf_loss *= 0.01  # Angstroms to nm

    dist_mat_loss = framediff_local_atomic_context_loss(
        denoised_bb,
        bb,
        res_batch,
        res_mask,
        r=local_atomic_dist_r
    )

    if square_aux_loss_time_factor:
        scaled_dist_mat_loss = dist_mat_loss / (norm_scale ** 2) * 0.01
        scaled_backbone_mse = backbone_mse / (norm_scale ** 2) * 0.01
    else:
        # this seems to work well
        scaled_dist_mat_loss = dist_mat_loss / norm_scale * 0.01
        scaled_backbone_mse = backbone_mse / norm_scale * 0.01

    if use_fafe:
        seqwise_weight = 1
        if polar_upweight:
            polar_mask = res_data['polar_mask'].float().view(batch_size, -1)
            seqwise_weight = seqwise_weight * (polar_mask+1)[..., None].expand(-1, -1, gt_frames.shape[-1])
        if sidechain_upweight:
            polar_mask = res_data['polar_mask'].float().view(batch_size, -1)
            sidechain_weight = torch.ones(gt_frames.shape[-1], device=device)
            sidechain_weight[1:] += 1
            sidechain_weight = sidechain_weight[None, None].expand(*polar_mask.shape[:2], -1)
            seqwise_weight = seqwise_weight * sidechain_weight


        if isinstance(seqwise_weight, torch.Tensor):
            seqwise_weight = seqwise_weight.view(batch_size, -1)
            framepair_weight = seqwise_weight[..., None] * seqwise_weight[..., None, :]
        else:
            framepair_weight = seqwise_weight

        if use_fafe_l2:
            fafe = fafe_loss_l2(
                pred_frames=denoised_frames.view(batch_size, -1),
                gt_frames=gt_frames.view(batch_size, -1),
                frame_mask=res_mask.view(batch_size, -1).repeat_interleave(denoised_frames.shape[-1], dim=-1),
                framepair_weight=framepair_weight
            )
            scaled_fafe = fafe / norm_scale
        else:
            fafe = fafe_loss(
                pred_frames=denoised_frames.view(batch_size, -1),
                gt_frames=gt_frames.view(batch_size, -1),
                frame_mask=res_mask.view(batch_size, -1).repeat_interleave(denoised_frames.shape[-1], dim=-1),
            )
            scaled_fafe = fafe / norm_scale
    else:
        fafe = torch.zeros(batch_size, device=res_mask.device)

    scaled_fafe = fafe / (norm_scale ** 2)

    ret = {
        "rot_vf_loss": rot_vf_loss,
        "trans_vf_loss": trans_vf_loss,
        "unscaled_rot_vf_loss": unscaled_rot_vf_loss,
        "unscaled_trans_vf_loss": unscaled_trans_vf_loss,
        "pred_trans_mse": pred_frame_trans_mse,
        "pred_bb_mse": backbone_mse,
        "scaled_pred_bb_mse": scaled_backbone_mse,
        "ref_trans_mse": ref_frame_trans_mse,
        "dist_mat_loss": dist_mat_loss,
        "scaled_dist_mat_loss": scaled_dist_mat_loss,
        "fafe": fafe,
        "scaled_fafe": scaled_fafe
    }
    ret.update(seqwise_loss)
    return ret


# adapted in part from https://github.com/mooninrain/FAFE/blob/main/losses/fafe.py
def fafe_loss(
    pred_frames,
    gt_frames,
    frame_mask,
    rot_scale: float = 1.0,
    trans_scale: float = 20.0,
    dist_clamp: float | None = 20.,
    eps_so3: float = 1e-6,
):
    def geodesic_dist(rots1, rots2):
        R_diff = torch.einsum("...ij,...jk->...ik", rots1.transpose(-2, -1), rots2)
        R_diff_trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        return torch.acos(
            torch.clamp(
                (R_diff_trace - 1) / 2,
                min=-1 + eps_so3,
                max=1 - eps_so3
            )
        )

    gt_framepairs = gt_frames[..., None].invert().compose(gt_frames[..., None, :])
    pred_framepairs = pred_frames[..., None].invert().compose(pred_frames[..., None, :])
    mask = frame_mask[..., None] * frame_mask[..., None, :]

    trans_dist = torch.linalg.vector_norm(pred_framepairs.get_trans() - gt_framepairs.get_trans(), dim=-1)
    rot_dist = geodesic_dist(pred_framepairs.get_rots().get_rot_mats(), gt_framepairs.get_rots().get_rot_mats())

    trans_dist_loss = torch.sum(
        trans_dist.clamp(max=dist_clamp) * mask,
        dim=(-2, -1),
    ) / mask.sum(dim=(-2, -1))

    clamp_mask = trans_dist > dist_clamp
    rotpair_mask = mask * clamp_mask
    rot_dist_loss = torch.sum(
        rot_dist * rotpair_mask,
        dim=(-2, -1),
    ) / torch.clamp(rotpair_mask.sum(dim=(-2, -1)), min=1)


    return trans_dist_loss / trans_scale + rot_dist_loss / rot_scale


# adapted in part from https://github.com/mooninrain/FAFE/blob/main/losses/fafe.py
def fafe_loss_l2(
    pred_frames,
    gt_frames,
    frame_mask,
    framepair_weight = 1.,
    rot_scale: float = 1.0,
    trans_scale: float = 20.0,
    dist_clamp: float | None = 20.,
    eps_so3: float = 1e-6,
):
    def geodesic_dist(rots1, rots2):
        R_diff = torch.einsum("...ij,...jk->...ik", rots1.transpose(-2, -1), rots2)
        R_diff_trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        return torch.acos(
            torch.clamp(
                (R_diff_trace - 1) / 2,
                min=-1 + eps_so3,
                max=1 - eps_so3
            )
        )

    gt_framepairs = gt_frames[..., None].invert().compose(gt_frames[..., None, :])
    pred_framepairs = pred_frames[..., None].invert().compose(pred_frames[..., None, :])
    mask = frame_mask[..., None] * frame_mask[..., None, :]
    mask = mask * (1 - torch.eye(frame_mask.shape[-1], device=mask.device))[None]

    trans_dist = torch.linalg.vector_norm(pred_framepairs.get_trans() - gt_framepairs.get_trans(), dim=-1)
    trans_dist = trans_dist.clamp(max=dist_clamp)
    rot_dist = geodesic_dist(pred_framepairs.get_rots().get_rot_mats(), gt_framepairs.get_rots().get_rot_mats())

    trans_dist = trans_dist * framepair_weight
    rot_dist = rot_dist * framepair_weight

    trans_dist_loss = torch.sum(
        trans_dist**2 * mask,
        dim=(-2, -1),
    ) / mask.sum(dim=(-2, -1))

    # clamp_mask = trans_dist > dist_clamp
    # rotpair_mask = mask * clamp_mask
    rot_dist_loss = torch.sum(
        rot_dist**2 * mask,
        dim=(-2, -1),
    ) / torch.clamp(mask.sum(dim=(-2, -1)), min=1)


    return torch.sqrt(trans_dist_loss / trans_scale**2 + rot_dist_loss / rot_scale**2)


