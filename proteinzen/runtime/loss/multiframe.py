import torch
import numpy as np
import torch.nn.functional as F

from proteinzen.data.openfold.residue_constants import restypes
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.model.utils.graph import batchwise_to_nodewise, get_data_lens

from .utils import _nodewise_to_graphwise
from .atomic.atomic import framediff_local_atomic_context_loss
from .frames import angle_axis_rot_vf_loss

from proteinzen.stoch_interp import so3_utils as so3_fm_utils


def multiframe_fm_loss_reduced(
    batch,
    denoiser_outputs,
    t_norm_clip=0.9,
    sep_rot_loss=True,
    polar_upweight=False,
    sidechain_upweight=False,
    rot_vf_angle_loss_weight=0.5,
    rot_cap_loss_weight=0.,
    fafe_l2_block_mask_size=1,
    downweight_K=False
):
    res_data = batch['residue']
    res_mask = res_data['res_mask']
    rigids_noising_mask = res_data['rigids_noising_mask']

    res_batch = res_data.batch
    device = res_mask.device
    total_mask = res_mask[..., None] & rigids_noising_mask

    if isinstance(res_data['rigids_t'], torch.Tensor):
        noised_frames = ru.Rigid.from_tensor_7(res_data['rigids_t'].to(device))
    else:
        noised_frames = res_data['rigids_t']
        dtype = noised_frames._trans.dtype
        noised_frames = ru.Rigid(
            rots=noised_frames._rots.to(device=device, dtype=dtype),
            trans=noised_frames._trans.to(device)
        )
    # total_mask = total_mask[..., None].expand([-1, noised_frames.shape[-1]])

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
    backbone_mse = _nodewise_to_graphwise(backbone_mse, res_batch, bb_mask & total_mask[..., [0]])

    batch_size = batch.num_graphs
    rigidwise_t = batch['rigidwise_t']
    rigidwise_norm_scale = 1 - torch.min(
        rigidwise_t, torch.as_tensor(t_norm_clip)
    )
    rots_t = noised_frames.get_rots().get_rot_mats()
    rots_1_pred = denoised_frames.get_rots().get_rot_mats()
    rots_1 = gt_frames.get_rots().get_rot_mats()
    pred_rot_vf = so3_fm_utils.calc_rot_vf(rots_t, rots_1_pred)
    gt_rot_vf = so3_fm_utils.calc_rot_vf(rots_t, rots_1)

    seqwise_weight = 1
    if polar_upweight:
        seqwise_weight = seqwise_weight * (res_data["polar_mask"].float() + 1)[..., None]
        if downweight_K:
            seqwise_weight[res_data.seq == restypes.index("K")] = 0.5
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
            rigidwise_norm_scale,
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
                rigidwise_norm_scale,
            )
    else:
        rot_vf_loss = torch.square(pred_rot_vf - gt_rot_vf).sum(dim=-1)
        unscaled_rot_vf_loss = rot_vf_loss
        rot_vf_loss = rot_vf_loss * seqwise_weight
        rot_vf_loss = rot_vf_loss / (rigidwise_norm_scale ** 2)
        rot_vf_loss = _nodewise_to_graphwise(rot_vf_loss, res_data.batch, total_mask)
        with torch.no_grad():
            unscaled_rot_vf_loss = unscaled_rot_vf_loss / (rigidwise_norm_scale ** 2)[..., None]
            unscaled_rot_vf_loss = _nodewise_to_graphwise(unscaled_rot_vf_loss, res_data.batch, total_mask)

    trans_1_pred = denoised_frames.get_trans()
    trans_1 = gt_frames.get_trans()
    trans_vf_loss = torch.square(trans_1_pred - trans_1).sum(dim=-1) / (rigidwise_norm_scale ** 2)
    unscaled_trans_vf_loss = trans_vf_loss
    trans_vf_loss = trans_vf_loss * seqwise_weight
    trans_vf_loss = _nodewise_to_graphwise(trans_vf_loss, res_data.batch, total_mask)
    trans_vf_loss *= 0.01  # Angstroms to nm
    unscaled_trans_vf_loss = _nodewise_to_graphwise(unscaled_trans_vf_loss, res_data.batch, total_mask)
    unscaled_trans_vf_loss *= 0.01  # Angstroms to nm

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

    fafe = fafe_loss_l2(
        pred_frames=denoised_frames.view(batch_size, -1),
        gt_frames=gt_frames.view(batch_size, -1),
        frame_mask=res_mask.view(batch_size, -1).repeat_interleave(denoised_frames.shape[-1], dim=-1),
        framepair_weight=framepair_weight,
        block_diag_size=fafe_l2_block_mask_size
    )

    flat_norm_scale = rigidwise_norm_scale.view(batch_size, -1)
    # this is quite arbitrary
    # the intuition is that the "lower noise" frame should provide some information
    # to the "higher noise" frame, so the pairwise fafe weighting
    # should be somewhere in between
    pair_norm_scale = (flat_norm_scale[..., None] + flat_norm_scale[..., None, :]) / 2
    framepair_weight = framepair_weight * pair_norm_scale

    scaled_fafe = fafe_loss_l2(
        pred_frames=denoised_frames.view(batch_size, -1),
        gt_frames=gt_frames.view(batch_size, -1),
        frame_mask=res_mask.view(batch_size, -1).repeat_interleave(denoised_frames.shape[-1], dim=-1),
        framepair_weight=framepair_weight,
        block_diag_size=fafe_l2_block_mask_size
    )


    ret = {
        "rot_vf_loss": rot_vf_loss,
        "trans_vf_loss": trans_vf_loss,
        "unscaled_rot_vf_loss": unscaled_rot_vf_loss,
        "unscaled_trans_vf_loss": unscaled_trans_vf_loss,
        "pred_trans_mse": pred_frame_trans_mse,
        "pred_bb_mse": backbone_mse,
        "ref_trans_mse": ref_frame_trans_mse,
        "fafe": fafe,
        "scaled_fafe": scaled_fafe
    }
    ret.update(seqwise_loss)
    return ret


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
    block_diag_size=1,
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
    # mask diagonal
    if block_diag_size == 1:
        mask = mask * (1 - torch.eye(frame_mask.shape[-1], device=mask.device))[None]
    else:
        assert mask.shape[-1] % block_diag_size == 0
        repeats = mask.shape[-1] // block_diag_size
        block_mask = torch.block_diag(*[torch.ones(block_diag_size, block_diag_size) for _ in range(repeats)])
        mask = mask * (1 - block_mask.to(mask.device))[None]

    trans_dist = torch.linalg.vector_norm(pred_framepairs.get_trans() - gt_framepairs.get_trans(), dim=-1)
    clamp_mask = trans_dist > dist_clamp
    trans_dist = trans_dist.clamp(max=dist_clamp)
    rot_dist = geodesic_dist(pred_framepairs.get_rots().get_rot_mats(), gt_framepairs.get_rots().get_rot_mats())

    trans_dist = trans_dist * framepair_weight
    rot_dist = rot_dist * framepair_weight

    trans_dist_loss = torch.sum(
        trans_dist**2 * mask,
        dim=(-2, -1),
    ) / mask.sum(dim=(-2, -1))

    rotpair_mask = mask * clamp_mask
    rot_dist_loss = torch.sum(
        rot_dist**2 * rotpair_mask,
        dim=(-2, -1),
    ) / torch.clamp(mask.sum(dim=(-2, -1)), min=1)


    return torch.sqrt(trans_dist_loss / trans_scale**2 + rot_dist_loss / rot_scale**2)

