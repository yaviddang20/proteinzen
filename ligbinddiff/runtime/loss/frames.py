import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric.utils as pygu
from torch_geometric.nn import radius_graph

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.atom_reps import atom14_sidechain_angles, atom14_residue_angles
from ligbinddiff.model.utils.graph import batchwise_to_nodewise, get_data_lens

from .utils import _nodewise_to_graphwise
from .atomic.common import atom91_to_atom14
from .atomic.atomic import framediff_local_atomic_context_loss
from .atomic.interresidue import backbone_dihedrals_loss, chain_constraints_loss, bb_clash_loss
from .openfold import compute_fape

from ligbinddiff.stoch_interp.interpolate import so3_utils as so3_fm_utils

def all_atom_fape_loss(
    pred_atom14,
    gt_atom14,
    pred_rigids,
    gt_rigids,
    batch,
    atom14_mask,
):
    ret = []
    for i in range(batch.max().item()+1):
        subset = (batch == i)
        fape = compute_fape(
            pred_frames=pred_rigids[subset],
            target_frames=gt_rigids[subset],
            frames_mask=atom14_mask[subset, 1],
            pred_positions=pred_atom14[subset],
            target_positions=gt_atom14[subset],
            positions_mask=atom14_mask[subset],
            length_scale=10.,
            l1_clamp_distance=10.
        )
        ret.append(fape)
    loss = torch.cat(ret, dim=-1)
    return _nodewise_to_graphwise(loss, batch, atom14_mask[:, 1])


def angle_axis_rot_score_loss(
        pred_rot_score,
        ref_rot_score,
        score_scaling,
        t,
        batch,
        res_mask,
        angle_loss_weight=0.5,
        angle_loss_t_threshold=0.2):
    gt_rot_score = ref_rot_score

    gt_rot_angle = torch.norm(gt_rot_score, dim=-1, keepdim=True)
    gt_rot_axis = gt_rot_score / (gt_rot_angle + 1e-6)

    pred_rot_angle = torch.norm(pred_rot_score, dim=-1, keepdim=True)
    pred_rot_axis = pred_rot_score / (pred_rot_angle + 1e-6)

    # Separate loss on the axis
    axis_loss = torch.square(gt_rot_axis - pred_rot_axis).sum(dim=-1)

    # Separate loss on the angle
    angle_loss = (gt_rot_angle - pred_rot_angle)**2
    angle_loss = torch.sum(
        angle_loss / score_scaling[:, None]**2,
        dim=-1
    )
    angle_loss *= angle_loss_weight
    angle_loss *= t > angle_loss_t_threshold
    rot_loss = angle_loss + axis_loss
    return _nodewise_to_graphwise(rot_loss, batch, res_mask)


def angle_axis_rot_vf_loss(
        pred_rot_vf,
        ref_rot_vf,
        batch,
        res_mask,
        norm_scale,
        angle_loss_weight=0.5,
        eps=1e-8):
    pred_rot_vf = pred_rot_vf / norm_scale[batch, None]
    ref_rot_vf = ref_rot_vf / norm_scale[batch, None]

    gt_rot_angle = torch.norm(ref_rot_vf, dim=-1, keepdim=True)
    gt_rot_axis = ref_rot_vf / (gt_rot_angle + eps)

    pred_rot_angle = torch.norm(pred_rot_vf, dim=-1, keepdim=True)
    pred_rot_axis = pred_rot_vf / (pred_rot_angle + eps)

    # Separate loss on the axis
    axis_loss = torch.square(gt_rot_axis - pred_rot_axis).sum(dim=-1)

    # Separate loss on the angle
    angle_loss = (gt_rot_angle - pred_rot_angle)**2
    angle_loss = torch.sum(
        angle_loss,
        dim=-1
    )
    rot_loss = angle_loss * angle_loss_weight + axis_loss
    return _nodewise_to_graphwise(rot_loss, batch, res_mask)


def edge_index_dist_loss(pred_x_ca, ref_x_ca, edge_index, data_lens, x_mask, eps=1e-8):
    pred_src = pred_x_ca[edge_index[1]]
    pred_dst = pred_x_ca[edge_index[0]]
    pred_dist = torch.linalg.vector_norm(pred_src - pred_dst + eps, dim=-1)

    ref_src = ref_x_ca[edge_index[1]]
    ref_dst = ref_x_ca[edge_index[0]]
    ref_dist = torch.linalg.vector_norm(ref_src - ref_dst + eps, dim=-1)

    dist_mse = torch.square(pred_dist - ref_dist)
    return _nodewise_to_graphwise(dist_mse[~x_mask], data_lens, x_mask)


def full_dist_mat_loss(pred_x_ca, ref_x_ca, batch, x_mask, eps=1e-8):
    ret = []
    for i in range(batch.max().item()+1):
        select = (batch == i)
        pred = pred_x_ca[select]
        ref = ref_x_ca[select]

        pred_dist_mat = torch.linalg.vector_norm(pred[:, None] - pred[None] + eps, dim=-1)
        ref_dist_mat = torch.linalg.vector_norm(ref[:, None] - ref[None] + eps, dim=-1)
        mask = ~x_mask[select]
        edge_mask = mask[None] * mask[..., None]

        dist_mse = torch.square(pred_dist_mat / (ref_dist_mat + eps) - 1)
        dist_mse = torch.sum(dist_mse * edge_mask) / edge_mask.long().sum()
        ret.append(dist_mse)
    return torch.stack(ret, dim=0)


def bb_frame_diffusion_loss(batch,
                            denoiser_outputs):
    res_data = batch['residue']
    res_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    res_batch = res_data.batch
    device = res_mask.device
    total_mask = res_mask & noising_mask

    if isinstance(res_data['rigids_t'], torch.Tensor):
        noised_bb_frames = ru.Rigid.from_tensor_7(res_data['rigids_t'].to(device))
    else:
        noised_bb_frames = res_data['rigids_t']
        dtype = noised_bb_frames._trans.dtype
        noised_bb_frames = ru.Rigid(
            rots=noised_bb_frames._rots.to(device=device, dtype=dtype),
            trans=noised_bb_frames._trans.to(device)
        )

    denoised_bb_frames = denoiser_outputs['final_rigids']

    X_ca = res_data['res_ca']
    pred_frame_X_ca = denoised_bb_frames.get_trans()
    ref_frame_X_ca = noised_bb_frames.get_trans()
    pred_X_ca_se = torch.square(X_ca - pred_frame_X_ca).sum(dim=-1)
    pred_X_ca_mse = _nodewise_to_graphwise(pred_X_ca_se, res_batch, total_mask)
    ref_X_ca_se = torch.square(X_ca - ref_frame_X_ca).sum(dim=-1)
    ref_X_ca_mse = _nodewise_to_graphwise(ref_X_ca_se, res_batch, total_mask)

    bb = res_data['atom37'][:, (0, 1, 2, 4, 3)]
    bb_mask = res_data['atom37_mask'][:, (0, 1, 2, 4, 3)].bool()
    denoised_bb = denoiser_outputs['denoised_bb']
    backbone_mse = torch.square(denoised_bb - bb).sum(dim=-1)
    backbone_mse = _nodewise_to_graphwise(backbone_mse, res_batch, bb_mask)

    pred_rot_score, pred_trans_score = denoiser_outputs['pred_bb_score']
    ref_rot_score = res_data['rot_score']
    ref_trans_score = res_data['trans_score']

    rot_score_scaling = res_data['rot_score_scaling']
    trans_score_scaling = res_data['trans_score_scaling']

    trans_score_scaling_nodewise = trans_score_scaling
    rot_score_scaling_nodewise = rot_score_scaling

    # Translation score loss
    trans_score_se = (ref_trans_score - pred_trans_score)**2 * total_mask[..., None]
    trans_score_loss = (trans_score_se / trans_score_scaling_nodewise[:, None]**2).sum(dim=-1)
    trans_score_loss = _nodewise_to_graphwise(trans_score_loss, res_batch, total_mask)

    # Rotation score loss
    rot_score_loss = angle_axis_rot_score_loss(
        pred_rot_score,
        ref_rot_score,
        rot_score_scaling_nodewise,
        batchwise_to_nodewise(res_data['t'], res_batch),
        res_batch,
        res_mask
    )

    dist_mat_loss = framediff_local_atomic_context_loss(
        denoised_bb,
        bb,
        res_batch,
        res_mask
    )

    # bb_dihedrals_loss = backbone_dihedrals_loss(
    #     denoised_bb[:, :3],
    #     bb[:, :3],
    #     data_lens,
    #     ~noising_mask)

    # bb_conn_lens, bb_conn_angles = chain_constraints_loss(
    #     denoised_bb[:, :3],
    #     bb[:, :3],
    #     data_lens,
    #     res_data.batch,
    #     ~noising_mask)

    # edge_dist_loss = full_dist_mat_loss(pred_frame_X_ca, ref_frame_X_ca, res_data.batch, ~res_mask)

    return {
        "rot_score_loss": rot_score_loss,
        "trans_score_loss": trans_score_loss,
        "pred_x_ca_mse": pred_X_ca_mse,
        "pred_bb_mse": backbone_mse,
        "ref_x_ca_mse": ref_X_ca_mse,
        "dist_mat_loss": dist_mat_loss,
        # "edge_dist_loss": edge_dist_loss,
        # "bb_dihedrals_loss": bb_dihedrals_loss,
        # "bb_conn_mse": bb_conn_lens,
        # "bb_angles_loss": bb_conn_angles
    }


def bb_frame_fm_loss(batch,
                     denoiser_outputs,
                     t_norm_clip=0.9,
                     sep_rot_loss=False,
                     local_atomic_dist_r=6,
                     square_aux_loss_time_factor=False):
    res_data = batch['residue']
    res_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    mask = res_mask & noising_mask
    res_batch = res_data.batch
    device = res_mask.device
    total_mask = res_mask & noising_mask

    if isinstance(res_data['rigids_t'], torch.Tensor):
        noised_bb_frames = ru.Rigid.from_tensor_7(res_data['rigids_t'].to(device))
    else:
        noised_bb_frames = res_data['rigids_t']
        dtype = noised_bb_frames._trans.dtype
        noised_bb_frames = ru.Rigid(
            rots=noised_bb_frames._rots.to(device=device, dtype=dtype),
            trans=noised_bb_frames._trans.to(device)
        )

    denoised_bb_frames = denoiser_outputs['final_rigids']
    bb_frames = ru.Rigid.from_tensor_7(res_data['rigids_1'].to(device))

    X_ca = res_data['res_ca']
    pred_frame_X_ca = denoised_bb_frames.get_trans()
    ref_frame_X_ca = noised_bb_frames.get_trans()
    pred_X_ca_se = torch.square(X_ca - pred_frame_X_ca).sum(dim=-1)
    pred_X_ca_mse = _nodewise_to_graphwise(pred_X_ca_se, res_batch, total_mask)
    ref_X_ca_se = torch.square(X_ca - ref_frame_X_ca).sum(dim=-1)
    ref_X_ca_mse = _nodewise_to_graphwise(ref_X_ca_se, res_batch, total_mask)

    bb = res_data['atom37'][:, (0, 1, 2, 4, 3)]
    bb_mask = res_data['atom37_mask'][:, (0, 1, 2, 4, 3)].bool()
    denoised_bb = denoiser_outputs['denoised_bb']
    backbone_mse = torch.square(denoised_bb - bb).sum(dim=-1)
    backbone_mse = _nodewise_to_graphwise(backbone_mse, res_batch, bb_mask & mask[..., None])

    t = batch['t']
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )
    t = batchwise_to_nodewise(t, res_data.batch)
    nodewise_norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )
    rots_t = noised_bb_frames.get_rots().get_rot_mats()
    rots_1_pred = denoised_bb_frames.get_rots().get_rot_mats()
    rots_1 = bb_frames.get_rots().get_rot_mats()
    pred_rot_vf = so3_fm_utils.calc_rot_vf(rots_t, rots_1_pred)
    gt_rot_vf = so3_fm_utils.calc_rot_vf(rots_t, rots_1)
    if sep_rot_loss:
        rot_vf_loss = angle_axis_rot_vf_loss(pred_rot_vf, gt_rot_vf, res_data.batch, mask, norm_scale)
    else:
        rot_vf_loss = torch.square(pred_rot_vf - gt_rot_vf).sum(dim=-1)
        rot_vf_loss = _nodewise_to_graphwise(rot_vf_loss, res_data.batch, mask)
        rot_vf_loss = rot_vf_loss / (norm_scale ** 2)

    trans_1_pred = denoised_bb_frames.get_trans()
    trans_1 = bb_frames.get_trans()
    trans_vf_loss = torch.square(trans_1_pred - trans_1).sum(dim=-1) / (nodewise_norm_scale ** 2)
    trans_vf_loss = _nodewise_to_graphwise(trans_vf_loss, res_data.batch, mask)
    trans_vf_loss *= 0.01  # Angstroms to nm

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

    # testing this
    # scaled_dist_mat_loss = dist_mat_loss * 0.01
    # scaled_backbone_mse = backbone_mse * 0.01

    # this does not work well
    # scaled_dist_mat_loss = dist_mat_loss / (norm_scale ** 2) * 0.01
    # scaled_backbone_mse = backbone_mse / (norm_scale ** 2) * 0.01

    # bb_dihedrals_loss = backbone_dihedrals_loss(
    #     denoised_bb[:, :3],
    #     bb[:, :3],
    #     data_lens,
    #     ~noising_mask)

    # bb_conn_lens, bb_conn_angles = chain_constraints_loss(
    #     denoised_bb[:, :3],
    #     bb[:, :3],
    #     data_lens,
    #     res_data.batch,
    #     ~noising_mask)

    # edge_dist_loss = full_dist_mat_loss(pred_frame_X_ca, ref_frame_X_ca, res_data.batch, ~res_mask)

    return {
        "rot_vf_loss": rot_vf_loss,
        "trans_vf_loss": trans_vf_loss,
        "pred_x_ca_mse": pred_X_ca_mse,
        "pred_bb_mse": backbone_mse,
        "scaled_pred_bb_mse": scaled_backbone_mse,
        "ref_x_ca_mse": ref_X_ca_mse,
        "dist_mat_loss": dist_mat_loss,
        "scaled_dist_mat_loss": scaled_dist_mat_loss,
        # "edge_dist_loss": edge_dist_loss,
        # "bb_dihedrals_loss": bb_dihedrals_loss,
        # "bb_conn_mse": bb_conn_lens,
        # "bb_angles_loss": bb_conn_angles
    }


def bb_plddt_loss(batch,
                  denoiser_outputs,
                  local_atomic_dist_r=15,
                  cutoffs=[0.5, 1, 2, 4],
                  max_num_neighbors=1000,
                  eps=1e-8):
    res_data = batch['residue']
    res_mask = res_data['res_mask']
    pred_bb = denoiser_outputs['denoised_bb']
    ref_bb = res_data['atom37'][:, (0, 1, 2, 4, 3)]

    flat_ref_bb = ref_bb[res_mask].reshape(-1, 3)
    flat_pred_bb = pred_bb[res_mask].reshape(-1, 3)
    batch_expand = res_data.batch[res_mask].repeat_interleave(5, dim=0)
    node_idx_expand = torch.arange(res_data.num_nodes, device=batch_expand.device)[res_mask].repeat_interleave(5, dim=0)
    edge_index = radius_graph(flat_ref_bb, local_atomic_dist_r, batch_expand, max_num_neighbors=max_num_neighbors)

    pred_dist_vec = flat_pred_bb[edge_index[0]] - flat_pred_bb[edge_index[1]]
    pred_dists = torch.linalg.vector_norm(pred_dist_vec + eps, dim=-1)

    ref_dist_vec = flat_ref_bb[edge_index[0]] - flat_ref_bb[edge_index[1]]
    ref_dists = torch.linalg.vector_norm(ref_dist_vec + eps, dim=-1)

    dist_error = torch.abs(pred_dists - ref_dists)
    dist_cutoffs = torch.as_tensor(cutoffs, device=dist_error.device)
    preserved = dist_error[..., None] < dist_cutoffs[None]
    edge_to_res_idx = node_idx_expand[edge_index[1]]
    lddt = pygu.scatter(
        preserved.float(),
        edge_to_res_idx,
        dim=0,
        dim_size=ref_bb.shape[0],
        reduce='mean'
    )
    lddt = lddt.mean(dim=-1)
    assert not lddt.requires_grad

    plddt_bin_logits = denoiser_outputs['plddt_logits']
    num_bins = plddt_bin_logits.shape[-1]
    plddt_bin_values = (torch.arange(num_bins, device=plddt_bin_logits.device) * 2 + 1) / 100
    plddt = torch.sum(torch.softmax(plddt_bin_logits, dim=-1) * plddt_bin_values, dim=-1)
    lddt_bins = torch.arange(num_bins - 1, device=plddt_bin_logits.device) / num_bins
    lddt_discritized = torch.bucketize(lddt, lddt_bins)
    plddt_loss = F.cross_entropy(plddt_bin_logits, lddt_discritized, reduction='none')

    graphwise_plddt_loss = _nodewise_to_graphwise(plddt_loss, res_data.batch, res_mask)
    graphwise_plddt = _nodewise_to_graphwise(plddt, res_data.batch, res_mask)
    graphwise_lddt = _nodewise_to_graphwise(lddt, res_data.batch, res_mask)

    return {
        "plddt_loss": graphwise_plddt_loss,
        "plddt": graphwise_plddt * 100,
        "lddt": graphwise_lddt * 100
    }

def _radius_of_gyration(rigids: ru.Rigid, batch, mask):
    trans = rigids.get_trans()
    nodewise_center = ru.batchwise_center(rigids, batch, mask)

    dev = torch.sum((trans - nodewise_center) ** 2, dim=-1)
    dev = dev[mask]
    rg = pygu.scatter(
        dev,
        index=batch[mask],
        dim=0,
        dim_size=int(batch.max().item()+1),
        reduce='mean'
    )
    return rg


def rg_loss(batch,
            denoiser_outputs,
            t_max_clip=0.5):
    res_data = batch['residue']
    res_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    mask = res_mask & noising_mask
    device = res_mask.device

    denoised_bb_frames = denoiser_outputs['final_rigids'].scale_translation(0.1)
    bb_frames = ru.Rigid.from_tensor_7(res_data['rigids_1'].to(device)).scale_translation(0.1)

    gt_rg = _radius_of_gyration(bb_frames, res_data.batch, mask)
    pred_rg = _radius_of_gyration(denoised_bb_frames, res_data.batch, mask)

    rg_diff = torch.square(gt_rg - pred_rg)
    t = batch['t'].clip(max=t_max_clip)

    loss = rg_diff / ((1 - t)**2)
    return {
        "rg_mse": loss
    }


def frame_traj_loss(batch,
                    denoiser_outputs):
    res_data = batch['residue']
    res_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    mask = res_mask & noising_mask
    device = res_mask.device

    bb_frames = ru.Rigid.from_tensor_7(res_data['rigids_1'].to(device))
    bb_traj = denoiser_outputs['intermediate_rigids']

    gt_X_ca = bb_frames.get_trans()
    traj_loss = 0
    # skip the last one since we're already counting it
    for rigid in bb_traj[:-1]:
        traj_X_ca = rigid.get_trans()
        traj_loss += torch.square(traj_X_ca - gt_X_ca).sum(dim=-1)

    traj_loss = _nodewise_to_graphwise(traj_loss, res_data.batch, mask)
    traj_loss *= 0.01  # Angstroms to nm
    return {
        "rigid_traj_loss": traj_loss
    }


def bb_frame_clash_loss(batch,
                        denoiser_outputs,
                        t_norm_clip=0.9,
                        loss_clip=50):
    res_data = batch['residue']
    res_mask = res_data['res_mask']
    noising_mask = res_data['noising_mask']
    mask = res_mask & noising_mask

    denoised_bb = denoiser_outputs['denoised_bb']
    clash_loss = bb_clash_loss(
        denoised_bb,
        mask,
        res_data.batch
    )
    t = batch['t']
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )
    scaled_clash_loss = clash_loss.clip(max=loss_clip) / norm_scale

    return {
        "bb_clash_loss": clash_loss,
        "scaled_bb_clash_loss": scaled_clash_loss
    }


def sparse_fape_loss(
    pred_atom14,
    gt_atom14,
    alt_atom14,
    rigids,
    batch,
    atom14_mask,
    r=10,
    eps=1e-6,
    max_num_neighbors=100,
):
    res_mask = atom14_mask[:, 1]
    # re: ambiguous atom14 naming
    # im gonna use a heuristic for the ref atom14
    # where we'll just take the ref residue atom ordering
    # as the one which is lowest in rmsd to the predicted structure
    # in a lot of cases this probably won't hold but it will at least work
    # in low rmsd regimes i think
    with torch.no_grad():
        pred_gt_diff = torch.square(pred_atom14 - gt_atom14).sum(dim=-1)
        pred_alt_diff = torch.square(pred_atom14 - alt_atom14).sum(dim=-1)
        pred_gt_mse = torch.sum(pred_gt_diff * atom14_mask, dim=-1)
        pred_alt_mse = torch.sum(pred_alt_diff * atom14_mask, dim=-1)

        gt_over_alt = pred_gt_mse < pred_alt_mse
        ref_atom14 = gt_atom14 * gt_over_alt[..., None, None] + alt_atom14 * ~gt_over_alt[..., None, None]

    flat_ref_atom14 = ref_atom14[atom14_mask]
    flat_pred_atom14 = pred_atom14[atom14_mask]
    batch_expand = batch[..., None].expand(-1, atom14_mask.shape[-1])[atom14_mask]
    edge_index = radius_graph(flat_ref_atom14, r, batch_expand, max_num_neighbors=max_num_neighbors)

    atom_to_rigid = torch.arange(pred_atom14.shape[0], device=pred_atom14.device)
    atom_to_rigid = atom_to_rigid[..., None].expand(-1, 14)
    flat_atom_to_rigid = atom_to_rigid[atom14_mask]
    flat_rigids = rigids[flat_atom_to_rigid]

    pred_dist_vec = flat_pred_atom14[edge_index[0]] - flat_pred_atom14[edge_index[1]]
    pred_dists = torch.linalg.vector_norm(pred_dist_vec + eps, dim=-1)

    ref_dist_vec = flat_ref_atom14[edge_index[0]] - flat_ref_atom14[edge_index[1]]
    ref_dists = torch.linalg.vector_norm(ref_dist_vec + eps, dim=-1)

    dist_se = torch.square(pred_dists - ref_dists)

    edge_batch = batch_expand[edge_index[1]]
    num_graph = batch_expand.max().item() + 1
    graphwise_dist_se = scatter(
        dist_se,
        edge_batch,
        dim=0,
        dim_size=num_graph
    )
    graphwise_num_edges = scatter(
        torch.ones_like(edge_batch),
        edge_batch,
        dim=0,
        dim_size=num_graph
    )
    graphwise_num_res = scatter(
        res_mask.long(),
        batch,
        dim=0,
        dim_size=num_graph
    )
    return graphwise_dist_se / (graphwise_num_edges - graphwise_num_res)
