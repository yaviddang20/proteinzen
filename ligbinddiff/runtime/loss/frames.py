import torch
import numpy as np

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.atom_reps import atom14_sidechain_angles, atom14_residue_angles

from .utils import _elemwise_to_graphwise, _nodewise_to_graphwise
from .atomic import _atom91_to_atom14
from .openfold import compute_fape

def all_atom_fape_loss(pred_atom91,
                  ref_atom91,
                  seq,
                  data_lens,
                  x_mask,
                  no_bb=True,
                  eps=1e-6):
    angle_atom_idx_select = []
    angle_store = atom14_sidechain_angles if no_bb else atom14_residue_angles
    for atom_list in angle_store.values():
        angle_atom_idx_select += atom_list
    angle_atom_idx_select = torch.as_tensor(angle_atom_idx_select, device=ref_atom91.device).long() # n_frame x 3

    pred_atom14, _ = _atom91_to_atom14(pred_atom91, seq)
    ref_atom14, atom14_mask = _atom91_to_atom14(ref_atom91, seq)
    atom14_mask = atom14_mask.any(dim=-1)

    frame_mask = atom14_mask[:, angle_atom_idx_select].any(dim=-1)  # n_res x n_frame
    ref_frame_atoms = ref_atom14[:, angle_atom_idx_select]  # n_res x n_frame x 3 x 3
    pred_frame_atoms = pred_atom14[:, angle_atom_idx_select]  # n_res x n_frame x 3 x 3

    all_atom_fape = []
    splits = [0] + np.cumsum(data_lens).tolist()
    for i in range(len(splits)-1):
        start = splits[i]
        end = splits[i+1]
        subset_ref_frame_atoms = ref_frame_atoms[start:end]
        subset_pred_frame_atoms = pred_frame_atoms[start:end]
        subset_ref_atom14 = ref_atom14[start:end]
        subset_pred_atom14 = pred_atom14[start:end]
        subset_x_mask = x_mask[start:end]
        subset_atom14_mask = atom14_mask[start:end]
        subset_frame_mask = frame_mask[start:end]

        flat_ref_frame_atoms = subset_ref_frame_atoms[~subset_x_mask].view(-1, 3, 3)
        flat_pred_frame_atoms = subset_pred_frame_atoms[~subset_x_mask].view(-1, 3, 3)
        flat_frame_mask = subset_frame_mask[~subset_x_mask].view(-1)
        dummy_frame_mask = torch.zeros_like(flat_frame_mask).bool()[~flat_frame_mask]

        flat_atom14_mask = subset_atom14_mask[~subset_x_mask].view(-1)
        flat_pred_atom14 = subset_pred_atom14[~subset_x_mask].view(-1, 3)
        flat_pred_atom14 = flat_pred_atom14[~flat_atom14_mask]
        flat_ref_atom14 = subset_ref_atom14[~subset_x_mask].view(-1, 3)
        flat_ref_atom14 = flat_ref_atom14[~flat_atom14_mask]
        dummy_atom14_mask = torch.zeros_like(flat_atom14_mask).bool()[~flat_atom14_mask]

        flat_ref_frames = ru.Rigid.from_3_points(
            flat_ref_frame_atoms[..., 0, :],
            flat_ref_frame_atoms[..., 1, :],
            flat_ref_frame_atoms[..., 2, :]
        )[~flat_frame_mask]
        flat_pred_frames = ru.Rigid.from_3_points(
            flat_pred_frame_atoms[..., 0, :],
            flat_pred_frame_atoms[..., 1, :],
            flat_pred_frame_atoms[..., 2, :]
        )[~flat_frame_mask]


        subset_all_atom_fape = compute_fape(
            flat_pred_frames,
            flat_ref_frames,
            ~dummy_frame_mask,
            flat_pred_atom14,
            flat_ref_atom14,
            ~dummy_atom14_mask,
            l1_clamp_distance=10
        )
        all_atom_fape.append(subset_all_atom_fape.unsqueeze(0))

    all_atom_fape = torch.cat(all_atom_fape, dim=0)
    return all_atom_fape
    return _elemwise_to_graphwise(all_atom_fape, data_lens, frame_mask)

def angle_axis_rot_loss(
        pred_rot_score,
        ref_rot_score,
        score_scaling,
        t,
        data_lens,
        x_mask,
        angle_loss_weight=0.5,
        angle_loss_t_threshold=0.2):
    gt_rot_score = ref_rot_score[~x_mask]
    pred_rot_score = pred_rot_score[~x_mask]
    score_scaling = score_scaling[~x_mask]

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
    angle_loss *= t[~x_mask] > angle_loss_t_threshold
    rot_loss = angle_loss + axis_loss
    return _nodewise_to_graphwise(rot_loss, data_lens, x_mask)
