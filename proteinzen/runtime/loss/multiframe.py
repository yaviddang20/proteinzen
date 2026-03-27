import torch
from itertools import permutations as _iperms

from proteinzen.openfold.utils import rigid_utils as ru
from proteinzen.stoch_interp import so3_utils as so3_fm_utils
import torch.nn.functional as F


def sym_permute_gt_rigids(pred_trans, gt_rigids, inputs):
    """Permute GT atom rigids within each symmetry equivalence class to minimise
    the squared distance to the predicted positions.

    For each batch element and each symmetry group (e.g. two identical methyl
    groups), we enumerate all permutations of the group's GT atom positions and
    select the assignment whose total squared distance to the predicted positions
    is smallest.  Both translation and rotation are swapped together so the full
    rigid is consistently permuted.

    Groups with size > 6 are skipped (>720 permutations; rare and expensive).

    Parameters
    ----------
    pred_trans : Tensor (B, N, 3)
    gt_rigids  : Rigid  (B, N)
    inputs     : batch dict — must contain 'sym_groups', 'sym_group_sizes',
                 'num_sym_groups'

    Returns
    -------
    Rigid (B, N) with permuted translations/rotations for symmetric atoms.
    """
    if 'sym_groups' not in inputs or int(inputs['num_sym_groups'].max()) == 0:
        return gt_rigids

    sym_groups      = inputs['sym_groups']       # (B, max_sym, max_gsz)
    sym_group_sizes = inputs['sym_group_sizes']  # (B, max_sym)
    num_sym_groups  = inputs['num_sym_groups']   # (B,)

    B = pred_trans.shape[0]

    pred_np          = pred_trans.detach().cpu().numpy()
    gt_trans_orig    = gt_rigids.get_trans()                         # (B, N, 3)
    gt_rotmat_orig   = gt_rigids.get_rots().get_rot_mats()           # (B, N, 3, 3)
    gt_trans_np      = gt_trans_orig.detach().cpu().numpy()
    sym_groups_cpu   = sym_groups.cpu().numpy()
    sym_gsizes_cpu   = sym_group_sizes.cpu().numpy()
    num_sym_cpu      = num_sym_groups.cpu().numpy()

    gt_trans_new   = gt_trans_orig.clone()
    gt_rotmat_new  = gt_rotmat_orig.clone()

    for b in range(B):
        n_groups = int(num_sym_cpu[b])
        for s in range(n_groups):
            gsz = int(sym_gsizes_cpu[b, s])
            if gsz < 2 or gsz > 6:
                continue
            atom_idxs = sym_groups_cpu[b, s, :gsz].tolist()

            pred_pos = pred_np[b, atom_idxs]   # (gsz, 3)
            gt_pos   = gt_trans_np[b, atom_idxs]  # (gsz, 3)

            best_perm = list(range(gsz))
            best_dist = float('inf')
            for perm in _iperms(range(gsz)):
                perm = list(perm)
                dist = float(((pred_pos - gt_pos[perm]) ** 2).sum())
                if dist < best_dist:
                    best_dist = dist
                    best_perm = perm

            perm_idxs = [atom_idxs[i] for i in best_perm]
            gt_trans_new[b, atom_idxs]  = gt_trans_orig[b, perm_idxs]
            gt_rotmat_new[b, atom_idxs] = gt_rotmat_orig[b, perm_idxs]

    new_rots = ru.Rotation(rot_mats=gt_rotmat_new)
    return ru.Rigid(new_rots, gt_trans_new)


def compute_gt_delta_torsion(rigids_pred, rigids_gt, bond_idx):
      """
      Compute GT delta torsion at each bond end:
      the rotation needed to go from pred frame to GT frame,
      projected onto the bond axis.
   
      rigids_pred: Rigid [B, N] — the x_1 prediction
      rigids_gt:   Rigid [B, N] — the clean ground truth
      bond_idx:    [num_bonds, 3] from BondDeltaTorsionHead
      """
      b, i, j = bond_idx[:, 0], bond_idx[:, 1], bond_idx[:, 2]

      trans_pred = rigids_pred.get_trans()
      bond_axis = F.normalize(trans_pred[b, j] - trans_pred[b, i], dim=-1)  # [num_bonds, 3]

      rots_pred = rigids_pred.get_rots().get_rot_mats()  # [B, N, 3, 3]
      rots_gt   = rigids_gt.get_rots().get_rot_mats()

      ref = torch.tensor([1., 0., 0.], device=bond_axis.device)

      def perp_projection(rots, b_idx, n_idx, axis):
          # rotate reference vector by atom's frame, project perp to bond axis
          local_x = (rots[b_idx, n_idx] @ ref.unsqueeze(-1)).squeeze(-1)  # [num_bonds, 3]
          perp = local_x - (local_x * axis).sum(-1, keepdim=True) * axis
          return F.normalize(perp + 1e-8, dim=-1)

      perp_pred_i = perp_projection(rots_pred, b, i, bond_axis)
      perp_gt_i   = perp_projection(rots_gt,   b, i, bond_axis)
      perp_pred_j = perp_projection(rots_pred, b, j, bond_axis)
      perp_gt_j   = perp_projection(rots_gt,   b, j, bond_axis)

      # sin/cos of Δθ at each end via cross and dot product
      sin_i = (torch.cross(perp_pred_i, perp_gt_i, dim=-1) * bond_axis).sum(-1)
      cos_i = (perp_pred_i * perp_gt_i).sum(-1)
      sin_j = (torch.cross(perp_pred_j, perp_gt_j, dim=-1) * bond_axis).sum(-1)
      cos_j = (perp_pred_j * perp_gt_j).sum(-1)

      gt_i = torch.stack([sin_i, cos_i], dim=-1)  # [num_bonds, 2]
      gt_j = torch.stack([sin_j, cos_j], dim=-1)

      return gt_i, gt_j
      

def delta_torsion_loss(pred_i, pred_j, gt_i, gt_j):
      """Simple MSE on sin/cos — equivalent to 1 - cos(Δ_pred - Δ_gt)."""
      return F.mse_loss(pred_i, gt_i) + F.mse_loss(pred_j, gt_j)


def ring_planarity_loss(inputs, denoiser_outputs, eps=1e-8):
    """Penalize deviation of predicted aromatic ring atoms from their best-fit plane.

    Loss is the smallest eigenvalue of the centered covariance of ring atom positions,
    normalized by ring size. Zero for a perfectly planar ring.
    """
    if 'ring_masks' not in inputs:
        return torch.tensor(0.0, device=inputs['rigids']['rigids_1'].device)

    ring_masks = inputs['ring_masks'].float()   # (B, R, N)
    num_rings = inputs['num_rings']             # (B,)
    B, R, N = ring_masks.shape

    if R == 0 or num_rings.max() == 0:
        return torch.tensor(0.0, device=ring_masks.device)

    trans = denoiser_outputs['denoised_rigids'].get_trans()  # (B, N, 3)

    n_atoms = ring_masks.sum(-1, keepdim=True).clamp(min=1)  # (B, R, 1)
    centroid = (trans[:, None, :, :] * ring_masks[..., None]).sum(-2) / n_atoms  # (B, R, 3)
    p = trans[:, None, :, :] - centroid[:, :, None, :]       # (B, R, N, 3)
    p_masked = p * ring_masks[..., None]                      # (B, R, N, 3)

    # Covariance matrix of ring atom positions
    cov = torch.einsum('brni,brnj->brij', p_masked, p_masked)  # (B, R, 3, 3)

    # Smallest eigenvalue = variance perpendicular to best-fit plane
    eigvals = torch.linalg.eigvalsh(cov + eps * torch.eye(3, device=cov.device))  # (B, R, 3)
    planarity_dev = eigvals[..., 0] / n_atoms.squeeze(-1)  # (B, R)

    valid = (torch.arange(R, device=ring_masks.device)[None, :] < num_rings[:, None]).float()
    loss = (planarity_dev * valid).sum(-1) / valid.sum(-1).clamp(min=1)  # (B,)
    return loss.mean()


def bond_rotation_pos_loss(inputs, denoiser_outputs, gt_trans=None):
    """MSE between bond-rotation-updated positions and GT positions for atom rigids.

    Parameters
    ----------
    gt_trans : Tensor (B, N, 3), optional
        Pre-computed (possibly sym-permuted) GT translations.  Falls back to
        parsing rigids_1 from inputs if not provided.
    """
    if 'bond_updated_trans' not in denoiser_outputs:
        return torch.tensor(0.0, device=inputs['rigids']['rigids_1'].device)

    updated_trans = denoiser_outputs['bond_updated_trans']  # (B, N, 3)
    if gt_trans is None:
        gt_trans = ru.Rigid.from_tensor_7(inputs['rigids']['rigids_1']).get_trans()

    atom_mask = inputs['rigids']['rigids_is_atom_mask'].bool()  # (B, N)
    valid_mask = inputs['rigids']['rigids_mask'].bool() & atom_mask

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=updated_trans.device)

    diff_sq = ((updated_trans - gt_trans) ** 2).sum(-1)  # (B, N)
    loss = (diff_sq * valid_mask).sum(-1) / valid_mask.sum(-1).clamp(min=1)  # (B,)
    return loss.mean()


def bond_length_rmse(inputs, denoiser_outputs):
    rigids_data = inputs['rigids']
    rigids_mask = rigids_data['rigids_mask']
    rigids_noising_mask = rigids_data['rigids_noising_mask']
    total_mask = rigids_mask * rigids_noising_mask  # [B, L_pad]

    denoised_rigids = denoiser_outputs['denoised_rigids']
    gt_rigids = ru.Rigid.from_tensor_7(rigids_data['rigids_1'])

    gt_frame_trans = gt_rigids.get_trans()    # [B, L_pad, 3]
    pred_frame_trans = denoised_rigids.get_trans()  # [B, L_pad, 3]
    
    B, L_pad, _ = gt_frame_trans.shape

    # Compute all pairwise distances
    gt_dists = torch.linalg.norm(
        gt_frame_trans[:, :, None, :] - gt_frame_trans[:, None, :, :],
        dim=-1
    )  # [B, L_pad, L_pad]
    pred_dists = torch.linalg.norm(
        pred_frame_trans[:, :, None, :] - pred_frame_trans[:, None, :, :],
        dim=-1
    )  # [B, L_pad, L_pad]

    # Build padded bond matrix
    token_bonds = inputs['token']['token_bonds']  # [L, L] or [B, L, L]
    
    if token_bonds.dim() == 2:
        token_bonds = token_bonds.unsqueeze(0).expand(B, -1, -1)
    
    L = token_bonds.shape[-1]
    pad = L_pad - L

    if pad > 0:
        token_bonds = torch.nn.functional.pad(
            token_bonds,
            (0, pad, 0, pad),
            value=0
        )

    token_bonds_mask = token_bonds > 0  # [B, L_pad, L_pad]
    
    # Only keep upper triangle to avoid double-counting
    triu_mask = torch.triu(torch.ones(L_pad, L_pad, device=token_bonds.device, dtype=torch.bool), diagonal=1)
    token_bonds_mask = token_bonds_mask & triu_mask[None, :, :]  # [B, L_pad, L_pad]
    
    # Apply position mask: both endpoints must be valid
    pos_mask = total_mask[:, :, None] & total_mask[:, None, :]  # [B, L_pad, L_pad]
    final_mask = token_bonds_mask & pos_mask
    
    # Extract valid bonds across all batches
    gt_bonds = gt_dists[final_mask]  # [N_total_bonds]
    pred_bonds = pred_dists[final_mask]  # [N_total_bonds]
    
    if gt_bonds.numel() == 0:
        return torch.tensor(0.0, device=gt_dists.device)
    
    # RMSE across all bonds in all batches
    rmse = torch.sqrt(torch.mean(torch.square(gt_bonds - pred_bonds)))
    
    return rmse  # scalar


def bond_angle_rmse(inputs, denoiser_outputs, eps=1e-8):
    rigids_data = inputs["rigids"]
    rigids_mask = rigids_data["rigids_mask"].bool()
    rigids_noising_mask = rigids_data["rigids_noising_mask"].bool()
    total_mask = rigids_mask & rigids_noising_mask
    
    denoised_rigids = denoiser_outputs["denoised_rigids"]
    gt_rigids = ru.Rigid.from_tensor_7(rigids_data["rigids_1"])
    
    gt_xyz = gt_rigids.get_trans()
    pred_xyz = denoised_rigids.get_trans()
    B, L_pad, _ = gt_xyz.shape
    device = gt_xyz.device
    
    # Pad token_bonds
    token_bonds = inputs["token"]["token_bonds"]
    if token_bonds.dim() == 2:
        token_bonds = token_bonds.unsqueeze(0).expand(B, -1, -1)
    
    L = token_bonds.shape[-1]
    pad = L_pad - L
    if pad > 0:
        token_bonds = F.pad(token_bonds, (0, pad, 0, pad), value=0)
    
    bond_mask = (token_bonds > 0)
    
    # Remove diagonal and apply masks
    diag = torch.eye(L_pad, device=device, dtype=torch.bool)[None, :, :]
    bond_mask = bond_mask & ~diag
    bond_mask = bond_mask & total_mask[:, :, None] & total_mask[:, None, :]
    
    # Edge list
    b_e, i_e, j_e = torch.where(bond_mask)
    if b_e.numel() == 0:
        return gt_xyz.new_zeros(())
    
    # Group by center (b, j)
    key = b_e * L_pad + j_e
    perm = torch.argsort(key)
    key = key[perm]
    b_e = b_e[perm]
    i_e = i_e[perm]
    j_e = j_e[perm]
    
    uniq_key, counts = torch.unique_consecutive(key, return_counts=True)
    
    # Build triplets
    trip_b = []
    trip_i = []
    trip_j = []
    trip_k = []
    
    start = 0
    counts_list = counts.tolist()  # Single CPU transfer
    for c in counts_list:
        end = start + c
        if c >= 2:
            neigh = i_e[start:end]
            b0 = b_e[start]
            j0 = j_e[start]
            
            a, b = torch.triu_indices(c, c, offset=1, device=device)
            trip_b.append(b0.expand(a.shape[0]))
            trip_j.append(j0.expand(a.shape[0]))
            trip_i.append(neigh[a])
            trip_k.append(neigh[b])
        start = end
    
    if len(trip_b) == 0:
        return gt_xyz.new_zeros(())
    
    b_idx = torch.cat(trip_b)
    i_idx = torch.cat(trip_i)
    j_idx = torch.cat(trip_j)
    k_idx = torch.cat(trip_k)
    
    # Gather positions
    gt_i = gt_xyz[b_idx, i_idx]
    gt_j = gt_xyz[b_idx, j_idx]
    gt_k = gt_xyz[b_idx, k_idx]
    
    pred_i = pred_xyz[b_idx, i_idx]
    pred_j = pred_xyz[b_idx, j_idx]
    pred_k = pred_xyz[b_idx, k_idx]
    
    # Vectors
    gt_v1 = gt_i - gt_j
    gt_v2 = gt_k - gt_j
    pred_v1 = pred_i - pred_j
    pred_v2 = pred_k - pred_j
    
    # Normalize
    gt_v1 = gt_v1 / (torch.linalg.norm(gt_v1, dim=-1, keepdim=True) + eps)
    gt_v2 = gt_v2 / (torch.linalg.norm(gt_v2, dim=-1, keepdim=True) + eps)
    pred_v1 = pred_v1 / (torch.linalg.norm(pred_v1, dim=-1, keepdim=True) + eps)
    pred_v2 = pred_v2 / (torch.linalg.norm(pred_v2, dim=-1, keepdim=True) + eps)
    
    # Angles
    gt_cos = (gt_v1 * gt_v2).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
    pred_cos = (pred_v1 * pred_v2).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
    
    gt_angle = torch.acos(gt_cos)
    pred_angle = torch.acos(pred_cos)
    
    rmse = torch.sqrt(torch.mean((gt_angle - pred_angle) ** 2))
    
    return rmse


def angle_axis_rot_vf_loss_dense(
        pred_rot_vf,
        ref_rot_vf,
        rigids_mask,
        rigidwise_norm_scale,
        angle_loss_weight=0.5,
        weight=None,
        eps=1e-8):
    pred_rot_vf = pred_rot_vf / rigidwise_norm_scale[..., None]
    ref_rot_vf = ref_rot_vf / rigidwise_norm_scale[..., None]

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

    if weight is not None:
        rot_loss = rot_loss * weight

    num_rigids_per_batch = rigids_mask.long().sum(dim=-1).clip(min=1)
    rot_loss = (rot_loss * rigids_mask).sum(dim=-1) / num_rigids_per_batch

    return rot_loss


def multiframe_fm_loss_dense_batch(
    inputs,
    denoiser_outputs,
    t_norm_clip=0.9,
    sep_rot_loss=True,
    use_euclidean_for_rots=False,
    rigidwise_weight=1,
    rot_vf_angle_loss_weight=0.5,
    fafe_l2_block_mask_size=1,
    trans_rigidwise_weight=1,
    rot_rigidwise_weight=1,
    direct_rot_vf_loss=False,
    direct_rot_vf_loss_scale=1,
    upweight_atomic=False,
    scale_bond_length_loss=False,
    scale_bond_angle_loss=False,
    scale_ring_planarity_loss=False,
):
    rigids_data = inputs['rigids']
    rigids_mask = rigids_data['rigids_mask']
    rigids_noising_mask = rigids_data['rigids_noising_mask']
    noised_rigids = ru.Rigid.from_tensor_7(rigids_data['rigids_t'])

    total_mask = rigids_mask * rigids_noising_mask

    if upweight_atomic:
        rigid_is_atomized = inputs['rigids']['rigids_is_atom_mask'].float()
        atom_rigid_upweight = (1 - rigid_is_atomized) + 10 * (rigid_is_atomized)
        rigidwise_weight = rigidwise_weight * atom_rigid_upweight

    denoised_rigids = denoiser_outputs['denoised_rigids']
    gt_rigids = ru.Rigid.from_tensor_7(rigids_data['rigids_1'])

    # Permute GT atom rigids within symmetry equivalence classes so the loss
    # is not penalising arbitrary labelling of identical atoms.
    gt_rigids = sym_permute_gt_rigids(denoised_rigids.get_trans(), gt_rigids, inputs)

    num_rigids_per_batch = rigids_mask.long().sum(dim=-1).clip(min=1)

    gt_frame_trans = gt_rigids.get_trans()
    pred_frame_trans = denoised_rigids.get_trans()
    ref_frame_trans = noised_rigids.get_trans()
    pred_frame_trans_se = torch.square(gt_frame_trans - pred_frame_trans).sum(dim=-1)
    pred_frame_trans_mse = (pred_frame_trans_se * total_mask).sum(-1) / num_rigids_per_batch
    ref_frame_trans_se = torch.square(gt_frame_trans - ref_frame_trans).sum(dim=-1)
    ref_frame_trans_mse = (ref_frame_trans_se * total_mask).sum(-1) / num_rigids_per_batch

    
    
    t = inputs['t']
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )
    rots_t = noised_rigids.get_rots().get_rot_mats()
    rots_1_pred = denoised_rigids.get_rots().get_rot_mats()
    rots_1 = gt_rigids.get_rots().get_rot_mats()

    if use_euclidean_for_rots:
        def geodesic_dist(rots1, rots2):
            R_diff = torch.einsum("...ij,...jk->...ik", rots1.transpose(-2, -1), rots2)
            R_diff_trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            return torch.acos(
                torch.clamp(
                    (R_diff_trace - 1) / 2,
                    min=-1 + 1e-6,
                    max=1 - 1e-6
                )
            )
        # pred_rotvec_1 = so3_fm_utils.rotmat_to_rotvec(rots_1_pred)
        # gt_rotvec_1 = so3_fm_utils.rotmat_to_rotvec(rots_1)
        # rot_vf_loss = torch.square(pred_rotvec_1 - gt_rotvec_1).sum(dim=-1)
        rot_vf_loss = (geodesic_dist(rots_1, rots_1_pred) / torch.pi) ** 2
        unscaled_rot_vf_loss = rot_vf_loss
        rot_vf_loss = rot_vf_loss * rigidwise_weight
        rot_vf_loss = rot_vf_loss / (norm_scale[..., None] ** 2)
        rot_vf_loss = (rot_vf_loss * total_mask).sum(dim=-1) / num_rigids_per_batch
        with torch.no_grad():
            unscaled_rot_vf_loss = unscaled_rot_vf_loss / (norm_scale ** 2)[..., None]
            unscaled_rot_vf_loss = (unscaled_rot_vf_loss * total_mask).sum(dim=-1) / num_rigids_per_batch

    else:
        if direct_rot_vf_loss:
            pred_rot_vf = denoiser_outputs['pred_rot_vf']
            gt_rot_vf = rigids_data['gt_rot_vf']
            # print("pred", pred_rot_vf)
            # print("gt", gt_rot_vf)
            # rot_vf_angle_loss_weight = rot_vf_angle_loss_weight / (2.4 ** 2)  # this is roughly the mean of the vector field magnitudes, squared

            unscaled_rot_vf_loss = torch.square(pred_rot_vf - gt_rot_vf).sum(dim=-1)
            rot_vf_loss = unscaled_rot_vf_loss * rot_rigidwise_weight * rigidwise_weight
            rot_vf_loss = torch.sum(rot_vf_loss * total_mask, dim=-1) / total_mask.sum(dim=-1).clip(min=1)
            rot_vf_loss = rot_vf_loss * direct_rot_vf_loss_scale
            with torch.no_grad():
                unscaled_rot_vf_loss = torch.sum(unscaled_rot_vf_loss * total_mask, dim=-1) / total_mask.sum(dim=-1).clip(min=1)

            # rot_vf_loss = angle_axis_rot_vf_loss_dense(
            #     pred_rot_vf,
            #     gt_rot_vf,
            #     total_mask,
            #     torch.ones_like(norm_scale) / rot_rigidwise_weight,
            #     weight=rigidwise_weight ,
            #     angle_loss_weight=rot_vf_angle_loss_weight,
            # )
            # with torch.no_grad():
            #     unscaled_rot_vf_loss = angle_axis_rot_vf_loss_dense(
            #         pred_rot_vf,
            #         gt_rot_vf,
            #         total_mask,
            #         torch.ones_like(norm_scale),
            #     )
        else:
            pred_rot_vf = so3_fm_utils.calc_rot_vf(rots_t, rots_1_pred)
            gt_rot_vf = so3_fm_utils.calc_rot_vf(rots_t, rots_1)

            if sep_rot_loss:
                rot_vf_loss = angle_axis_rot_vf_loss_dense(
                    pred_rot_vf,
                    gt_rot_vf,
                    total_mask,
                    norm_scale / rot_rigidwise_weight,
                    weight=rigidwise_weight ,
                    angle_loss_weight=rot_vf_angle_loss_weight,
                )
                with torch.no_grad():
                    unscaled_rot_vf_loss = angle_axis_rot_vf_loss_dense(
                        pred_rot_vf,
                        gt_rot_vf,
                        total_mask,
                        norm_scale,
                    )
            else:
                rot_vf_loss = torch.square(pred_rot_vf - gt_rot_vf).sum(dim=-1)
                unscaled_rot_vf_loss = rot_vf_loss
                rot_vf_loss = rot_vf_loss * rigidwise_weight
                rot_vf_loss = rot_vf_loss / (norm_scale[..., None] ** 2)
                rot_vf_loss = (rot_vf_loss * total_mask).sum(dim=-1) / num_rigids_per_batch
                with torch.no_grad():
                    unscaled_rot_vf_loss = unscaled_rot_vf_loss / (norm_scale ** 2)[..., None]
                    unscaled_rot_vf_loss = (unscaled_rot_vf_loss * total_mask).sum(dim=-1) / num_rigids_per_batch

    trans_1_pred = denoised_rigids.get_trans()
    trans_1 = gt_rigids.get_trans()
    trans_vf_loss = torch.square(trans_1_pred - trans_1).sum(dim=-1) / (norm_scale ** 2)
    unscaled_trans_vf_loss = trans_vf_loss
    trans_vf_loss = trans_vf_loss * rigidwise_weight * trans_rigidwise_weight
    trans_vf_loss = (trans_vf_loss * total_mask).sum(dim=-1) / num_rigids_per_batch
    trans_vf_loss *= 0.01  # Angstroms to nm
    unscaled_trans_vf_loss = (unscaled_trans_vf_loss * total_mask).sum(dim=-1) / num_rigids_per_batch
    unscaled_trans_vf_loss *= 0.01  # Angstroms to nm


    raw_bond_length = bond_length_rmse(inputs, denoiser_outputs)
    raw_bond_angle = bond_angle_rmse(inputs, denoiser_outputs)
    raw_ring_plan = ring_planarity_loss(inputs, denoiser_outputs)
    raw_bond_rot = bond_rotation_pos_loss(inputs, denoiser_outputs, gt_trans=gt_rigids.get_trans())

    bond_length_loss = raw_bond_length / (norm_scale ** 2) if scale_bond_length_loss else raw_bond_length
    bond_angle_loss = raw_bond_angle / (norm_scale ** 2) if scale_bond_angle_loss else raw_bond_angle
    ring_plan_loss = raw_ring_plan / (norm_scale ** 2) if scale_ring_planarity_loss else raw_ring_plan
    bond_rot_loss = raw_bond_rot / (norm_scale ** 2)

    # torch.set_printoptions(threshold=1000001)
    # print(trans_1_pred, trans_1)

    if isinstance(rigidwise_weight, torch.Tensor):
        framepair_weight = rigidwise_weight[..., None] * rigidwise_weight[..., None, :]
    else:
        framepair_weight = rigidwise_weight

    fafe = fafe_loss_l2(
        pred_frames=denoised_rigids,
        gt_frames=gt_rigids,
        frame_mask=rigids_mask,
        framepair_weight=framepair_weight,
        block_diag_size=fafe_l2_block_mask_size
    )
    scaled_fafe = fafe / norm_scale.squeeze(-1)

    ret = {
        "rot_vf_loss": rot_vf_loss,
        "trans_vf_loss": trans_vf_loss,
        "unscaled_rot_vf_loss": unscaled_rot_vf_loss,
        "unscaled_trans_vf_loss": unscaled_trans_vf_loss,
        "pred_trans_mse": pred_frame_trans_mse,
        "ref_trans_mse": ref_frame_trans_mse,
        "fafe": fafe,
        "scaled_fafe": scaled_fafe,
        "bond_length_rmse": bond_length_loss,
        "bond_angle_rmse": bond_angle_loss,
        "ring_planarity_loss": ring_plan_loss,
        "bond_rot_mse": bond_rot_loss,
        "unscaled_bond_length_rmse": raw_bond_length.detach(),
        "unscaled_bond_angle_rmse": raw_bond_angle.detach(),
        "unscaled_ring_planarity_loss": raw_ring_plan.detach(),
        "unscaled_bond_rot_mse": raw_bond_rot.detach(),
    }
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
    ) / mask.sum(dim=(-2, -1)).clip(min=1)

    rotpair_mask = mask * clamp_mask
    rot_dist_loss = torch.sum(
        rot_dist**2 * rotpair_mask,
        dim=(-2, -1),
    ) / torch.clamp(mask.sum(dim=(-2, -1)), min=1)

    return torch.sqrt(trans_dist_loss / trans_scale**2 + rot_dist_loss / rot_scale**2 + eps_so3)

