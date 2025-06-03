import torch

from ..utils import _nodewise_to_graphwise

def atom10_local_mse_loss(
        gt_atom14,
        alt_atom14,
        gt_rigids,
        pred_atom10_local,
        batch,
        atom14_gt_mask,
        atom14_alt_gt_mask,
        res_mask,
        res_noising_mask):
    gt_mask = torch.ones_like(atom14_gt_mask) * res_noising_mask[..., None] * res_mask[..., None]
    alt_mask = torch.ones_like(atom14_alt_gt_mask) * res_noising_mask[..., None] * res_mask[..., None]
    gt_atom10_local = gt_rigids[..., None].invert_apply(gt_atom14[..., 4:, :]) * gt_mask[..., 4:, None]
    alt_atom10_local = gt_rigids[..., None].invert_apply(alt_atom14[..., 4:, :]) * alt_mask[..., 4:, None]

    gt_diff = torch.square(gt_atom10_local - pred_atom10_local).sum(dim=-1)
    alt_diff = torch.square(alt_atom10_local - pred_atom10_local).sum(dim=-1)
    apply_loss = res_noising_mask * res_mask
    num_atoms = torch.ones_like(gt_mask)[..., 4:] * apply_loss[..., None]
    num_atoms = num_atoms.sum(dim=-1)

    # we take the min mse per residue
    gt_poswise_diff = gt_diff.sum(dim=-1)
    alt_poswise_diff = alt_diff.sum(dim=-1)
    min_diff = torch.min(gt_poswise_diff, alt_poswise_diff)   # n_node

    return _nodewise_to_graphwise(min_diff / num_atoms.clip(min=1), batch, apply_loss)


def atom10_local_dist_loss(
        gt_atom14,
        alt_atom14,
        gt_rigids,
        pred_atom10_local,
        batch,
        atom14_gt_mask,
        atom14_alt_gt_mask,
        res_mask,
        res_noising_mask):
    gt_mask = atom14_gt_mask * res_noising_mask[..., None] * res_mask[..., None]
    alt_mask = atom14_alt_gt_mask * res_noising_mask[..., None] * res_mask[..., None]

    gt_atom14 = gt_atom14 * gt_mask[..., None]
    alt_atom14 = alt_atom14 * gt_mask[..., None]

    gt_atom10_local = gt_rigids[..., None].invert_apply(gt_atom14[..., 4:, :])
    alt_atom10_local = gt_rigids[..., None].invert_apply(alt_atom14[..., 4:, :])

    gt_dists = torch.square(gt_atom10_local - pred_atom10_local).sum(dim=-1) * gt_mask[..., 4:]
    alt_diff = torch.square(alt_atom10_local - pred_atom10_local).sum(dim=-1) * alt_mask[..., 4:]
    apply_loss = res_noising_mask * res_mask
    num_atoms = torch.ones_like(gt_mask)[..., 4:] * apply_loss[..., None]
    num_atoms = num_atoms.sum(dim=-1)

    # we take the min mse per residue
    gt_poswise_diff = gt_diff.sum(dim=-1)
    alt_poswise_diff = alt_diff.sum(dim=-1)
    min_diff = torch.min(gt_poswise_diff, alt_poswise_diff)   # n_node

    return _nodewise_to_graphwise(min_diff / num_atoms.clip(min=1), batch, apply_loss)