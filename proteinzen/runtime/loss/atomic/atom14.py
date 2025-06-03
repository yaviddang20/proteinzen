import torch

from ..utils import _nodewise_to_graphwise

def atom14_mse_loss(gt_atom14,
                    alt_atom14,
                    pred_atom14,
                    batch,
                    atom14_gt_mask,
                    atom14_alt_gt_mask,
                    noising_mask,
                    no_bb=False,
                    seqwise_weight=None,
                    ignore_symmetry=False):
    gt_mask = atom14_gt_mask * noising_mask[..., None]
    alt_mask = atom14_alt_gt_mask * noising_mask[..., None]
    if no_bb:
        gt_atom14 = gt_atom14[..., 4:, :]
        alt_atom14 = alt_atom14[..., 4:, :]
        gt_mask = gt_mask[..., 4:]
        alt_mask = alt_mask[..., 4:]
        pred_atom14 = pred_atom14[..., 4:, :]


    if ignore_symmetry:
        gt_diff = torch.square(gt_atom14 - pred_atom14).sum(dim=-1) * gt_mask

        if seqwise_weight is not None:
            gt_diff = gt_diff * seqwise_weight

        return _nodewise_to_graphwise(gt_diff, batch, gt_mask)

    else:
        gt_diff = torch.square(gt_atom14 - pred_atom14).sum(dim=-1) * gt_mask
        alt_diff = torch.square(gt_atom14 - pred_atom14).sum(dim=-1) * alt_mask

        # we take the min mse per residue
        gt_poswise_diff = gt_diff.sum(dim=-1)
        alt_poswise_diff = alt_diff.sum(dim=-1)
        choose_gt = (gt_poswise_diff < alt_poswise_diff)  # n_node

        min_diff = choose_gt[..., None] * gt_diff + (~choose_gt)[..., None] * alt_diff
        if seqwise_weight is not None:
            min_diff = min_diff * seqwise_weight

        return _nodewise_to_graphwise(min_diff, batch, gt_mask)


def chi_loss(gt_chi_vecs,
             alt_chi_vecs,
             pred_chi_vecs,
             batch,
             chi_mask):
    gt_diff = torch.square(gt_chi_vecs - pred_chi_vecs).sum(dim=-1) * chi_mask
    alt_diff = torch.square(alt_chi_vecs - pred_chi_vecs).sum(dim=-1) * chi_mask
    num_chis = chi_mask.int().sum(dim=-1)

    # we take the min dist per residue
    gt_poswise_diff = gt_diff.sum(dim=-1) / torch.where(num_chis>0, num_chis, 1)
    alt_poswise_diff = alt_diff.sum(dim=-1) / torch.where(num_chis>0, num_chis, 1)
    min_diff = torch.min(gt_poswise_diff, alt_poswise_diff)  # n_node

    ret = _nodewise_to_graphwise(min_diff, batch, (num_chis > 0))
    return ret
