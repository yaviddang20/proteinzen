
import torch

from ligbinddiff.runtime.loss.utils import _mask, _nodewise_to_graphwise
from ligbinddiff.utils.fiber import compact_fiber_to_nl
from ligbinddiff.utils.type_l import type_l_apply, type_l_sub




def zernike_coeff_loss(ref_density,
                       pred_density,
                       num_nodes,
                       mask,
                       n_channels=1,
                       channel_weights=None,
                       eps=1e-6):
    """ Invariant MSE loss on Zernike coeffs
    Average of norms of the difference between all type-l vectors """
    ref_density = compact_fiber_to_nl(ref_density, n_channels=n_channels)
    pred_density = compact_fiber_to_nl(pred_density, n_channels=n_channels)

    # only compute residues where all atoms are present
    ref_density = type_l_apply(lambda x: _mask(x, mask), ref_density)
    pred_density = type_l_apply(lambda x: _mask(x, mask), pred_density)

    # print({k: v.abs().max() for k, v in ref_density.items()})
    # print({k: v.abs().max() for k, v in pred_density.items()})
    diff = type_l_sub(ref_density, pred_density)

    # eps for numerical stability
    diff = type_l_apply(lambda t: t + eps, diff)

    square_diff = type_l_apply(torch.square, diff)
    loss = 0
    numel = 0
    for (n, l), elems in square_diff.items():
        if (n - l) % 2 == 1:
            continue
        mags = elems.sum(dim=-1).sqrt()
        if channel_weights is not None:
            # print(mags.shape, channel_weights[~mask].shape)
            mags = mags * _mask(channel_weights, mask)
        loss = loss + mags#.sum()
        numel = numel + (~mask).long()

    per_graph_loss = _nodewise_to_graphwise(loss, num_nodes, numel)
    return per_graph_loss
