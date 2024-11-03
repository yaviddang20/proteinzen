from proteinzen.runtime.loss.utils import _nodewise_to_graphwise, vec_norm


import numpy as np
import torch
import torch.nn.functional as F

import torch_geometric.utils as pygu


def so3_embedding_mse(ref_so3, pred_so3, num_nodes, x_mask, scaling=None):
    vec_diff = ref_so3.embedding[~x_mask] - pred_so3.embedding[~x_mask]
    splits = []
    for lmax in ref_so3.lmax_list:
        for l in range(lmax+1):
            splits.append(2*l+1)
    vec_diffs = vec_diff.split(splits, dim=1)
    vec_diff_norms = [vec_norm(v.transpose(-1, -2)) for v in vec_diffs]  # vector dim as final dim
    nodewise_loss = torch.cat(vec_diff_norms, dim=-1).sum(dim=-1)

    if scaling is not None:
        nodewise_loss = nodewise_loss * scaling[~x_mask]

    return _nodewise_to_graphwise(nodewise_loss, num_nodes, x_mask)


def so3_embedding_kl(so3_mu, so3_logvar, num_nodes, x_mask):
    splits = []
    for lmax in so3_mu.lmax_list:
        for l in range(lmax+1):
            splits.append(2*l+1)
    split_mu = so3_mu.embedding[~x_mask].split(splits, dim=1)
    split_logvar = so3_logvar.embedding[~x_mask].split(splits, dim=1)
    kl_div = []
    for m, mu, logvar in zip(splits, split_mu, split_logvar):
        comp_kl_div = -0.5 * (logvar - mu.square() - logvar.exp() + 1)
        kl_div.append(comp_kl_div.sum(dim=(-1, -2)))

    return [_nodewise_to_graphwise(kl, num_nodes, x_mask) for kl in kl_div]


def scalars_kl_div(mu, logvar, batch, mask):
    kl_div = -0.5 * (logvar - mu.square() - logvar.exp() + 1)
    kl_div = kl_div.sum(dim=-1)
    return _nodewise_to_graphwise(kl_div, batch, mask)


def nodewise_kl_div(x, batch, mask, eps=1e-8):
    x = x * mask[..., None]
    norm_factor = pygu.scatter(mask.float(), index=batch, reduce='sum')[..., None]
    mu = pygu.scatter(x, index=batch, reduce='sum') / norm_factor.clip(min=1)
    var = pygu.scatter(torch.square(x - mu[batch] + eps), index=batch, reduce='sum')
    var = var / (norm_factor-1).clip(min=1)
    logvar = torch.log(var)
    kl_div = -0.5 * (logvar - mu.square() - var + 1)
    kl_div = kl_div.sum(dim=-1)
    return kl_div, mu.mean(dim=-1), var.mean(dim=-1)


def gaussian_nll(data, mu, logvar, batch, mask, reduction='mean'):
    ll = -0.5 * (logvar + torch.square(data - mu) / torch.exp(logvar) + np.log(np.pi * 2))
    if reduction == 'mean':
        return _nodewise_to_graphwise(-ll, batch, mask[..., None].expand(*[-1 for _ in range(len(mask.shape))], ll.shape[-1]))
    elif reduction == 'sum':
        return _nodewise_to_graphwise(-ll.sum(dim=-1), batch, mask)
    else:
        raise ValueError(f"unsupported reduction {reduction}")