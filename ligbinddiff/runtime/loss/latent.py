from ligbinddiff.runtime.loss.utils import _nodewise_to_graphwise, vec_norm


import torch


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
    kl_div = 0
    for m, mu, logvar in zip(splits, split_mu, split_logvar):
        comp_kl_div = -0.5 * (logvar.sum(dim=-2) - mu.square().sum(dim=-2) - logvar.exp().sum(dim=-2) + m)
        kl_div = kl_div + comp_kl_div.mean(dim=-1)

    return _nodewise_to_graphwise(kl_div, num_nodes, x_mask)
