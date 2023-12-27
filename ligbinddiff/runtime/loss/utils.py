import torch

from torch_geometric.utils import scatter

def _mask(tensor, mask):
    extra_ones = [1 for _ in range(len(tensor.shape) - len(mask.shape))]
    mask_long = mask.long.view(list(mask.shape) + extra_ones)
    return tensor * (1 - mask_long)


def vec_norm(tensor, mask_nans=True, eps=1e-6, dim=-1):
    if mask_nans:
        tensor = torch.nan_to_num(tensor)
    norm = torch.sum(tensor * tensor, dim=dim)
    norm = (norm + eps).sqrt()
    return norm


def _sum_except(tensor, dim=0):
    if len(tensor.shape) < 2:
        return tensor
    other_dims = [i for i in range(tensor.dim()) if i != dim]
    return tensor.sum(dim=other_dims)


def _nodewise_to_graphwise(node_elem_tensor, batch, node_elem_mask: torch.BoolTensor, reduction='mean'):
    """
    Args
    -----
    node_elem_tensor: torch.Tensor (n_node, ...)

    batch: torch.Tensor (n_node,)

    node_elem_mask: torch.Tensor (n_node, ...)
    """
    assert node_elem_tensor.shape == node_elem_mask.shape, (node_elem_tensor.shape, node_elem_mask.shape)

    num_graphs = batch.max().item()+1
    if num_graphs == 1:
        ret_denom = node_elem_mask.long().sum()
        # prevent a div-by-zero error if there are no elements
        ret_denom = torch.where(ret_denom>0, ret_denom, 1)
        ret = (node_elem_tensor * node_elem_mask).sum() / ret_denom
        return ret.unsqueeze(0)

    nodewise_sum = _sum_except(node_elem_tensor * node_elem_mask, dim=0)
    ret_sum = scatter(
        nodewise_sum,
        batch,
        dim=0,
        dim_size=num_graphs
    )
    if reduction == 'sum':
        return ret_sum
    elif reduction == 'mean':
        nodewise_num_elem = _sum_except(node_elem_mask.long(), dim=0)
        ret_denom = scatter(
            nodewise_num_elem,
            batch,
            dim=0,
            dim_size=num_graphs
        )
        # prevent a div-by-zero error if there are no elements
        ret = ret_sum / torch.where(ret_denom>0, ret_denom, 1)
        return ret
    else:
        raise ValueError('reduction must be mean or sum')
