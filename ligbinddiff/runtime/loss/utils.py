import torch


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


def _elemwise_to_graphwise(elemwise_tensor, nodes_per_graph, node_elem_mask, reduction='mean'):
    if len(nodes_per_graph) == 1:
        return elemwise_tensor.mean().unsqueeze(0)

    reswise_num_elem = (~node_elem_mask).long().sum(dim=-1)
    num_elem_per_graph = [t.sum().item() for t in torch.split(reswise_num_elem, nodes_per_graph)]
    if reduction == 'mean':
        graphwise_tensor = torch.cat([t.mean().unsqueeze(0) for t in elemwise_tensor.split(num_elem_per_graph)])
    elif reduction == 'sum':
        graphwise_tensor = torch.cat([t.sum().unsqueeze(0) for t in elemwise_tensor.split(num_elem_per_graph)])
    else:
        raise ValueError('reduction must be mean or sum')
    return graphwise_tensor


def _nodewise_to_graphwise(nodewise_tensor, nodes_per_graph, node_mask, reduction='mean'):
    if len(nodes_per_graph) == 1:
        return nodewise_tensor.mean().unsqueeze(0)
    reswise_num = (~node_mask).long()
    num_node_per_graph = [t.sum().item() for t in torch.split(reswise_num, nodes_per_graph)]
    if reduction == 'mean':
        graphwise_tensor = torch.cat([t.mean().unsqueeze(0) for t in nodewise_tensor.split(num_node_per_graph)])
    elif reduction == 'sum':
        graphwise_tensor = torch.cat([t.sum().unsqueeze(0) for t in nodewise_tensor.split(num_node_per_graph)])
    else:
        raise ValueError('reduction must be mean or sum')
    return graphwise_tensor
