import numpy as np
import torch


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu', dtype=None):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device, dtype=dtype)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    RBF = RBF.view(*D.shape, -1)
    return RBF


def _node_positional_embeddings(node_index,
                           num_embeddings,
                           device='cpu'):
    # From https://github.com/jingraham/neurips19-graph-protein-design

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = node_index.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def _edge_positional_embeddings(edge_index,
                           num_embeddings,
                           device='cpu'):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E
