""" Featurize only the sidechains """
import math

import torch
import torch.nn.functional as F


def _normalize(tensor, dim=-1, eps=1e-8):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor+eps, dim=dim, keepdim=True)))


def _dihedrals(X, eps=1e-7):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    # X = torch.reshape(X[..., :, :3, :], [3*X.shape[-2], 3])
    X = X[..., :, :3, :].flatten(start_dim=-3, end_dim=-2)
    dX = X[..., 1:, :] - X[..., :-1, :]
    U = _normalize(dX, dim=-1)
    u_2 = U[..., :-2, :]
    u_1 = U[..., 1:-1, :]
    u_0 = U[..., 2:, :]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, list(D.shape[:-1]) + [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], -1)
    return D_features


def _orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def _ideal_virtual_Cb(X_bb):
    # from ProteinMPNN paper computation of Cb
    N, Ca, C = [X_bb[..., i, :] for i in range(3)]
    b = Ca - N
    c = C- Ca
    a = torch.cross(b, c)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    return Cb