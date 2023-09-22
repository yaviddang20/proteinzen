""" Featurize only the sidechains """
import math
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
from torch_geometric.data import Data, HeteroData

from ligbinddiff.utils.atom_reps import atom14_to_atom91, atom14_residue_bonds, atom14_atomic_row, atom14_atomic_period, restype_3to1, atom_to_atomic_period, atom_to_atomic_row
# from ligbinddiff.utils.fiber import nl_to_fiber

# TODO: brought this in to avoid importing dgl in the utils, do this in a less hacky way
def nl_to_fiber(Z):
    """ Convert from Zernike coeffs to a Fiber dict (no n) """
    fiber_dict = {}
    for (n,l), coeffs in Z.items():
        if l not in fiber_dict:
            fiber_dict[l] = []
        fiber_dict[l].append((n, coeffs))

    for l, coeff_list in fiber_dict.items():
        coeff_list = sorted(coeff_list, key=lambda p: p[0], reverse=True)  # order from largest to smallest n
        coeff_list = list(zip(*coeff_list))[1]  # take only coeffs
        fiber_dict[l] = torch.cat(coeff_list, dim=-2)  # fuse coeffs into one tensor

    return fiber_dict


def _normalize(tensor, dim=-1, eps=1e-8):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor+eps, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _dihedrals(X, eps=1e-7):
    # From https://github.com/jingraham/neurips19-graph-protein-design

    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def _positional_embeddings(edge_index,
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


def featurize_density(protein,
                      zernike_transform,
                      letter_to_num,
                      channel_atoms=True,
                      channel_values=None,
                      bb_density=False,
                      num_rbf=16,
                      num_positional_embeddings=16,
                      top_k=30,
                      device='cpu'):
    name = protein['name']
    with torch.no_grad():
        atom14 = protein['coords']
        atom14_mask = protein['coords_mask']
        atom91, atom91_mask = atom14_to_atom91(protein['seq'], atom14)
        atom14 = torch.as_tensor(atom14,
                                 device=device, dtype=torch.float32)
        atom14_mask = torch.as_tensor(atom14_mask,
                                      device=device, dtype=torch.bool)
        atom91 = torch.as_tensor(atom91,
                                 device=device, dtype=torch.float32)
        atom91_mask = torch.as_tensor(atom91_mask,
                                      device=device, dtype=torch.bool)
        seq = torch.as_tensor([letter_to_num[a] for a in protein['seq']],
                               device=device, dtype=torch.long)
        # backbone coords
        coords = atom14[:, :4]
        X_cb = _ideal_virtual_Cb(coords)

        x_mask = torch.isfinite(coords.sum(dim=(1,2)))
        x_mask = ~x_mask
        coords[x_mask] = np.inf
        atom14_mask[x_mask] = True
        atom91_mask[x_mask] = True

        X_ca = coords[:, 1]
        edge_index = torch_cluster.knn_graph(X_ca, k=top_k)

        pos_embeddings = _positional_embeddings(edge_index, num_positional_embeddings)
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf, device=device)

        dihedrals = _dihedrals(coords)
        orientations = _orientations(X_ca)
        sidechains = _sidechains(coords)

        node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)

        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                (node_s, node_v, edge_s, edge_v))

        # densities should be computed centered at C_beta
        if channel_atoms:
            # we could do this using atom14 but it's not clear to me that's any faster
            # since we can't take advantage of double masking to avoid taking sequence as input
            atom_mask = atom91_mask.any(dim=-1)
            channel_values = torch.as_tensor(channel_values, device=device, dtype=torch.float)
            if not bb_density:
                atom91_ = atom91[..., 4:, :]
                atom_mask_ = atom_mask[..., 4:]
                channel_values_ = channel_values[..., 4:]
            else:
                atom91_ = atom91
                atom_mask_ = atom_mask
                channel_values_ = channel_values
            Z = zernike_transform.forward_transform(atom91_, X_cb, atom_mask_, point_value=channel_values_)
        else:
            atom_mask = atom14_mask.any(dim=-1)
            Z = zernike_transform.forward_transform(atom14, X_cb, atom_mask)
        fiber_dict = nl_to_fiber(Z)

        graph = Data(x=X_ca,
                     x_mask=x_mask,
                     x_cb=X_cb,
                     seq=seq,
                     bb_s=node_s,
                     bb_v=node_v,
                     edge_index=edge_index,
                     edge_s=edge_s,
                     edge_v=edge_v,
                     rel_pos=torch.nan_to_num(E_vectors),
                     atom91_centered=atom91 - X_ca.unsqueeze(-2),
                     atom91_mask=atom91_mask,
                     name=name,
                     density=fiber_dict)

    return graph


def featurize_atomic(protein,
                     letter_to_num,
                     num_rbf=16,
                     num_positional_embeddings=16,
                     top_k=30,
                     device='cpu'):
    name = protein['name']
    with torch.no_grad():
        atom14 = protein['coords']
        atom14_mask = protein['coords_mask']
        atom91, atom91_mask = atom14_to_atom91(protein['seq'], atom14)
        atom14 = torch.as_tensor(atom14,
                                    device=device, dtype=torch.float32)
        atom14_mask = torch.as_tensor(atom14_mask,
                                        device=device, dtype=torch.bool)
        atom91 = torch.as_tensor(atom91,
                                    device=device, dtype=torch.float32)
        atom91_mask = torch.as_tensor(atom91_mask,
                                        device=device, dtype=torch.bool)
        seq = torch.as_tensor([letter_to_num[a] for a in protein['seq']],
                                device=device, dtype=torch.long)
        # backbone coords
        coords = atom14[:, :4]
        X_cb = _ideal_virtual_Cb(coords)

        x_mask = torch.isfinite(coords.sum(dim=(1,2)))
        x_mask = ~x_mask
        coords[x_mask] = np.inf
        atom14_mask[x_mask] = True
        atom91_mask[x_mask] = True

        X_ca = coords[:, 1]
        edge_index = torch_cluster.knn_graph(X_ca, k=top_k)

        pos_embeddings = _positional_embeddings(edge_index, num_positional_embeddings)
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf, device=device)

        dihedrals = _dihedrals(coords)
        orientations = _orientations(X_ca)
        sidechains = _sidechains(coords)

        node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)

        nan_to_num = partial(torch.nan_to_num, neginf=0.0, posinf=0.0)

        node_s, node_v, edge_s, edge_v = map(nan_to_num,
                (node_s, node_v, edge_s, edge_v))


        graph = Data(x=nan_to_num(X_ca),
                     x_mask=x_mask,
                     x_cb=nan_to_num(X_cb),
                     seq=seq,
                     bb=nan_to_num(coords),
                     bb_s=nan_to_num(node_s),
                     bb_v=nan_to_num(node_v),
                     edge_index=edge_index,
                     edge_s=nan_to_num(edge_s),
                     edge_v=nan_to_num(edge_v),
                     atom91_centered=nan_to_num(atom91 - X_ca.unsqueeze(-2)),
                     atom91_mask=atom91_mask,
                     name=name)

    return graph

def featurize_cross_scale_atomic(protein,
                                 letter_to_num,
                                 num_rbf=16,
                                 num_positional_embeddings=16,
                                 top_k=30,
                                 max_period=18,
                                 max_row=5,
                                 device='cpu'):
    name = protein['name']
    with torch.no_grad():
        atom14 = protein['coords']
        atom14_mask = protein['coords_mask']
        atom91, atom91_mask = atom14_to_atom91(protein['seq'], atom14)
        atom14 = torch.as_tensor(atom14,
                                    device=device, dtype=torch.float32)
        atom14_mask = torch.as_tensor(atom14_mask,
                                        device=device, dtype=torch.bool)
        atom91 = torch.as_tensor(atom91,
                                    device=device, dtype=torch.float32)
        atom91_mask = torch.as_tensor(atom91_mask,
                                        device=device, dtype=torch.bool)
        seq = torch.as_tensor([letter_to_num[a] for a in protein['seq']],
                                device=device, dtype=torch.long)
        # backbone coords
        coords = atom14[:, :4]
        X_cb = _ideal_virtual_Cb(coords)

        x_mask = torch.isfinite(coords.sum(dim=(1,2)))
        x_mask = ~x_mask
        coords[x_mask] = np.inf
        atom14_mask[x_mask] = True
        atom91_mask[x_mask] = True

        X_ca = coords[:, 1]
        edge_index = torch_cluster.knn_graph(X_ca, k=top_k)

        pos_embeddings = _positional_embeddings(edge_index, num_positional_embeddings)
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf, device=device)

        dihedrals = _dihedrals(coords)
        orientations = _orientations(X_ca)
        sidechains = _sidechains(coords)

        node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)

        nan_to_num = partial(torch.nan_to_num, neginf=0.0, posinf=0.0)

        node_s, node_v, edge_s, edge_v = map(nan_to_num,
                (node_s, node_v, edge_s, edge_v))

        atom_type_embedding = atom14_atom_type_embedding(seq, atom14_mask, letter_to_num, max_period=max_period, max_row=max_row)
        atoms_flat, atomic_radius_edge_index, backbone_atoms_select = atom_radius_graph(atom14, atom14_mask)
        atom14_to_ca_edge_index = atom14_to_ca_graph(atom14, atom14_mask)

        residue_dict = {
            "x": nan_to_num(X_ca),
            "x_mask": x_mask,
            "x_cb": nan_to_num(X_cb),
            "seq": seq,
            "bb_s": nan_to_num(node_s),
            "bb_v": nan_to_num(node_v),
            "edge_index": edge_index,
            "edge_s": nan_to_num(edge_s),
            "edge_v": nan_to_num(edge_v),
            "atom91_centered": nan_to_num(atom91 - X_ca.unsqueeze(-2)),
            "atom91_mask": atom91_mask,
        }

        atomic_dict = {
            "x": atoms_flat,
            "atom_embedding": atom_type_embedding,
            "backbone_atoms_select": backbone_atoms_select,
            "atomic_radius_edge_index": atomic_radius_edge_index,
            "atom14_to_ca_edge_index": atom14_to_ca_edge_index,
        }

        graph = HeteroData(
            atomic=atomic_dict,
            residue=residue_dict,
            name=name)

    return graph


def residue_bond_graph(atom14_mask, seq, letter_to_num):
    assert len(atom14_mask.shape) == 2
    # compute residue bond graphs
    num_atoms_per_residue = ~atom14_mask.sum(dim=-1)
    atom_idx_offset = torch.cat([
        torch.zeros(1),
        torch.cumsum(num_atoms_per_residue, dim=0)[:-1]
    ], dim=0)

    residue_bonds_mask = torch.zeros(
        len(atom14_residue_bonds.keys()),
        sum(len(bonds) for bonds in atom14_residue_bonds.values())
    )
    offset = 0
    residue_bonds = []
    for aa, bonds in atom14_residue_bonds.items():
        bonds_forward = torch.as_tensor(bonds).T
        bonds_reverse = torch.flip(bonds_forward, (0,))
        bonds_idx = torch.cat([bonds_forward, bonds_reverse], dim=0)
        residue_bonds.append(bonds_idx)
        aa_idx = letter_to_num[aa]
        residue_bonds_mask[aa_idx, offset:offset+2*len(bonds)] = bonds_idx
        offset += 2*len(bonds)
    residue_bonds = torch.cat(residue_bonds, dim=-1)

    residue_bond_edges = residue_bonds[seq] + atom_idx_offset[:, None]
    residue_bond_edge_mask = residue_bonds_mask[seq]
    residue_bond_edge_index = residue_bond_edges[residue_bond_edge_mask]

    return residue_bond_edge_index


def atom_radius_graph(atom14, atom14_mask, radius=6):
    assert len(atom14_mask.shape) == 2
    residue_atoms = atom14[~atom14_mask]
    atom_radius_edge_index = torch_cluster.radius_graph(residue_atoms, r=radius)

    backbone_atoms_select = torch.zeros_like(atom14_mask).bool()
    backbone_atoms_select[:, :4] = True
    backbone_atoms_select = backbone_atoms_select[~atom14_mask]

    return residue_atoms, atom_radius_edge_index, backbone_atoms_select


def atom14_to_ca_graph(atom14, atom14_mask):
    assert len(atom14_mask.shape) == 2
    num_res = atom14.shape[0]
    num_atoms_per_residue = ~atom14_mask.sum(dim=-1)
    atom_idx_offset = torch.cat([
        torch.zeros(1),
        torch.cumsum(num_atoms_per_residue, dim=0)[:-1]
    ], dim=0)

    residue_dst = torch.arange(14)
    residue_src = torch.ones(14)  # ca is index 1
    residue_graph = torch.stack([
        residue_dst,
        residue_src
    ], dim=-1)
    residue_graph = residue_graph[residue_dst != 1]  # no self edge to ca
    atom14_to_ca_graph = residue_graph.unsqueeze(0).expand(num_res, -1) + atom_idx_offset

    atom14_mask_no_ca = atom14_mask[:, residue_dst != 1]
    atom14_to_ca_edge_index = atom14_to_ca_graph[atom14_mask_no_ca]
    return atom14_to_ca_edge_index


def atom14_atom_type_embedding(seq, atom14_mask, letter_to_num, max_period=None, max_row=None):
    if max_period is None:
        max_period = max(atom_to_atomic_period.values())
    if max_row is None:
        max_row = max(atom_to_atomic_row.values())
    one_hot_period = torch.eye(max_period+1)
    one_hot_row = torch.eye(max_row+1)

    num_aa = max(letter_to_num.values()) + 1

    period_store = torch.zeros(
        num_aa,
        14
    )
    row_store = torch.zeros(
        num_aa,
        14
    )
    for aa, periods in atom14_atomic_period.items():
        aa_idx = letter_to_num(restype_3to1[aa])
        period_store[aa_idx] = torch.as_tensor(periods)
    for aa, row in atom14_atomic_period.items():
        aa_idx = letter_to_num(restype_3to1[aa])
        row_store[aa_idx] = torch.as_tensor(row)

    atom_periods = period_store[seq]
    atom_rows = row_store[seq]
    period_embedding = one_hot_period[atom_periods]
    row_embedding = one_hot_row[atom_rows]
    atom_type_embedding = torch.cat([period_embedding, row_embedding], dim=-1)  # n_nodes x 14 x n_period+n_row

    return atom_type_embedding[~atom14_mask]  # n_atoms x n_period+n_row
