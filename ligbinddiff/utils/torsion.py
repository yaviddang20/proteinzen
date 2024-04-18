""" Adapted from Torsional Diffusion / DiffDock """
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx, scatter


# TODO: there's probably a better way to do this import
from ligbinddiff.stoch_interp.interpolate.so3_utils import rotvec_to_rotmat


def get_transformation_mask(pyg_data):
    G = to_networkx(pyg_data.to_homogeneous(), to_undirected=False)
    to_rotate = []
    edges = pyg_data['ligand', 'ligand'].edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate


def modify_conformer_torsion_angles_batch(
        pos,
        edge_index,
        node_batch,
        mask_rotate,
        torsion_updates,
        eps=1e-8):
    pos = pos.clone()
    if edge_index.shape[0] == 2:
        edge_index = edge_index.T
    assert edge_index.shape[0] == torsion_updates.shape[0]
    edge_batch = node_batch[edge_index[:, 0]]
    node_offset = scatter(
        torch.ones_like(node_batch).long(),
        node_batch,
        dim=0
    )
    node_offset = F.pad(node_offset[:-1], (1, 0), value=0)
    node_offset = torch.cumsum(node_offset, dim=0)
    edge_offset = scatter(
        torch.ones_like(edge_batch).long(),
        edge_batch,
        dim=0
    )
    edge_offset = F.pad(edge_offset[:-1], (1, 0), value=0)
    edge_offset = torch.cumsum(edge_offset, dim=0)

    for idx_edge, e in enumerate(edge_index):
        u, v = e[0], e[1]
        data_batch = edge_batch[idx_edge]
        edge_mask_rotate = mask_rotate[data_batch][idx_edge - edge_offset[data_batch]]
        offset = node_offset[data_batch]
        node_select = (node_batch == data_batch)
        # print(u,v, offset, data_batch, node_offset)

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not edge_mask_rotate[u - offset]
        assert edge_mask_rotate[v - offset]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec / torch.linalg.norm(rot_vec + eps, dim=-1, keepdims=True) * torsion_updates[idx_edge:idx_edge + 1]
        rot_mat = rotvec_to_rotmat(rot_vec[None]).squeeze(0)

        select = torch.zeros_like(node_select).bool()
        select[node_select] = edge_mask_rotate

        pos[select] = (pos[select] - pos[v:v + 1]) @ rot_mat.T + pos[v:v + 1]

    return pos


def bdot(a, b):
    return torch.sum(a*b, dim=-1, keepdim=True)


def get_torsion_angles(dihedral, batch_pos, batch_size):
    batch_pos = batch_pos.reshape(batch_size, -1, 3)

    c, a, b, d = dihedral[:, 0], dihedral[:, 1], dihedral[:, 2], dihedral[:, 3]
    c_project_ab = batch_pos[:,a] + bdot(batch_pos[:,c] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) / bdot(batch_pos[:,b] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) * (batch_pos[:,b] - batch_pos[:,a])
    d_project_ab = batch_pos[:,a] + bdot(batch_pos[:,d] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) / bdot(batch_pos[:,b] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) * (batch_pos[:,b] - batch_pos[:,a])
    dshifted = batch_pos[:,d] - d_project_ab + c_project_ab
    cos = bdot(dshifted - c_project_ab, batch_pos[:,c] - c_project_ab) / (
                torch.norm(dshifted - c_project_ab, dim=-1, keepdim=True) * torch.norm(batch_pos[:,c] - c_project_ab, dim=-1,
                                                                                       keepdim=True))
    cos = torch.clamp(cos, -1 + 1e-5, 1 - 1e-5)
    angle = torch.acos(cos)
    sign = torch.sign(bdot(torch.cross(dshifted - c_project_ab, batch_pos[:,c] - c_project_ab), batch_pos[:,b] - batch_pos[:,a]))
    torsion_angles = (angle * sign).squeeze(-1)
    return torsion_angles


def get_dihedrals(data):
    edge_index, edge_mask = data['ligand', 'ligand'].edge_index, data['ligand'].edge_mask
    edge_list = [[] for _ in range(torch.max(edge_index) + 1)]

    for p in edge_index.T:
        edge_list[p[0]].append(p[1])

    rot_bonds = [(p[0], p[1]) for i, p in enumerate(edge_index.T) if edge_mask[i]]

    dihedral = []
    for a, b in rot_bonds:
        c = edge_list[a][0] if edge_list[a][0] != b else edge_list[a][1]
        d = edge_list[b][0] if edge_list[b][0] != a else edge_list[b][1]
        dihedral.append((c.item(), a.item(), b.item(), d.item()))
    # dihedral_numpy = np.asarray(dihedral)
    # print(dihedral_numpy.shape)
    dihedral = torch.tensor(dihedral)
    return dihedral