""" Adapted from Torsional Diffusion / DiffDock """
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx, scatter


# TODO: there's probably a better way to do this import
from proteinzen.stoch_interp.interpolate.so3_utils import rotvec_to_rotmat


def get_transformation_mask(pyg_data):
    G = to_networkx(pyg_data.to_homogeneous(), to_undirected=False)
    to_rotate = []
    edges = pyg_data['ligand', 'ligand'].edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]
        # print(edges[i], edges[i+1])

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                    # print([[], l], sorted(nx.connected_components(G2), key=len))
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                    # print([l, []], sorted(nx.connected_components(G2), key=len))
                continue
        to_rotate.append([])
        to_rotate.append([])
        # print([[], []])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1
    assert idx == np.sum(mask_edges)

    return mask_edges, mask_rotate


def modify_conformer_torsion_angles_batch_iterative(
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
    assert edge_index.shape[0] == sum([t.shape[0] for t in mask_rotate])
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
        if edge_mask_rotate[u - offset]:
            print("u", edge_mask_rotate, u, v, offset)
        assert not edge_mask_rotate[u - offset]
        if not edge_mask_rotate[v - offset]:
            print("v", edge_mask_rotate, u, v, offset)
        assert edge_mask_rotate[v - offset]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec / torch.linalg.norm(rot_vec + eps, dim=-1, keepdims=True) * torsion_updates[idx_edge:idx_edge + 1]
        rot_mat = rotvec_to_rotmat(rot_vec[None]).squeeze(0)

        print(rot_vec, pos[v:v + 1])

        select = torch.zeros_like(node_select).bool()
        select[node_select] = edge_mask_rotate

        pos[select] = (pos[select] - pos[v:v + 1]) @ rot_mat.T + pos[v:v + 1]

    return pos


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
    assert edge_index.shape[0] == sum([t.shape[0] for t in mask_rotate])

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

    for i, (u,v) in enumerate(edge_index):
        e_b = edge_batch[i]
        e_offset = edge_offset[e_b]
        n_offset = node_offset[e_b]
        edge_mask_rotate = mask_rotate[e_b][i - e_offset]
        assert not edge_mask_rotate[u - n_offset], (edge_mask_rotate, u, v, n_offset, e_b)
        assert edge_mask_rotate[v - n_offset], (edge_mask_rotate, u, v, n_offset, e_b)

    # rot_mats = torch.eye(3, device=rot_mats.device)[None].expand(rot_vecs.shape[0], -1, -1).float()

    num_rot_bond = [t.shape[0] for t in mask_rotate]

    for idx_edge in range(max(num_rot_bond)):
        # restrict node selections to molecules which have rotatable bonds at this idx
        has_rot_bond = [idx_edge < n for n in num_rot_bond]
        data_batch = torch.arange(len(mask_rotate), device=edge_index.device)[has_rot_bond]
        node_subset = (node_batch[:, None] == data_batch[None]).any(dim=-1)
        # select rotatable bonds from edge_index
        select_bond = [torch.zeros(n).bool() for n in num_rot_bond]
        for t in select_bond:
            if t.shape[0] > idx_edge:
                t[idx_edge] = True
        select_bond = torch.cat(select_bond).to(edge_index.device)
        # select nodes which rotate
        edge_mask_rotate_list = [mask_rotate[b][idx_edge] for b in data_batch]
        edge_mask_rotate = torch.cat([mask_rotate[b][idx_edge] for b in data_batch], dim=-1)
        edge_mask_count = torch.stack([mask_rotate[b][idx_edge].sum() for b in data_batch])

        # ensure bonds are correct
        bonds = edge_index[select_bond]
        for (u,v), edge_mask_rotate_item in zip(bonds, edge_mask_rotate_list):
            assert node_batch[u] == node_batch[v]
            offset = node_offset[node_batch[u]]
            # print(u, v, offset, edge_mask_rotate_item)
            # check if need to reverse the edge, v should be connected to the part that gets rotated
            # print(u, v, offset, edge_mask_rotate_item)
            assert not edge_mask_rotate_item[u - offset], (edge_mask_rotate_item, u, v, offset, node_batch[u], has_rot_bond, idx_edge)
            assert edge_mask_rotate_item[v - offset], (edge_mask_rotate_item, u, v, offset, node_batch[u], has_rot_bond, idx_edge)

        v = bonds[:, 1]

        select = torch.zeros_like(node_subset).bool()
        select[node_subset] = edge_mask_rotate

        trans = pos[v]
        # print(trans.shape, edge_mask_count.shape)
        trans_expand = torch.repeat_interleave(trans, edge_mask_count, dim=0)
        # select rot mats
        rot_vecs = pos[edge_index[select_bond, 0]] - pos[edge_index[select_bond, 1]]
        rot_vecs = rot_vecs / torch.linalg.norm(rot_vecs + eps, dim=-1, keepdims=True) * torsion_updates[select_bond, None]
        rot_mats = rotvec_to_rotmat(rot_vecs)
        rot_mat_expand = torch.repeat_interleave(rot_mats, edge_mask_count, dim=0)
        # print(rot_mat_expand.shape, trans_expand.shape, pos[select].shape)

        # print(rot_vecs[select_bond], trans)

        pos[select] = torch.matmul(
            (pos[select] - trans_expand).unsqueeze(-2),
            rot_mat_expand.transpose(-1, -2)
        ).squeeze(-2) + trans_expand
        # pos[select] = (pos[select] - trans_expand) + trans_expand

    return pos

def gen_conformer_update_instructs(
    edge_index,
    node_batch,
    mask_rotate
):
    if edge_index.shape[0] == 2:
        edge_index = edge_index.T
    assert edge_index.shape[0] == sum([t.shape[0] for t in mask_rotate])
    output_cache = {
        "edge_index": edge_index,
    }

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

    for i, (u,v) in enumerate(edge_index):
        e_b = edge_batch[i]
        e_offset = edge_offset[e_b]
        n_offset = node_offset[e_b]
        edge_mask_rotate = mask_rotate[e_b][i - e_offset]
        assert not edge_mask_rotate[u - n_offset], (edge_mask_rotate, u, v, n_offset, e_b)
        assert edge_mask_rotate[v - n_offset], (edge_mask_rotate, u, v, n_offset, e_b)

    num_rot_bond = [t.shape[0] for t in mask_rotate]
    output_cache["num_rot_bond"] = num_rot_bond
    for i in range(max(num_rot_bond)):
        output_cache[i] = {}

    for idx_edge in range(max(num_rot_bond)):
        # restrict node selections to molecules which have rotatable bonds at this idx
        has_rot_bond = [idx_edge < n for n in num_rot_bond]
        data_batch = torch.arange(len(mask_rotate), device=edge_index.device)[has_rot_bond]
        node_subset = (node_batch[:, None] == data_batch[None]).any(dim=-1)
        # select rotatable bonds from edge_index
        select_bond = [torch.zeros(n).bool() for n in num_rot_bond]
        for t in select_bond:
            if t.shape[0] > idx_edge:
                t[idx_edge] = True
        select_bond = torch.cat(select_bond).to(edge_index.device)
        # select nodes which rotate
        edge_mask_rotate = torch.cat([mask_rotate[b][idx_edge] for b in data_batch], dim=-1)
        edge_mask_count = torch.stack([mask_rotate[b][idx_edge].sum() for b in data_batch])

        # ensure bonds are correct
        bonds = edge_index[select_bond]
        select = torch.zeros_like(node_subset).bool()
        select[node_subset] = edge_mask_rotate

        output_cache[idx_edge]["bonds"] = bonds
        output_cache[idx_edge]["edge_mask_count"] = edge_mask_count
        output_cache[idx_edge]["select_node"] = select
        output_cache[idx_edge]["select_bond"] = select_bond

    return output_cache


def modify_conformer_torsion_angles_batch_cached(
        pos,
        torsion_updates,
        update_instructs,
        eps=1e-8):
    pos = pos.clone()

    edge_index = update_instructs['edge_index']
    num_rot_bond = update_instructs['num_rot_bond']

    for idx_edge in range(max(num_rot_bond)):
        bonds = update_instructs[idx_edge]["bonds"]
        edge_mask_count = update_instructs[idx_edge]["edge_mask_count"]
        select_bond = update_instructs[idx_edge]["select_bond"]
        select_node = update_instructs[idx_edge]["select_node"]
        v = bonds[:, 1]

        trans = pos[v]
        # print(trans.shape, edge_mask_count.shape)
        trans_expand = torch.repeat_interleave(trans, edge_mask_count, dim=0)
        # select rot mats
        rot_vecs = pos[bonds[:, 0]] - pos[bonds[:, 1]]
        rot_vecs = rot_vecs / torch.linalg.norm(rot_vecs + eps, dim=-1, keepdims=True) * torsion_updates[select_bond, None]
        rot_mats = rotvec_to_rotmat(rot_vecs)
        rot_mat_expand = torch.repeat_interleave(rot_mats, edge_mask_count, dim=0)
        # print(rot_mat_expand.shape, trans_expand.shape, pos[select].shape)

        # print(pos)
        # print(pos.dtype, select_node.dtype, trans_expand.dtype, rot_mat_expand.dtype)
        new_pos = torch.zeros_like(pos)
        new_pos[select_node] = torch.matmul(
            (pos[select_node] - trans_expand).unsqueeze(-2),
            rot_mat_expand.transpose(-1, -2)
        ).squeeze(-2) + trans_expand
        new_pos[~select_node] = pos[~select_node]
        # pos[select_node] = torch.matmul(
        #     (pos[select_node] - trans_expand).unsqueeze(-2),
        #     rot_mat_expand.transpose(-1, -2)
        # ).squeeze(-2) + trans_expand
        pos = new_pos

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