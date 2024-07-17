import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.utils as pygu

import ligbinddiff.utils.openfold.rigid_utils as ru
from ligbinddiff.data.constants.atom14 import atom14_atom_props, atom14_bond_props, atom14_bond_edge_indicies, atom14_bond_mask, peptide_bond_props
from ligbinddiff.data.openfold.residue_constants import restype_atom14_mask



def atom14_to_atomic(atom14, atom14_mask):
    num_nodes = atom14.shape[0]
    atom_res_idx = torch.arange(num_nodes, device=atom14.device)[..., None].expand(-1, 14)
    atom_res_idx = atom_res_idx[atom14_mask]
    flat_atoms = atom14[atom14_mask]
    ca_select = torch.zeros(num_nodes, 14, device=atom14.device, dtype=torch.bool)
    ca_select[:, 1] = True
    ca_select = ca_select[atom14_mask]

    return flat_atoms, atom_res_idx, ca_select


def atomic_to_atom14(atoms, atom14_mask):
    num_nodes = atom14_mask.shape[0]
    atom14 = torch.zeros(num_nodes, 14, atoms.shape[-1], device=atoms.device)
    atom14[atom14_mask] = atoms
    return atom14

def gen_bond_graph(seq, atom14_mask, res_batch):
    atom_props_store = torch.as_tensor(atom14_atom_props, device=seq.device)
    bond_props_store = torch.as_tensor(atom14_bond_props, device=seq.device)
    bond_edge_index_store = torch.as_tensor(atom14_bond_edge_indicies, device=seq.device)
    select_bonds_store = torch.as_tensor(atom14_bond_mask, device=seq.device)
    select_bonds = select_bonds_store[seq]

    ideal_atom14_mask = torch.as_tensor(restype_atom14_mask, device=seq.device).bool()[seq]
    # select just bonds with proper atom offset per residue bond set
    atom_offset = torch.cumsum(atom14_mask.sum(dim=-1), dim=0)
    atom_offset = F.pad(atom_offset[:-1], (1, 0), value=0.0)
    all_possible_bonds = bond_edge_index_store[seq]
    all_possible_bonds = atom_offset[..., None, None] + all_possible_bonds
    bond_edge_index = all_possible_bonds[select_bonds]
    bond_props = bond_props_store[seq][select_bonds]

    # we have to add the backbone peptide bonds
    peptide_bonds = []
    num_graphs = res_batch.max().item() + 1
    for i in range(num_graphs):
        select = (res_batch == i)
        N_idx = atom_offset[select]
        C_idx = N_idx + 2
        N_mask = atom14_mask[select, 0]
        C_mask = atom14_mask[select, 2]
        bb_bonds = torch.stack(
            (N_idx[1:], C_idx[:-1])
        )
        bb_bond_mask = N_mask[1:] & C_mask[:-1]
        peptide_bonds.append(bb_bonds[:, bb_bond_mask])
        peptide_bonds.append(bb_bonds[:, bb_bond_mask].flip(0))
    bond_edge_index = torch.cat([bond_edge_index.T] + peptide_bonds, dim=-1)
    peptide_props = torch.as_tensor(peptide_bond_props, device=seq.device)
    peptide_props = peptide_props.expand(
        sum([t.shape[-1] for t in peptide_bonds]),
        -1
    )
    bond_props = torch.cat([bond_props, peptide_props], dim=0)

    # get flat atom props
    atom14_props = atom_props_store[seq]
    atom_props, atom_res_batch, ca_select = atom14_to_atomic(atom14_props, atom14_mask)
    atom_batch = res_batch[atom_res_batch]

    # we remove and renumber edges based on which atoms exist
    # ideal_atom_idx = torch.arange(ideal_atom14_mask.long().sum().item(), device=seq.device).long()
    # atom_idx = ideal_atom_idx[atom14_mask[ideal_atom14_mask]]
    # print(ideal_atom_idx.shape, atom_idx.shape)
    atom_idx = torch.arange(atom14_mask.long().sum().item(), device=seq.device).long()

    # torch.set_printoptions(threshold=10000000)
    # print(atom_idx, bond_edge_index)

    # print(seq)
    # print(ideal_atom14_mask.sum(), atom14_mask.sum())
    # print(atom_idx.shape, atom_props.shape)
    # print(res_batch.shape, atom14_mask[:, 1].sum())

    bond_edge_index, bond_props = pygu.subgraph(
        atom_idx,
        edge_index=bond_edge_index,
        edge_attr=bond_props,
        relabel_nodes=True)


    return {
        "atom_props": atom_props,
        "atom_batch": atom_batch,
        "atom_res_batch": atom_res_batch,
        "ca_select": ca_select,
        "bond_props": bond_props,
        "bond_edge_index": bond_edge_index,
    }

