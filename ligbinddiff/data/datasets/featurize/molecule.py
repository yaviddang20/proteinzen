from typing import Dict

import numpy as np
import torch
import tree

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdchem import HybridizationType

from torch_geometric.data import HeteroData
import torch_geometric.utils as pygu

## data constants
element_to_period_row = {
    'H' : (1, 1),
    'Li': (2, 2),
    'B' : (13, 2),
    'C' : (14, 2),
    'N' : (15, 2),
    'O' : (16, 2),
    'F' : (17, 2),
    'Na': (1, 3),
    'Mg': (2, 3),
    'Al': (13, 3),
    'Si': (14, 3),
    'P' : (15, 3),
    'S' : (16, 3),
    'Cl': (17, 3),
    'K' : (1, 4),
    'Ca': (2, 4),
    'V' : (5, 4),
    'Cr': (6, 4),
    'Mn': (7, 4),
    'Cu': (11, 4),
    'Zn': (12, 4),
    'Ga': (13, 4),
    'Ge': (14, 4),
    'As': (15, 4),
    'Se': (16, 4),
    'Br': (17, 4),
    'Ag': (11, 5),
    'In': (13, 5),
    'Sb': (15, 5),
    'I' : (17, 5),
    #'Gd': (,),
    'Pt': (10, 6),
    'Au': (11, 6),
    'Hg': (12, 6),
    'Bi': (15, 6),
}
periods, rows = zip(*element_to_period_row.values())
unique_periods, unique_rows = sorted(set(periods)), sorted(set(rows))

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}

hybridization = {
    HybridizationType.SP: 0,
    HybridizationType.SP2: 1,
    HybridizationType.SP3: 2,
    HybridizationType.SP3D: 3,
    HybridizationType.SP3D2: 4,
}

max_num_bonds = 6
ring_size_min = 3
ring_size_max = 8
num_rings_max = 3
max_abs_formal_charge = 5
max_bond_order = 3


# ordered (featurization, max_value)
prop_featurization = {
    "atom_period": ("categorical", max(unique_periods)),  # 13
    "atom_row": ("categorical", max(unique_rows)),  # 6
    "atom_chirality": ("ordinal", 1),  # 1
    "atom_hybridization": ("categorical", len(hybridization)),  # 5
    "atom_implicit_hs": ("ordinal", max_num_bonds),  # 1
    "atom_formal_charge": ("ordinal", 5),  # 1
    "atom_aromatic": None,  # 1
    "atom_ring_props": None,  # 7
    "atom_degree": ("ordinal", max_num_bonds),  # 1
    "bond_order": ("ordinal", max_bond_order),  # 1
    "bond_aromatic": None,  # 1
    "bond_type": ("categorical", max(bonds.values())),  # 4
    "bond_conjugated": None,  # 1
}


# helper functions
def safe_lookup(val, store, default):
    if val in store:
        return store[val]
    else:
        return default


def bucket(val, options, default=None):
    if default is None:
        default = len(options)
    if val in options:
        return options.index(val)
    else:
        return default


# properties
def conformer_props(conformer, implicit_H=True):
    # if implicit_H is enabled, we make all Hs implicit in the structure
    # to simulate PDB structures with no H info
    # and also we can save ourselves some compute
    # we remove Hs from the graph but store implicit Hs (# of Hs per heavy atom)
    # by adding the already-implicit valence of the heavy atom with the number
    # of explicit Hs already attached
    # we then just skip over all H atom features and any bonds with Hs

    mol = conformer.GetOwningMol()
    ringinfo = mol.GetRingInfo()
    # atom features
    atom_period = []
    atom_row = []
    atom_chirality = []
    atom_hybridization = []
    atom_implicit_Hs = []
    atom_formal_charge = []
    atom_aromatic = []
    atom_ring_props = []
    atom_degree = []

    h_idx = []

    for idx, atom in enumerate(mol.GetAtoms()):
        element = atom.GetSymbol()

        if implicit_H and element == 'H':
            h_idx.append(idx)
            continue

        period, row = element_to_period_row[element]
        # element embedding
        atom_period.append(unique_periods.index(period))
        atom_row.append(unique_rows.index(row))
        # chirality
        atom_chirality.append(
            safe_lookup(atom.GetChiralTag(), chirality, default=0)
        )
        # hybridization
        atom_hybridization.append(
            safe_lookup(atom.GetHybridization(), hybridization,
                        default=len(hybridization))
        )
        # implicit valence
        num_Hs = atom.GetImplicitValence() + atom.GetNumExplicitHs()
        atom_implicit_Hs.append(
            bucket(num_Hs, list(range(max_num_bonds+1)))
        )
        atom_degree.append(
            bucket(atom.GetDegree(), list(range(max_num_bonds+1)))
        )
        atom_formal_charge.append(atom.GetFormalCharge())
        atom_aromatic.append(int(atom.GetIsAromatic()))
        atom_ring_props.append(
            [
                int(ringinfo.IsAtomInRingOfSize(idx, ring_size))
                for ring_size in range(ring_size_min, ring_size_max+1)
            ] # if atom is in a ring of size X
            + [bucket(ringinfo.NumAtomRings(idx), list(range(num_rings_max+1)))] # how many rings is the atom in
        )

    # get 3D coords, removing Hs
    atom_pos = conformer.GetPositions()
    atom_idx_map = np.arange(mol.GetNumAtoms())
    if implicit_H:
        mol_H_mask = np.array(
            [atom.GetSymbol() != 'H' for atom in mol.GetAtoms()]
        )
        atom_idx_map = atom_idx_map[mol_H_mask]
        atom_pos = atom_pos[mol_H_mask]

    # bond features
    bond_edge_index = []
    bond_order = []
    bond_aromatic = []
    bond_type = []
    bond_conjugated = []
    for bond in mol.GetBonds():
        src_idx = bond.GetBeginAtomIdx()
        dst_idx = bond.GetEndAtomIdx()


        if implicit_H and (src_idx in h_idx or dst_idx in h_idx):
            continue
        elif implicit_H:
            src_idx = np.nonzero(atom_idx_map == src_idx)[0][0]
            dst_idx = np.nonzero(atom_idx_map == dst_idx)[0][0]

        # undirected graph
        bond_edge_index.append([src_idx, dst_idx])
        bond_edge_index.append([dst_idx, src_idx])
        # do everything twice bc undirected graph
        for _ in range(2):
            bond_order.append(bond.GetBondTypeAsDouble())
            bond_aromatic.append(bond.GetIsAromatic())
            bond_conjugated.append(bond.GetIsConjugated())
            bond_type.append(
                safe_lookup(
                    bond.GetBondType(),
                    bonds,
                    default=0
                )
            )


    props = {
        "atom_pos": atom_pos,                       # N x 3
        "atom_period": atom_period,                 # N, categorical
        "atom_row": atom_row,                       # N, categorical
        "atom_chirality": atom_chirality,           # N, ordinal
        "atom_hybridization": atom_hybridization,   # N, categorical
        "atom_implicit_hs": atom_implicit_Hs,       # N, ordinal
        "atom_formal_charge": atom_formal_charge,   # N, ordinal
        "atom_aromatic": atom_aromatic,             # N, bool
        "atom_ring_props": atom_ring_props,         # N x 7, 1-6 bool, 7 ordinal
        "atom_degree": atom_degree,                 # N, ordinal
        "bond_edge_index": bond_edge_index,         # E x 2
        "bond_order": bond_order,                   # E, ordinal
        "bond_aromatic": bond_aromatic,             # E, bool
        "bond_type": bond_type,                     # E, categorical
        "bond_conjugated": bond_conjugated,         # E, bool
    }
    props = {
        k: np.array(v)
        for k, v in props.items()
    }

    return props

def one_hot(cat_feats, max_val):
    one_hot_mat = torch.eye(max_val+1)
    return one_hot_mat[cat_feats]

def featurize_props(mol_props: Dict, center=True):
    mol_props = tree.map_structure(
        torch.as_tensor,
        mol_props
    )

    data = HeteroData()
    atom_pos = mol_props['atom_pos']
    atom_pos = atom_pos - atom_pos.mean(dim=0)[None]
    data["ligand"].x = atom_pos
    data["ligand"].atom_pos = atom_pos
    data["ligand"].atom_mask = torch.ones(atom_pos.shape[0]).bool()
    data["ligand", "ligand"].edge_index = mol_props["bond_edge_index"].T

    for prop, prop_type in prop_featurization.items():
        if "atom" in prop:
            if prop_type is None:
                data["ligand"][prop] = mol_props[prop].float()
            elif prop_type[0] == "categorical":
                max_val = prop_type[1]
                data["ligand"][prop] = one_hot(mol_props[prop], max_val).float()
            elif prop_type[0] == "ordinal":
                max_val = prop_type[1]
                data["ligand"][prop] = (mol_props[prop] / max_val).float()
        elif "bond" in prop:
            if prop_type is None:
                data["ligand", "ligand"][prop] = mol_props[prop].float()
            elif prop_type[0] == "categorical":
                max_val = prop_type[1]
                data["ligand", "ligand"][prop] = one_hot(mol_props[prop], max_val).float()
            elif prop_type[0] == "ordinal":
                max_val = prop_type[1]
                data["ligand", "ligand"][prop] = (mol_props[prop] / max_val).float()

    return data

