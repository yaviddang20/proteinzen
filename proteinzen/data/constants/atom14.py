import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from proteinzen.data.openfold.residue_constants import restype_name_to_atom14_names, restype_order, restype_1to3
from proteinzen.data.datasets.featurize.molecule import featurize_props, mol_props, prop_featurization


### taken from diffdock https://github.com/gcorso/DiffDock/blob/main/datasets/constants.py

amino_acid_smiles = {
    'PHE': '[NH3+]CC(=O)N[C@@H](Cc1ccccc1)C(=O)NCC(=O)O',
    'MET': 'CSCC[C@H](NC(=O)C[NH3+])C(=O)NCC(=O)O',
    'TYR': '[NH3+]CC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)NCC(=O)O',
    'ILE': 'CC[C@H](C)[C@H](NC(=O)C[NH3+])C(=O)NCC(=O)O',
    'TRP': '[NH3+]CC(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)NCC(=O)O',
    'THR': 'C[C@@H](O)[C@H](NC(=O)C[NH3+])C(=O)NCC(=O)O',
    'CYS': '[NH3+]CC(=O)N[C@@H](CS)C(=O)NCC(=O)O',
    'ALA': 'C[C@H](NC(=O)C[NH3+])C(=O)NCC(=O)O',
    'LYS': '[NH3+]CCCC[C@H](NC(=O)C[NH3+])C(=O)NCC(=O)O',
    'PRO': '[NH3+]CC(=O)N1CCC[C@H]1C(=O)NCC(=O)O',
    'LEU': 'CC(C)C[C@H](NC(=O)C[NH3+])C(=O)NCC(=O)O',
    'GLY': '[NH3+]CC(=O)NCC(=O)NCC(=O)O',
    'ASP': '[NH3+]CC(=O)N[C@@H](CC(=O)O)C(=O)NCC(=O)O',
    'HIS': '[NH3+]CC(=O)N[C@@H](Cc1c[nH]c[nH+]1)C(=O)NCC(=O)O',
    'VAL': 'CC(C)[C@H](NC(=O)C[NH3+])C(=O)NCC(=O)O',
    'SER': '[NH3+]CC(=O)N[C@@H](CO)C(=O)NCC(=O)O',
    'ARG': 'NC(=[NH2+])NCCC[C@H](NC(=O)C[NH3+])C(=O)NCC(=O)O',
    'GLU': '[NH3+]CC(=O)N[C@@H](CCC(=O)O)C(=O)NCC(=O)O',
    'GLN': 'NC(=O)CC[C@H](NC(=O)C[NH3+])C(=O)NCC(=O)O',
    'ASN': 'NC(=O)C[C@H](NC(=O)C[NH3+])C(=O)NCC(=O)O',
 }

cg_rdkit_indices = {
    'PHE': {4: 'N', 5: 'CA', 13: 'C', 14: 'O', 6: 'CB', 7: 'CG', 8: 'CD1', 12: 'CD2', 9: 'CE1', 11: 'CE2', 10: 'CZ'},
    'MET': {5: 'N', 4: 'CA', 10: 'C', 11: 'O', 3: 'CB', 2: 'CG', 1: 'SD', 0: 'CE'},
    'TYR': {4: 'N', 5: 'CA', 14: 'C', 15: 'O', 6: 'CB', 7: 'CG', 8: 'CD1', 13: 'CD2', 9: 'CE1', 12: 'CE2', 10: 'CZ', 11: 'OH'},
    'ILE': {5: 'N', 4: 'CA', 10: 'C', 11: 'O', 2: 'CB', 1: 'CG1', 3: 'CG2', 0: 'CD1'},
    'TRP': {4: 'N', 5: 'CA', 16: 'C', 17: 'O', 6: 'CB', 7: 'CG', 8: 'CD1', 15: 'CD2', 9: 'NE1', 10: 'CE2', 14: 'CE3', 11: 'CZ2', 13: 'CZ3', 12: 'CH2'},
    'THR': {4: 'N', 3: 'CA', 9: 'C', 10: 'O', 1: 'CB', 2: 'OG1', 0: 'CG2'},
    'CYS': {4: 'N', 5: 'CA', 8: 'C', 9: 'O', 6: 'CB', 7: 'SG'},
    'ALA': {2: 'N', 1: 'CA', 7: 'C', 8: 'O', 0: 'CB'},
    'LYS': {6: 'N', 5: 'CA', 11: 'C', 12: 'O', 4: 'CB', 3: 'CG', 2: 'CD', 1: 'CE', 0: 'NZ'},
    'PRO': {4: 'N', 8: 'CA', 9: 'C', 10: 'O', 7: 'CB', 6: 'CG', 5: 'CD'},
    'LEU': {5: 'N', 4: 'CA', 10: 'C', 11: 'O', 3: 'CB', 1: 'CG', 0: 'CD1', 2: 'CD2'},
    'GLY': {4: 'N', 5: 'CA', 6: 'C', 7: 'O'},
    'ASP': {4: 'N', 5: 'CA', 10: 'C', 11: 'O', 6: 'CB', 7: 'CG', 8: 'OD1', 9: 'OD2'},
    'HIS': {4: 'N', 5: 'CA', 12: 'C', 13: 'O', 6: 'CB', 7: 'CG', 11: 'ND1', 8: 'CD2', 10: 'CE1', 9: 'NE2'},
    'VAL': {4: 'N', 3: 'CA', 9: 'C', 10: 'O', 1: 'CB', 0: 'CG1', 2: 'CG2'},
    'SER': {4: 'N', 5: 'CA', 8: 'C', 9: 'O', 6: 'CB', 7: 'OG'},
    'ARG': {8: 'N', 7: 'CA', 13: 'C', 14: 'O', 6: 'CB', 5: 'CG', 4: 'CD', 3: 'NE', 1: 'CZ', 0: 'NH1', 2: 'NH2'},
    'GLU': {4: 'N', 5: 'CA', 11: 'C', 12: 'O', 6: 'CB', 7: 'CG', 8: 'CD', 9: 'OE1', 10: 'OE2'},
    'GLN': {6: 'N', 5: 'CA', 11: 'C', 12: 'O', 4: 'CB', 3: 'CG', 1: 'CD', 2: 'OE1', 0: 'NE2'},
    'ASN': {5: 'N', 4: 'CA', 10: 'C', 11: 'O', 3: 'CB', 1: 'CG', 2: 'OD1', 0: 'ND2'}
}


# Backbond bonds, in form (src, dst)

backbone_bonds = [
    ('CA', 'N'),
    ('CA', 'C'),
    ('C',  'O')
]

# Bonds in sidechains, in form (src, dst)
sidechain_bonds = {
    'ALA': [('CA', 'CB')],
    'ARG': [
        ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'NE'),
        ('NE', 'CZ'), ('CZ', 'NH1'), ('CZ', 'NH2')
    ],
    'ASN': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'ND2')],
    'ASP': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'OD2')],
    'CYS': [('CA', 'CB'), ('CB', 'SG')],
    'GLN': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CG', 'NE2')],
    'GLU': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CG', 'OE2')],
    'GLY': [],
    'HIS': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'ND1'), ('CG', 'CD2'), ('ND1', 'CE1'), ('CD2', 'NE2'), ('CE1', 'NE2')],
    'ILE': [('CA', 'CB'), ('CB', 'CG1'), ('CB', 'CG2'), ('CG1', 'CD1')],
    'LEU': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2')],
    'LYS': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'CE'), ('CE', 'NZ')],
    'MET': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'SD'), ('SD', 'CE')],
    'PHE': [
        ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'),
        ('CD1', 'CE1'), ('CD2', 'CE2'), ('CE1', 'CZ'), ('CE2', 'CZ')
    ],
    'PRO': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'N')],
    'SER': [('CA', 'CB'), ('CB', 'OG')],
    'THR': [('CA', 'CB'), ('CB', 'OG1'), ('CB', 'CG2')],
    'TRP': [
        ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'),
        ('CD1', 'NE1'), ('CD2', 'CE2'), ('NE1', 'CE2'),
        ('CD2', 'CE3'), ('CE2', 'CZ2'), ('CE3', 'CZ3'),
        ('CZ2', 'CH2'), ('CZ3', 'CH2')
    ],
    'TYR': [
        ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'),
        ('CD1', 'CE1'), ('CD2', 'CE2'), ('CE1', 'CZ'), ('CE2', 'CZ'),
        ('CZ', 'OH')
    ],
    'VAL': [('CA', 'CB'), ('CB', 'CG1'), ('CB', 'CG2')],
}

residue_bonds = {
    aa: backbone_bonds + bonds
    for aa, bonds in sidechain_bonds.items()
}

# stored as ((atom1_idx, atom2_idx), bond_order,
atom14_backbone_bonds = [
    (1, 0), # ('CA', 'N'),
    (1, 2), # ('CA', 'C'),
    (2, 3), # ('C',  'O')
]

atom14_sidechain_bonds = {}
for res_3lt, bonds in sidechain_bonds.items():
    bond_idxs = []
    atom_order = restype_name_to_atom14_names[res_3lt]
    for (atom1, atom2) in bonds:
        idx1 = atom_order.index(atom1)
        idx2 = atom_order.index(atom2)
        bond_idxs.append((idx1, idx2))
    atom14_sidechain_bonds[res_3lt] = bond_idxs


def _gen_props_stores(atom_keys, bond_keys):
    atom14_idx_to_rdkit_idx = {}
    for res_3lt, rdkit_mapping in cg_rdkit_indices.items():
        mapping = {}
        atom14_order = restype_name_to_atom14_names[res_3lt]
        for rdkit_idx, atom_name in rdkit_mapping.items():
            atom14_idx = atom14_order.index(atom_name)
            mapping[atom14_idx] = rdkit_idx
        atom14_idx_to_rdkit_idx[res_3lt] = mapping

    atom_props_list = []
    bond_props_list = []
    bond_props_mask = np.zeros(
        (
            len(restype_order),
            max([len(bonds) for bonds in residue_bonds.values()]) * 2,
        ),
        dtype=bool
    )
    bond_edge_index_store = np.zeros(
        (
            len(restype_order),
            max([len(bonds) for bonds in residue_bonds.values()]) * 2,
            2
        ),
        dtype=np.int64
    )

    # we use these to compute the peptide bond features, since we can't assign peptide bonds
    # until we have the actual data
    # this is not very clean but im lazy, so taking this from the glycine iteration
    peptide_bond_idx = None
    peptide_bond_features = None

    for i, res_1lt in enumerate(restype_order):
        res_3lt = restype_1to3[res_1lt]
        res_smiles = amino_acid_smiles[res_3lt]
        res_mol = Chem.MolFromSmiles(res_smiles)
        props = mol_props(res_mol, implicit_H=False)
        # add a dummy conformation for compatibility with library code
        props['atom_pos'] = np.zeros((res_mol.GetNumAtoms(), 3))
        pyg_graph = featurize_props(props)
        res_atom14_to_rdkit_idx = atom14_idx_to_rdkit_idx[res_3lt]
        gather_atoms = [i for (i, _) in sorted(res_atom14_to_rdkit_idx.items(), key=lambda x: x[0])]

        atom_features = [
            pyg_graph['ligand'][key].numpy(force=True)
            for key in atom_keys
        ]
        atom_features = [
            arr[..., None] if len(arr.shape) == 1 else arr
            for arr in atom_features
        ]
        atom_features = np.concatenate(atom_features, axis=-1)
        atom_features = atom_features[gather_atoms]
        atom_props_list.append(atom_features)

        bond_edge_index = pyg_graph["ligand", "ligand"].edge_index.T.tolist()
        res_rdkit_idx_to_atom14_idx = {v: k for k,v in res_atom14_to_rdkit_idx.items()}
        atom14_edge_index = []
        gather_bonds = []
        for bond_idx, (atom1_idx, atom2_idx) in enumerate(bond_edge_index):
            # extrac peptide bond idx if looking at gly
            if res_3lt == 'GLY' and (atom1_idx, atom2_idx) in [(2, 4), (4,2)]:
                peptide_bond_idx = bond_idx

            if atom1_idx not in res_rdkit_idx_to_atom14_idx or atom2_idx not in res_rdkit_idx_to_atom14_idx:
                continue

            atom1_new_idx = res_rdkit_idx_to_atom14_idx[atom1_idx]
            atom2_new_idx = res_rdkit_idx_to_atom14_idx[atom2_idx]
            atom14_edge_index.append([atom1_new_idx, atom2_new_idx])
            gather_bonds.append(bond_idx)
        assert len(atom14_edge_index) == len(residue_bonds[res_3lt])*2, [len(atom14_edge_index), len(residue_bonds[res_3lt]), res_3lt]

        bond_features = [
            pyg_graph['ligand', 'ligand'][key].numpy(force=True)
            for key in bond_keys
        ]
        bond_features = [
            arr[..., None] if len(arr.shape) == 1 else arr
            for arr in bond_features
        ]
        bond_features = np.concatenate(bond_features, axis=-1)

        # capture the peptide bond features
        if peptide_bond_features is None:
            peptide_bond_features = bond_features[restype_order["G"], peptide_bond_idx]

        bond_features = bond_features[gather_bonds]
        bond_props_list.append(bond_features)
        bond_props_mask[i, :len(gather_bonds)] = True
        bond_edge_index_store[i, :len(gather_bonds)] = np.array(atom14_edge_index)


    num_atom_features = max([t.shape[-1] for t in atom_props_list])
    num_bond_features = max([t.shape[-1] for t in bond_props_list])
    atom_props_store = np.zeros((len(restype_order), 14, num_atom_features))
    bond_props_store = np.zeros(
        (
            len(restype_order),
            max([len(bonds) for bonds in residue_bonds.values()])*2,
            num_bond_features
        )
    )
    for i, _ in enumerate(restype_order):
        atom_features = atom_props_list[i]
        bond_features = bond_props_list[i]
        atom_props_store[i, :atom_features.shape[0]] = atom_features
        bond_props_store[i, :bond_features.shape[0]] = bond_features

    return atom_props_store, bond_props_store, bond_edge_index_store, bond_props_mask, peptide_bond_features

atom14_atom_props, atom14_bond_props, atom14_bond_edge_indicies, atom14_bond_mask, peptide_bond_props = _gen_props_stores(
    atom_keys=[k for k in prop_featurization if k.startswith("atom")],
    bond_keys=[k for k in prop_featurization if k.startswith("bond")]
)