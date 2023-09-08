""" Utils for handling sidechain atomic representations

Many parts adapted from from https://github.com/deepmind/alphafold/blob/main/alphafold/common/residue_constants.py

TODO: generate all constants that are procedurally generated to outputs of generating functions
"""
import numpy as np

# i'm roughly using https://physlab.lums.edu.pk/images/f/f6/Franck_ref2.pdf values for now
# but i should really figure out which reference to use
van_der_waals_radius = {
    'C': 1.70,
    'N': 1.80,
    'O': 1.50,
    'S': 1.80,
}

alphabet = "ACDEFGHIKLMNPQRSTVWY"
letter_to_num = {letter: alphabet.index(letter) for letter in alphabet}
num_to_letter = {v: k for k,v in letter_to_num.items()}

### Canonical AA constants
restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}


# NB: restype_3to1 differs from Bio.PDB.protein_letters_3to1 by being a simple
# 1-to-1 mapping of 3 letter names to one letter names. The latter contains
# many more, and less common, three letter names as keys and maps many of these
# to the same one letter name (including 'X' and 'U' which we don't use here).
restype_3to1 = {v: k for k, v in restype_1to3.items()}


### Atom quantities


# A list of atoms (excluding hydrogen) for each AA sidechain. PDB naming convention.
sidechain_atoms = {
    'ALA': ['CB'],
    'ARG': ['CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2'],
    'ASN': ['CB', 'CG',  'OD1', 'ND2'],
    'ASP': ['CB', 'CG',  'OD1', 'OD2'],
    'CYS': ['CB', 'SG'],
    'GLN': ['CB', 'CG',  'CD',  'OE1', 'NE2'],
    'GLU': ['CB', 'CG',  'CD',  'OE1', 'OE2'],
    'GLY': [],
    'HIS': ['CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2'],
    'ILE': ['CB', 'CG1', 'CG2', 'CD1'],
    'LEU': ['CB', 'CG',  'CD1', 'CD2'],
    'LYS': ['CB', 'CG',  'CD',  'CE',  'NZ'],
    'MET': ['CB', 'CG',  'SD',  'CE'],
    'PHE': ['CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'PRO': ['CB', 'CG',  'CD'],
    'SER': ['CB', 'OG'],
    'THR': ['CB', 'OG1', 'CG2'],
    'TRP': ['CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH'],
    'VAL': ['CB', 'CG1', 'CG2'],
}


### Bond quantities


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
    'HIS': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'ND1'), ('CG', 'CD2'), ('ND1', 'CE1'), ('CD2', 'NE2')],
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


### Bond angle quantities



sidechain_bond_angles = {
    aa: [] for aa in sidechain_bonds.keys()
}
for aa, store in sidechain_bond_angles.items():
    bonds = sidechain_bonds[aa]
    for b1 in bonds:
        for b2 in bonds:
            if b1[-1] == b2[0]:
                store.append((b1[0], b1[1], b2[1]))
            elif b1[0] == b2[0] and b1[-1] != b2[-1]:
                forward = (b1[1], b1[0], b2[1])
                reverse = (b2[1], b1[0], b1[1])
                if forward not in store and reverse not in store:
                    store.append(forward)
            elif b1[1] == b2[1] and b1[0] != b2[0]:
                forward = (b1[0], b1[1], b2[0])
                reverse = (b2[0], b1[1], b1[0])
                if forward not in store and reverse not in store:
                    store.append(forward)


### Torision quantities


# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
chi_angles_atoms = {
    'ALA': [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'CYS': [['N', 'CA', 'CB', 'SG']],
    'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLY': [],
    'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
            ['CB', 'CG', 'SD', 'CE']],
    'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    'SER': [['N', 'CA', 'CB', 'OG']],
    'THR': [['N', 'CA', 'CB', 'OG1']],
    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'VAL': [['N', 'CA', 'CB', 'CG1']],
}

# The following chi angles are pi periodic: they can be rotated by a multiple
# of pi without affecting the structure.
chi_pi_periodic = [
    [],  # ALA
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0],  # ASN
    [0.0, 1.0],  # ASP
    [0.0],  # CYS
    [0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 1.0],  # GLU
    [],  # GLY
    [0.0, 0.0],  # HIS
    [0.0, 0.0],  # ILE
    [0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0],  # MET
    [0.0, 1.0],  # PHE
    [0.0, 0.0],  # PRO
    [0.0],  # SER
    [0.0],  # THR
    [0.0, 0.0],  # TRP
    [0.0, 1.0],  # TYR
    [0.0],  # VAL
]
for chi_atoms, chi_angles in zip(chi_angles_atoms.values(), chi_pi_periodic):
    assert len(chi_atoms) == len(chi_angles)



# all_torsions = {
#     # if we do all sidechain torsions, we still need the chi1 reference to ensure
#     # everything's in the right reference frame
#     aa: [chis[0]] if len(chis) > 0 else [] for aa, chis in chi_angles_atoms.items()
# }
# for aa, store in all_torsions.items():
#     bonds = sidechain_bonds[aa]
#     for b1, b2, b3 in zip(bonds, bonds, bonds):
#         b1 = tuple(b1)
#         b2 = tuple(b2)
#         b3 = tuple(b3)
#         if len(set((b1, b2, b3))) != 3:
#             continue
#
#         if b1[-1] == b2[0] and b2[0] == b3[-1]:
#             store.append((b1[0], b1[1], b2[1], b3[1]))
#         elif b1[0] == b2[0] and b2[0] == b3[-1]:
#             forward = (b1[1], b1[0], b2[1], b3[1])
#             reverse = (b3[1], b2[1], b1[0], b1[1])
#             if forward not in store and reverse not in store:
#                 store.append(forward)
#         elif b1[1] == b2[1] and b1[0] != b2[0]:
#             forward = (b1[0], b1[1], b2[0])
#             reverse = (b2[0], b1[1], b1[0])
#             if forward not in store and reverse not in store:
#                 store.append(forward)


### Atom37 Quantities


atom37_atom_label = [
    "N", "CA", "C", "O",
    "CB",
    "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
    "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
    "CZ", "CZ2", "CZ3", "NZ",
    "CH2", "NH1", "NH2", "OH"
]


### Atom91 quantities


# atom91 ordering constants and atom masks
atom91_atom_label = ['N', 'C', 'CA', 'O']
atom91_res_order = []
for res, atom_list in sidechain_atoms.items():
    atom91_res_order.append(res)
    atom91_atom_label += atom_list

atom91_carbon_mask = [atom.startswith('C') for atom in atom91_atom_label]
atom91_nitrogen_mask = [atom.startswith('N') for atom in atom91_atom_label]
atom91_oxygen_mask = [atom.startswith('O') for atom in atom91_atom_label]
atom91_sulfur_mask = [atom.startswith('S') for atom in atom91_atom_label]
atom91_atom_masks = {
    "C": atom91_carbon_mask,
    "N": atom91_nitrogen_mask,
    "O": atom91_oxygen_mask,
    "S": atom91_sulfur_mask,
}


atom91_start_end = {}
_start = 4  # first 4 slots are N C CA O
for res, atom_list in sidechain_atoms.items():
    atom91_start_end[res] = (_start, _start + len(atom_list))
    _start += len(atom_list)
assert _start == 91,  f"error in generating atom91 lookup 'atom91_start_end', _start={_start}"


atom91_bonds = {}
for res_3lt, bonds in sidechain_bonds.items():
    bond_idxs = []
    offset = atom91_start_end[res_3lt][0]
    atom_order = sidechain_atoms[res_3lt]
    for (atom1, atom2) in bonds:
        if atom1 == 'CA':  # initial bond
            idx1 = atom91_atom_label.index(atom1)
        else:
            idx1 = atom_order.index(atom1) + offset

        if atom2 == 'N':  # proline
            idx2 = atom91_atom_label.index(atom2)
        else:
            idx2 = atom_order.index(atom2) + offset

        bond_idxs.append((idx1, idx2))
    atom91_bonds[res_3lt] = bond_idxs


def _gen_nonbonded_atom_pairs():
    bonds = []
    for b_list in atom91_bonds.values():
        bonds += [tuple(b) for b in b_list]
    bonds = set(bonds)

    nonbonded_pairs = []
    for start, end in atom91_start_end.values():
        for i in range(start, end):
            for j in range(start, end):
                if i == j:
                    continue
                if (i, j) not in bonds and (j, i) not in bonds:
                    nonbonded_pairs.append((i, j))

    return nonbonded_pairs
nonbonded_sidechain_atom_pairs = _gen_nonbonded_atom_pairs()


atom91_angles = {}
for res_3lt, angles in sidechain_bond_angles.items():
    angle_idxs = []
    offset = atom91_start_end[res_3lt][0]
    atom_order = sidechain_atoms[res_3lt]
    for (atom1, atom2, atom3) in angles:
        if atom1 == 'CA':  # initial bond
            idx1 = atom91_atom_label.index(atom1)
        else:
            idx1 = atom_order.index(atom1) + offset

        idx2 = atom_order.index(atom2) + offset
        if atom3 == 'N':  # proline
            idx3 = atom91_atom_label.index(atom3)
        else:
            idx3 = atom_order.index(atom3) + offset

        angle_idxs.append((idx1, idx2, idx3))
    atom91_angles[res_3lt] = angle_idxs


chi_atom_idxs = {}
_bb_atoms = ['N', 'CA', 'C', 'O']
for res_3lt, chi_atoms_list in chi_angles_atoms.items():
    atom_idxs = []
    offset = atom91_start_end[res_3lt][0]
    atom_order = sidechain_atoms[res_3lt]
    for atoms in chi_atoms_list:
        idxs = []
        for atom in atoms:
            if atom in _bb_atoms:
                idxs.append(_bb_atoms.index(atom))
            else:
                idxs.append(atom_order.index(atom) + offset)

        atom_idxs.append(tuple(idxs))
    chi_atom_idxs[res_3lt] = atom_idxs


### Atom14 quantities


# A compact atom encoding with 14 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_atom14_names = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'UNK': ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
}

atom14_bonds = {}
for res_3lt, bonds in sidechain_bonds.items():
    bond_idxs = []
    atom_order = sidechain_atoms[res_3lt]
    for (atom1, atom2) in bonds:
        if atom1 == 'CA':  # initial bond
            idx1 = 1
        else:
            idx1 = atom_order.index(atom1) + 4

        if atom2 == 'N':  # proline
            idx2 = 0
        else:
            idx2 = atom_order.index(atom2) + 4

        bond_idxs.append((idx1, idx2))
    atom14_bonds[res_3lt] = bond_idxs


### Conversion functions


def atom14_to_atom91(seq, atom14):
    """ Convert a tensor from atom14 representation to atom91 representation

    Args
    ----
    seq: str (len n_res)
        sequence of AAs
    atom14: np.ndarray (n_res x 14 x 3)
        coordinates of residue atoms in atom14 form

    Returns
    -------
    atom91: np.ndarray (n_res x 91 x 3)
        coordinates of residue atoms in atom91 form
    """
    n_res = len(seq)
    atom91 = np.full((n_res, 91, 3), np.nan)
    atom91[:, :4] = atom14[:, :4]  # add N CA C O
    for i, res_1lt in enumerate(seq):
        res_3lt = restype_1to3[res_1lt]
        sidechain_len = len(sidechain_atoms[res_3lt])
        sidechain = atom14[i, 4:4+sidechain_len]
        atom91_start, atom91_end = atom91_start_end[res_3lt]
        atom91[i, atom91_start:atom91_end] = sidechain

    atom91_mask = np.isnan(atom91)
    # atom91[atom91_mask] = 0.

    return atom91, atom91_mask


def atom37_to_atom91(seq, atom37):
    """ Convert a tensor from atom37 representation to atom91 representation

    Args
    ----
    seq: str (len n_res)
        sequence of AAs
    atom14: np.ndarray (n_res x 37 x 3)
        coordinates of residue atoms in atom37 form

    Returns
    -------
    atom91: np.ndarray (n_res x 91 x 3)
        coordinates of residue atoms in atom91 form
    """
    n_res = len(seq)
    atom91 = np.full((n_res, 91, 3), np.nan)
    atom91[:, :4] = atom37[:, :4]  # add N CA C O
    for i, res_1lt in enumerate(seq):
        res_3lt = restype_1to3[res_1lt]
        atom91_atoms = sidechain_atoms[res_3lt]
        atom91_start, atom91_end = atom91_start_end[res_3lt]
        atom91_loc_in_atom37 = [atom37_atom_label.index(atom) for atom in atom91_atoms]
        sidechain = atom37[i, atom91_loc_in_atom37]
        atom91[i, atom91_start:atom91_end] = sidechain

    atom91_mask = np.isnan(atom91)
    # atom91[atom91_mask] = 0.

    return atom91, atom91_mask

def atom37_to_atom14(seq, atom37):
    """ Convert a tensor from atom37 representation to atom91 representation

    Args
    ----
    seq: str (len n_res)
        sequence of AAs
    atom14: np.ndarray (n_res x 37 x 3)
        coordinates of residue atoms in atom37 form

    Returns
    -------
    atom14: np.ndarray (n_res x 14 x 3)
        coordinates of residue atoms in atom91 form
    """
    n_res = len(seq)
    atom14 = np.full((n_res, 14, 3), np.nan)
    atom14[:, :4] = atom37[:, :4]  # add N CA C O
    for i, res_1lt in enumerate(seq):
        res_3lt = restype_1to3[res_1lt]
        atom14_atoms = sidechain_atoms[res_3lt]
        atom14_loc_in_atom37 = [atom37_atom_label.index(atom) for atom in atom14_atoms]
        sidechain = atom37[i, atom14_loc_in_atom37]
        atom14[i, 4:4+len(atom14_atoms)] = sidechain

    atom14_mask = np.isnan(atom14)
    # atom14[atom14_mask] = 0.

    return atom14, atom14_mask
