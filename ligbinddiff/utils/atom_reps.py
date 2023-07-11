""" Utils for handling sidechain atomic representations """
import numpy as np

alphabet = "ACDEFGHIKLMNPQRSTVWY"
letter_to_num = {letter: alphabet.index(letter) for letter in alphabet}

### Canonical AA constants
### many parts adapted from from https://github.com/deepmind/alphafold/blob/main/alphafold/common/residue_constants.py
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


atom37_atom_label = [
    "N", "CA", "C", "O",
    "CB",
    "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
    "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
    "CZ", "CZ2", "CZ3", "NZ",
    "CH2", "NH1", "NH2", "OH"
]


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
