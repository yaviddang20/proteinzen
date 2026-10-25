"""Per-atom/per-residue solvent-accessible surface area (SASA) and relative
SASA (RASA) burial-state bins, computed directly from a Structure's atom
array (no PDB round-trip) via freesasa, mirroring the approach already used
in _scripts/compare_crops.py::_calc_ligand_sasa.

Bin thresholds follow RFD3's Supplementary Methods: buried <=0.06, partially
buried <=0.2, exposed >0.2, unassigned when no data is available.
"""
import traceback

import numpy as np

from proteinzen.boltz.data import const
from proteinzen.boltz.data.types import Structure

RASA_UNASSIGNED = 0
RASA_BURIED = 1
RASA_PARTIAL = 2
RASA_EXPOSED = 3

BURIED_MAX = 0.06
PARTIAL_MAX = 0.2

# Tien et al. 2013, theoretical per-residue max accessible surface area (Angstrom^2)
RESIDUE_MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

# single-char chain ID pool for freesasa's PDB-style chain label field
_CHAIN_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _decode_atom_name(raw) -> str:
    return ''.join(chr(int(b) + 32) for b in raw if b != 0).strip()


def compute_atom_sasa(struct: Structure) -> np.ndarray:
    """Per-atom raw SASA (Angstrom^2) via freesasa, built directly from
    struct.atoms/residues/chains. NaN for atoms where is_present is False,
    or for every atom if freesasa is unavailable or the calculation fails.
    """
    n_atoms = len(struct.atoms)
    sasa = np.full(n_atoms, np.nan, dtype=np.float32)

    try:
        import freesasa
        freesasa.setVerbosity(freesasa.silent)
    except ImportError:
        return sasa

    fs = freesasa.Structure()
    inserted_atom_idx = []  # freesasa insertion order -> original atom index

    for chain_idx, chain in enumerate(struct.chains):
        chain_char = _CHAIN_POOL[chain_idx % len(_CHAIN_POOL)]
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        for res_idx in range(res_start, res_end):
            res = struct.residues[res_idx]
            res_name = str(res["name"]).strip()[:3]
            res_num = str(res_idx % 9999)
            atom_start = int(res["atom_idx"])
            atom_end = atom_start + int(res["atom_num"])
            for atom_idx in range(atom_start, atom_end):
                atom = struct.atoms[atom_idx]
                if not atom["is_present"]:
                    continue
                name = _decode_atom_name(atom["name"])
                if not name:
                    continue
                x, y, z = (float(c) for c in atom["coords"])
                try:
                    fs.addAtom(name, res_name, res_num, chain_char, x, y, z)
                except Exception:
                    continue
                inserted_atom_idx.append(atom_idx)

    if fs.nAtoms() == 0:
        return sasa

    try:
        result = freesasa.calc(fs)
    except Exception:
        traceback.print_exc()
        return sasa

    for i, atom_idx in enumerate(inserted_atom_idx):
        sasa[atom_idx] = result.atomArea(i)

    return sasa


def rasa_bin_from_fraction(rasa: np.ndarray) -> np.ndarray:
    """NaN/missing -> 0 (unassigned); <=BURIED_MAX -> 1 (buried);
    <=PARTIAL_MAX -> 2 (partial); >PARTIAL_MAX -> 3 (exposed)."""
    bins = np.zeros(rasa.shape, dtype=np.int8)
    valid = np.isfinite(rasa)
    bins[valid & (rasa <= BURIED_MAX)] = RASA_BURIED
    bins[valid & (rasa > BURIED_MAX) & (rasa <= PARTIAL_MAX)] = RASA_PARTIAL
    bins[valid & (rasa > PARTIAL_MAX)] = RASA_EXPOSED
    return bins


def compute_residue_rasa_bins(struct: Structure, atom_sasa: np.ndarray) -> np.ndarray:
    """Per-STANDARD-residue RASA bin, shape (n_residues,). Sums atom_sasa over
    each residue's atoms and divides by that residue's tabulated max ASA;
    residues not in RESIDUE_MAX_ASA (non-standard/ligand residues) are left
    unassigned -- use compute_atom_rasa_bins for those instead.
    """
    n_res = len(struct.residues)
    rasa = np.full(n_res, np.nan, dtype=np.float32)
    for i, res in enumerate(struct.residues):
        max_asa = RESIDUE_MAX_ASA.get(str(res["name"]).strip())
        if max_asa is None:
            continue
        atom_start = int(res["atom_idx"])
        atom_end = atom_start + int(res["atom_num"])
        res_sasa = atom_sasa[atom_start:atom_end]
        if not np.isfinite(res_sasa).any():
            continue
        rasa[i] = np.nansum(res_sasa) / max_asa
    return rasa_bin_from_fraction(rasa)


def compute_atom_rasa_bins(struct: Structure, atom_sasa: np.ndarray) -> np.ndarray:
    """Per-ATOM RASA bin, shape (n_atoms,) -- for ligand/atomized tokens,
    which have no residue-level reference max ASA. Normalizes each atom's
    SASA by its own isolated-sphere max (4*pi*r^2) from its VdW radius.
    """
    radii = np.asarray(const.vdw_radii, dtype=np.float32)
    elements = struct.atoms["element"].astype(np.int64)
    valid_element = (elements >= 1) & (elements <= len(radii))
    r = np.where(valid_element, radii[np.clip(elements - 1, 0, len(radii) - 1)], np.nan)
    max_sasa = 4.0 * np.pi * r ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        rasa = atom_sasa / max_sasa
    return rasa_bin_from_fraction(rasa)
