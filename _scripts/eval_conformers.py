#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Imports
# ============================================================

import argparse
import glob
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import sys
from collections import defaultdict
from itertools import product as iproduct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign, rdmolfiles, rdMolTransforms
from rdkit.Chem import rdchem, rdmolops
from tqdm.auto import tqdm
from xtb.interface import Calculator, Param

RDLogger.DisableLog('rdApp.*')

# ============================================================
# Constants
# ============================================================

ANGSTROM_TO_BOHR   = 1.8897259886
HARTREE_TO_KCALMOL = 627.509474
HARTREE_TO_EV      = 27.2114
XTB_BIN            = "xtb"

# ============================================================
# I/O helpers
# ============================================================

class _Tee:
    """Write to multiple file-like objects simultaneously (e.g. stdout + log file)."""
    def __init__(self, *files):
        self._files = files
    def write(self, data):
        for f in self._files:
            f.write(data)
    def flush(self):
        for f in self._files:
            f.flush()


def _clean_pdb_block(block):
    """Strip END and CONECT records from a MolToPDBBlock — keeps multi-MODEL files clean."""
    lines = block.splitlines(keepends=True)
    return "".join(l for l in lines if not l.startswith(("END", "CONECT")))


def save_best_pair(m1, m2, out_path):
    m1 = Chem.RemoveHs(m1)
    m2 = Chem.RemoveHs(m2)
    with open(out_path, "w") as f:
        f.write("MODEL        1\n")
        f.write(_clean_pdb_block(rdmolfiles.MolToPDBBlock(m1, flavor=4)))
        f.write("ENDMDL\n")
        f.write("MODEL        2\n")
        f.write(_clean_pdb_block(rdmolfiles.MolToPDBBlock(m2, flavor=4)))
        f.write("ENDMDL\n")
        f.write("END\n")

# ============================================================
# Molecule key / matching
# ============================================================

def mol_to_match_key(mol, stereo=False):
    """
    Canonical SMILES key for matching.
    stereo=False: strips stereo and charges — connectivity-only (loose fallback).
    stereo=True:  preserves stereo and charges — exact match tier.
    """
    mol = Chem.RemoveHs(mol)
    if not stereo:
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            atom.SetFormalCharge(0)
        try:
            Chem.SanitizeMol(rw)
        except Exception:
            try:
                Chem.SanitizeMol(rw, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception:
                pass
        mol = rw.GetMol()
    else:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
    return Chem.MolToSmiles(mol, isomericSmiles=stereo, canonical=True)


def canon_smi_from_mol(mol, stereo=True):
    """Canonical SMILES for display — same neutralization as matching key."""
    return mol_to_match_key(mol, stereo=stereo)


def load_pdb(path, perceive_stereo=False):
    mol = Chem.MolFromPDBFile(path, removeHs=True, sanitize=False)
    if mol is None:
        return None
    if not perceive_stereo:
        for atom in mol.GetAtoms():
            atom.SetChiralTag(rdchem.ChiralType.CHI_UNSPECIFIED)
        for bond in mol.GetBonds():
            bond.SetStereo(rdchem.BondStereo.STEREONONE)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if perceive_stereo:
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    return mol


def _parse_remark_smiles(pdb_path):
    """Return the SMILES string from a REMARK SMILES line, or None if absent."""
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("REMARK SMILES "):
                return line.split("REMARK SMILES ", 1)[1].strip()
            if line.startswith("ATOM") or line.startswith("HETATM"):
                break  # no REMARK before coordinate records
    return None


def _apply_smiles_template(mol, smiles):
    """Use AssignBondOrdersFromTemplate to transfer bond orders and formal
    charges from the SMILES template onto the PDB-loaded mol.
    Returns the fixed mol, or the original mol on failure."""
    try:
        template = AllChem.MolFromSmiles(smiles)
        if template is None:
            return mol
        template = Chem.RemoveHs(template)
        mol_noH = Chem.RemoveHs(mol)
        fixed = AllChem.AssignBondOrdersFromTemplate(template, mol_noH)
        fixed = Chem.AddHs(fixed, addCoords=True)
        return fixed
    except Exception:
        return mol


def load_and_prepare(pdb_path):
    smiles = _parse_remark_smiles(pdb_path)

    mol = load_pdb(pdb_path, perceive_stereo=False)
    if mol is None:
        return None, None, None, None, None
    if smiles is not None:
        mol = _apply_smiles_template(mol, smiles)
    else:
        mol = Chem.AddHs(mol)

    display_smi_no_stereo = canon_smi_from_mol(mol, stereo=False)
    match_no_stereo = mol_to_match_key(mol, stereo=False)

    mol_stereo = load_pdb(pdb_path, perceive_stereo=True)
    if mol_stereo is not None:
        if smiles is not None:
            mol_stereo = _apply_smiles_template(mol_stereo, smiles)
        else:
            mol_stereo = Chem.AddHs(mol_stereo)
        display_smi_stereo = canon_smi_from_mol(mol_stereo, stereo=True)
        match_stereo = mol_to_match_key(mol_stereo, stereo=True)
    else:
        display_smi_stereo = display_smi_no_stereo
        match_stereo = match_no_stereo

    return mol, match_no_stereo, match_stereo, display_smi_no_stereo, display_smi_stereo

# ============================================================
# Geometry metrics
# ============================================================

def heavy_atom_rmsd(mol_ref, mol_gen):
    ref = Chem.RemoveHs(mol_ref)
    gen = Chem.RemoveHs(mol_gen)
    try:
        return rdMolAlign.GetBestRMS(ref, gen)
    except RuntimeError:
        return float('inf')


def get_atom_map(mol_ref, mol_gen):
    """Substructure match mol_ref onto mol_gen.
    Returns a tuple t where t[mol_ref_atom_idx] = mol_gen_atom_idx, or None if no match."""
    ref_noH = Chem.RemoveHs(mol_ref)
    gen_noH = Chem.RemoveHs(mol_gen)
    matches = gen_noH.GetSubstructMatches(ref_noH, uniquify=False)
    return matches[0] if matches else None


def bond_length_mae(mol_ref, mol_gen, atom_map):
    """atom_map[mol_ref_idx] = mol_gen_idx"""
    mol_ref = Chem.RemoveHs(mol_ref)
    mol_gen = Chem.RemoveHs(mol_gen)
    conf_r = mol_ref.GetConformer()
    conf_g = mol_gen.GetConformer()

    errs = []
    for b in mol_ref.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        d_r = conf_r.GetAtomPosition(i).Distance(conf_r.GetAtomPosition(j))
        d_g = conf_g.GetAtomPosition(atom_map[i]).Distance(conf_g.GetAtomPosition(atom_map[j]))
        errs.append(abs(d_r - d_g))

    return float(np.mean(errs))


def angle(a, b, c):
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def bond_angle_mae(mol_ref, mol_gen, atom_map):
    """atom_map[mol_ref_idx] = mol_gen_idx"""
    mol_ref = Chem.RemoveHs(mol_ref)
    mol_gen = Chem.RemoveHs(mol_gen)
    conf_r = mol_ref.GetConformer()
    conf_g = mol_gen.GetConformer()

    errs = []
    for atom in mol_ref.GetAtoms():
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) < 2:
            continue
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                a, b, c = nbrs[i], atom.GetIdx(), nbrs[j]
                ar = angle(conf_r.GetAtomPosition(a), conf_r.GetAtomPosition(b), conf_r.GetAtomPosition(c))
                ag = angle(conf_g.GetAtomPosition(atom_map[a]), conf_g.GetAtomPosition(atom_map[b]), conf_g.GetAtomPosition(atom_map[c]))
                errs.append(abs(ar - ag))
    return float(np.mean(errs))


def atom_priority(mol, idx, exclude_idx, depth=4):
    """CIP-like recursive priority WITHOUT atom-index tiebreaker.
    Atoms with identical chemical environments return identical priority tuples,
    which is used to group symmetric substituents."""
    if depth == 0:
        return (mol.GetAtomWithIdx(idx).GetAtomicNum(),)
    nbr_priorities = sorted(
        [atom_priority(mol, n.GetIdx(), idx, depth - 1)
         for n in mol.GetAtomWithIdx(idx).GetNeighbors()
         if n.GetIdx() != exclude_idx],
        reverse=True,
    )
    return (mol.GetAtomWithIdx(idx).GetAtomicNum(),) + tuple(x for p in nbr_priorities for x in p)


def _equiv_groups(mol, center_idx, exclude_idx):
    """Return list-of-lists: neighbors of center_idx (excluding exclude_idx)
    grouped by chemical equivalence (same atom_priority).  Each group is a
    list of atom indices that are symmetrically interchangeable."""
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(center_idx).GetNeighbors()
            if n.GetIdx() != exclude_idx]
    buckets = defaultdict(list)
    for idx in nbrs:
        buckets[atom_priority(mol, idx, center_idx)].append(idx)
    return list(buckets.values())


def torsion_mae(mol_ref, mol_gen, atom_map):
    """Torsion MAE over rotatable bonds.

    For each bond i-j, we pick one representative atom from each set of
    equivalent substituents on atom i and atom j.  When substituents are
    symmetric (e.g. two identical branches), we enumerate all valid choices
    and take the minimum torsion error — avoiding penalising the model for
    correctly generating a conformation but with substituents labelled in a
    different-but-equivalent order.

    atom_map[mol_ref_idx] = mol_gen_idx
    """
    mol_ref = Chem.RemoveHs(mol_ref)
    mol_gen = Chem.RemoveHs(mol_gen)
    conf_r = mol_ref.GetConformer()
    conf_g = mol_gen.GetConformer()

    torsions = []
    for bond in mol_ref.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue

        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ai, aj = mol_ref.GetAtomWithIdx(i), mol_ref.GetAtomWithIdx(j)
        if ai.GetDegree() < 2 or aj.GetDegree() < 2:
            continue

        # Groups of equivalent substituents on each end
        groups_i = _equiv_groups(mol_ref, i, j)
        groups_j = _equiv_groups(mol_ref, j, i)

        # Pick the highest-priority group on each end, then try all atoms in
        # that group as the torsion anchor — take the minimum error.
        best_group_i = max(groups_i, key=lambda g: atom_priority(mol_ref, g[0], i))
        best_group_j = max(groups_j, key=lambda g: atom_priority(mol_ref, g[0], j))

        best_diff = float('inf')
        for a, d in iproduct(best_group_i, best_group_j):
            tr = rdMolTransforms.GetDihedralDeg(conf_r, a, i, j, d)
            tg = rdMolTransforms.GetDihedralDeg(conf_g, atom_map[a], atom_map[i], atom_map[j], atom_map[d])
            diff = abs(tr - tg) % 360
            best_diff = min(best_diff, min(diff, 360 - diff))

        if np.isfinite(best_diff):
            torsions.append(best_diff)

    return float(np.mean(torsions)) if torsions else 0.0


def heavy_atom_rmsf(mol_list, align_first=True):
    if len(mol_list) < 2:
        return 0.0

    mols_noH = [Chem.RemoveHs(mol) for mol in mol_list]
    n_atoms = mols_noH[0].GetNumAtoms()
    if not all(mol.GetNumAtoms() == n_atoms for mol in mols_noH):
        return None

    coords_list = [
        np.array([mol.GetConformer().GetAtomPosition(i) for i in range(n_atoms)])
        for mol in mols_noH
    ]
    coords_array = np.stack(coords_list, axis=0)

    if align_first:
        ref_coords = [coords_array[0]]
        for i in range(1, coords_array.shape[0]):
            mol_aligned = Chem.Mol(mols_noH[i])
            rdMolAlign.AlignMol(mol_aligned, mols_noH[0])
            conf_aligned = mol_aligned.GetConformer()
            ref_coords.append(np.array([conf_aligned.GetAtomPosition(j) for j in range(n_atoms)]))
        coords_array = np.stack(ref_coords, axis=0)

    mean_coords = np.mean(coords_array, axis=0)
    squared_deviations = np.sum((coords_array - mean_coords[None]) ** 2, axis=2)
    rmsf_per_atom = np.sqrt(np.mean(squared_deviations, axis=0))
    return float(np.mean(rmsf_per_atom))


def clash_count(mol, scale=0.75):
    conf = mol.GetConformer()
    pts = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    vdw = np.array([
        rdchem.GetPeriodicTable().GetRvdw(a.GetAtomicNum())
        for a in mol.GetAtoms()
    ])

    n = mol.GetNumAtoms()
    clashes = 0
    for i in range(n):
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 1:
            continue
        for j in range(i + 1, n):
            if mol.GetAtomWithIdx(j).GetAtomicNum() == 1:
                continue
            path = rdmolops.GetShortestPath(mol, i, j)
            if path is not None and len(path) <= 4:
                continue
            if np.linalg.norm(pts[i] - pts[j]) < scale * (vdw[i] + vdw[j]):
                clashes += 1
    return clashes

# ============================================================
# Energy
# ============================================================

def mmff_energy(mol):
    props = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    return ff.CalcEnergy()


def xtb_energy(mol):
    """Calculate GFN2-xTB single-point energy. Returns energy in kcal/mol, or NaN on failure."""
    try:
        # mol = Chem.RemoveHs(mol)
        mol = Chem.AddHs(mol, addCoords=True)

        numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32)
        conf = mol.GetConformer()
        positions = np.array(
            [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())],
            dtype=np.float64
        ) * ANGSTROM_TO_BOHR  # Å → Bohr

        calc = Calculator(Param.GFN2xTB, numbers, positions)
        calc.set_verbosity(0)
        res = calc.singlepoint()
        return res.get_energy() * HARTREE_TO_KCALMOL
    except Exception:
        return float('nan')


def xtb_homo_lumo(mol):
    """Unrelaxed GFN2-xTB HOMO, LUMO, and gap (eV). Returns (homo, lumo, gap) or (nan, nan, nan)."""
    try:
        mol = Chem.AddHs(mol, addCoords=True)
        numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32)
        conf = mol.GetConformer()
        positions = np.array(
            [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())],
            dtype=np.float64
        ) * ANGSTROM_TO_BOHR
        calc = Calculator(Param.GFN2xTB, numbers, positions)
        calc.set_verbosity(0)
        res = calc.singlepoint()
        eigs = res.get_orbital_eigenvalues()
        occs = res.get_orbital_occupations()
        homo_idx = int(np.where(occs > 0)[0][-1])
        homo = float(eigs[homo_idx]) * HARTREE_TO_EV
        lumo = float(eigs[homo_idx + 1]) * HARTREE_TO_EV
        return homo, lumo, lumo - homo
    except Exception:
        return float('nan'), float('nan'), float('nan')


def _mol_to_xyz_string(mol):
    """Write mol (with Hs) to an xyz-format string."""
    n = mol.GetNumAtoms()
    conf = mol.GetConformer()
    lines = [str(n), ""]
    for i, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        lines.append("%s %.8f %.8f %.8f" % (atom.GetSymbol(), p.x, p.y, p.z))
    return "\n".join(lines)


def xtb_relax(mol):
    """Relax mol with GFN2-xTB binary (--opt).

    Returns (relaxed_mol, strain_kcal, homo_eV, lumo_eV, gap_eV).
    relaxed_mol has updated conformer coordinates.
    strain_kcal = E(unrelaxed) - E(relaxed) in kcal/mol.
    HOMO/LUMO/gap are from the relaxed geometry.
    Returns (None, nan, nan, nan, nan) on failure.
    """
    nan5 = (None, float('nan'), float('nan'), float('nan'), float('nan'))
    try:
        mol = Chem.AddHs(mol, addCoords=True)
        e_unrelaxed = xtb_energy(mol)  # kcal/mol

        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_path = os.path.join(tmpdir, "mol.xyz")
            with open(xyz_path, "w") as f:
                f.write(_mol_to_xyz_string(mol))

            result = subprocess.run(
                [XTB_BIN, "mol.xyz", "--gfn", "2", "--opt", "--json"],
                cwd=tmpdir, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                return nan5

            with open(os.path.join(tmpdir, "xtbout.json")) as f:
                data = json.load(f)
            e_relaxed_kcal = data["total energy"] * HARTREE_TO_KCALMOL
            strain_kcal = e_unrelaxed - e_relaxed_kcal

            homo_eV = float('nan')
            lumo_eV = float('nan')
            gap_eV  = data.get("HOMO-LUMO gap / eV", float('nan'))
            orb = data.get("orbital energies / eV", [])
            occ = data.get("fractional occupation", [])
            if orb and occ:
                occupied = [i for i, o in enumerate(occ) if o > 0]
                if occupied:
                    hi = occupied[-1]
                    homo_eV = float(orb[hi])
                    if hi + 1 < len(orb):
                        lumo_eV = float(orb[hi + 1])

            # update conformer from xtbopt.xyz
            with open(os.path.join(tmpdir, "xtbopt.xyz")) as f:
                opt_lines = f.readlines()
            relaxed_mol = Chem.RWMol(mol)
            conf = relaxed_mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                parts = opt_lines[2 + i].split()
                conf.SetAtomPosition(i, (float(parts[1]), float(parts[2]), float(parts[3])))

            return relaxed_mol.GetMol(), strain_kcal, homo_eV, lumo_eV, gap_eV

    except Exception:
        return nan5

# ============================================================
# Boltzmann weights / energy-per-path helpers
# ============================================================

def load_boltzmann_weights(mol_id):
    """Load (conformer_id -> boltzmann_weight) from record JSON. Returns {} on failure."""
    mid = mol_id[1:3]
    for split in ("train", "val", "test"):
        record_path = GEOM_DATADIR / split / "records" / mid / f"{mol_id}.json"
        if record_path.exists():
            with record_path.open() as f:
                rec = json.load(f)
            return {cid: w for cid, w in zip(rec["ids"], rec["boltzmann_weights"])}
    return {}


def _energy_for_path(p):
    mol, *_ = load_and_prepare(p)
    if mol is None:
        return str(p), float('nan')
    return str(p), xtb_energy(mol)


def _gen_energies_for_group(mol_id, paths):
    return mol_id, [_energy_for_path(p) for p in sorted(paths)]


def _ref_energies_for_group(mol_id, paths):
    """Select top-5 conformers by Boltzmann weight, compute energies, return (path, energy, weight)."""
    bw_map = load_boltzmann_weights(mol_id)

    weighted = []
    for p in paths:
        cid = Path(p).stem   # ids in record are "{mol_id}_{i}"
        w = bw_map.get(cid, 0.0)
        weighted.append((w, p))

    top5 = sorted(weighted, key=lambda x: x[0], reverse=True)[:5]

    entries = []
    for w, p in top5:
        _, e = _energy_for_path(p)
        entries.append((str(p), e, w))
    return mol_id, entries

# ============================================================
# Coverage / aggregation helpers
# ============================================================

def pairwise_rmsd_matrix(ref_mols, gen_mols):
    D = np.zeros((len(ref_mols), len(gen_mols)), dtype=float)
    for i, r in enumerate(ref_mols):
        for j, g in enumerate(gen_mols):
            D[i, j] = heavy_atom_rmsd(r, g)
    return D


def compute_coverage_amr(D, delta):
    """Compute AMR-R, COV-R, AMR-P, COV-P from an RMSD matrix, ignoring inf entries."""
    if D.size == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    min_ref = np.where(np.isfinite(D), D, np.inf).min(axis=1)
    min_gen = np.where(np.isfinite(D), D, np.inf).min(axis=0)
    valid_ref = np.isfinite(min_ref)
    valid_gen = np.isfinite(min_gen)
    amr_r = float(min_ref[valid_ref].mean()) if valid_ref.any() else float('nan')
    cov_r = float((min_ref[valid_ref] < delta).mean()) if valid_ref.any() else float('nan')
    amr_p = float(min_gen[valid_gen].mean()) if valid_gen.any() else float('nan')
    cov_p = float((min_gen[valid_gen] < delta).mean()) if valid_gen.any() else float('nan')
    return amr_r, cov_r, amr_p, cov_p


def compute_geometry_stats(ref_mols, gen_mols, gen_paths, D, out_align_dir, out_relax_dir=None):
    """Per-conformer geometry metrics; returns (best_rmsd, best_bl, best_ba, mean_tor, mean_clash, mean_strain, rmsf)."""
    per_rmsd, per_bl, per_ba, per_tor, per_clash, per_strain = [], [], [], [], [], []

    for j, (gen, p) in enumerate(zip(gen_mols, gen_paths)):
        col = D[:, j]
        finite_rows = np.where(np.isfinite(col))[0]
        if len(finite_rows) == 0:
            continue
        i = int(finite_rows[np.argmin(col[finite_rows])])
        ref = ref_mols[i]

        per_rmsd.append(D[i, j])
        atom_map = get_atom_map(gen, ref)
        if atom_map is not None:
            try:
                per_bl.append(bond_length_mae(gen, ref, atom_map))
                per_ba.append(bond_angle_mae(gen, ref, atom_map))
                per_tor.append(torsion_mae(gen, ref, atom_map))
            except Exception:
                pass
        per_clash.append(clash_count(gen))
        relaxed_mol, strain, _, _, _ = xtb_relax(gen)
        if np.isfinite(strain):
            per_strain.append(strain)

        if out_align_dir is not None:
            try:
                probe = Chem.RemoveHs(Chem.Mol(gen))
                ref_a = Chem.RemoveHs(Chem.Mol(ref))
                rdMolAlign.AlignMol(probe, ref_a)
                out_name = f"{p.stem}_BEST_rmsd{D[i, j]:.3f}.pdb"
                save_best_pair(probe, ref_a, out_align_dir / out_name)
            except Exception:
                pass

        if out_relax_dir is not None and relaxed_mol is not None and np.isfinite(strain):
            try:
                unrelaxed = Chem.RemoveHs(Chem.Mol(gen))
                relaxed = Chem.RemoveHs(Chem.Mol(relaxed_mol))
                rdMolAlign.AlignMol(relaxed, unrelaxed)
                out_name = f"{p.stem}_strain{strain:.2f}kcal.pdb"
                save_best_pair(unrelaxed, relaxed, out_relax_dir / out_name)
            except Exception:
                pass

    if not per_rmsd:
        return None
    rmsf_val = heavy_atom_rmsf(gen_mols, align_first=True) if len(gen_mols) >= 2 else None
    return (
        float(np.min(per_rmsd)),
        float(np.min(per_bl))     if per_bl     else float('nan'),
        float(np.min(per_ba))     if per_ba     else float('nan'),
        float(np.mean(per_tor))   if per_tor    else float('nan'),
        float(np.mean(per_clash)),
        float(np.mean(per_strain)) if per_strain else float('nan'),
        rmsf_val,
    )


def compute_reference_rmsf_mean(ref_key_to_mols):
    vals = []
    for mols in ref_key_to_mols.values():
        if len(mols) < 2:
            continue
        try:
            v = heavy_atom_rmsf(mols, align_first=True)
            if v is not None and not np.isnan(v):
                vals.append(v)
        except Exception:
            pass
    return float(np.mean(vals)) if vals else None


def make_metric_lists():
    return {k: [] for k in ['amr_r', 'cov_r', 'amr_p', 'cov_p', 'rmsd', 'bl', 'ba', 'tor', 'clash', 'strain', 'rmsf']}


def append_metrics(agg, metrics):
    amr_r, cov_r, amr_p, cov_p, rmsd, bl, ba, tor, clash, strain, rmsf_val = metrics
    for key, val in [('amr_r', amr_r), ('cov_r', cov_r), ('amr_p', amr_p), ('cov_p', cov_p),
                     ('rmsd', rmsd), ('bl', bl), ('ba', ba), ('tor', tor), ('clash', clash), ('strain', strain)]:
        if np.isfinite(val):
            agg[key].append(val)
    if rmsf_val is not None:
        agg['rmsf'].append(rmsf_val)


def mean_or_nan(lst):
    return float(np.mean(lst)) if lst else float('nan')


def compute_metrics_for_group(paths, ref_key_no_stereo_to_mols, ref_key_stereo_to_mols, out_align_dir, out_relax_dir=None):
    RDLogger.DisableLog('rdApp.*')

    gen_mols, gen_paths, gen_stereo_flags = [], [], []
    ref_mols_for_group = None
    mol_id = paths[0].stem.rsplit("_", 1)[0] if paths else None

    connectivity_no_match = []
    stereo_no_match = []
    n_attempted = 0
    n_matched_no_stereo = 0
    n_matched_stereo = 0

    for p in sorted(paths):
        mol, match_no_stereo, match_stereo, display_smi_no_stereo, display_smi_stereo = load_and_prepare(p)
        if mol is None:
            continue

        n_attempted += 1
        ref_mols = ref_key_no_stereo_to_mols.get(match_no_stereo)
        if ref_mols is None:
            connectivity_no_match.append((mol_id, display_smi_no_stereo))
            continue

        n_matched_no_stereo += 1
        stereo_ok = ref_key_stereo_to_mols.get(match_stereo) is not None
        if stereo_ok:
            n_matched_stereo += 1
        else:
            stereo_no_match.append((mol_id, display_smi_stereo))

        gen_mols.append(mol)
        gen_paths.append(p)
        gen_stereo_flags.append(stereo_ok)
        ref_mols_for_group = ref_mols

    empty = (None, None, connectivity_no_match, stereo_no_match,
             n_attempted, n_matched_no_stereo, n_matched_stereo)
    if not gen_mols or ref_mols_for_group is None:
        return empty

    D = pairwise_rmsd_matrix(ref_mols_for_group, gen_mols)

    amr_r, cov_r, amr_p, cov_p = compute_coverage_amr(D, DELTA)
    geom_conn = compute_geometry_stats(ref_mols_for_group, gen_mols, gen_paths, D, out_align_dir, out_relax_dir)
    if geom_conn is None:
        return empty
    metrics_conn = (amr_r, cov_r, amr_p, cov_p) + geom_conn

    stereo_idx = [j for j, ok in enumerate(gen_stereo_flags) if ok]
    if stereo_idx:
        D_stereo = D[:, stereo_idx]
        gen_mols_s = [gen_mols[j] for j in stereo_idx]
        gen_paths_s = [gen_paths[j] for j in stereo_idx]
        amr_r_s, cov_r_s, amr_p_s, cov_p_s = compute_coverage_amr(D_stereo, DELTA)
        geom_stereo = compute_geometry_stats(ref_mols_for_group, gen_mols_s, gen_paths_s, D_stereo, None)
        metrics_stereo = (amr_r_s, cov_r_s, amr_p_s, cov_p_s) + (geom_stereo if geom_stereo else (float('nan'),) * 7)
    else:
        metrics_stereo = None

    return (metrics_conn, metrics_stereo,
            connectivity_no_match, stereo_no_match,
            n_attempted, n_matched_no_stereo, n_matched_stereo)

# ============================================================
# Torsion plot helpers
# ============================================================

def assign_atom_names(mol_h):
    """Assign unique names (C1, O1, N2, …) to each heavy atom by element order."""
    counters = defaultdict(int)
    names = {}
    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        counters[sym] += 1
        names[atom.GetIdx()] = f"{sym}{counters[sym]}"
    return names


def get_torsion_quads(mol_h):
    """
    Return [(a, b, c, d), …] for every rotatable bond in mol_h:
      - non-ring single bonds where both endpoints have degree >= 2
      - covers ALL such bonds (no further filtering)
    a and d are the highest CIP-priority heavy neighbours of b and c respectively,
    using the atom_priority() function defined earlier in the notebook.
    """
    quads = []
    seen = set()
    for bond in mol_h.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        b, c = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        key = (min(b, c), max(b, c))
        if key in seen:
            continue
        seen.add(key)
        ab, ac = mol_h.GetAtomWithIdx(b), mol_h.GetAtomWithIdx(c)
        if ab.GetDegree() < 2 or ac.GetDegree() < 2:
            continue
        a = max(
            (n.GetIdx() for n in ab.GetNeighbors() if n.GetIdx() != c),
            key=lambda idx: atom_priority(mol_h, idx, b),
        )
        d = max(
            (n.GetIdx() for n in ac.GetNeighbors() if n.GetIdx() != b),
            key=lambda idx: atom_priority(mol_h, idx, c),
        )
        quads.append((a, b, c, d))
    return quads


def circular_diff(ang1, ang2):
    """Minimum circular difference between two dihedral angles in degrees."""
    diff = abs(ang1 - ang2) % 360
    return min(diff, 360 - diff)


def conf_torsion_angles(mol_h, quads):
    """Return {quad: dihedral_deg} for each quad in quads."""
    conf = mol_h.GetConformer()
    out = {}
    for a, b, c, d in quads:
        try:
            out[(a, b, c, d)] = rdMolTransforms.GetDihedralDeg(conf, a, b, c, d)
        except Exception:
            pass
    return out


def circular_std(angles_deg):
    """Circular standard deviation of a list of angles in degrees.
    Returns degrees.  Uses the formula: sqrt(-2 * ln(R)) where R is the
    mean resultant length of the unit vectors."""
    if len(angles_deg) < 2:
        return float('nan')
    rads = np.deg2rad(angles_deg)
    R = np.abs(np.mean(np.exp(1j * rads)))
    R = np.clip(R, 0.0, 1.0 - 1e-10)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(R))))

# ============================================================
# User Settings
# ============================================================

_parser = argparse.ArgumentParser()
_parser.add_argument("--mode", choices=["train", "test", "val", "xl_processed", "all"], default="all")
_parser.add_argument("--model_name", default="geom_identityRot_256_conformer_3std_bondlength")
_args = _parser.parse_args()

model_name = _args.model_name
_modes     = ["train", "test", "xl_processed"] if _args.mode == "all" else [_args.mode]


def _sampling_base(mode):
    """Return the base sampling directory for a given mode."""
    repo = Path(__file__).resolve().parent.parent
    if mode == "xl_processed":
        return repo / "sampling" / "xl_processed"
    return repo / "sampling" / f"geom_conformer_{mode}"

DELTA        = 0.75   # RMSD threshold for coverage (Å)
N_JOBS       = max(1, mp.cpu_count() // 2)
USE_PARALLEL = True
GEOM_DATADIR = Path(os.environ.get("REPO_ROOT", "/datastor1/dy4652/proteinzen")) / "data" / "geom_drugs_conformers"

# ============================================================
# Main — geometry evaluation
# ============================================================

for mode in _modes:
    _base = _sampling_base(mode)
    REF_GLOB  = str(_base / "conformer_mols" / "*.pdb")
    PRED_GLOB = str(_base / model_name / "samples" / "*.pdb")
    OUT_ALIGN_DIR = _base / model_name / "aligned_pairs"
    if OUT_ALIGN_DIR.exists():
        shutil.rmtree(OUT_ALIGN_DIR)
    OUT_ALIGN_DIR.mkdir(parents=True)
    OUT_RELAX_DIR = _base / model_name / "relaxed_pairs"
    if OUT_RELAX_DIR.exists():
        shutil.rmtree(OUT_RELAX_DIR)
    OUT_RELAX_DIR.mkdir(parents=True)
    OUT_STATS_DIR = _base / model_name / "eval_stats"
    if OUT_STATS_DIR.exists():
        shutil.rmtree(OUT_STATS_DIR)
    OUT_STATS_DIR.mkdir(parents=True)
    _stats_log = open(OUT_STATS_DIR / f"eval_{mode}.txt", 'w')
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _stats_log)

    print(f"\n{'='*60}")
    print(f"  MODE: {mode.upper()}")
    print(f"{'='*60}")

    # ---- Load reference molecules ----
    print("Loading reference molecules...")

    ref_key_no_stereo_to_mols = defaultdict(list)
    ref_key_stereo_to_mols = defaultdict(list)
    ref_mol_id_to_display_smis = {}

    for ref_path in glob.glob(REF_GLOB):
        mol, match_no_stereo, match_stereo, display_smi_no_stereo, display_smi_stereo = load_and_prepare(ref_path)
        assert mol is not None, f"Failed to load ref: {ref_path}"
        ref_key_no_stereo_to_mols[match_no_stereo].append(mol)
        ref_key_stereo_to_mols[match_stereo].append(mol)
        mol_id = Path(ref_path).stem.rsplit("_", 1)[0]
        if mol_id not in ref_mol_id_to_display_smis:
            ref_mol_id_to_display_smis[mol_id] = (display_smi_no_stereo, display_smi_stereo)

    print(f"  Ref: {sum(len(v) for v in ref_key_no_stereo_to_mols.values())} conformers across {len(ref_mol_id_to_display_smis)} molecules")
    print(f"  Ref unique connectivity keys (no stereo): {len(ref_key_no_stereo_to_mols)}")
    print(f"  Ref unique stereo keys:                   {len(ref_key_stereo_to_mols)}")

    # ---- Collect generated molecules ----
    print("Collecting generated PDBs...")

    pred_paths = [Path(p) for p in glob.glob(PRED_GLOB)]
    groups = defaultdict(list)
    for p in pred_paths:
        mol_id = p.stem.rsplit("_", 1)[0]
        groups[mol_id].append(p)

    print(f"  Gen: {sum(len(v) for v in groups.values())} conformers across {len(groups)} molecules")

    # ---- Run computation ----
    print("Running metric computation (gen vs. ref)...")

    if USE_PARALLEL:
        results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(compute_metrics_for_group)(paths, ref_key_no_stereo_to_mols, ref_key_stereo_to_mols, OUT_ALIGN_DIR, OUT_RELAX_DIR)
            for paths in groups.values()
        )
    else:
        results = [
            compute_metrics_for_group(paths, ref_key_no_stereo_to_mols, ref_key_stereo_to_mols, OUT_ALIGN_DIR, OUT_RELAX_DIR)
            for paths in groups.values()
        ]

    # ---- Aggregate ----
    agg_conn   = make_metric_lists()
    agg_stereo = make_metric_lists()
    connectivity_no_match_all = []
    stereo_no_match_all = []
    total_attempted = 0
    total_matched_no_stereo = 0
    total_matched_stereo = 0

    for res in results:
        metrics_conn, metrics_stereo, conn_nm, stereo_nm, n_att, n_conn, n_stereo = res
        total_attempted += n_att
        total_matched_no_stereo += n_conn
        total_matched_stereo += n_stereo
        connectivity_no_match_all.extend(conn_nm)
        stereo_no_match_all.extend(stereo_nm)
        if metrics_conn is not None:
            append_metrics(agg_conn, metrics_conn)
        if metrics_stereo is not None:
            append_metrics(agg_stereo, metrics_stereo)

    ref_rmsf_mean = compute_reference_rmsf_mean(ref_key_no_stereo_to_mols)

    # ---- Summary ----
    print(f"\n--- Gen vs. Ref: Conformer Match Summary ({total_attempted} gen conformers, {sum(len(v) for v in ref_key_no_stereo_to_mols.values())} ref conformers) ---")
    print(f"  Gen matched ref — connectivity only (no stereo):  {total_matched_no_stereo}/{total_attempted}")
    print(f"  Gen matched ref — connectivity + stereo (exact):  {total_matched_stereo}/{total_attempted}")
    print(f"  Gen no-match — wrong connectivity:                {total_attempted - total_matched_no_stereo}/{total_attempted}")
    print(f"  Gen no-match — correct connectivity, wrong stereo:{total_matched_no_stereo - total_matched_stereo}/{total_attempted}")

    for label, agg in [("Gen/Ref metrics — connectivity-matched (no stereo)", agg_conn),
                       ("Gen/Ref metrics — stereo-matched (exact)",           agg_stereo)]:
        print(f"\n--- {label} ---")
        print(f"  AMR-R:          {mean_or_nan(agg['amr_r']):.4f}")
        print(f"  COV-R:          {mean_or_nan(agg['cov_r']):.4f}")
        print(f"  AMR-P:          {mean_or_nan(agg['amr_p']):.4f}")
        print(f"  COV-P:          {mean_or_nan(agg['cov_p']):.4f}")
        print(f"  Best RMSD:      {mean_or_nan(agg['rmsd']):.4f}")
        print(f"  Best BL MAE:    {mean_or_nan(agg['bl']):.4f}")
        print(f"  Best BA MAE:    {mean_or_nan(agg['ba']):.4f}")
        print(f"  Torsion MAE:    {mean_or_nan(agg['tor']):.4f}")
        print(f"  Clashes:        {mean_or_nan(agg['clash']):.4f}")
        print(f"  Strain (kcal):  {mean_or_nan(agg['strain']):.4f}")
        print(f"  RMSF (gen):     {mean_or_nan(agg['rmsf']):.4f}")

    print(f"\n  RMSF (ref conformers, all):  {ref_rmsf_mean}")

    # ---- No-match details ----
    if connectivity_no_match_all:
        print(f"\n--- No-match: wrong connectivity ({len(connectivity_no_match_all)} conformers) ---")
        by_name = defaultdict(list)
        for mol_id, gen_smi in connectivity_no_match_all:
            by_name[mol_id].append(gen_smi)
        for mol_id, gen_smis in sorted(by_name.items()):
            ref_smis = ref_mol_id_to_display_smis.get(mol_id)
            ref_smi = ref_smis[0] if ref_smis else None
            print(f"  [{mol_id}]  ({len(gen_smis)} conformers)")
            print(f"    ref: {ref_smi}")
            for smi in sorted(set(gen_smis)):
                print(f"    gen: {smi}")
            print()
    else:
        print("\n--- No-match (connectivity): none ---")

    if stereo_no_match_all:
        print(f"\n--- No-match: wrong stereo, correct connectivity ({len(stereo_no_match_all)} conformers) ---")
        by_name = defaultdict(list)
        for mol_id, gen_smi_stereo in stereo_no_match_all:
            by_name[mol_id].append(gen_smi_stereo)
        for mol_id, gen_smis in sorted(by_name.items()):
            ref_smis = ref_mol_id_to_display_smis.get(mol_id)
            ref_smi_stereo = ref_smis[1] if ref_smis else None
            print(f"  [{mol_id}]  ({len(gen_smis)} conformers)")
            print(f"    ref: {ref_smi_stereo}")
            for smi in sorted(set(gen_smis)):
                print(f"    gen: {smi}")
            print()
    else:
        print("\n--- No-match (stereo): none ---")

    no_match_mol_ids = set(mol_id for mol_id, _ in connectivity_no_match_all) | \
                       set(mol_id for mol_id, _ in stereo_no_match_all)
    perfect = [(mol_id, ref_mol_id_to_display_smis.get(mol_id, ('?', '?'))[1])
               for mol_id in sorted(groups.keys())
               if mol_id not in no_match_mol_ids
               and mol_id in ref_mol_id_to_display_smis]
    if perfect:
        print(f"\n--- Perfectly matched molecules ({len(perfect)}/{len(groups)}, all conformers stereo-matched) ---")
        for mol_id, smi in perfect:
            print(f"  [{mol_id}]  {smi}")
    else:
        print("\n--- Perfectly matched molecules: none ---")

    sys.stdout = _orig_stdout
    _stats_log.close()
    print(f"Eval stats saved to {OUT_STATS_DIR / f'eval_{mode}.txt'}")

# ============================================================
# Main — xTB energy evaluation
# ============================================================

print("Computing GFN2-xTB energies for generated molecules...")

# Generated molecules
gen_results = Parallel(n_jobs=N_JOBS, backend="loky")(
    delayed(_gen_energies_for_group)(mol_id, paths)
    for mol_id, paths in tqdm(groups.items(), desc="gen molecules")
)
gen_energies = defaultdict(list)
for mol_id, entries in gen_results:
    gen_energies[mol_id] = entries  # list of (path, energy)

# Reference molecules
ref_id_to_paths = defaultdict(list)
for ref_path in glob.glob(REF_GLOB):
    mol_id = Path(ref_path).stem.rsplit("_", 1)[0]
    ref_id_to_paths[mol_id].append(ref_path)

ref_results = Parallel(n_jobs=N_JOBS, backend="loky")(
    delayed(_ref_energies_for_group)(mol_id, paths)
    for mol_id, paths in tqdm(ref_id_to_paths.items(), desc="ref molecules")
)
ref_energies = defaultdict(list)
for mol_id, entries in ref_results:
    ref_energies[mol_id] = entries  # list of (path, energy, boltzmann_weight)

print("Done.")

perfect_mol_ids = set(mol_id for mol_id, _ in perfect)

pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

rows = []
for mol_id in sorted(set(gen_energies) | set(ref_energies)):
    if mol_id not in perfect_mol_ids:
        continue
    gen_e = [e for _, e in gen_energies.get(mol_id, []) if np.isfinite(e)]
    ref_entries = [(e, w) for _, e, w in ref_energies.get(mol_id, []) if np.isfinite(e)]
    ref_e = [e for e, w in ref_entries]
    ref_w = [w for e, w in ref_entries]

    gen_min     = np.min(gen_e)  if gen_e else float('nan')
    gen_mean    = np.mean(gen_e) if gen_e else float('nan')
    ref_min     = np.min(ref_e)  if ref_e else float('nan')
    ref_bw_mean = float(np.average(ref_e, weights=ref_w)) if ref_e and sum(ref_w) > 0 else float('nan')

    rows.append({
        'mol_id':         mol_id[:16] + '…',
        'n_gen':          len(gen_e),
        'gen_min':        gen_min,
        'gen_mean':       gen_mean,
        'n_ref':          len(ref_e),
        'ref_min':        ref_min,
        'ref_bw_mean':    ref_bw_mean,
        'Δmin (gen-ref)': gen_min - ref_min,
    })

df = pd.DataFrame(rows)
print(f"Perfectly matched molecules: {len(df)}/{len(groups)}")

if df.empty:
    print("  (no perfectly matched molecules — skipping energy aggregate)")
else:
    print(df.to_string(index=False))
    print(f"\n--- Aggregate (kcal/mol, perfectly matched only) ---")
    print(f"  Mean gen_min  energy:     {df['gen_min'].mean():.2f}")
    print(f"  Mean ref_min  energy:     {df['ref_min'].mean():.2f}")
    print(f"  Mean ref BW-mean energy:  {df['ref_bw_mean'].mean():.2f}")
    print(f"  Mean Δmin (gen-ref):      {df['Δmin (gen-ref)'].mean():.2f}")
    df.to_csv(OUT_STATS_DIR / f"energy_stats_{mode}.csv", index=False)
    print(f"Energy stats saved to {OUT_STATS_DIR / f'energy_stats_{mode}.csv'}")

# ============================================================
# Main — min-energy aligned pairs
# ============================================================

OUT_MIN_ENERGY_DIR = _sampling_base(mode) / model_name / "min_energy_aligned_pairs"
if OUT_MIN_ENERGY_DIR.exists():
    shutil.rmtree(OUT_MIN_ENERGY_DIR)
OUT_MIN_ENERGY_DIR.mkdir(parents=True)

print("Writing min-energy aligned pairs...")
for mol_id in sorted(perfect_mol_ids):
    gen_entries   = [(p, e) for p, e in gen_energies.get(mol_id, []) if np.isfinite(e)]
    ref_entries_w = [(p, e, w) for p, e, w in ref_energies.get(mol_id, []) if np.isfinite(e)]
    if not gen_entries or not ref_entries_w:
        print(f"  skipping {mol_id[:16]}… (missing energies)")
        continue

    gen_min_path, gen_min_e = min(gen_entries, key=lambda x: x[1])
    ref_min_path, ref_min_e = min(ref_entries_w, key=lambda x: x[1])[:2]
    delta_e = gen_min_e - ref_min_e

    gen_mol, *_ = load_and_prepare(gen_min_path)
    ref_mol, *_ = load_and_prepare(ref_min_path)
    if gen_mol is None or ref_mol is None:
        print(f"  skipping {mol_id[:16]}… (load failed)")
        continue

    try:
        probe = Chem.RemoveHs(Chem.Mol(gen_mol))
        ref_a = Chem.RemoveHs(Chem.Mol(ref_mol))
        rmsd = rdMolAlign.AlignMol(probe, ref_a)
        sign = "+" if delta_e >= 0 else ""
        out_name = f"{mol_id}_dE{sign}{delta_e:.1f}_rmsd{rmsd:.3f}.pdb"
        save_best_pair(probe, ref_a, OUT_MIN_ENERGY_DIR / out_name)
        print(f"  {mol_id[:16]}…  ΔE={sign}{delta_e:.1f} kcal/mol  RMSD={rmsd:.3f}")
    except Exception as ex:
        print(f"  {mol_id[:16]}… failed: {ex}")

print(f"Done. Written to {OUT_MIN_ENERGY_DIR}")

# ============================================================
# Main — torsion spread + plots
# ============================================================

OUT_PLOTS_DIR = _sampling_base(mode) / model_name / "torsion_plots"
if OUT_PLOTS_DIR.exists():
    shutil.rmtree(OUT_PLOTS_DIR)
OUT_PLOTS_DIR.mkdir(parents=True)

# Per molecule: circular std of torsion angles across ref vs gen conformers.
# Only perfectly-matched molecules are included.
torsion_results = {}   # mol_id -> {torsion_name: {'ref': float, 'gen': float}}

for mol_id in sorted(perfect_mol_ids):
    ref_ents = ref_energies.get(mol_id, [])
    if not ref_ents:
        continue
    ref_mol, *_ = load_and_prepare(ref_ents[0][0])
    if ref_mol is None:
        continue
    ref_h = Chem.RemoveHs(ref_mol)

    quads = get_torsion_quads(ref_h)
    if not quads:
        continue

    atom_names = assign_atom_names(ref_h)
    quad_name  = {(a, b, c, d): f"{atom_names[a]}-{atom_names[b]}-{atom_names[c]}-{atom_names[d]}"
                  for a, b, c, d in quads}

    ref_angle_lists = defaultdict(list)
    for path, _, _ in ref_ents:
        mol, *_ = load_and_prepare(path)
        if mol is None:
            continue
        angs = conf_torsion_angles(Chem.RemoveHs(mol), quads)
        for q, v in angs.items():
            ref_angle_lists[q].append(v)

    gen_ents = [(p, e) for p, e in gen_energies.get(mol_id, []) if np.isfinite(e)]
    gen_angle_lists = defaultdict(list)
    for path, _ in gen_ents:
        mol, *_ = load_and_prepare(path)
        if mol is None:
            continue
        angs = conf_torsion_angles(Chem.RemoveHs(mol), quads)
        for q, v in angs.items():
            gen_angle_lists[q].append(v)

    mol_data = {}
    for q in quads:
        name    = quad_name[q]
        ref_val = circular_std(ref_angle_lists[q])
        gen_val = circular_std(gen_angle_lists[q])
        mol_data[name] = {'ref': ref_val, 'gen': gen_val}

    torsion_results[mol_id] = mol_data

print(f"Torsion spread computed for {len(torsion_results)} perfectly-matched molecules.")

for mol_id, mol_data in sorted(torsion_results.items()):
    if not mol_data:
        continue

    names    = list(mol_data.keys())
    ref_vals = [mol_data[n]['ref'] for n in names]
    gen_vals = [mol_data[n]['gen'] for n in names]

    n     = len(names)
    x     = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, n * 0.85), 5))

    ax.bar(x - width / 2, ref_vals, width, label='Ref circular std (all ref conformers)',
           color='steelblue', alpha=0.85)
    ax.bar(x + width / 2, gen_vals, width, label='Gen circular std (all gen conformers)',
           color='darkorange', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_xlabel('Torsion (a–b–c–d)  |  all rotatable non-ring single bonds', labelpad=4)
    ax.set_ylabel('Circular std of torsion angles across conformers (°)')
    ax.set_ylim(0, 185)
    ax.axhline(180, color='grey', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

    smi   = ref_mol_id_to_display_smis.get(mol_id, ('?', '?'))[1]
    title = f"{mol_id[:20]}…    {smi}"
    ax.set_title(title, fontsize=8)

    plt.tight_layout()
    out_path = OUT_PLOTS_DIR / f"{mol_id}_torsion_std.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
