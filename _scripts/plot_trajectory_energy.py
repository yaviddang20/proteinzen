#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot GFN2-xTB energy vs timestep for the last N clean trajectory steps.

For each molecule, loads all clean_traj PDB files, takes the last N_LAST models
(closest to clean data, t~1), computes xTB single-point energy, and saves a plot to:
  sampling/geom_conformer_{mode}/{model_name}/energy_plots/{name}.png

Trajectory order: ts = linspace(0, 1, num_timesteps), iterating ts[1:].
  - model 0 in PDB = prediction at t~0 (from near-noise)
  - model -1 in PDB = prediction at t=1 (clean)
  => last N_LAST models are closest to clean data.
"""

import os
import sys
import glob
import numpy as np

# Limit xTB OpenMP threads — unbounded parallelism causes slowdowns for small molecules
os.environ.setdefault("OMP_NUM_THREADS", "4")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from rdkit import Chem, RDLogger
from xtb.interface import Calculator, Param

RDLogger.DisableLog('rdApp.*')

# ============================================================
# SETTINGS
# ============================================================
mode = "train"
model_name = "geom_identityRot_256_conformer_3std_bondlength"
T_START = 0.75  # evaluate from this t value to t=1 (clean)

BASE_DIR = Path(__file__).parent.parent
TRAJ_DIR    = BASE_DIR / f"sampling/geom_conformer_{mode}/{model_name}/traj"
SAMPLES_DIR = BASE_DIR / f"sampling/geom_conformer_{mode}/{model_name}/samples"
ENERGY_PLOTS_DIR = BASE_DIR / f"sampling/geom_conformer_{mode}/{model_name}/energy_plots"

ENERGY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ANGSTROM_TO_BOHR   = 1.8897259886
HARTREE_TO_KCALMOL = 627.509474


# ============================================================
# xTB single-point energy
# ============================================================
def xtb_energy(mol):
    """GFN2-xTB single-point energy in kcal/mol. Returns NaN on failure."""
    try:
        mol = Chem.RemoveHs(mol)
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
        return res.get_energy() * HARTREE_TO_KCALMOL
    except Exception:
        return float('nan')


# ============================================================
# Multi-model PDB parsing
# ============================================================
def parse_models(pdb_path):
    """
    Parse a multi-MODEL PDB file.
    Returns a list of lists of ATOM/HETATM lines, one inner list per MODEL.
    """
    models = []
    current = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('MODEL'):
                current = []
            elif line.startswith('ENDMDL'):
                if current:
                    models.append(current)
            elif line.startswith(('ATOM', 'HETATM')):
                current.append(line)
    # Trailing atoms without MODEL/ENDMDL wrapper
    if current and not models:
        models.append(current)
    return models


def coords_from_atom_lines(atom_lines):
    """Parse x, y, z from PDB ATOM/HETATM lines. Returns (N, 3) float64 array."""
    coords = []
    for line in atom_lines:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coords.append((x, y, z))
    return np.array(coords, dtype=np.float64)


def mol_with_coords(template_mol, coords):
    """
    Return a copy of template_mol (heavy atoms) with a new conformer from coords.
    coords is (N_heavy, 3).
    """
    mol = Chem.RWMol(template_mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol.GetMol()


# ============================================================
# Discover molecules via traj files
# ============================================================
traj_files_all = sorted(TRAJ_DIR.glob("*_clean_traj.pdb"))
# Group by molecule name (strip _<sample_idx>_clean_traj.pdb suffix)
name_to_trajs = {}
for p in traj_files_all:
    # filename: {name}_{sample_idx}_clean_traj.pdb
    stem = p.stem  # e.g. "abc123_0_clean_traj"
    # Remove trailing "_clean_traj"
    base = stem.replace("_clean_traj", "")
    # Split off last underscore-separated token = sample_idx
    parts = base.rsplit("_", 1)
    name = parts[0]
    name_to_trajs.setdefault(name, []).append(p)

print(f"Found {len(name_to_trajs)} unique molecules in {TRAJ_DIR}")

# ============================================================
# Main loop
# ============================================================
for mol_idx, (name, traj_paths) in enumerate(sorted(name_to_trajs.items())):
    out_path = ENERGY_PLOTS_DIR / f"{name}.png"
    if out_path.exists():
        print(f"[{mol_idx+1}/{len(name_to_trajs)}] {name}: skipping (exists)")
        continue

    # Load template mol (heavy atoms + connectivity) from first sample PDB
    sample_pdbs = sorted(SAMPLES_DIR.glob(f"{name}_*.pdb"))
    if not sample_pdbs:
        print(f"[{mol_idx+1}/{len(name_to_trajs)}] {name}: no sample PDB, skipping")
        continue

    template = Chem.MolFromPDBFile(str(sample_pdbs[0]), removeHs=True, sanitize=False)
    if template is None:
        print(f"[{mol_idx+1}/{len(name_to_trajs)}] {name}: failed to load template, skipping")
        continue
    try:
        Chem.SanitizeMol(template)
    except Exception:
        pass
    n_heavy = template.GetNumAtoms()

    all_energies = []
    n_total_models = None

    for traj_path in sorted(traj_paths):
        models = parse_models(str(traj_path))
        if not models:
            continue
        if n_total_models is None:
            n_total_models = len(models)

        # model i -> t = (i+1)/n_total_models; select i where t >= T_START
        start_idx = int(np.ceil(T_START * n_total_models)) - 1
        start_idx = max(start_idx, 0)
        selected_models = models[start_idx:]

        energies = []
        for atom_lines in selected_models:
            coords = coords_from_atom_lines(atom_lines)
            if len(coords) != n_heavy:
                energies.append(float('nan'))
                continue
            m = mol_with_coords(template, coords)
            energies.append(xtb_energy(m))

        all_energies.append(energies)

    if not all_energies or n_total_models is None:
        print(f"[{mol_idx+1}/{len(name_to_trajs)}] {name}: no valid traj data, skipping")
        continue

    # t values: model i -> t = (i+1)/n_total_models
    first_idx = max(int(np.ceil(T_START * n_total_models)) - 1, 0)
    model_indices = np.arange(first_idx, n_total_models)
    t_values = (model_indices + 1) / n_total_models

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, energies in enumerate(all_energies):
        valid = np.array(energies, dtype=float)
        ax.plot(t_values, valid, marker='o', markersize=4, linewidth=1.2, label=f"sample {i}")

    ax.set_xlabel("Timestep  t  (0 = noise, 1 = clean)", fontsize=11)
    ax.set_ylabel("GFN2-xTB Energy (kcal/mol)", fontsize=11)
    ax.set_title(f"{name[:32]}...\nt = {T_START:.2f} to 1.0 ({len(model_indices)} steps)", fontsize=10)
    if len(all_energies) <= 8:
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=100)
    plt.close(fig)

    print(f"[{mol_idx+1}/{len(name_to_trajs)}] {name}: saved {out_path.name}")

print("Done.")
