#!/usr/bin/env python
"""Evaluate protein-pocket-conditioned ligand generation on Plinder.

For each system:
  1. Load the GT structure from its npz (protein pocket + ligand).
  2. Load the generated PDB(s) (same pocket template, generated ligand).
  3. Kabsch-align the generated protein pocket onto the GT protein pocket.
     (Because sampling centers coordinates on the protein COM, this alignment
     is theoretically just a translation, but we do Kabsch for robustness.)
  4. Apply the same rigid transform to the generated ligand.
  5. Compute:
       - pocket_rmsd : RMSD between the pocket-aligned generated ligand and
                       the GT ligand, using positional atom correspondence
                       (no further ligand alignment).
       - best_rmsd   : symmetry-aware best-RMSD via rdMolAlign.GetBestRMS.
                       Both the GT and generated ligand use the topology from the
                       generated PDB's CONECT records. Falls back to inf on failure.

Usage
-----
python _scripts/eval_plinder_pocket.py \\
    --samples-dir ./pocket_samples/samples \\
    --data-dir    /path/to/plinder_processed/val \\
    --max-protein-residues 100 \\
    --delta 2.0 5.0
"""

import argparse
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolAlign
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proteinzen.boltz.data import const
from proteinzen.runtime.sampling.protein_pocket import (
    _crop_protein_to_pocket,
    load_structure_from_npz,
)

RDLogger.DisableLog("rdApp.*")


# ============================================================
# Kabsch alignment
# ============================================================

def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation R and translation t that aligns P onto Q (source → target).

    P, Q : (N, 3) float64 arrays.
    Returns (R, t) where  Q_approx = (R @ P.T).T + t.
    """
    cp = P.mean(0)
    cq = Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cq - R @ cp
    return R, t


def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ coords.T).T + t


def pos_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=-1))))


# ============================================================
# PDB parsing
# ============================================================

def parse_pdb_atoms(pdb_path: str):
    """Return (prot_coords, lig_coords, lig_serial_to_local, conect) from a PDB file.

    prot_coords : (N_prot, 3) float64 — ATOM records.
    lig_coords  : (N_lig,  3) float64 — HETATM records.
    lig_serial_to_local : dict mapping PDB serial number → 0-based HETATM index.
    conect      : list of (local_i, local_j) bond pairs for the ligand.
    """
    prot, lig = [], []
    lig_serial_to_local: dict[int, int] = {}
    raw_conects: list[tuple[int, int]] = []

    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].rstrip()
            if rec == "ATOM":
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    prot.append((x, y, z))
                except ValueError:
                    pass
            elif rec == "HETATM":
                try:
                    serial = int(line[6:11])
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    lig_serial_to_local[serial] = len(lig)
                    lig.append((x, y, z))
                except ValueError:
                    pass
            elif rec == "CONECT":
                # "CONECT" atom1 [atom2 atom3 atom4] — multiple partners on one line
                try:
                    nums = [int(line[6 + 5 * i : 11 + 5 * i]) for i in range(4)
                            if line[6 + 5 * i : 11 + 5 * i].strip()]
                    src = nums[0]
                    for dst in nums[1:]:
                        raw_conects.append((src, dst))
                except ValueError:
                    pass

    prot_arr = np.array(prot, dtype=np.float64) if prot else np.zeros((0, 3))
    lig_arr  = np.array(lig,  dtype=np.float64) if lig  else np.zeros((0, 3))

    # Keep only ligand–ligand CONECT bonds (deduplicated, directed i < j)
    conect_set: set[tuple[int, int]] = set()
    for s1, s2 in raw_conects:
        if s1 in lig_serial_to_local and s2 in lig_serial_to_local:
            a, b = lig_serial_to_local[s1], lig_serial_to_local[s2]
            conect_set.add((min(a, b), max(a, b)))

    return prot_arr, lig_arr, conect_set


def ligand_mol_from_pdb(pdb_path: str):
    """Build an RDKit mol for the ligand using HETATM records and CONECT bonds.

    Returns (mol, coords) where coords is (N_lig, 3) in PDB-file coordinates,
    or (None, None) on failure.
    """
    prot_coords, lig_coords, conect = parse_pdb_atoms(pdb_path)
    n_lig = len(lig_coords)
    if n_lig == 0:
        return None, None

    # Build a fake PDB block with only the HETATM atoms, renumbered 1..n_lig,
    # plus CONECT records using the local indices.
    lines = []
    for i, (x, y, z) in enumerate(lig_coords):
        lines.append(
            f"HETATM{i+1:5d}  C   LIG A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
        )
    for a, b in conect:
        lines.append(f"CONECT{a+1:5d}{b+1:5d}")
    lines.append("END")
    block = "\n".join(lines)

    mol = Chem.MolFromPDBBlock(block, removeHs=True, sanitize=False)
    if mol is None:
        return None, None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    return mol, lig_coords


def _clone_with_coords(mol, coords: np.ndarray):
    """Clone mol and set its conformer to coords (N, 3)."""
    if mol is None or mol.GetNumAtoms() != len(coords):
        return None
    rw = Chem.RWMol(Chem.Mol(mol))
    conf = rw.GetConformer()
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    return rw.GetMol()


def best_rmsd_rdkit(mol_template, gt_coords: np.ndarray, gen_aligned_coords: np.ndarray) -> float:
    """Symmetry-aware RMSD using GetBestRMS.

    mol_template : RDKit mol whose topology is used for both GT and gen mols.
    gt_coords    : (N, 3) GT ligand coordinates.
    gen_aligned_coords : (N, 3) pocket-aligned generated ligand coordinates.

    Returns inf on any failure.
    """
    gt_mol  = _clone_with_coords(mol_template, gt_coords)
    gen_mol = _clone_with_coords(mol_template, gen_aligned_coords)
    if gt_mol is None or gen_mol is None:
        return float("inf")
    try:
        return float(rdMolAlign.GetBestRMS(Chem.RemoveHs(gt_mol), Chem.RemoveHs(gen_mol)))
    except Exception:
        return float("inf")


# ============================================================
# Extract GT coords from cropped Structure
# ============================================================

def extract_gt_coords(struct):
    """Return (prot_coords, lig_coords) as (N, 3) float64 arrays.

    Only is_present atoms are included — matching the PDB writer behaviour.
    """
    protein_id    = const.chain_type_ids["PROTEIN"]
    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]

    prot_list, lig_list = [], []
    for chain in struct.chains[struct.mask]:
        mol = int(chain["mol_type"])
        a0  = int(chain["atom_idx"])
        atoms = struct.atoms[a0 : a0 + int(chain["atom_num"])]
        coords  = atoms["coords"]
        present = atoms["is_present"].astype(bool)
        if mol == protein_id:
            prot_list.append(coords[present])
        elif mol == nonpolymer_id:
            lig_list.append(coords[present])

    prot = np.concatenate(prot_list, 0) if prot_list else np.zeros((0, 3))
    lig  = np.concatenate(lig_list,  0) if lig_list  else np.zeros((0, 3))
    return prot.astype(np.float64), lig.astype(np.float64)


# ============================================================
# Per-system evaluation
# ============================================================

def eval_system(system_id: str, gen_pdb_paths, gt_struct):
    """Evaluate all generated samples for one system.

    Returns list of dicts: system_id, sample_idx, pocket_rmsd, best_rmsd, note.
    """
    gt_prot, gt_lig = extract_gt_coords(gt_struct)
    n_gt_prot = len(gt_prot)
    n_gt_lig  = len(gt_lig)

    if n_gt_lig == 0:
        return []

    records = []
    for idx, pdb_path in enumerate(sorted(gen_pdb_paths)):
        gen_prot, gen_lig, _ = parse_pdb_atoms(str(pdb_path))

        if len(gen_prot) != n_gt_prot:
            records.append(dict(
                system_id=system_id, sample_idx=idx,
                pocket_rmsd=np.inf, best_rmsd=np.inf,
                note=f"protein atom count mismatch: gen={len(gen_prot)} gt={n_gt_prot}",
            ))
            continue

        if len(gen_lig) != n_gt_lig:
            records.append(dict(
                system_id=system_id, sample_idx=idx,
                pocket_rmsd=np.inf, best_rmsd=np.inf,
                note=f"ligand atom count mismatch: gen={len(gen_lig)} gt={n_gt_lig}",
            ))
            continue

        # Kabsch: align generated protein onto GT protein
        R, t = kabsch(gen_prot, gt_prot)
        gen_lig_aligned = apply_transform(gen_lig, R, t)

        p_rmsd = pos_rmsd(gen_lig_aligned, gt_lig)

        # Best RMSD: build mol from CONECT records (shared topology for GT and gen)
        mol_template, _ = ligand_mol_from_pdb(str(pdb_path))
        b_rmsd = best_rmsd_rdkit(mol_template, gt_lig, gen_lig_aligned)

        records.append(dict(
            system_id=system_id, sample_idx=idx,
            pocket_rmsd=p_rmsd, best_rmsd=b_rmsd,
            note="",
        ))

    return records


def _eval_system_job(system_id: str, gen_pdb_paths, npz_path: str, include_h: bool, max_protein_residues: int):
    """Top-level worker for joblib — loads GT and evaluates one system."""
    RDLogger.DisableLog("rdApp.*")
    try:
        gt_struct = load_structure_from_npz(npz_path, include_h=include_h)
    except Exception as e:
        return system_id, [], f"npz load error: {e}"
    gt_struct = _crop_protein_to_pocket(gt_struct, max_protein_residues)
    records = eval_system(system_id, gen_pdb_paths, gt_struct)
    return system_id, records, None


# ============================================================
# Aggregation
# ============================================================

def mean_finite(vals):
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    return float(arr.mean()) if len(arr) else float("nan")


def cov(vals, delta):
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    return float((arr < delta).mean()) if len(arr) else float("nan")


def min_per_system(records_by_system, key):
    mins = []
    for recs in records_by_system.values():
        vals = [r[key] for r in recs if np.isfinite(r[key])]
        if vals:
            mins.append(min(vals))
    return mins


def mean_per_system(records_by_system, key):
    means = []
    for recs in records_by_system.values():
        vals = [r[key] for r in recs if np.isfinite(r[key])]
        if vals:
            means.append(float(np.mean(vals)))
    return means


def print_block(label, pocket_vals, best_vals, deltas):
    print(f"\n--- {label} ---")
    print(f"  n            : {len(pocket_vals)}")
    print(f"  pocket RMSD  : {mean_finite(pocket_vals):.3f} Å  (mean)")
    for d in deltas:
        print(f"  COV pocket < {d:.1f}Å : {cov(pocket_vals, d)*100:.1f}%")
    n_best = sum(np.isfinite(v) for v in best_vals)
    if n_best:
        print(f"  best RMSD    : {mean_finite(best_vals):.3f} Å  (mean, {n_best}/{len(best_vals)} finite)")
        for d in deltas:
            print(f"  COV best   < {d:.1f}Å : {cov(best_vals, d)*100:.1f}%")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples-dir", type=Path, required=True,
                        help="Directory of generated PDB files ({system_id}_{i}.pdb)")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Plinder processed split directory (manifest.json + structures/)")
    parser.add_argument("--max-protein-residues", type=int, default=100,
                        help="Protein crop radius used during sampling (default: 100)")
    parser.add_argument("--include-h", action="store_true",
                        help="Keep hydrogen atoms (match the --include-h flag used at sample time)")
    parser.add_argument("--delta", type=float, nargs="+", default=[2.0, 5.0],
                        help="RMSD thresholds for coverage reporting (Å; default: 2.0 5.0)")
    parser.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() // 2),
                        help="Parallel workers (default: half of CPU count)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-system details during evaluation")
    args = parser.parse_args()

    # ---- manifest ----
    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"manifest.json not found at {manifest_path}")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    system_ids_in_manifest = {rec["id"] for rec in manifest}
    print(f"Manifest: {len(system_ids_in_manifest)} systems")

    # ---- collect generated PDB files ----
    pdb_files = sorted(args.samples_dir.glob("*.pdb"))
    print(f"Generated PDBs: {len(pdb_files)}")

    groups: dict[str, list[Path]] = defaultdict(list)
    unmatched = []
    for p in pdb_files:
        stem = p.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            groups[parts[0]].append(p)
        else:
            unmatched.append(p.name)

    if unmatched:
        print(f"  Warning: {len(unmatched)} PDB(s) had unrecognised names — skipped")

    common = sorted(set(groups) & system_ids_in_manifest)
    print(f"  Systems with generated samples : {len(groups)}")
    print(f"  Systems evaluated              : {len(common)}")
    extra   = set(groups) - system_ids_in_manifest
    missing = system_ids_in_manifest - set(groups)
    if extra:
        print(f"  Warning: {len(extra)} sampled systems not in manifest")
    if missing:
        print(f"  Note: {len(missing)} manifest systems have no samples")

    # ---- build job list ----
    jobs = []
    for sid in common:
        mid = sid[1:3]
        npz_path = args.data_dir / "structures" / mid / f"{sid}.npz"
        if not npz_path.exists():
            print(f"  SKIP {sid}: npz not found")
            continue
        jobs.append((sid, groups[sid], str(npz_path)))

    print(f"  Running {len(jobs)} systems with {args.n_jobs} workers...")

    # ---- parallel evaluation ----
    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(_eval_system_job)(sid, pdbs, npz, args.include_h, args.max_protein_residues)
        for sid, pdbs, npz in tqdm(jobs, desc="evaluating")
    )

    # ---- collect results ----
    all_records: list[dict] = []
    records_by_system: dict[str, list[dict]] = {}
    n_atom_mismatch = 0

    for sid, sys_records, err in results:
        if err:
            print(f"  SKIP {sid}: {err}")
            continue
        if not sys_records:
            if args.verbose:
                print(f"  SKIP {sid}: no valid samples")
            continue

        mismatch_count = sum(1 for r in sys_records if r["note"])
        n_atom_mismatch += mismatch_count
        if args.verbose:
            for r in sys_records:
                if r["note"]:
                    print(f"  [{sid}] sample {r['sample_idx']}: {r['note']}")
            pk = [r["pocket_rmsd"] for r in sys_records if np.isfinite(r["pocket_rmsd"])]
            if pk:
                print(
                    f"  {sid}: {len(pk)} samples, "
                    f"pocket RMSD min={min(pk):.2f} mean={np.mean(pk):.2f}"
                )

        records_by_system[sid] = sys_records
        all_records.extend(sys_records)

    if n_atom_mismatch:
        print(f"\nWarning: {n_atom_mismatch} samples had atom-count mismatches (skipped)")

    if not records_by_system:
        print("No systems evaluated — check paths.")
        return

    # ---- aggregate ----
    all_pocket = [r["pocket_rmsd"] for r in all_records]
    all_best   = [r["best_rmsd"]   for r in all_records]

    sys_min_pocket  = min_per_system(records_by_system, "pocket_rmsd")
    sys_min_best    = min_per_system(records_by_system, "best_rmsd")
    sys_mean_pocket = mean_per_system(records_by_system, "pocket_rmsd")
    sys_mean_best   = mean_per_system(records_by_system, "best_rmsd")

    deltas = args.delta

    print(f"\n{'='*60}")
    print(f"  PLINDER POCKET EVAL  —  {len(records_by_system)} systems")
    print(f"{'='*60}")

    print_block("Per-sample (all samples pooled)", all_pocket, all_best, deltas)
    print_block(
        "Per-system best sample (min pocket RMSD per system)",
        sys_min_pocket, sys_min_best, deltas,
    )
    print_block(
        "Per-system mean (mean pocket RMSD per system)",
        sys_mean_pocket, sys_mean_best, deltas,
    )

    # ---- per-system table ----
    print(f"\n--- Per-system summary (sorted by min pocket RMSD) ---")
    header = f"  {'system_id':<42} {'n':>4} {'min_pk':>7} {'mean_pk':>8} {'min_bst':>8}"
    print(header)
    rows = []
    for sid, recs in records_by_system.items():
        pk  = [r["pocket_rmsd"] for r in recs if np.isfinite(r["pocket_rmsd"])]
        bst = [r["best_rmsd"]   for r in recs if np.isfinite(r["best_rmsd"])]
        rows.append((
            sid,
            len(recs),
            min(pk)  if pk  else np.inf,
            float(np.mean(pk)) if pk else np.inf,
            min(bst) if bst else np.inf,
        ))
    rows.sort(key=lambda x: x[2])
    for sid, n, mn_pk, avg_pk, mn_bst in rows:
        def _fmt(v): return f"{v:.3f}" if np.isfinite(v) else "  inf"
        print(f"  {sid:<42} {n:>4} {_fmt(mn_pk):>7} {_fmt(avg_pk):>8} {_fmt(mn_bst):>8}")


if __name__ == "__main__":
    main()
