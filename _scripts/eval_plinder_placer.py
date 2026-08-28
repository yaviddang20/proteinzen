#!/usr/bin/env python
"""Evaluate PLINDER placer task — sidechain RMSD.

The placer task fixes the backbone and generates sidechains.  Each sample PDB
contains the predicted all-atom structure; GT is loaded from the processed npz.

Metrics:
  sc_rmsd       : Kabsch-aligned (on Cα) RMSD over sidechain heavy atoms (non N/CA/C/O)
  lig_rmsd      : same alignment applied to ligand heavy atoms
  combined_rmsd : sidechain + ligand atoms together under same alignment
  COV @1Å / @2Å : fraction of samples below threshold

Usage
-----
python _scripts/eval_plinder_placer.py \\
    --samples-dir ./sampling/plinder_pocket_train/placer/<model>/samples \\
    --data-dir    plinder_pocket_processed/train \\
    [--delta 1.0 2.0] [--n-jobs 8] [--verbose]
"""

import argparse
import json
import multiprocessing as mp
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proteinzen.boltz.data import const
from proteinzen.runtime.sampling.protein_pocket import (
    load_structure_from_npz,
    _crop_protein_to_pocket,
)

_GPU_SUFFIX = re.compile(r'_gpu\d+_batch\d+_idx\d+')

_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


# ============================================================
# Kabsch
# ============================================================

def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation R and translation t that aligns P onto Q."""
    cp, cq = P.mean(0), Q.mean(0)
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
# GT extraction from npz
# ============================================================

def _decode_atom_name(name_bytes) -> str:
    return "".join(chr(int(c) + 32) for c in name_bytes if c != 0)


def extract_gt_atoms(struct):
    """Return (prot_atom_names, prot_coords, lig_coords) for present heavy atoms."""
    protein_id    = const.chain_type_ids["PROTEIN"]
    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]
    prot_names, prot_coords, lig_coords = [], [], []
    for chain in struct.chains[struct.mask]:
        mol = int(chain["mol_type"])
        a0 = int(chain["atom_idx"])
        atoms = struct.atoms[a0 : a0 + int(chain["atom_num"])]
        for atom in atoms:
            if not atom["is_present"]:
                continue
            if atom["element"] == 1:  # hydrogen
                continue
            xyz = atom["coords"].astype(np.float64)
            if mol == protein_id:
                prot_names.append(_decode_atom_name(atom["name"]))
                prot_coords.append(xyz)
            elif mol == nonpolymer_id:
                lig_coords.append(xyz)
    prot_arr = np.stack(prot_coords) if prot_coords else np.zeros((0, 3), dtype=np.float64)
    lig_arr  = np.stack(lig_coords)  if lig_coords  else np.zeros((0, 3), dtype=np.float64)
    return prot_names, prot_arr, lig_arr


# ============================================================
# PDB parsing — ATOM and HETATM coords in file order
# ============================================================

def parse_pdb_coords(pdb_path: str):
    """Return (prot_coords, lig_coords) from ATOM and HETATM records."""
    prot, lig = [], []
    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].rstrip()
            try:
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except (ValueError, IndexError):
                continue
            if rec == "ATOM":
                prot.append(xyz)
            elif rec == "HETATM":
                lig.append(xyz)
    prot_arr = np.array(prot, dtype=np.float64) if prot else np.zeros((0, 3), dtype=np.float64)
    lig_arr  = np.array(lig,  dtype=np.float64) if lig  else np.zeros((0, 3), dtype=np.float64)
    return prot_arr, lig_arr


# ============================================================
# Per-sample evaluation
# ============================================================

def eval_sample(pdb_path: str, gt_prot_names: list, gt_prot: np.ndarray, gt_lig: np.ndarray):
    gen_prot, gen_lig = parse_pdb_coords(pdb_path)

    if len(gen_prot) != len(gt_prot):
        raise ValueError(f"protein atom count mismatch: gen={len(gen_prot)} gt={len(gt_prot)}")
    if len(gt_prot) == 0:
        raise ValueError("no protein atoms in GT")

    is_bb = np.array([n in _BACKBONE_ATOMS for n in gt_prot_names], dtype=bool)
    is_ca = np.array([n == "CA" for n in gt_prot_names], dtype=bool)

    if is_ca.sum() < 3:
        raise ValueError(f"too few Cα atoms for alignment: {is_ca.sum()}")

    R, t = kabsch(gen_prot[is_ca], gt_prot[is_ca])
    gen_prot_aligned = apply_transform(gen_prot, R, t)

    n_sc = int((~is_bb).sum())
    sc_rmsd = pos_rmsd(gt_prot[~is_bb], gen_prot_aligned[~is_bb]) if n_sc > 0 else float("nan")

    lig_rmsd = float("nan")
    combined_rmsd = float("nan")
    if len(gt_lig) > 0 and len(gen_lig) == len(gt_lig):
        gen_lig_aligned = apply_transform(gen_lig, R, t)
        lig_rmsd = pos_rmsd(gt_lig, gen_lig_aligned)
        if n_sc > 0:
            gt_combined  = np.concatenate([gt_prot[~is_bb],          gt_lig],          axis=0)
            gen_combined = np.concatenate([gen_prot_aligned[~is_bb],  gen_lig_aligned],  axis=0)
            combined_rmsd = pos_rmsd(gt_combined, gen_combined)
    elif len(gt_lig) > 0:
        lig_rmsd = float("inf")
        combined_rmsd = float("inf")

    return {"sc_rmsd": sc_rmsd, "lig_rmsd": lig_rmsd, "combined_rmsd": combined_rmsd,
            "n_sc_atoms": n_sc, "n_lig_atoms": len(gt_lig)}


# ============================================================
# Per-system job
# ============================================================

def _eval_system_job(system_id: str, pdb_paths: list, npz_path: str, max_protein_residues: int):
    try:
        struct = load_structure_from_npz(npz_path, include_h=False)
        struct = _crop_protein_to_pocket(struct, max_protein_residues)
        gt_prot_names, gt_prot, gt_lig = extract_gt_atoms(struct)
    except Exception as e:
        return system_id, [], f"npz load error: {e}", None

    records = []
    first_error = None
    for idx, p in enumerate(sorted(pdb_paths)):
        try:
            r = eval_sample(str(p), gt_prot_names, gt_prot, gt_lig)
            records.append({
                "system_id": system_id,
                "sample_idx": idx,
                "sc_rmsd": r["sc_rmsd"],
                "lig_rmsd": r["lig_rmsd"],
                "combined_rmsd": r["combined_rmsd"],
                "n_sc_atoms": r["n_sc_atoms"],
                "n_lig_atoms": r["n_lig_atoms"],
                "note": "",
            })
        except Exception as e:
            note = str(e)
            if first_error is None:
                first_error = f"{p.name}: {note}"
            records.append({
                "system_id": system_id,
                "sample_idx": idx,
                "sc_rmsd": float("inf"),
                "lig_rmsd": float("inf"),
                "combined_rmsd": float("inf"),
                "n_sc_atoms": 0,
                "n_lig_atoms": 0,
                "note": note,
            })
    return system_id, records, None, first_error


# ============================================================
# Aggregation helpers
# ============================================================

def mean_finite(vals):
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    return float(arr.mean()) if len(arr) else float("nan")


def cov(vals, delta):
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    return float((arr < delta).mean()) if len(arr) else float("nan")


def _min_per_system(records_by_system, key):
    mins = []
    for recs in records_by_system.values():
        vals = [r[key] for r in recs if np.isfinite(r[key])]
        if vals:
            mins.append(min(vals))
    return mins


def _mean_per_system(records_by_system, key):
    means = []
    for recs in records_by_system.values():
        vals = [r[key] for r in recs if np.isfinite(r[key])]
        if vals:
            means.append(float(np.mean(vals)))
    return means


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples-dir", type=Path, required=True,
                        help="Directory of generated PDB files")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Plinder processed split dir (manifest.json + structures/)")
    parser.add_argument("--max-protein-residues", type=int, default=20,
                        help="Pocket crop size used during sampling (default: 20)")
    parser.add_argument("--delta", type=float, nargs="+", default=[1.0, 2.0],
                        help="RMSD thresholds for COV reporting (Å; default: 1.0 2.0)")
    parser.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() // 2),
                        help="Parallel workers (default: half of CPU count)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path to write results JSON")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-system details")
    args = parser.parse_args()

    # ---- manifest ----
    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"manifest.json not found at {manifest_path}")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    system_ids_in_manifest = {rec["id"] for rec in manifest}
    print(f"Manifest: {len(system_ids_in_manifest)} systems")

    # ---- collect PDB files ----
    pdb_files = sorted(args.samples_dir.glob("*.pdb"))
    print(f"Generated PDBs: {len(pdb_files)}")

    groups: dict[str, list[Path]] = defaultdict(list)
    unmatched = []
    for p in pdb_files:
        stem = p.stem
        m = _GPU_SUFFIX.search(stem)
        if m:
            groups[stem[:m.start()]].append(p)
        else:
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                groups[parts[0]].append(p)
            else:
                unmatched.append(p.name)

    if unmatched:
        print(f"  Warning: {len(unmatched)} PDB(s) had unrecognised names — skipped")

    common = sorted(set(groups) & system_ids_in_manifest)
    extra   = set(groups) - system_ids_in_manifest
    missing = system_ids_in_manifest - set(groups)
    if extra:
        print(f"  Warning: {len(extra)} sampled systems not in manifest")
    if missing:
        print(f"  Note: {len(missing)} manifest systems have no samples")

    print(f"  Systems with generated samples : {len(groups)}")
    print(f"  Systems evaluated              : {len(common)}")

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
        delayed(_eval_system_job)(sid, pdbs, npz, args.max_protein_residues)
        for sid, pdbs, npz in tqdm(jobs, desc="evaluating")
    )

    # ---- collect ----
    all_records: list[dict] = []
    records_by_system: dict[str, list[dict]] = {}
    n_errors = 0
    first_errors_shown = 0

    for sid, sys_records, err, first_err in results:
        if err:
            print(f"  SKIP {sid}: {err}")
            continue
        if not sys_records:
            continue

        err_count = sum(1 for r in sys_records if r["note"])
        n_errors += err_count
        if first_err and first_errors_shown < 5:
            print(f"  [sample error] {sid} — {first_err}")
            first_errors_shown += 1
            if args.verbose:
            for r in sys_records:
                if r["note"]:
                    print(f"  [{sid}] sample {r['sample_idx']}: {r['note']}")
            sc_vals   = [r["sc_rmsd"]       for r in sys_records if np.isfinite(r["sc_rmsd"])]
            lig_vals  = [r["lig_rmsd"]      for r in sys_records if np.isfinite(r["lig_rmsd"])]
            comb_vals = [r["combined_rmsd"] for r in sys_records if np.isfinite(r["combined_rmsd"])]
            if sc_vals:
                msg = (f"  {sid}: {len(sc_vals)} samples, "
                       f"sc min={min(sc_vals):.3f} mean={np.mean(sc_vals):.3f} Å")
                if lig_vals:
                    msg += f"  lig min={min(lig_vals):.3f} mean={np.mean(lig_vals):.3f} Å"
                if comb_vals:
                    msg += f"  comb min={min(comb_vals):.3f} mean={np.mean(comb_vals):.3f} Å"
                print(msg)

        records_by_system[sid] = sys_records
        all_records.extend(sys_records)

    if n_errors:
        print(f"\nWarning: {n_errors} samples failed (skipped in summary)")

    if not records_by_system:
        print("No systems evaluated — check paths.")
        return

    # ---- aggregate ----
    all_sc   = [r["sc_rmsd"]       for r in all_records]
    all_lig  = [r["lig_rmsd"]      for r in all_records]
    all_comb = [r["combined_rmsd"] for r in all_records]

    sys_min_sc    = _min_per_system(records_by_system, "sc_rmsd")
    sys_mean_sc   = _mean_per_system(records_by_system, "sc_rmsd")
    sys_min_lig   = _min_per_system(records_by_system, "lig_rmsd")
    sys_mean_lig  = _mean_per_system(records_by_system, "lig_rmsd")
    sys_min_comb  = _min_per_system(records_by_system, "combined_rmsd")
    sys_mean_comb = _mean_per_system(records_by_system, "combined_rmsd")

    deltas = args.delta
    n_sys  = len(records_by_system)
    n_samp = sum(1 for v in all_sc if np.isfinite(v))

    def _block(label, sc_vals, lig_vals, comb_vals):
        print(f"\n--- {label} ---")
        print(f"  n              : {len([v for v in sc_vals if np.isfinite(v)])}")
        print(f"  sc_rmsd  mean  : {mean_finite(sc_vals):.3f} Å")
        for d in deltas:
            print(f"  COV sc  < {d:.1f}Å  : {cov(sc_vals, d)*100:.1f}%")
        n_lig = sum(1 for v in lig_vals if np.isfinite(v))
        if n_lig:
            print(f"  lig_rmsd mean  : {mean_finite(lig_vals):.3f} Å")
            for d in deltas:
                print(f"  COV lig < {d:.1f}Å  : {cov(lig_vals, d)*100:.1f}%")
        n_comb = sum(1 for v in comb_vals if np.isfinite(v))
        if n_comb:
            print(f"  combined mean  : {mean_finite(comb_vals):.3f} Å")
            for d in deltas:
                print(f"  COV comb< {d:.1f}Å  : {cov(comb_vals, d)*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"  PLINDER PLACER EVAL  —  {n_sys} systems,  {n_samp} samples")
    print(f"{'='*60}")

    _block("Per-sample (all samples pooled)",        all_sc,       all_lig,      all_comb)
    _block("Per-system best sample (min sc_rmsd)",   sys_min_sc,   sys_min_lig,  sys_min_comb)
    _block("Per-system mean sample",                 sys_mean_sc,  sys_mean_lig, sys_mean_comb)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "n_systems": n_sys,
            "n_samples": n_samp,
            "deltas": deltas,
            "per_sample": {
                "sc_rmsd_mean":       mean_finite(all_sc),
                "lig_rmsd_mean":      mean_finite(all_lig),
                "combined_rmsd_mean": mean_finite(all_comb),
                "cov_sc":   {f"{d:.1f}": cov(all_sc,   d) for d in deltas},
                "cov_lig":  {f"{d:.1f}": cov(all_lig,  d) for d in deltas},
                "cov_comb": {f"{d:.1f}": cov(all_comb, d) for d in deltas},
            },
            "per_system_best": {
                "sc_rmsd_mean":       mean_finite(sys_min_sc),
                "lig_rmsd_mean":      mean_finite(sys_min_lig),
                "combined_rmsd_mean": mean_finite(sys_min_comb),
                "cov_sc":   {f"{d:.1f}": cov(sys_min_sc,   d) for d in deltas},
                "cov_lig":  {f"{d:.1f}": cov(sys_min_lig,  d) for d in deltas},
                "cov_comb": {f"{d:.1f}": cov(sys_min_comb, d) for d in deltas},
            },
            "per_system_mean": {
                "sc_rmsd_mean":       mean_finite(sys_mean_sc),
                "lig_rmsd_mean":      mean_finite(sys_mean_lig),
                "combined_rmsd_mean": mean_finite(sys_mean_comb),
                "cov_sc":   {f"{d:.1f}": cov(sys_mean_sc,   d) for d in deltas},
                "cov_lig":  {f"{d:.1f}": cov(sys_mean_lig,  d) for d in deltas},
                "cov_comb": {f"{d:.1f}": cov(sys_mean_comb, d) for d in deltas},
            },
            "samples": [
                {k: v for k, v in r.items() if k != "note" or v}
                for r in all_records
            ],
        }
        with open(args.out, "w") as fh:
            json.dump(out_data, fh, indent=2)
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
