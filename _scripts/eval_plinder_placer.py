#!/usr/bin/env python
"""Evaluate PLINDER placer task — sidechain RMSD.

The placer task fixes the backbone and generates sidechains.  Each sample PDB
has two MODEL blocks: MODEL 1 = GT, MODEL 2 = predicted.  Backbone atoms
(N, CA, C, O) are identical between models; sidechain atoms (everything else
in ATOM records) are generated.

Metrics reported:
  sc_rmsd  : RMSD over all sidechain heavy atoms in ATOM records
  COV @1Å / @2Å : fraction of samples with sc_rmsd below threshold

Usage
-----
python _scripts/eval_plinder_placer.py \\
    --samples-dir ./sampling/plinder_placer/<model>/samples \\
    --data-dir    plinder_placer_processed/val \\
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

_GPU_SUFFIX = re.compile(r'_gpu\d+_batch\d+_idx\d+')

_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


# ============================================================
# PDB parsing
# ============================================================

def _parse_models(pdb_path: str):
    """Parse a two-MODEL PDB.

    Returns two lists of atom dicts (one per model), each dict having:
      chain, resid, resname, atom_name, element, xyz.
    Only ATOM records are returned (not HETATM).
    """
    models = []
    current = []
    in_model = False

    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].rstrip()
            if rec == "MODEL":
                in_model = True
                current = []
            elif rec == "ENDMDL":
                models.append(current)
                in_model = False
            elif rec == "ATOM" and in_model:
                try:
                    atom_name = line[12:16].strip()
                    resname   = line[17:20].strip()
                    chain     = line[21]
                    resid     = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    element = line[76:78].strip() if len(line) > 76 else atom_name[0]
                    if not element:
                        element = atom_name.lstrip("0123456789")[0] if atom_name else "C"
                    current.append({
                        "chain": chain, "resid": resid, "resname": resname,
                        "atom_name": atom_name, "element": element.capitalize(),
                        "xyz": np.array([x, y, z], dtype=np.float64),
                    })
                except (ValueError, IndexError):
                    pass

    if not in_model and not models and current:
        # PDB without MODEL records — treat as single model
        models.append(current)

    return models


def _atoms_to_coord_dict(atoms):
    """Map (chain, resid, atom_name) → xyz."""
    return {(a["chain"], a["resid"], a["atom_name"]): a["xyz"] for a in atoms}


def eval_sample(pdb_path: str):
    """Evaluate one sample PDB.

    Returns dict with sc_rmsd (Å) and n_sc_atoms, or raises on failure.
    """
    models = _parse_models(pdb_path)
    if len(models) < 2:
        raise ValueError(f"Expected 2 MODEL blocks, got {len(models)}")

    gt_atoms   = models[0]
    pred_atoms = models[1]

    if len(gt_atoms) != len(pred_atoms):
        raise ValueError(
            f"Atom count mismatch: GT={len(gt_atoms)} pred={len(pred_atoms)}"
        )

    # build coord dict from GT; iterate pred in same order
    gt_coord   = _atoms_to_coord_dict(gt_atoms)
    pred_coord = _atoms_to_coord_dict(pred_atoms)

    common_keys = set(gt_coord) & set(pred_coord)
    sc_keys = [k for k in common_keys if k[2] not in _BACKBONE_ATOMS]

    if not sc_keys:
        raise ValueError("No sidechain atoms found")

    sc_gt   = np.stack([gt_coord[k]   for k in sc_keys])
    sc_pred = np.stack([pred_coord[k] for k in sc_keys])

    rmsd = float(np.sqrt(np.mean(np.sum((sc_gt - sc_pred) ** 2, axis=-1))))
    return {"sc_rmsd": rmsd, "n_sc_atoms": len(sc_keys)}


# ============================================================
# Per-system evaluation
# ============================================================

def _eval_system_job(system_id: str, pdb_paths: list):
    """Evaluate all samples for one system. Returns (system_id, records, err)."""
    records = []
    for idx, p in enumerate(sorted(pdb_paths)):
        try:
            r = eval_sample(str(p))
            records.append({
                "system_id": system_id,
                "sample_idx": idx,
                "sc_rmsd": r["sc_rmsd"],
                "n_sc_atoms": r["n_sc_atoms"],
                "note": "",
            })
        except Exception as e:
            records.append({
                "system_id": system_id,
                "sample_idx": idx,
                "sc_rmsd": float("inf"),
                "n_sc_atoms": 0,
                "note": str(e),
            })
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
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Plinder processed split dir (for manifest filtering). "
                             "If omitted, all PDBs in samples-dir are evaluated.")
    parser.add_argument("--delta", type=float, nargs="+", default=[1.0, 2.0],
                        help="RMSD thresholds for COV reporting (Å; default: 1.0 2.0)")
    parser.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() // 2),
                        help="Parallel workers (default: half of CPU count)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path to write results JSON")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-system details")
    args = parser.parse_args()

    # ---- optional manifest ----
    system_ids_in_manifest = None
    if args.data_dir is not None:
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

    if system_ids_in_manifest is not None:
        common = sorted(set(groups) & system_ids_in_manifest)
        extra   = set(groups) - system_ids_in_manifest
        missing = system_ids_in_manifest - set(groups)
        if extra:
            print(f"  Warning: {len(extra)} sampled systems not in manifest")
        if missing:
            print(f"  Note: {len(missing)} manifest systems have no samples")
    else:
        common = sorted(groups)

    print(f"  Systems with generated samples : {len(groups)}")
    print(f"  Systems evaluated              : {len(common)}")

    jobs = [(sid, groups[sid]) for sid in common]
    print(f"  Running {len(jobs)} systems with {args.n_jobs} workers...")

    # ---- parallel evaluation ----
    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(_eval_system_job)(sid, pdbs)
        for sid, pdbs in tqdm(jobs, desc="evaluating")
    )

    # ---- collect ----
    all_records: list[dict] = []
    records_by_system: dict[str, list[dict]] = {}
    n_errors = 0

    for sid, sys_records, err in results:
        if err:
            print(f"  SKIP {sid}: {err}")
            continue
        if not sys_records:
            continue

        err_count = sum(1 for r in sys_records if r["note"])
        n_errors += err_count
        if args.verbose:
            for r in sys_records:
                if r["note"]:
                    print(f"  [{sid}] sample {r['sample_idx']}: {r['note']}")
            vals = [r["sc_rmsd"] for r in sys_records if np.isfinite(r["sc_rmsd"])]
            if vals:
                print(
                    f"  {sid}: {len(vals)} samples, "
                    f"sc_rmsd min={min(vals):.3f} mean={np.mean(vals):.3f} Å"
                )

        records_by_system[sid] = sys_records
        all_records.extend(sys_records)

    if n_errors:
        print(f"\nWarning: {n_errors} samples failed (skipped)")

    if not records_by_system:
        print("No systems evaluated — check paths.")
        return

    # ---- aggregate ----
    all_sc   = [r["sc_rmsd"] for r in all_records]
    sys_min  = _min_per_system(records_by_system, "sc_rmsd")
    sys_mean = _mean_per_system(records_by_system, "sc_rmsd")

    deltas = args.delta
    n_sys = len(records_by_system)
    n_samp = sum(1 for v in all_sc if np.isfinite(v))

    print(f"\n{'='*60}")
    print(f"  PLINDER PLACER EVAL  —  {n_sys} systems,  {n_samp} samples")
    print(f"{'='*60}")

    print(f"\n--- Per-sample (all samples pooled) ---")
    print(f"  n            : {n_samp}")
    print(f"  sc_rmsd mean : {mean_finite(all_sc):.3f} Å")
    for d in deltas:
        print(f"  COV < {d:.1f}Å     : {cov(all_sc, d)*100:.1f}%")

    print(f"\n--- Per-system best sample (min sc_rmsd) ---")
    print(f"  n            : {len(sys_min)}")
    print(f"  sc_rmsd mean : {mean_finite(sys_min):.3f} Å")
    for d in deltas:
        print(f"  COV < {d:.1f}Å     : {cov(sys_min, d)*100:.1f}%")

    print(f"\n--- Per-system mean sample ---")
    print(f"  n            : {len(sys_mean)}")
    print(f"  sc_rmsd mean : {mean_finite(sys_mean):.3f} Å")
    for d in deltas:
        print(f"  COV < {d:.1f}Å     : {cov(sys_mean, d)*100:.1f}%")

    # ---- optional JSON output ----
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "n_systems": n_sys,
            "n_samples": n_samp,
            "deltas": deltas,
            "per_sample": {
                "sc_rmsd_mean": mean_finite(all_sc),
                "cov": {f"{d:.1f}": cov(all_sc, d) for d in deltas},
            },
            "per_system_best": {
                "sc_rmsd_mean": mean_finite(sys_min),
                "cov": {f"{d:.1f}": cov(sys_min, d) for d in deltas},
            },
            "per_system_mean": {
                "sc_rmsd_mean": mean_finite(sys_mean),
                "cov": {f"{d:.1f}": cov(sys_mean, d) for d in deltas},
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
