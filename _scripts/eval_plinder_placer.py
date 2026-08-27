#!/usr/bin/env python
"""Evaluate PLINDER placer task — sidechain RMSD.

The placer task fixes the backbone and generates sidechains.  Each sample PDB
contains the predicted all-atom structure; GT is loaded from the processed npz.

Metrics:
  sc_rmsd  : Kabsch-aligned (on Cα) RMSD over sidechain heavy atoms (non N/CA/C/O)
  COV @1Å / @2Å : fraction of samples with sc_rmsd below threshold

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
from proteinzen.runtime.sampling.protein_pocket import load_structure_from_npz

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

def extract_gt_protein_atoms(struct):
    """Return (atom_names, coords) for present heavy protein atoms in struct order."""
    protein_id = const.chain_type_ids["PROTEIN"]
    names, coords = [], []
    for chain in struct.chains[struct.mask]:
        if int(chain["mol_type"]) != protein_id:
            continue
        a0 = int(chain["atom_idx"])
        atoms = struct.atoms[a0 : a0 + int(chain["atom_num"])]
        for atom in atoms:
            if not atom["is_present"]:
                continue
            if atom["element"] == 1:  # hydrogen
                continue
            name = "".join(chr(int(c)) for c in atom["name"]).strip()
            names.append(name)
            coords.append(atom["coords"].astype(np.float64))
    return names, (np.stack(coords) if coords else np.zeros((0, 3), dtype=np.float64))


# ============================================================
# PDB parsing — ATOM records only, flat coord array
# ============================================================

def parse_pdb_prot_coords(pdb_path: str) -> np.ndarray:
    """Return (N, 3) float64 of all ATOM-record coords in file order."""
    coords = []
    with open(pdb_path) as fh:
        for line in fh:
            if line[:6].rstrip() != "ATOM":
                continue
            try:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                pass
    return np.array(coords, dtype=np.float64) if coords else np.zeros((0, 3), dtype=np.float64)


# ============================================================
# Per-sample evaluation
# ============================================================

def eval_sample(pdb_path: str, gt_atom_names: list, gt_coords: np.ndarray):
    gen_coords = parse_pdb_prot_coords(pdb_path)

    if len(gen_coords) != len(gt_coords):
        raise ValueError(
            f"atom count mismatch: gen={len(gen_coords)} gt={len(gt_coords)}"
        )
    if len(gt_coords) == 0:
        raise ValueError("no protein atoms in GT")

    is_bb = np.array([n in _BACKBONE_ATOMS for n in gt_atom_names], dtype=bool)
    is_ca = np.array([n == "CA" for n in gt_atom_names], dtype=bool)
    n_sc = int((~is_bb).sum())

    if is_ca.sum() < 3:
        raise ValueError(f"too few Cα atoms for alignment: {is_ca.sum()}")
    if n_sc == 0:
        raise ValueError("no sidechain atoms (all Gly?)")

    R, t = kabsch(gen_coords[is_ca], gt_coords[is_ca])
    gen_aligned = apply_transform(gen_coords, R, t)

    rmsd = pos_rmsd(gt_coords[~is_bb], gen_aligned[~is_bb])
    return {"sc_rmsd": rmsd, "n_sc_atoms": n_sc}


# ============================================================
# Per-system job
# ============================================================

def _eval_system_job(system_id: str, pdb_paths: list, npz_path: str):
    try:
        struct = load_structure_from_npz(npz_path, include_h=False)
        gt_atom_names, gt_coords = extract_gt_protein_atoms(struct)
    except Exception as e:
        return system_id, [], f"npz load error: {e}"

    records = []
    first_error = None
    for idx, p in enumerate(sorted(pdb_paths)):
        try:
            r = eval_sample(str(p), gt_atom_names, gt_coords)
            records.append({
                "system_id": system_id,
                "sample_idx": idx,
                "sc_rmsd": r["sc_rmsd"],
                "n_sc_atoms": r["n_sc_atoms"],
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
                "n_sc_atoms": 0,
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
        delayed(_eval_system_job)(sid, pdbs, npz)
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
            vals = [r["sc_rmsd"] for r in sys_records if np.isfinite(r["sc_rmsd"])]
            if vals:
                print(
                    f"  {sid}: {len(vals)} samples, "
                    f"sc_rmsd min={min(vals):.3f} mean={np.mean(vals):.3f} Å"
                )

        records_by_system[sid] = sys_records
        all_records.extend(sys_records)

    if n_errors:
        print(f"\nWarning: {n_errors} samples failed (skipped in summary)")

    if not records_by_system:
        print("No systems evaluated — check paths.")
        return

    # ---- aggregate ----
    all_sc   = [r["sc_rmsd"] for r in all_records]
    sys_min  = _min_per_system(records_by_system, "sc_rmsd")
    sys_mean = _mean_per_system(records_by_system, "sc_rmsd")

    deltas = args.delta
    n_sys  = len(records_by_system)
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
