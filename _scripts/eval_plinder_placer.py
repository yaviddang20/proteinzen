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
from scipy import stats as scipy_stats
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

    ca_rmsd = pos_rmsd(gt_prot[is_ca], gen_prot_aligned[is_ca])
    aa_rmsd = pos_rmsd(gt_prot, gen_prot_aligned)
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

    return {"ca_rmsd": ca_rmsd, "aa_rmsd": aa_rmsd, "sc_rmsd": sc_rmsd,
            "lig_rmsd": lig_rmsd, "combined_rmsd": combined_rmsd,
            "n_sc_atoms": n_sc, "n_lig_atoms": len(gt_lig)}


# ============================================================
# Per-system job
# ============================================================

def _eval_system_job(system_id: str, pdb_paths: list, npz_path: str, max_protein_residues: int,
                     pred_lig_rmsds: dict):
    try:
        struct = load_structure_from_npz(npz_path, include_h=False)
        struct = _crop_protein_to_pocket(struct, max_protein_residues)
        gt_prot_names, gt_prot, gt_lig = extract_gt_atoms(struct)
    except Exception as e:
        return system_id, [], f"npz load error: {e}", None

    records = []
    first_error = None
    for idx, p in enumerate(sorted(pdb_paths)):
        pred_rmsd = pred_lig_rmsds.get(p.stem)
        try:
            r = eval_sample(str(p), gt_prot_names, gt_prot, gt_lig)
            records.append({
                "system_id": system_id,
                "sample_idx": idx,
                "pdb_stem": p.stem,
                "ca_rmsd": r["ca_rmsd"],
                "aa_rmsd": r["aa_rmsd"],
                "sc_rmsd": r["sc_rmsd"],
                "lig_rmsd": r["lig_rmsd"],
                "combined_rmsd": r["combined_rmsd"],
                "n_sc_atoms": r["n_sc_atoms"],
                "n_lig_atoms": r["n_lig_atoms"],
                "pred_lig_rmsd": pred_rmsd,
                "note": "",
            })
        except Exception as e:
            note = str(e)
            if first_error is None:
                first_error = f"{p.name}: {note}"
            records.append({
                "system_id": system_id,
                "sample_idx": idx,
                "pdb_stem": p.stem,
                "ca_rmsd": float("inf"),
                "aa_rmsd": float("inf"),
                "sc_rmsd": float("inf"),
                "lig_rmsd": float("inf"),
                "combined_rmsd": float("inf"),
                "n_sc_atoms": 0,
                "n_lig_atoms": 0,
                "pred_lig_rmsd": pred_rmsd,
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


def _select_by_pred_lig_rmsd(records_by_system):
    selected_lig = []
    n_has_pred = 0
    for recs in records_by_system.values():
        eligible = [r for r in recs if r.get("pred_lig_rmsd") is not None
                    and np.isfinite(r["lig_rmsd"])]
        if not eligible:
            continue
        n_has_pred += 1
        best = min(eligible, key=lambda r: r["pred_lig_rmsd"])
        selected_lig.append(best["lig_rmsd"])
    return selected_lig, n_has_pred


def _pred_vs_true_lig_rmsd_pairs(all_records):
    pairs = []
    for r in all_records:
        p = r.get("pred_lig_rmsd")
        t = r.get("lig_rmsd")
        if p is not None and t is not None and np.isfinite(p) and np.isfinite(t):
            pairs.append((float(p), float(t)))
    return pairs


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
    parser.add_argument("--metadata-path", type=Path, default=None,
                        help="Path to samples_metadata.json (default: {samples-dir}/samples_metadata.json). "
                             "If absent or pred_lig_rmsd is missing, PLACER selection metrics are skipped.")
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

    meta_path = args.metadata_path or (args.samples_dir / "samples_metadata.json")
    if not meta_path.exists():
        meta_path = args.samples_dir.parent / "samples_metadata.json"
    pred_lig_rmsds: dict[str, float | None] = {}
    if meta_path.exists():
        with open(meta_path) as fh:
            meta = json.load(fh)
        for key, entry in meta.items():
            stem = Path(key).name
            v = entry.get("pred_lig_rmsd")
            if v is not None:
                pred_lig_rmsds[stem] = float(v)
        n_with_pred = sum(1 for v in pred_lig_rmsds.values() if v is not None)
        print(f"  pred_lig_rmsd found for {n_with_pred}/{len(meta)} samples in metadata")
    else:
        print(f"  samples_metadata.json not found — PLACER selection metrics will be skipped")

    print(f"  Running {len(jobs)} systems with {args.n_jobs} workers...")

    # ---- parallel evaluation ----
    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(_eval_system_job)(sid, pdbs, npz, args.max_protein_residues, pred_lig_rmsds)
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
            ca_vals   = [r["ca_rmsd"]       for r in sys_records if np.isfinite(r["ca_rmsd"])]
            aa_vals   = [r["aa_rmsd"]       for r in sys_records if np.isfinite(r["aa_rmsd"])]
            sc_vals   = [r["sc_rmsd"]       for r in sys_records if np.isfinite(r["sc_rmsd"])]
            lig_vals  = [r["lig_rmsd"]      for r in sys_records if np.isfinite(r["lig_rmsd"])]
            comb_vals = [r["combined_rmsd"] for r in sys_records if np.isfinite(r["combined_rmsd"])]
            if sc_vals:
                msg = (f"  {sid}: {len(sc_vals)} samples, "
                       f"ca min={min(ca_vals):.3f} mean={np.mean(ca_vals):.3f} Å  "
                       f"aa min={min(aa_vals):.3f} mean={np.mean(aa_vals):.3f} Å  "
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
    all_ca   = [r["ca_rmsd"]       for r in all_records]
    all_aa   = [r["aa_rmsd"]       for r in all_records]
    all_sc   = [r["sc_rmsd"]       for r in all_records]
    all_lig  = [r["lig_rmsd"]      for r in all_records]
    all_comb = [r["combined_rmsd"] for r in all_records]

    sys_min_ca    = _min_per_system(records_by_system, "ca_rmsd")
    sys_mean_ca   = _mean_per_system(records_by_system, "ca_rmsd")
    sys_min_aa    = _min_per_system(records_by_system, "aa_rmsd")
    sys_mean_aa   = _mean_per_system(records_by_system, "aa_rmsd")
    sys_min_sc    = _min_per_system(records_by_system, "sc_rmsd")
    sys_mean_sc   = _mean_per_system(records_by_system, "sc_rmsd")
    sys_min_lig   = _min_per_system(records_by_system, "lig_rmsd")
    sys_mean_lig  = _mean_per_system(records_by_system, "lig_rmsd")
    sys_min_comb  = _min_per_system(records_by_system, "combined_rmsd")
    sys_mean_comb = _mean_per_system(records_by_system, "combined_rmsd")

    deltas = args.delta
    n_sys  = len(records_by_system)
    n_samp = sum(1 for v in all_sc if np.isfinite(v))

    def _block(label, ca_vals, aa_vals, sc_vals, lig_vals, comb_vals):
        print(f"\n--- {label} ---")
        print(f"  n              : {len([v for v in sc_vals if np.isfinite(v)])}")
        print(f"  ca_rmsd  mean  : {mean_finite(ca_vals):.3f} Å")
        print(f"  aa_rmsd  mean  : {mean_finite(aa_vals):.3f} Å")
        print(f"  sc_rmsd  mean  : {mean_finite(sc_vals):.3f} Å")
        for d in deltas:
            print(f"  COV ca  < {d:.1f}Å  : {cov(ca_vals, d)*100:.1f}%")
            print(f"  COV aa  < {d:.1f}Å  : {cov(aa_vals, d)*100:.1f}%")
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

    _block("Per-sample (all samples pooled)",
           all_ca,      all_aa,      all_sc,       all_lig,      all_comb)
    _block("Per-system best sample (min sc_rmsd)",
           sys_min_ca,  sys_min_aa,  sys_min_sc,   sys_min_lig,  sys_min_comb)
    _block("Per-system mean sample",
           sys_mean_ca, sys_mean_aa, sys_mean_sc,  sys_mean_lig, sys_mean_comb)

    placer_out: dict = {}
    pairs = _pred_vs_true_lig_rmsd_pairs(all_records)
    if pairs:
        pred_arr = np.array([p for p, _ in pairs])
        true_arr = np.array([t for _, t in pairs])
        mae = float(np.mean(np.abs(pred_arr - true_arr)))
        spearman_r, spearman_p = scipy_stats.spearmanr(pred_arr, true_arr)
        pearson_r, pearson_p = scipy_stats.pearsonr(pred_arr, true_arr)

        sel_lig, n_sys_with_pred = _select_by_pred_lig_rmsd(records_by_system)
        oracle_lig = sys_min_lig  # best achievable per system

        print(f"\n--- PLACER selection (pred_lig_rmsd head) ---")
        print(f"  systems with pred_lig_rmsd     : {n_sys_with_pred}")
        print(f"  sample pairs (pred vs true lig) : {len(pairs)}")
        print(f"  pred_lig_rmsd MAE              : {mae:.3f} Å")
        print(f"  Spearman r                     : {spearman_r:.3f}  (p={spearman_p:.2e})")
        print(f"  Pearson  r                     : {pearson_r:.3f}  (p={pearson_p:.2e})")
        print(f"  -- Selection by min pred_lig_rmsd (n={len(sel_lig)} systems) --")
        print(f"  selected lig_rmsd mean         : {mean_finite(sel_lig):.3f} Å")
        for d in deltas:
            print(f"  COV lig < {d:.1f}Å (selected)     : {cov(sel_lig, d)*100:.1f}%")
        print(f"  -- Oracle (min true lig_rmsd) --")
        print(f"  oracle lig_rmsd mean           : {mean_finite(oracle_lig):.3f} Å")
        for d in deltas:
            print(f"  COV lig < {d:.1f}Å (oracle)       : {cov(oracle_lig, d)*100:.1f}%")

        placer_out = {
            "n_systems_with_pred": n_sys_with_pred,
            "n_pairs": len(pairs),
            "pred_lig_rmsd_mae": mae,
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "selection_by_pred": {
                "lig_rmsd_mean": mean_finite(sel_lig),
                "cov_lig": {f"{d:.1f}": cov(sel_lig, d) for d in deltas},
            },
            "oracle": {
                "lig_rmsd_mean": mean_finite(oracle_lig),
                "cov_lig": {f"{d:.1f}": cov(oracle_lig, d) for d in deltas},
            },
        }
    else:
        print(f"\n  [PLACER selection metrics skipped — no pred_lig_rmsd values found in {meta_path}]")

    def _sumline(label, val_str):
        return f"  {label:<40}: {val_str}"

    slines = [
        f"PLINDER PLACER EVAL — {n_sys} systems, {n_samp} samples",
        "",
        "Per-sample (all samples pooled)",
    ]
    for d in deltas:
        slines.append(_sumline(f"COV sc  < {d:.1f} Å", f"{cov(all_sc,  d)*100:.1f}%"))
        slines.append(_sumline(f"COV lig < {d:.1f} Å", f"{cov(all_lig, d)*100:.1f}%"))
    slines += [
        _sumline("sc_rmsd  mean",  f"{mean_finite(all_sc):.3f} Å"),
        _sumline("lig_rmsd mean",  f"{mean_finite(all_lig):.3f} Å"),
        "",
        "Per-system best sample (min sc_rmsd)",
    ]
    for d in deltas:
        slines.append(_sumline(f"COV sc  < {d:.1f} Å (best)", f"{cov(sys_min_sc,  d)*100:.1f}%"))
        slines.append(_sumline(f"COV lig < {d:.1f} Å (best)", f"{cov(sys_min_lig, d)*100:.1f}%"))
    slines += [
        _sumline("sc_rmsd  mean (best)",  f"{mean_finite(sys_min_sc):.3f} Å"),
        _sumline("lig_rmsd mean (best)",  f"{mean_finite(sys_min_lig):.3f} Å"),
    ]
    if placer_out:
        slines += [
            "",
            "PLACER selection (by pred_lig_rmsd)",
            _sumline("pred_lig_rmsd MAE",  f"{placer_out['pred_lig_rmsd_mae']:.3f} Å"),
            _sumline("Spearman r",          f"{placer_out['spearman_r']:.3f}  (p={placer_out['spearman_p']:.2e})"),
            _sumline("Pearson  r",          f"{placer_out['pearson_r']:.3f}  (p={placer_out['pearson_p']:.2e})"),
        ]
        for d in deltas:
            slines.append(_sumline(f"COV lig < {d:.1f} Å (selected)", f"{placer_out['selection_by_pred']['cov_lig'][f'{d:.1f}']*100:.1f}%"))
            slines.append(_sumline(f"COV lig < {d:.1f} Å (oracle)",   f"{placer_out['oracle']['cov_lig'][f'{d:.1f}']*100:.1f}%"))
        slines += [
            _sumline("lig_rmsd mean (selected)", f"{placer_out['selection_by_pred']['lig_rmsd_mean']:.3f} Å"),
            _sumline("lig_rmsd mean (oracle)",   f"{placer_out['oracle']['lig_rmsd_mean']:.3f} Å"),
        ]
    summary = "\n".join(slines) + "\n"
    print("\n" + summary)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "n_systems": n_sys,
            "n_samples": n_samp,
            "deltas": deltas,
            "per_sample": {
                "ca_rmsd_mean":       mean_finite(all_ca),
                "aa_rmsd_mean":       mean_finite(all_aa),
                "sc_rmsd_mean":       mean_finite(all_sc),
                "lig_rmsd_mean":      mean_finite(all_lig),
                "combined_rmsd_mean": mean_finite(all_comb),
                "cov_ca":   {f"{d:.1f}": cov(all_ca,   d) for d in deltas},
                "cov_aa":   {f"{d:.1f}": cov(all_aa,   d) for d in deltas},
                "cov_sc":   {f"{d:.1f}": cov(all_sc,   d) for d in deltas},
                "cov_lig":  {f"{d:.1f}": cov(all_lig,  d) for d in deltas},
                "cov_comb": {f"{d:.1f}": cov(all_comb, d) for d in deltas},
            },
            "per_system_best": {
                "ca_rmsd_mean":       mean_finite(sys_min_ca),
                "aa_rmsd_mean":       mean_finite(sys_min_aa),
                "sc_rmsd_mean":       mean_finite(sys_min_sc),
                "lig_rmsd_mean":      mean_finite(sys_min_lig),
                "combined_rmsd_mean": mean_finite(sys_min_comb),
                "cov_ca":   {f"{d:.1f}": cov(sys_min_ca,   d) for d in deltas},
                "cov_aa":   {f"{d:.1f}": cov(sys_min_aa,   d) for d in deltas},
                "cov_sc":   {f"{d:.1f}": cov(sys_min_sc,   d) for d in deltas},
                "cov_lig":  {f"{d:.1f}": cov(sys_min_lig,  d) for d in deltas},
                "cov_comb": {f"{d:.1f}": cov(sys_min_comb, d) for d in deltas},
            },
            "per_system_mean": {
                "ca_rmsd_mean":       mean_finite(sys_mean_ca),
                "aa_rmsd_mean":       mean_finite(sys_mean_aa),
                "sc_rmsd_mean":       mean_finite(sys_mean_sc),
                "lig_rmsd_mean":      mean_finite(sys_mean_lig),
                "combined_rmsd_mean": mean_finite(sys_mean_comb),
                "cov_ca":   {f"{d:.1f}": cov(sys_mean_ca,   d) for d in deltas},
                "cov_aa":   {f"{d:.1f}": cov(sys_mean_aa,   d) for d in deltas},
                "cov_sc":   {f"{d:.1f}": cov(sys_mean_sc,   d) for d in deltas},
                "cov_lig":  {f"{d:.1f}": cov(sys_mean_lig,  d) for d in deltas},
                "cov_comb": {f"{d:.1f}": cov(sys_mean_comb, d) for d in deltas},
            },
            "placer_selection": placer_out if placer_out else None,
            "samples": [
                {k: v for k, v in r.items() if k != "note" or v}
                for r in all_records
            ],
        }
        with open(args.out, "w") as fh:
            json.dump(out_data, fh, indent=2)
        summary_path = args.out.with_name(args.out.stem + "_summary.txt")
        summary_path.write_text(summary)
        print(f"Results: {args.out}")
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
