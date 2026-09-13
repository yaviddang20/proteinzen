#!/usr/bin/env python
"""Pallatom-style evaluation for the ligand_cond task: fixed-ligand, generated-protein
binder design, evaluated via Boltz2 single-sequence (msa=empty, no templates/pocket
constraints) refolding self-consistency.

Reuses eval_plinder.py's ligand_cond machinery (eval_ligand_cond_sample, KNOWN_SMILES,
run_refolding, etc. -- including its Cα-only Kabsch alignment with the ligand carried
along rigidly, never independently re-aligned, and its RDKit-substructure-match-based
chemically-valid atom correspondence for ligand RMSD). This script adds only what's
specific to the Pallatom-Ligand benchmark: grouping one combined samples/ directory by
CCD code, applying the three Pallatom success-rate formulas, and reporting fractions
overall and per ligand (with denominators and prediction failures).

Usage
-----
  # 1) generate the sampling tasks (100 designs x 8 ligands):
  python _scripts/make_ligand_cond_yaml.py \\
      --ligand-codes FAD FMN SAM DOG SRO LDP IAI OQO \\
      --num-samples 100 --out-yaml sampling/plinder/pallatom/val

  # 2) sample.py (see run_pallatom_ligand_cond.sh for the full invocation)

  # 3) this script:
  python _scripts/eval_pallatom_ligand_cond.py \\
      --samples-dir sampling/plinder/pallatom/<model>/samples \\
      --out-dir eval/pallatom_ligand_cond/<model>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from eval_plinder import KNOWN_SMILES, _GPU_SUFFIX, eval_ligand_cond_sample  # noqa: E402


def _ligand_code_for(pdb_path: Path) -> str:
    """Recover the CCD code from a sample filename built by
    make_ligand_cond_yaml.py (task name f"{code}_conf{i}") + sample.py's
    "{task_name}_gpu{rank}_batch{idx}_idx{sid}.pdb" naming."""
    stem = pdb_path.stem
    m = _GPU_SUFFIX.search(stem)
    sid = stem[: m.start()] if m else stem.rsplit("_", 1)[0]
    return sid.split("_conf")[0]


def _finite(v) -> bool:
    try:
        return np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def compute_success(r: dict) -> dict:
    protein_rmsd = r.get("ca_rmsd")
    protein_plddt = r.get("prot_plddt")
    ligand_displacement = r.get("lig_displacement")
    ligand_plddt = r.get("lig_plddt")
    ligand_rmsd = r.get("lig_rmsd")

    fold_success = bool(
        _finite(protein_rmsd) and _finite(protein_plddt)
        and protein_rmsd < 2 and protein_plddt > 80
    )
    pocket_success = bool(
        fold_success and _finite(ligand_displacement) and _finite(ligand_plddt)
        and ligand_displacement < 4 and ligand_plddt > 80
    )
    pose_success = bool(fold_success and _finite(ligand_rmsd) and ligand_rmsd < 2)
    return {"fold_success": fold_success, "pocket_success": pocket_success, "pose_success": pose_success}


def _report(results: list[dict], label: str) -> list[str]:
    n = len(results)
    n_failed = sum(1 for r in results if "boltz_error" in r or not _finite(r.get("ca_rmsd")))
    n_fold = sum(1 for r in results if r.get("fold_success"))
    n_pocket = sum(1 for r in results if r.get("pocket_success"))
    n_pose = sum(1 for r in results if r.get("pose_success"))
    lines = [f"{label}: n={n}  prediction_failures={n_failed}"]
    for name, count in [("fold_success", n_fold), ("pocket_success", n_pocket), ("pose_success", n_pose)]:
        pct = f"{count/n*100:.1f}%" if n else "n/a"
        lines.append(f"  {name:<15}: {count}/{n}  ({pct})")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="The same directory sample.py wrote samples/ into -- eval "
                             "artifacts (refold_inputs/, per_sample/, results.json, "
                             "summary.txt) are written alongside samples/ in this same "
                             "directory, not a separate eval tree (matches run_eval_plinder.sh).")
    parser.add_argument("--samples-dir", type=Path, default=None,
                        help="Directory of generated PDBs for ALL ligand codes together. "
                             "Defaults to {out_dir}/samples if not set.")
    parser.add_argument("--ligand-codes", nargs="+", default=None,
                        help="Restrict to these ligand codes only. Default: auto-discover "
                             "from the sample filenames themselves (whatever "
                             "make_ligand_cond_yaml.py was actually run with) -- no need "
                             "to repeat the ligand list here.")
    parser.add_argument("--boltz-cache", type=Path, default=None)
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--aggregate-only", action="store_true", default=False,
                        help="Skip evaluating any not-yet-cached samples; just aggregate what's cached.")
    args = parser.parse_args()

    args.samples_dir = args.samples_dir or (args.out_dir / "samples")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    refold_input_dir = args.out_dir / "refold_inputs"
    refold_output_dir = args.out_dir / "refold_outputs"
    per_sample_dir = args.out_dir / "per_sample"
    per_sample_dir.mkdir(exist_ok=True)

    pdb_files = sorted(args.samples_dir.glob("*.pdb"))
    if not pdb_files:
        sys.exit(f"No PDB files found in {args.samples_dir}")

    by_code: dict[str, list[Path]] = {}
    for p in pdb_files:
        by_code.setdefault(_ligand_code_for(p), []).append(p)

    ligand_codes = args.ligand_codes or sorted(by_code.keys())
    unknown = [c for c in ligand_codes if c not in by_code]
    if unknown:
        print(f"Warning: requested ligand code(s) with no samples found: {unknown}")
    for code in ligand_codes:
        print(f"  {code}: {len(by_code.get(code, []))} samples found")

    all_results = []
    for code in ligand_codes:
        files = by_code.get(code, [])
        if not files:
            continue
        smiles = KNOWN_SMILES.get(code)
        if smiles is None:
            print(f"  {code}: no SMILES in KNOWN_SMILES -- skipping entirely")
            continue
        for pdb_path in tqdm(files, desc=code):
            cache_path = per_sample_dir / f"{pdb_path.stem}.json"
            if not args.overwrite and cache_path.exists():
                r = json.loads(cache_path.read_text())
            elif args.aggregate_only:
                continue
            else:
                r = eval_ligand_cond_sample(
                    pdb_path=pdb_path, smiles=smiles,
                    refold_input_dir=refold_input_dir, refold_output_dir=refold_output_dir,
                    boltz_cache=args.boltz_cache, run_pb=False, skip_fold=False,
                    contact_cutoff=args.contact_cutoff,
                )
                r["ligand_code"] = code
                cache_path.write_text(json.dumps(r, indent=2, default=str))
            r.setdefault("ligand_code", code)
            r.update(compute_success(r))
            all_results.append(r)

    (args.out_dir / "results.json").write_text(json.dumps(all_results, indent=2, default=str))

    report = [
        "=" * 70,
        "Pallatom-style criteria, Boltz2 single-sequence evaluation",
        "=" * 70,
        "",
    ]
    report.extend(_report(all_results, "OVERALL"))
    report.append("")
    for code in ligand_codes:
        code_results = [r for r in all_results if r.get("ligand_code") == code]
        if not code_results:
            continue
        report.append(f"--- {code} ---")
        report.extend(_report(code_results, code))
        report.append("")

    summary = "\n".join(report) + "\n"
    print(summary)
    (args.out_dir / "summary.txt").write_text(summary)
    print(f"Results: {args.out_dir / 'results.json'}")
    print(f"Summary: {args.out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
