#!/usr/bin/env python
"""Pallatom-style evaluation for the ligand_cond task: fixed-ligand, generated-protein
binder design, evaluated via Boltz2 single-sequence (msa=empty, no templates/pocket
constraints) refolding self-consistency -- following Wang et al. 2026,
"Pallatom-Ligand: an All-Atom Diffusion Model for Designing Ligand-Binding Proteins"
(ICLR 2026), Section 4.1-4.2 (verified directly against the paper text).

Also reports (from the SAME refold, no extra work) the small-molecule success criteria
used by two related, directly-comparable-in-spirit papers, since the metrics they need
(iPTM, min cross-chain PAE) come from the exact same Boltz2 refold call already made for
Pallatom's own criteria -- there's no reason to force a choice of one protocol, so all
three are always computed and printed side by side, labeled by name:
  - RFdiffusion3 (Butcher et al. 2025, bioRxiv, Fig 3c caption): backbone RMSD<1.5A &
    ligand RMSD<5A & min chain-pair PAE<1.5 & iPTM>0.8. Uses AF3.
  - Proteina-Complexa (Didi et al. 2026, ICLR 2026, Sec 3.3/Appendix F): min-ipAE<2 &
    binder Ca RMSD<2A & ligand RMSD<5A. Uses RF3. "min ipAE" is defined there as "the
    minimum entry of the cross-chain elements in the pAE matrix" -- the same statistic
    RFD3 calls "min chain-pair PAE", just a different name/threshold/folding engine.
Both use looser ligand-RMSD (<5A) than Pallatom-Ligand's <2A pose criterion -- the
single biggest reason absolute numbers aren't comparable across all three. Substituting
Boltz2 for AF3/RF3 means none of these numbers are directly comparable to the papers'
own reported figures either way -- only structurally analogous.

Reuses eval_plinder.py's ligand_cond machinery (eval_ligand_cond_sample, KNOWN_SMILES,
run_refolding, etc. -- including its Cα-only Kabsch alignment with the ligand carried
along rigidly, never independently re-aligned, and its RDKit-substructure-match-based
chemically-valid atom correspondence for ligand RMSD). This script adds only what's
specific to the Pallatom-Ligand benchmark:
  - Grouping one combined samples/ directory by CCD code.
  - LigandMPNN redesign of non-pocket residues (6A cutoff, pocket residues fixed) on
    top of the model's own raw generated sequence -- the paper's protocol, since
    Pallatom-Ligand (like our own ligand_cond task) co-generates sequence+structure
    directly and only additionally redesigns the non-binding-interface residues.
  - The three Pallatom success formulas (fold/pocket/pose_success), computed for BOTH
    the raw sequence and the LigandMPNN-redesigned sequence, reported separately.
  - Per-ligand-code success rates, and an overall rate computed as the average of the
    per-ligand rates (matching the paper's "Avg." row), not a pooled fraction.

Note: the paper itself uses AlphaFold3 (no MSA, 5 models/sequence) for structure
prediction; we substitute Boltz2 (no MSA, 1 model/sequence) since AF3 isn't
self-hostable -- same protocol, different (open) folding engine, so absolute numbers
aren't directly comparable to the paper's, only structurally analogous. Boltz2's
single-model-per-input also sidesteps the paper's unspecified "which of 5 AF3 models
counts as the result" choice entirely.

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

RAW_SUCCESS_KEYS = [("fold_success", "fold_success"),
                    ("pocket_success", "pocket_success"),
                    ("pose_success", "pose_success")]
MPNN_SUCCESS_KEYS = [("fold_success", "fold_success_mpnn"),
                     ("pocket_success", "pocket_success_mpnn"),
                     ("pose_success", "pose_success_mpnn")]

RFD3_RAW_KEYS = [("rfd3_success", "rfd3_success")]
RFD3_MPNN_KEYS = [("rfd3_success", "rfd3_success_mpnn")]
COMPLEXA_RAW_KEYS = [("complexa_success", "complexa_success")]
COMPLEXA_MPNN_KEYS = [("complexa_success", "complexa_success_mpnn")]

# Raw fields (from eval_ligand_cond_sample's return dict) that every *current* code
# version is expected to produce. A per-sample cache JSON missing any of these predates
# that field (e.g. cached before min_ipae/RFD3/Complexa support was added) and is treated
# as stale -- add to this tuple whenever a new field is wired into eval_ligand_cond_sample
# / run_refolding so old caches get auto-backfilled instead of silently reporting
# missing metrics as failures.
REQUIRED_RAW_KEYS = ("min_ipae",)


def _cache_is_complete(r: dict) -> bool:
    return all(k in r for k in REQUIRED_RAW_KEYS)


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


def _success_flags(protein_rmsd, protein_plddt, ligand_displacement, ligand_plddt, ligand_rmsd):
    """The three Pallatom-Ligand success formulas, verbatim from the paper (Sec 4.1):
    Protein-Fold Success: Ca-RMSD < 2A and protein-pLDDT > 80.
    Ligand-Pocket Success: Protein-Fold Success and ligand-Dcenter < 4A and ligand-pLDDT > 80.
    Ligand-Pose Success: Protein-Fold Success and ligand-RMSD < 2A."""
    fold = bool(_finite(protein_rmsd) and _finite(protein_plddt)
                and protein_rmsd < 2 and protein_plddt > 80)
    pocket = bool(fold and _finite(ligand_displacement) and _finite(ligand_plddt)
                  and ligand_displacement < 4 and ligand_plddt > 80)
    pose = bool(fold and _finite(ligand_rmsd) and ligand_rmsd < 2)
    return fold, pocket, pose


def _rfd3_success(ca_rmsd, lig_rmsd, min_ipae, iptm) -> bool:
    """RFdiffusion3's small-molecule success criterion, verbatim (Butcher et al. 2025,
    Fig 3c caption): backbone RMSD<1.5A & backbone-aligned ligand RMSD<5A & min
    chain-pair PAE<1.5 & iPTM>0.8."""
    return bool(_finite(ca_rmsd) and _finite(lig_rmsd) and _finite(min_ipae) and _finite(iptm)
                and ca_rmsd < 1.5 and lig_rmsd < 5 and min_ipae < 1.5 and iptm > 0.8)


def _complexa_success(ca_rmsd, lig_rmsd, min_ipae) -> bool:
    """Proteina-Complexa's small-molecule success criterion, verbatim (Didi et al. 2026,
    Appendix F): min ipAE<2 & binder Ca (sc)RMSD<2A & binder-aligned ligand (sc)RMSD<5A."""
    return bool(_finite(ca_rmsd) and _finite(lig_rmsd) and _finite(min_ipae)
                and min_ipae < 2 and ca_rmsd < 2 and lig_rmsd < 5)


def compute_success(r: dict) -> dict:
    """Success flags for the raw generated sequence, plus (if LigandMPNN redesign was
    run) a second set for the redesigned sequence. Computes all three papers' success
    criteria every time (Pallatom-Ligand, RFdiffusion3, Proteina-Complexa) -- they all
    come free from the same refold, so there's no reason to make the caller pick one."""
    fold, pocket, pose = _success_flags(
        r.get("ca_rmsd"), r.get("prot_plddt"), r.get("lig_displacement"), r.get("lig_plddt"), r.get("lig_rmsd"),
    )
    out = {
        "fold_success": fold, "pocket_success": pocket, "pose_success": pose,
        "rfd3_success": _rfd3_success(r.get("ca_rmsd"), r.get("lig_rmsd"), r.get("min_ipae"), r.get("iptm")),
        "complexa_success": _complexa_success(r.get("ca_rmsd"), r.get("lig_rmsd"), r.get("min_ipae")),
    }

    if "lmpnn_ca_rmsd_best" in r:
        fold_m, pocket_m, pose_m = _success_flags(
            r.get("lmpnn_ca_rmsd_best"), r.get("lmpnn_prot_plddt_best"),
            r.get("lmpnn_lig_displacement_best"), r.get("lmpnn_lig_plddt_best"),
            r.get("lmpnn_lig_rmsd_best"),
        )
        out.update({"fold_success_mpnn": fold_m, "pocket_success_mpnn": pocket_m, "pose_success_mpnn": pose_m})
        out["rfd3_success_mpnn"] = _rfd3_success(
            r.get("lmpnn_ca_rmsd_best"), r.get("lmpnn_lig_rmsd_best"),
            r.get("lmpnn_min_ipae_best"), r.get("lmpnn_iptm_best"),
        )
        out["complexa_success_mpnn"] = _complexa_success(
            r.get("lmpnn_ca_rmsd_best"), r.get("lmpnn_lig_rmsd_best"), r.get("lmpnn_min_ipae_best"),
        )
    return out


def _ligand_rate(results: list[dict], key: str) -> float:
    n = len(results)
    return (sum(1 for r in results if r.get(key)) / n) if n else float("nan")


def _report_per_ligand(results: list[dict], label: str, success_keys) -> list[str]:
    n = len(results)
    n_failed = sum(1 for r in results if "boltz_error" in r or not _finite(r.get("ca_rmsd")))
    lines = [f"{label}: n={n}  prediction_failures={n_failed}"]
    for name, key in success_keys:
        count = sum(1 for r in results if r.get(key))
        pct = f"{count/n*100:.1f}%" if n else "n/a"
        lines.append(f"  {name:<15}: {count}/{n}  ({pct})")
    return lines


def _report_overall(by_code: dict, label: str, success_keys) -> list[str]:
    """Average of each ligand's own rate -- matches the paper's 'Avg.' row exactly
    (rather than a pooled fraction over all samples, which only coincides with this
    when every ligand has the same sample count)."""
    groups = [v for v in by_code.values() if v]
    n_total = sum(len(v) for v in groups)
    all_results = [r for v in groups for r in v]
    n_failed = sum(1 for r in all_results if "boltz_error" in r or not _finite(r.get("ca_rmsd")))
    lines = [f"{label}: n={n_total} across {len(groups)} ligand(s)  prediction_failures={n_failed}"]
    for name, key in success_keys:
        rates = [_ligand_rate(v, key) for v in groups]
        avg = float(np.mean(rates)) if rates else float("nan")
        lines.append(f"  {name:<15}: {avg*100:.1f}%  (average across {len(rates)} ligands)")
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
    _repo_root = Path(__file__).parent.parent
    _default_ligandmpnn = str(_repo_root / "LigandMPNN" / "run.py")
    parser.add_argument("--ligandmpnn-script", type=str, default=_default_ligandmpnn,
                        help="Path to LigandMPNN run.py, used for the paper's pocket-fixed "
                             f"redesign step (default: {_default_ligandmpnn}). Pass an "
                             "empty string or nonexistent path to skip the +MPNN variant "
                             "entirely and only report the raw generated-sequence success rates.")
    parser.add_argument("--mpnn-n-seqs", type=int, default=1,
                        help="Sequences designed by LigandMPNN per structure (default: 1, "
                             "matching the paper's protocol of exactly one designed "
                             "sequence -- not a best-of-N pick).")
    parser.add_argument("--mpnn-cutoff", type=float, default=6.0,
                        help="Distance (A) defining pocket residues that stay fixed during "
                             "LigandMPNN redesign (default: 6.0, matches the paper).")
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

    run_mpnn = bool(args.ligandmpnn_script and Path(args.ligandmpnn_script).exists())
    if not run_mpnn:
        print(f"Note: LigandMPNN script not found at {args.ligandmpnn_script!r} -- "
              f"only raw generated-sequence success rates will be reported.")

    results_by_code: dict[str, list[dict]] = {}
    for code in ligand_codes:
        files = by_code.get(code, [])
        if not files:
            continue
        smiles = KNOWN_SMILES.get(code)
        if smiles is None:
            print(f"  {code}: no SMILES in KNOWN_SMILES -- skipping entirely")
            continue
        code_results = []
        for pdb_path in tqdm(files, desc=code):
            cache_path = per_sample_dir / f"{pdb_path.stem}.json"
            r = None
            if not args.overwrite and cache_path.exists():
                cached = json.loads(cache_path.read_text())
                # aggregate_only never triggers a (re)compute, even for a stale cache --
                # it means "just report on whatever's on disk". Otherwise, a cache missing
                # a currently-expected field (see REQUIRED_RAW_KEYS) is auto-backfilled
                # below rather than silently reported as a metric failure.
                if args.aggregate_only or _cache_is_complete(cached):
                    r = cached
            if r is None:
                if args.aggregate_only:
                    continue
                # force=args.overwrite: --overwrite means a COMPLETE redo, including
                # re-running Boltz/LigandMPNN. Without it (e.g. backfilling a stale
                # cache's missing fields), eval_ligand_cond_sample still only re-derives
                # metrics from whatever Boltz/LigandMPNN output already exists on disk --
                # run_refolding/_run_ligandmpnn independently skip the actual subprocess
                # calls unless force=True.
                r = eval_ligand_cond_sample(
                    pdb_path=pdb_path, smiles=smiles,
                    refold_input_dir=refold_input_dir, refold_output_dir=refold_output_dir,
                    boltz_cache=args.boltz_cache, run_pb=False, skip_fold=False,
                    contact_cutoff=args.contact_cutoff,
                    ligandmpnn_script=args.ligandmpnn_script if run_mpnn else None,
                    mpnn_n_seqs=args.mpnn_n_seqs, mpnn_cutoff=args.mpnn_cutoff,
                    force=args.overwrite,
                )
                r["ligand_code"] = code
                cache_path.write_text(json.dumps(r, indent=2, default=str))
            r.setdefault("ligand_code", code)
            r.update(compute_success(r))
            code_results.append(r)
        results_by_code[code] = code_results

    all_results = [r for v in results_by_code.values() for r in v]
    (args.out_dir / "results.json").write_text(json.dumps(all_results, indent=2, default=str))

    benchmarks = [
        ("PALLATOM-LIGAND (Wang et al. 2026): fold/pocket/pose, AF3 in the paper", RAW_SUCCESS_KEYS, MPNN_SUCCESS_KEYS),
        ("RFDIFFUSION3 (Butcher et al. 2025): backbone_RMSD<1.5A & lig_RMSD<5A & "
         "min_chain_pair_PAE<1.5 & iPTM>0.8, AF3 in the paper", RFD3_RAW_KEYS, RFD3_MPNN_KEYS),
        ("PROTEINA-COMPLEXA (Didi et al. 2026): min_ipAE<2 & binder_RMSD<2A & "
         "lig_RMSD<5A, RF3 in the paper", COMPLEXA_RAW_KEYS, COMPLEXA_MPNN_KEYS),
    ]

    report = [
        "=" * 70,
        "Small-molecule binder success criteria, Boltz2 single-sequence refold",
        "(same refold for all three -- only the success formula differs; see docstring",
        " for why absolute numbers aren't directly comparable across papers)",
        "=" * 70,
        "",
    ]
    for label, raw_keys, mpnn_keys in benchmarks:
        report.append(f"### {label}")
        report.append("")
        report.append("--- RAW (model's own generated sequence, no redesign) ---")
        report.append("")
        report.extend(_report_overall(results_by_code, "OVERALL", raw_keys))
        report.append("")
        for code, code_results in results_by_code.items():
            if not code_results:
                continue
            report.extend(_report_per_ligand(code_results, code, raw_keys))
        report.append("")

        if run_mpnn:
            report.append(f"--- +MPNN (LigandMPNN redesign of non-pocket residues, "
                          f"{args.mpnn_cutoff:.0f}A cutoff) ---")
            report.append("")
            report.extend(_report_overall(results_by_code, "OVERALL", mpnn_keys))
            report.append("")
            for code, code_results in results_by_code.items():
                if not code_results:
                    continue
                report.extend(_report_per_ligand(code_results, code, mpnn_keys))
            report.append("")
        report.append("")

    summary = "\n".join(report) + "\n"
    print(summary)
    (args.out_dir / "summary.txt").write_text(summary)
    print(f"Results: {args.out_dir / 'results.json'}")
    print(f"Summary: {args.out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
