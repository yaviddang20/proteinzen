"""Generate sample.py task YAML for ligand-conditioned protein generation.

Filters a Plinder manifest by CCD ligand code(s) (matched against the last
token of the system ID, e.g. "1abc__1.1_A_SAM" → "SAM"), then writes a
single YAML of LigandPocketConditionedSampling tasks.

Target ligands (edit TARGET_LIGANDS to change defaults):
  SAM  — S-adenosyl methionine
  FAD  — flavin adenine dinucleotide
  IAI  — iodoacetamide / as present in Plinder

Usage
-----
  # Default target ligands (SAM, FAD, IAI):
  python _scripts/make_ligand_cond_yaml.py \\
      --data-dir /path/to/plinder_processed/val \\
      --out-yaml sampling/ligand_cond/val

  # Custom ligand codes:
  python _scripts/make_ligand_cond_yaml.py \\
      --data-dir /path/to/plinder_processed/val \\
      --out-yaml sampling/ligand_cond/val \\
      --ligand-codes SAM FAD NAD ATP

  # Then run sampling:
  python sample.py \\
      sampler.tasks_yaml=sampling/ligand_cond/val_ligand_cond.yaml \\
      model_dir=/path/to/run \\
      out_dir=./out/ligand_cond
"""

import argparse
import json
import random
from pathlib import Path

import yaml

TARGET_LIGANDS = ["SAM", "FAD", "IAI"]


def system_mid(system_id: str) -> str:
    return system_id[1:3]


def main():
    parser = argparse.ArgumentParser(
        description="Build a sample.py YAML for ligand-conditioned protein generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Plinder processed split directory (manifest.json + structures/)")
    parser.add_argument("--out-yaml", type=Path, required=True,
                        help="Output path stem; _ligand_cond.yaml is appended")
    parser.add_argument("--ligand-codes", nargs="+", default=None,
                        help=f"CCD codes to include (default: {TARGET_LIGANDS})")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Samples to generate per system (default: 10)")
    parser.add_argument("--n-systems", type=int, default=30,
                        help="Max systems per ligand code, sampled with seed=42 (default: 30)")
    parser.add_argument("--trans-std", type=float, default=3.0,
                        help="Translation noise std (default: 3.0)")
    parser.add_argument("--max-protein-residues", type=int, default=100,
                        help="Protein crop radius in residues (default: 100)")
    parser.add_argument("--include-h", action="store_true",
                        help="Include hydrogen atoms (default: False)")
    args = parser.parse_args()

    codes = {c.upper() for c in (args.ligand_codes or TARGET_LIGANDS)}

    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Group by ligand code
    by_code: dict[str, list[dict]] = {c: [] for c in codes}
    for rec in manifest:
        code = rec["id"].rsplit("_", 1)[-1].upper()
        if code in codes:
            by_code[code].append(rec)

    for code, recs in sorted(by_code.items()):
        print(f"  {code}: {len(recs)} systems in manifest")

    # Sample per code
    rng = random.Random(42)
    selected: list[dict] = []
    for code in sorted(codes):
        recs = by_code[code]
        if args.n_systems is not None and args.n_systems < len(recs):
            recs = rng.sample(recs, args.n_systems)
        selected.extend(recs)

    tasks = []
    skipped = 0
    for rec in selected:
        sid = rec["id"]
        npz_path = args.data_dir / "structures" / system_mid(sid) / f"{sid}.npz"
        if not npz_path.exists():
            skipped += 1
            continue
        tasks.append({
            "_target_": "proteinzen.runtime.sampling.protein_pocket.LigandPocketConditionedSampling",
            "name": sid,
            "npz_path": str(npz_path.resolve()),
            "num_samples": args.num_samples,
            "trans_std": args.trans_std,
            "include_h": args.include_h,
            "max_protein_residues": args.max_protein_residues,
        })

    if skipped:
        print(f"Skipped {skipped} systems whose npz files were not found.")

    out_path = args.out_yaml.parent / (args.out_yaml.name + "_ligand_cond.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(tasks, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote {len(tasks)} tasks → {out_path}")


if __name__ == "__main__":
    main()
