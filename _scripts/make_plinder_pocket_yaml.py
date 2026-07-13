"""
Generate sample.py task YAMLs for Plinder protein-ligand sampling.

Produces two output files:
  <out-yaml-stem>_protein_cond.yaml  — fix protein, generate ligand
  <out-yaml-stem>_ligand_cond.yaml   — fix ligand, generate protein

Usage
-----
# All systems in the val split:
python _scripts/make_plinder_pocket_yaml.py \
    --data-dir /path/to/plinder_processed/val \
    --out-yaml  sampling/plinder/val \
    --num-samples 10

# Restrict to specific system IDs:
python _scripts/make_plinder_pocket_yaml.py \
    --data-dir /path/to/plinder_processed/val \
    --system-ids 1abc__1.2_B_L 2xyz__1.1_A_L \
    --out-yaml sampling/plinder/val \
    --num-samples 10

# Then run sampling (protein-cond direction):
python sample.py \
    sampler.tasks_yaml=sampling/plinder/val_protein_cond.yaml \
    model_dir=/path/to/run \
    out_dir=./plinder_samples/protein_cond
"""

import argparse
import json
import random
from pathlib import Path

import yaml


def system_mid(system_id: str) -> str:
    return system_id[1:3]


def main():
    parser = argparse.ArgumentParser(
        description="Build sample.py YAMLs for both Plinder conditioning directions."
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Plinder processed split directory (contains manifest.json and structures/)")
    parser.add_argument("--out-yaml", type=Path, required=True,
                        help="Output path stem; suffixes _protein_cond.yaml and _ligand_cond.yaml are appended")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples to generate per system (default: 10)")
    parser.add_argument("--system-ids", nargs="+", default=None,
                        help="Restrict to these system IDs (default: all in manifest.json)")
    parser.add_argument("--n-systems", type=int, default=30,
                        help="Randomly sample this many systems (seed=42); default: 30")
    parser.add_argument("--trans-std", type=float, default=3.0,
                        help="Translation noise std (default: 3.0, matches plinder training)")
    parser.add_argument("--max-protein-residues", type=int, default=100,
                        help="Crop protein to this many residues closest to ligand (default: 100)")
    parser.add_argument("--include-h", action="store_true",
                        help="Include hydrogen atoms (default: False)")
    args = parser.parse_args()

    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    if args.system_ids:
        requested = set(args.system_ids)
        manifest = [r for r in manifest if r["id"] in requested]
        missing = requested - {r["id"] for r in manifest}
        if missing:
            print(f"Warning: {len(missing)} requested system(s) not in manifest: {sorted(missing)[:5]}...")

    if args.n_systems is not None and args.n_systems < len(manifest):
        rng = random.Random(42)
        manifest = rng.sample(manifest, args.n_systems)

    protein_cond_tasks = []
    ligand_cond_tasks = []
    skipped = 0

    for record in manifest:
        system_id = record["id"]
        mid = system_mid(system_id)
        npz_path = args.data_dir / "structures" / mid / f"{system_id}.npz"
        if not npz_path.exists():
            skipped += 1
            continue

        common = {
            "name": system_id,
            "npz_path": str(npz_path.resolve()),
            "num_samples": args.num_samples,
            "trans_std": args.trans_std,
            "include_h": args.include_h,
            "max_protein_residues": args.max_protein_residues,
        }

        protein_cond_tasks.append({
            "_target_": "proteinzen.runtime.sampling.protein_pocket.ProteinPocketConditionedSampling",
            **common,
        })
        ligand_cond_tasks.append({
            "_target_": "proteinzen.runtime.sampling.protein_pocket.LigandPocketConditionedSampling",
            **common,
        })

    if skipped:
        print(f"Skipped {skipped} systems whose npz files were not found.")

    out_stem = args.out_yaml
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    protein_cond_path = out_stem.parent / (out_stem.name + "_protein_cond.yaml")
    ligand_cond_path  = out_stem.parent / (out_stem.name + "_ligand_cond.yaml")

    with open(protein_cond_path, "w") as f:
        yaml.dump(protein_cond_tasks, f, default_flow_style=False, sort_keys=False)

    with open(ligand_cond_path, "w") as f:
        yaml.dump(ligand_cond_tasks, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote {len(protein_cond_tasks)} tasks to {protein_cond_path}")
    print(f"Wrote {len(ligand_cond_tasks)} tasks to {ligand_cond_path}")


if __name__ == "__main__":
    main()
