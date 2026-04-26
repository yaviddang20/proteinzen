"""
Generate a sample.py tasks YAML for protein-pocket-conditioned ligand generation
from a processed Plinder data directory.

Each entry in the YAML will be a `protein_pocket_conditioned` task pointing at
one system's npz file.  The resulting YAML can be passed directly to sample.py.

Usage
-----
# All systems in the val split:
python scripts/make_plinder_pocket_yaml.py \
    --data-dir /path/to/plinder/processed/val \
    --out-yaml  pocket_tasks.yaml \
    --num-samples 10

# Restrict to specific system IDs:
python scripts/make_plinder_pocket_yaml.py \
    --data-dir /path/to/plinder/processed/val \
    --system-ids 1abc__1.2_B_L 2xyz__1.1_A_L \
    --out-yaml pocket_tasks.yaml \
    --num-samples 10

# Then run sampling:
python sample.py \
    sampler.tasks_yaml=pocket_tasks.yaml \
    model_dir=/path/to/run \
    out_dir=./pocket_samples
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
        description="Build a sample.py YAML for Plinder protein-pocket-conditioned sampling."
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Plinder processed split directory (contains manifest.json and structures/)")
    parser.add_argument("--out-yaml", type=Path, required=True,
                        help="Output YAML path")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of ligand conformers to generate per system")
    parser.add_argument("--system-ids", nargs="+", default=None,
                        help="Restrict to these system IDs (default: all in manifest.json)")
    parser.add_argument("--n-systems", type=int, default=30,
                        help="Randomly sample this many systems (seed=42); default: 30")
    args = parser.parse_args()

    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}. Run plinder.py finalize first.")

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

    tasks = []
    skipped = 0
    for record in manifest:
        system_id = record["id"]
        mid = system_mid(system_id)
        npz_path = args.data_dir / "structures" / mid / f"{system_id}.npz"
        if not npz_path.exists():
            skipped += 1
            continue
        tasks.append({
            "task": "protein_pocket_conditioned",
            "name": system_id,
            "npz_path": str(npz_path.resolve()),
            "num_samples": args.num_samples,
        })

    if skipped:
        print(f"Skipped {skipped} systems whose npz files were not found.")

    args.out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_yaml, "w") as f:
        yaml.dump({"tasks": tasks}, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote {len(tasks)} tasks to {args.out_yaml}")


if __name__ == "__main__":
    main()
