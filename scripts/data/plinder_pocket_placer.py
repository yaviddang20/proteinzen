"""
Process Plinder systems (1-3 protein chains) into the plinder_pocket_placer dataset.

plinder.py only keeps systems with exactly one protein chain, filtering out systems
with 2-3 protein chains (`chain_filter_protein{2,3}_ligand1`). This script accepts
1-3-protein-chain systems; whenever a system has 2 or 3 protein chains, they're
fused into a single chain/entity — residues are concatenated in chain order under
one fresh, continuous residue index (no gap between the original chains) — so
multi-chain pockets can be used to train the pocket PLACER model alongside ordinary
single-chain systems, since PLACER only cares about the local pocket around the
ligand rather than original chain boundaries. `fuse_protein_chains` is a no-op for
single-protein-chain systems, so this is a strict superset of plinder.py's dataset.

Reuses plinder.py's parsing, filtering, and per-system processing wholesale; the
only difference is which protein-chain counts are accepted (1-3, instead of just 1)
and that `fuse_multi_chain=True` is passed through so `process_system` calls
`fuse_protein_chains` before featurization.

Output layout mirrors plinder.py: one subdirectory per split (train/val/test),
each containing structures/, records/, auth_maps/, manifest.json, plus a top-level
dataset_stats.yaml with per-split and total system counts.
"""

import argparse
import multiprocessing
import os
from pathlib import Path

import rdkit
import yaml

from plinder import load_annotation_table, load_clusters, load_split, process

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Plinder 2-3-protein-chain systems (fused into one chain) into the plinder_pocket_placer dataset."
    )
    parser.add_argument("--ccd-path", type=Path, default=Path(os.environ.get("REPO_ROOT", ".")) / "ccd.pkl",
                        help="Path to ccd.pkl (default: $REPO_ROOT/ccd.pkl)")
    parser.add_argument("--plinder-dir", type=Path, required=True,
                        help="Path to plinder data root (e.g. /mnt/scratch/.../plinder/2024-06/v2)")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Output root — one subdirectory per split (train/val/test)")
    parser.add_argument("--cluster-algorithm", type=str, default="communities")
    parser.add_argument("--cluster-directed", action="store_true", default=False)
    parser.add_argument("--cluster-metric", type=str, default="pli_qcov")
    parser.add_argument("--cluster-threshold", type=int, default=50)
    parser.add_argument("--num-processes", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--system-ids-file", type=Path, default=None,
                        help="Optional text file of allowed system IDs (one per line); from filter_plinder_pocket.py")
    parser.add_argument("--max-systems", type=int, default=None,
                        help="Cap number of systems per split (for debugging)")
    parser.add_argument("--dedupe-assemblies", action=argparse.BooleanOptionalAction, default=False,
                        help="Keep only the lowest-assembly-index system per apparent NCS-duplicate group "
                             "(same pdb_id/receptor-chain-count/ligand-chain-count, different assembly index). "
                             "KNOWN BUG: the grouping key doesn't verify ligand identity, so distinct systems "
                             "sharing a PDB entry + chain-count shape get incorrectly collapsed too. "
                             "Default: False until fixed; pass --dedupe-assemblies to enable anyway.")
    parser.add_argument("--dedup-cluster-threshold", type=int, default=95,
                        help="Keep only the best-resolution representative per cluster at this threshold "
                             "before processing. Uses the same algorithm/metric as --cluster-*. "
                             "Default: 95; pass --dedup-cluster-threshold 0 to disable.")
    parser.add_argument("--max-ligand-atoms", type=int, default=200,
                        help="Filter out systems whose ligand has more than this many total atoms, including H (default: 200)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Delete and recreate the output directory before processing")
    parser.add_argument("--pocket-data-dir", type=Path,
                        default=Path(os.environ.get("REPO_ROOT", ".")) / "plinder_pocket_alpha_spheres",
                        help="Directory of per-system alpha-sphere .npy files from filter_plinder_pocket.py")
    args = parser.parse_args()

    # Fixed for this dataset: accept 1-3 protein-chain systems; 2-3 protein chains get
    # fused into one chain (fuse_protein_chains is a no-op for single-protein-chain systems).
    args.allowed_protein_chain_counts = (1, 2, 3)
    args.fuse_multi_chain = True

    # Set rdkit pickle options
    pickle_option = rdkit.Chem.PropertyPickleOptions.AllProps
    rdkit.Chem.SetDefaultPickleProperties(pickle_option)

    # Load shared data once
    print("Loading clusters...")
    clusters = load_clusters(
        args.plinder_dir,
        algorithm=args.cluster_algorithm,
        directed=args.cluster_directed,
        metric=args.cluster_metric,
        threshold=args.cluster_threshold,
    )
    print(f"Loaded {len(clusters)} cluster assignments")

    print("Loading annotation table...")
    annotations = load_annotation_table(args.plinder_dir)
    print(f"Loaded {len(annotations)} annotation rows")

    print("Loading split...")
    split = load_split(args.plinder_dir)
    print(f"Loaded {len(split)} split assignments")

    split_counts = {}
    for split_name in ["train", "val", "test"]:
        print(f"\n=== Processing split: {split_name} ===")
        split_args = argparse.Namespace(**{**vars(args), "splits": [split_name], "outdir": args.outdir / split_name})
        split_counts[split_name] = process(split_args, clusters, annotations, split)

    stats = {
        **split_counts,
        "total": {
            "systems": sum(v["systems"] for v in split_counts.values()),
            "clusters": sum(v["clusters"] for v in split_counts.values()),
        },
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    with open(args.outdir / "dataset_stats.yaml", "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote dataset stats to {args.outdir / 'dataset_stats.yaml'}")
