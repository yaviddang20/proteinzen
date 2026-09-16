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
import collections
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
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
    parser.add_argument("--max-resolution", type=float, default=9.0,
                        help="Filter out systems whose entry resolution is missing or exceeds this value, in "
                             "Angstroms (matches AlphaFold2's training filter). Default: 9.0; pass "
                             "--max-resolution 0 to disable (keep everything, including null resolution).")
    parser.add_argument("--require-quality-pass", action="store_true", default=False,
                        help="Hard-filter out any system where Plinder's system_pass_validation_criteria is "
                             "False (or missing). Off by default. Note: with this on, every kept system's "
                             "system_pass_validation_criteria record field is trivially True, which makes the "
                             "training-time gate_low_quality_t soft-gate a no-op — use one or the other, not "
                             "both, unless you have a specific reason to.")
    parser.add_argument("--plip-sif", type=str, default=None,
                        help="Path to PLIP singularity .sif file. If provided alongside --include-waters, "
                             "water-mediated interaction counts are computed via PLIP and added to water_stats.json.")
    parser.add_argument("--include-waters", action="store_true", default=False,
                         help="Keep PLIP-detected interacting waters (residue HOH/DOD) as extra NONPOLYMER "
                              "chains, instead of stripping all water unconditionally (the default, to avoid "
                              "changing existing datasets/tasks). Waters are excluded from every "
                              "ligand-identification step (chain-count filter, SDF matching, rot-bond data, "
                              "interaction mask, fusion) and only reattached at the very end, so they never get "
                              "mistaken for the ligand. Whether they get noised during training is a separate, "
                              "task-level flag (noise_waters on SidechainRedesign/PocketPLACERTraining).")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Force reprocessing of every system (ignore cached output files) and, at the end "
                             "of a full/unrestricted run, delete any on-disk structures/records/auth_maps files "
                             "no longer in the new manifest. Never wipes the output directory upfront, so files "
                             "for still-valid systems stay in place the whole run — safe to run while something "
                             "else is training off this same directory. Pruning is skipped if combined with "
                             "--max-systems/--system-ids-file, since that would incorrectly delete valid systems "
                             "outside this restricted run's scope.")
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

    if args.include_waters:
        print("\nComputing water stats...")
        _WATER_NAMES = {"HOH", "DOD"}

        # collect (npz_path, system_id, plinder_system_cif) tuples
        system_entries = []
        for split_name in ["train", "val", "test"]:
            struct_root = args.outdir / split_name / "structures"
            for npz_path in struct_root.rglob("*.npz"):
                system_id = npz_path.stem
                cif_path = args.plinder_dir / "systems" / system_id / "system.cif"
                system_entries.append((npz_path, system_id, cif_path))

        def _water_stats_worker(entry):
            npz_path, system_id, cif_path = entry
            result = {"system_id": system_id, "n_waters": 0, "n_water_bridges": 0}
            try:
                data = np.load(npz_path, allow_pickle=False)
                residues = data["residues"]
                result["n_waters"] = int(np.isin(residues["name"], list(_WATER_NAMES)).sum())
            except Exception:
                pass
            if cif_path.exists() and args.plip_sif:
                try:
                    import subprocess, tempfile, os
                    script = (
                        "from openbabel import openbabel; "
                        "conv = openbabel.OBConversion(); conv.SetInAndOutFormats('cif','pdb'); "
                        f"mol = openbabel.OBMol(); conv.ReadFile(mol, '{cif_path}'); "
                        "import tempfile, os; f=tempfile.NamedTemporaryFile(suffix='.pdb',delete=False); tmp=f.name; f.close(); "
                        "conv.WriteFile(mol, tmp); "
                        "from plip.structure.preparation import PDBComplex; "
                        "pc = PDBComplex(); pc.load_pdb(tmp); pc.analyze(); "
                        "os.unlink(tmp); "
                        "total = sum(len(v.water_bridges) for v in pc.interaction_sets.values()); "
                        "print(total)"
                    )
                    r = subprocess.run(
                        ["singularity", "exec", "--bind", "/mnt/scratch",
                         args.plip_sif, "python3", "-c", script],
                        capture_output=True, text=True, timeout=60
                    )
                    lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip().lstrip('-').isdigit()]
                    if lines:
                        result["n_water_bridges"] = int(lines[-1])
                except Exception:
                    pass
            return result

        with multiprocessing.Pool(args.num_processes) as pool:
            results = list(pool.imap_unordered(_water_stats_worker, system_entries, chunksize=32))

        water_counts = collections.Counter(r["n_waters"] for r in results)
        bridge_counts = collections.Counter(r["n_water_bridges"] for r in results)
        water_stats = {
            "total_systems": len(results),
            "systems_with_waters": sum(v for k, v in water_counts.items() if k > 0),
            "water_count_histogram": {str(k): v for k, v in sorted(water_counts.items())},
            "systems_with_water_bridges": sum(v for k, v in bridge_counts.items() if k > 0),
            "water_bridge_count_histogram": {str(k): v for k, v in sorted(bridge_counts.items())},
        }
        with open(args.outdir / "water_stats.json", "w") as f:
            json.dump(water_stats, f, indent=2)
        print(f"Wrote water stats to {args.outdir / 'water_stats.json'}")
