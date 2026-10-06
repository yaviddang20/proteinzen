"""Delete old checkpoints, keeping best.ckpt, last.ckpt, the N most recent per
version, and (optionally) any checkpoint referenced by a saved sample run_config.yaml
-- sample.py writes the exact ckpt_path used for that run into <out_dir>/run_config.yaml,
so a checkpoint that isn't best/last/recent can still be the one behind samples you
actually care about.

Usage:
    python cleanup_checkpoints.py <outputs_dir> [--keep-last N] [--sampling-dir DIR] [--dry-run]

Example:
    python cleanup_checkpoints.py /mnt/scratch/user/daviyang/proteinzen/outputs --dry-run
    python cleanup_checkpoints.py /mnt/scratch/user/daviyang/proteinzen/outputs --keep-last 2 \\
        --sampling-dir /mnt/scratch/user/daviyang/proteinzen/sampling
"""
import argparse
import os
import re
from pathlib import Path

import yaml


# Leaf content dirs that can hold huge numbers of files but never contain
# run_config.yaml themselves -- pruned during the walk so this stays fast even
# on a sampling/ tree with hundreds of thousands of sample/trajectory files.
_PRUNE_DIR_NAMES = {
    "samples", "traj", "metadata", "mpnn_refold", "refold_inputs", "refold_outputs",
    "per_sample", "ligand_npz", "renders", "conformer_mols", "first_conformer_mols",
}
_PRUNE_DIR_PREFIXES = ("traj_", "gtalign_raw_", "_staged_")


def find_sampled_checkpoints(sampling_dir: Path) -> set[Path]:
    """Scan every run_config.yaml under sampling_dir (written by sample.py for each
    run) and collect the exact ckpt_path each one used. Prunes known-huge leaf
    content directories during the walk instead of a plain rglob, which would
    otherwise stat every sample/trajectory file just to check its name."""
    protected = set()
    for dirpath, dirnames, filenames in os.walk(sampling_dir):
        dirnames[:] = [
            d for d in dirnames
            if d not in _PRUNE_DIR_NAMES and not d.startswith(_PRUNE_DIR_PREFIXES)
        ]
        if "run_config.yaml" not in filenames:
            continue
        cfg_path = Path(dirpath) / "run_config.yaml"
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
        except Exception as e:
            print(f"  WARNING: couldn't parse {cfg_path}: {e}")
            continue
        ckpt_path = cfg.get("ckpt_path") if isinstance(cfg, dict) else None
        if ckpt_path:
            protected.add(Path(ckpt_path).resolve())
    return protected


def cleanup_version(ckpt_dir: Path, keep_last: int, dry_run: bool, sampled_protected: set[Path]):
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return 0, 0, 0

    protected_names = {"best.ckpt", "last.ckpt"}
    candidates = [c for c in ckpts if c.name not in protected_names]

    n_sample_protected = sum(1 for c in candidates if c.resolve() in sampled_protected)
    candidates = [c for c in candidates if c.resolve() not in sampled_protected]

    epoch_ckpts = sorted(
        candidates,
        key=lambda p: (
            # sort by step number if present, else by mtime
            int(m.group(1)) if (m := re.search(r'step=(\d+)', p.name)) else 0,
            p.stat().st_mtime,
        )
    )

    to_delete = epoch_ckpts[:-keep_last] if keep_last > 0 else epoch_ckpts
    freed = 0
    for c in to_delete:
        try:
            size = c.stat().st_size
            if not dry_run:
                c.unlink()
            freed += size
        except FileNotFoundError:
            pass

    return len(to_delete), freed, n_sample_protected


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outputs_dir", type=Path)
    parser.add_argument("--keep-last", type=int, default=1,
                        help="Number of most-recent epoch checkpoints to keep per version (default: 1)")
    parser.add_argument("--sampling-dir", type=Path, default=None,
                        help="Also protect any checkpoint referenced by a run_config.yaml "
                             "found anywhere under this directory (e.g. .../proteinzen/sampling).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be deleted without deleting")
    args = parser.parse_args()

    sampled_protected = set()
    if args.sampling_dir is not None:
        sampled_protected = find_sampled_checkpoints(args.sampling_dir)
        print(f"Found {len(sampled_protected)} distinct checkpoint(s) referenced by "
              f"run_config.yaml files under {args.sampling_dir}")

    ckpt_dirs = list(args.outputs_dir.rglob("checkpoints"))
    print(f"Found {len(ckpt_dirs)} checkpoint directories")
    if args.dry_run:
        print("DRY RUN — nothing will be deleted\n")

    total_deleted = 0
    total_freed = 0
    total_sample_protected = 0
    for ckpt_dir in sorted(ckpt_dirs):
        n, freed, n_sample_protected = cleanup_version(ckpt_dir, args.keep_last, args.dry_run, sampled_protected)
        total_sample_protected += n_sample_protected
        if n or n_sample_protected:
            verb = "Would delete" if args.dry_run else "Deleted"
            extra = f", protected {n_sample_protected} sample-referenced" if n_sample_protected else ""
            print(f"  {verb} {n} ckpts from {ckpt_dir.parent.name}/{ckpt_dir.name}  "
                  f"({freed / 1e9:.1f} GB){extra}")
            total_deleted += n
            total_freed += freed

    verb = "Would free" if args.dry_run else "Freed"
    print(f"\nTotal: {total_deleted} checkpoints removed, {verb} {total_freed / 1e9:.1f} GB, "
          f"{total_sample_protected} sample-referenced checkpoint(s) protected from deletion")


if __name__ == "__main__":
    main()
