"""Delete old checkpoints, keeping best.ckpt, last.ckpt, and the N most recent per version.

Usage:
    python cleanup_checkpoints.py <outputs_dir> [--keep-last N] [--dry-run]

Example:
    python cleanup_checkpoints.py /mnt/scratch/user/daviyang/proteinzen/outputs --dry-run
    python cleanup_checkpoints.py /mnt/scratch/user/daviyang/proteinzen/outputs --keep-last 2
"""
import argparse
import os
import re
from pathlib import Path


def cleanup_version(ckpt_dir: Path, keep_last: int, dry_run: bool):
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return 0, 0

    protected = {"best.ckpt", "last.ckpt"}
    epoch_ckpts = sorted(
        [c for c in ckpts if c.name not in protected],
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

    return len(to_delete), freed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outputs_dir", type=Path)
    parser.add_argument("--keep-last", type=int, default=1,
                        help="Number of most-recent epoch checkpoints to keep per version (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be deleted without deleting")
    args = parser.parse_args()

    ckpt_dirs = list(args.outputs_dir.rglob("checkpoints"))
    print(f"Found {len(ckpt_dirs)} checkpoint directories")
    if args.dry_run:
        print("DRY RUN — nothing will be deleted\n")

    total_deleted = 0
    total_freed = 0
    for ckpt_dir in sorted(ckpt_dirs):
        n, freed = cleanup_version(ckpt_dir, args.keep_last, args.dry_run)
        if n:
            verb = "Would delete" if args.dry_run else "Deleted"
            print(f"  {verb} {n} ckpts from {ckpt_dir.parent.name}/{ckpt_dir.name}  "
                  f"({freed / 1e9:.1f} GB)")
            total_deleted += n
            total_freed += freed

    verb = "Would free" if args.dry_run else "Freed"
    print(f"\nTotal: {total_deleted} checkpoints removed, {verb} {total_freed / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
