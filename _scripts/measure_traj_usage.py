"""Measure how much disk space is used by trajectory dumps vs. everything else,
under outputs/ and sampling/ -- sample.py writes per-run trajectories into a
"traj" dir (only when save_traj=true), and run_epoch_sample writes periodic
in-training sample dumps into "epoch_samples" dirs. Both can be large and are
candidates for cleanup separately from checkpoints/final samples.

Usage:
    python measure_traj_usage.py <root_dir> [--list]

Example:
    python measure_traj_usage.py /mnt/scratch/user/daviyang/proteinzen/outputs --list
    python measure_traj_usage.py /mnt/scratch/user/daviyang/proteinzen/sampling --list
"""
import argparse
import os
from pathlib import Path

_TRAJ_DIR_NAMES = {"traj", "epoch_samples"}
_TRAJ_DIR_PREFIXES = ("traj_", "gtalign_raw_")


def du_bytes(path: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--list", action="store_true",
                        help="Print every matched traj dir individually, not just the total")
    args = parser.parse_args()

    total_bytes = 0
    total_dirs = 0
    hits = []
    for dirpath, dirnames, _ in os.walk(args.root_dir):
        for d in list(dirnames):
            if d in _TRAJ_DIR_NAMES or d.startswith(_TRAJ_DIR_PREFIXES):
                full = Path(dirpath) / d
                size = du_bytes(full)
                hits.append((full, size))
                total_bytes += size
                total_dirs += 1
                dirnames.remove(d)  # don't descend into it, already measured

    if args.list:
        for full, size in sorted(hits, key=lambda x: -x[1]):
            print(f"  {size / 1e9:8.2f} GB  {full}")

    print(f"\nFound {total_dirs} trajectory-related dir(s) under {args.root_dir}")
    print(f"Total: {total_bytes / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
