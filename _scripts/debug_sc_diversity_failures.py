"""
Re-run generate_ensemble on a sample of previously-failed structures with
full tracebacks printed, instead of the batch driver's swallowed
`f"error: {e}"` (which came out empty for these).

Usage:
    python debug_sc_diversity_failures.py \
        --failed-ids sc_diversity_failed_ids.txt \
        --staging-dir plinder_pocket_sc_ensemble_staging_pdbs \
        --out-dir /tmp/sc_diversity_debug \
        --limit 30
"""
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sidechain_diversity import generate_ensemble


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--failed-ids", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    import pyrosetta
    pyrosetta.init("-mute all")

    ids = [l.strip() for l in args.failed_ids.read_text().splitlines() if l.strip()][: args.limit]
    for sid in ids:
        pdb_path = args.staging_dir / f"{sid}.pdb"
        try:
            generate_ensemble(pdb_path, args.out_dir / sid, n_samples=1, ntrials=1000, max_retries=2)
            print(f"{sid}: unexpectedly OK this time")
        except Exception:
            print(f"=== {sid} ===")
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()
