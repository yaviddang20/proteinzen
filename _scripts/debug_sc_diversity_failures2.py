"""
Like debug_sc_diversity_failures.py, but also builds/passes the custom
ligand residue-type set, for diagnosing failures that persist even after
custom ligand params are registered.

Usage:
    python debug_sc_diversity_failures2.py \
        --failed-ids sc_diversity_still_failed_ids.txt \
        --staging-dir plinder_pocket_sc_ensemble_staging_pdbs \
        --custom-ligand-sdf-dir sc_diversity_custom_ligand_sdfs \
        --out-dir /tmp/sc_diversity_debug2 \
        --limit 15
"""
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sidechain_diversity import generate_ensemble
from sidechain_diversity_batch import _build_custom_residue_type_set


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--failed-ids", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--custom-ligand-sdf-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    import pyrosetta
    pyrosetta.init("-mute all")
    rts = _build_custom_residue_type_set(args.custom_ligand_sdf_dir)

    ids = [l.strip() for l in args.failed_ids.read_text().splitlines() if l.strip()][: args.limit]
    for sid in ids:
        pdb_path = args.staging_dir / f"{sid}.pdb"
        try:
            generate_ensemble(pdb_path, args.out_dir / sid, n_samples=1, ntrials=1000, max_retries=2,
                               residue_type_set=rts)
            print(f"{sid}: unexpectedly OK this time")
        except Exception:
            print(f"=== {sid} ===")
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()
