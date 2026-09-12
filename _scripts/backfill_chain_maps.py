"""
Backfill chain_map.json (original Plinder chain_name -> gemmi single-char PDB
chain ID) for every system that already has a generated sidechain-diversity
ensemble. This mapping was computed transiently during the original
system.cif -> PDB conversion (prep_sc_diversity_inputs.py) but never saved --
re-derives it deterministically from the same source system.cif using the
same _remap_chains logic, so it exactly matches what's baked into the
already-written sample_*.pdb files.

Usage:
    python backfill_chain_maps.py \
        --ensembles-root plinder_pocket_processed_sc_ensembles \
        --plinder-dir plinder/2024-06/v2 \
        --num-workers 16
"""
import argparse
import json
import multiprocessing
from pathlib import Path

import gemmi

_CHAIN_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _remap_chains(st):
    """Same logic as prep_sc_diversity_inputs.py's _remap_chains -- must stay
    in sync so the recomputed mapping matches what's already in the PDBs."""
    mapping: dict[str, str] = {}
    idx = 0
    for model in st:
        for chain in model:
            if chain.name not in mapping:
                if len(chain.name) > 1:
                    while idx < len(_CHAIN_CHARS) and _CHAIN_CHARS[idx] in mapping.values():
                        idx += 1
                    mapping[chain.name] = _CHAIN_CHARS[idx] if idx < len(_CHAIN_CHARS) else chain.name[0]
                    idx += 1
                else:
                    mapping[chain.name] = chain.name
            chain.name = mapping[chain.name]
    return mapping


def _process_one(args):
    system_id, plinder_dir, out_dir = args
    cif_path = Path(plinder_dir) / "systems" / system_id / "system.cif"
    map_path = Path(out_dir) / "chain_map.json"
    try:
        st = gemmi.read_structure(str(cif_path))
        mapping = _remap_chains(st)
        map_path.write_text(json.dumps(mapping, indent=2))
        return system_id, None
    except Exception as e:
        return system_id, f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensembles-root", type=Path, required=True)
    parser.add_argument("--plinder-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    system_dirs = [
        d for d in args.ensembles_root.iterdir()
        if d.is_dir() and (d / "ensemble_stats.json").exists()
    ]
    print(f"{len(system_dirs)} systems with an ensemble")

    tasks = [(d.name, args.plinder_dir, d) for d in system_dirs]

    n_ok, failed = 0, []
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for system_id, err in pool.imap_unordered(_process_one, tasks, chunksize=32):
            if err is None:
                n_ok += 1
            else:
                failed.append((system_id, err))

    print(f"Wrote chain_map.json for {n_ok}/{len(system_dirs)} systems, {len(failed)} failed")
    if failed:
        (args.ensembles_root / "_chain_map_failures.json").write_text(json.dumps(failed, indent=2))
        print(f"Failure details -> {args.ensembles_root / '_chain_map_failures.json'}")
