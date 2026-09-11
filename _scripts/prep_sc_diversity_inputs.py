"""
Build the input PDB list for sidechain_diversity_batch.py from a
plinder_pocket_processed manifest: for each system_id, convert the raw
Plinder system.cif (full protein+ligand complex) to PDB via gemmi, matching
the format sidechain_diversity.py expects (PyRosetta pose_from_pdb).

Usage:
    python prep_sc_diversity_inputs.py \
        --manifest plinder_pocket_processed/train/manifest.json \
        --plinder-dir plinder/2024-06/v2 \
        --out-dir plinder_pocket_sc_ensemble_staging_pdbs \
        --list-out sc_diversity_full_train_list.txt \
        --num-workers 16
"""
import argparse
import json
import multiprocessing
from pathlib import Path
from typing import Optional

import gemmi

_CHAIN_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _remap_chains(st):
    """Remap multi-char chain names to single-char in-place. Returns old->new mapping."""
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


def _cif_to_pdb(cif_path: Path, pdb_path: Path) -> Optional[str]:
    """Convert mmCIF to PDB using gemmi. Returns None on success, else an error message."""
    try:
        st = gemmi.read_structure(str(cif_path))
        st.remove_hydrogens()
        _remap_chains(st)
        st.write_pdb(str(pdb_path))
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _convert_one(task):
    system_id, plinder_dir, out_dir = task
    cif_path = Path(plinder_dir) / "systems" / system_id / "system.cif"
    pdb_path = Path(out_dir) / f"{system_id}.pdb"
    if pdb_path.exists() and pdb_path.stat().st_size > 0:
        return system_id, str(pdb_path), None
    if not cif_path.exists():
        return system_id, None, "missing system.cif"
    err = _cif_to_pdb(cif_path, pdb_path)
    if err is None:
        return system_id, str(pdb_path), None
    return system_id, None, err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plinder-dir", type=Path, required=True, help="Path up to and including the version dir, e.g. plinder/2024-06/v2")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where converted PDBs are staged")
    parser.add_argument("--list-out", type=Path, required=True, help="Text file of successfully converted PDB paths, one per line")
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    manifest = json.load(open(args.manifest))
    system_ids = [r["id"] for r in manifest]
    print(f"{len(system_ids)} systems in manifest")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(sid, args.plinder_dir, args.out_dir) for sid in system_ids]

    ok, failed = [], []
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for system_id, pdb_path, err in pool.imap_unordered(_convert_one, tasks, chunksize=32):
            if err is None:
                ok.append(pdb_path)
            else:
                failed.append((system_id, err))

    with open(args.list_out, "w") as f:
        f.write("\n".join(sorted(ok)) + "\n")

    print(f"Converted {len(ok)}/{len(system_ids)}, {len(failed)} failed")
    if failed:
        fail_path = args.list_out.with_suffix(".failed.json")
        json.dump(failed, open(fail_path, "w"), indent=2)
        print(f"Failure details -> {fail_path}")
