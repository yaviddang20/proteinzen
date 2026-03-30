"""
Count max heavy atoms and max total atoms (heavy + H) across the GEOM drugs dataset.
Reads directly from NPZ files. Saves top 5 biggest molecules as PDB with hydrogens.

Usage:
  python3 _scripts/count_atoms.py [--split train|val|test|all] [--workers N]
"""
import argparse
import heapq
import json
import pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RDKIT_DIR = PROJECT_DIR / "data" / "rdkit" / "drugs"
DATA_DIR = PROJECT_DIR / "data" / "geom_drugs_conformers"
OUT_DIR = PROJECT_DIR / "sampling" / "count_atoms_top5"


def get_splits(split_arg):
    if split_arg == "all":
        return ["train", "val", "test"]
    return [split_arg]


def get_npz_paths(splits):
    paths = []
    seen = set()
    for split in splits:
        struct_dir = DATA_DIR / split / "structures"
        for p in struct_dir.glob("*/*.npz"):
            if not p.stem.endswith("_0"):
                continue
            mol_hash = p.stem[:-2]
            if mol_hash in seen:
                continue
            seen.add(mol_hash)
            paths.append(p)
    return paths


def process_npz(npz_path):
    try:
        data = np.load(npz_path)
        atoms = data["atoms"]
        heavy = int(np.sum(atoms["element"] != 1))
        total = len(atoms)
        return str(npz_path), heavy, total
    except Exception:
        return str(npz_path), 0, 0


def get_smiles(npz_path):
    npz_path = Path(npz_path)
    mol_hash = npz_path.stem[:-2]
    mid = mol_hash[1:3]
    # search across all splits
    for split in ["train", "val", "test"]:
        record_path = DATA_DIR / split / "records" / mid / f"{mol_hash}.json"
        if record_path.exists():
            rec = json.load(open(record_path))
            try:
                method = rec["structures"][0]["method"]
                if method.startswith("QM9:"):
                    return method[4:]
            except Exception:
                pass
    return None


def save_pdb_with_h(smiles, out_path):
    from rdkit import Chem
    fname = smiles.replace('/', '_') + '.pickle'
    if len(fname) > 255:
        import glob as _glob
        prefix = fname[:100].replace('[', '[[]').replace(']', '[]]')
        matches = [Path(p) for p in _glob.glob(str(RDKIT_DIR / f"{prefix}*.pickle"))]
        if not matches:
            return False
        pkl_path = matches[0]
    else:
        pkl_path = RDKIT_DIR / fname
    if not pkl_path.exists():
        return False
    d = pickle.load(open(str(pkl_path), "rb"))
    mol = d["conformers"][0]["rd_mol"]
    Chem.MolToPDBFile(mol, str(out_path))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--workers", type=int, default=cpu_count() // 2)
    args = parser.parse_args()

    splits = get_splits(args.split)
    npz_paths = get_npz_paths(splits)
    print(f"Found {len(npz_paths)} unique molecules across {splits}")

    # process in parallel
    with Pool(processes=args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(process_npz, npz_paths, chunksize=64)):
            results.append(r)
            if (i + 1) % 5000 == 0:
                cur_heavy = max(r[1] for r in results)
                cur_total = max(r[2] for r in results)
                print(f"  {i+1} checked, max_heavy={cur_heavy}, max_total={cur_total}")

    max_heavy = max(results, key=lambda x: x[1])
    max_total = max(results, key=lambda x: x[2])

    top5_total = heapq.nlargest(5, results, key=lambda x: x[2])
    top5_heavy = heapq.nlargest(5, results, key=lambda x: x[1])

    print(f"\nMax heavy atoms:      {max_heavy[1]}  ({Path(max_heavy[0]).stem})")
    print(f"Max total atoms (+H): {max_total[2]}  ({Path(max_total[0]).stem})")
    print(f"\nSuggested max_crop_rigids (next multiple of 16): {((max_total[2] - 1) // 16 + 1) * 16}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for label, top5, key_fn in [
        ("total", top5_total, lambda x: f"total{x[2]}_heavy{x[1]}"),
        ("heavy", top5_heavy, lambda x: f"heavy{x[1]}_total{x[2]}"),
    ]:
        print(f"\nTop 5 by {label} atoms:")
        for rank, (npz_str, heavy, total) in enumerate(top5, 1):
            smiles = get_smiles(npz_str)
            print(f"  {rank}. heavy={heavy} total={total}  {Path(npz_str).stem}")
            if smiles:
                out_pdb = OUT_DIR / f"top{label}{rank}_{key_fn((npz_str, heavy, total))}.pdb"
                ok = save_pdb_with_h(smiles, out_pdb)
                print(f"     -> {out_pdb.name}" if ok else "     -> pickle not found")
            else:
                print(f"     -> SMILES not found")


if __name__ == "__main__":
    main()
