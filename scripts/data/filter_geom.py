"""
Filter raw GEOM summary JSON by:
  - sanitizability of each conformer's rd_mol
  - no fragment molecules (single connected component)
  - connectivity consistency across conformers
  - stereo consistency across conformers

Reads:  datadir / summary_{dataset}.json
Writes: outdir  / filtered_summary_{dataset}.json
        outdir  / filter_errors_{dataset}.json
"""
import argparse
import json
import multiprocessing
import pickle
import traceback
from functools import partial
from pathlib import Path
from typing import Optional

import rdkit
from rdkit import Chem
from p_tqdm import p_umap
from tqdm import tqdm


def mol_to_smiles(mol, stereo: bool) -> str:
    mol = Chem.RemoveHs(mol)
    if stereo:
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    return Chem.MolToSmiles(mol, isomericSmiles=stereo, canonical=True)


def validate_entry(
    item: tuple[str, dict],
    datadir: Path,
) -> tuple[str, dict, Optional[str]]:
    """Returns (smiles, data, error_type). error_type is None if valid."""
    smiles, data = item

    try:
        data_path = datadir / data['pickle_path']
        with open(data_path, 'rb') as fp:
            data_dict = pickle.load(fp)
    except Exception:
        return (smiles, data, "load_error")

    conformer_data = data_dict.get('conformers', [])
    if not conformer_data:
        return (smiles, data, "no_conformers")

    # 1. Sanitize each conformer and get connectivity SMILES
    conformer_smiles = []
    for conformer in conformer_data:
        rd_mol = conformer['rd_mol']
        try:
            Chem.SanitizeMol(rd_mol)
            smi_no_stereo = mol_to_smiles(rd_mol, stereo=False)
        except Exception:
            return (smiles, data, "sanitization_error")
        conformer_smiles.append(smi_no_stereo)

    # 2. Check for fragments
    if "." in conformer_smiles[0]:
        return (smiles, data, "fragmented_molecule")

    # 3. Check connectivity consistency
    ref_no_stereo = conformer_smiles[0]
    for smi in conformer_smiles[1:]:
        if smi != ref_no_stereo:
            return (smiles, data, "inconsistent_connectivity")

    # 4. Check stereo consistency
    try:
        stereo_smiles = []
        for conformer in conformer_data:
            stereo_smiles.append(mol_to_smiles(conformer['rd_mol'], stereo=True))
    except Exception:
        return (smiles, data, "stereo_error")

    ref_stereo = stereo_smiles[0]
    for smi in stereo_smiles[1:]:
        if smi != ref_stereo:
            return (smiles, data, "inconsistent_stereo")

    return (smiles, data, None)


def filter_dataset(args) -> None:
    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = datadir / f"summary_{args.dataset}.json"
    print(f"Loading {summary_path} ...")
    with open(summary_path) as fp:
        raw = json.load(fp)

    metadata = [(smi, data) for smi, data in raw.items() if len(smi) > 1]
    print(f"Loaded {len(metadata)} entries.")

    pickle_option = rdkit.Chem.PropertyPickleOptions.AllProps
    rdkit.Chem.SetDefaultPickleProperties(pickle_option)

    max_processes = multiprocessing.cpu_count()
    num_processes = max(1, min(args.num_processes, max_processes, len(metadata)))
    parallel = num_processes > 1

    fn = partial(validate_entry, datadir=datadir)

    print("Filtering ...")
    if parallel:
        results = p_umap(fn, metadata, num_cpus=num_processes)
    else:
        results = [fn(item) for item in tqdm(metadata)]

    filtered = {}
    errors: dict[str, list[str]] = {}
    for smiles, data, error_type in results:
        if error_type is None:
            filtered[smiles] = data
        else:
            errors.setdefault(error_type, []).append(smiles)

    out_summary = outdir / f"filtered_summary_{args.dataset}.json"
    with out_summary.open("w") as fp:
        json.dump(filtered, fp, indent=4)
    print(f"Wrote {len(filtered)} entries to {out_summary}")

    out_errors = outdir / f"filter_errors_{args.dataset}.json"
    with out_errors.open("w") as fp:
        json.dump(errors, fp, indent=2)
    total_errors = sum(len(v) for v in errors.values())
    print(f"Errors written to {out_errors} ({total_errors} total):")
    for error_type, names in errors.items():
        print(f"  {error_type}: {len(names)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter GEOM conformer data.")
    parser.add_argument("--datadir", type=Path, required=True,
                        help="Directory containing summary_{dataset}.json and rdkit pickle files.")
    parser.add_argument("--dataset", type=str, required=True, choices=["qm9", "drugs"])
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Directory to write filtered_summary_{dataset}.json.")
    parser.add_argument("--num-processes", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    filter_dataset(args)
