"""
Modified from Boltz
https://github.com/jwohlwend/boltz/blob/main/scripts/process/rcsb.py
"""
import os
import argparse
import json
import multiprocessing
import pickle
import traceback
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rdkit
from rdkit import Chem
from mmcif import parse_mmcif
from p_tqdm import p_umap
from redis import Redis
from tqdm import tqdm
import hashlib

from proteinzen.boltz.data.types import ChainInfo, StructureInfo, InterfaceInfo, Record, Target, ConformerRecord, ConformerTarget

from proteinzen.data.featurize.mol.sampling import mol_to_struct, compute_rot_bond_fragments, compute_ring_atom_masks, compute_sym_groups

def hash_sequence(seq: str) -> str:
    """Hash a sequence."""
    return hashlib.sha256(seq.encode()).hexdigest()


def finalize(outdir: Path) -> None:
    """Run post-processing in main thread.

    Parameters
    ----------
    outdir : Path
        The output directory.

    """
    # Group records into a manifest
    records_dir = outdir / "records"

    failed_count = 0
    records = []
    for record in records_dir.rglob("*.json"):
        path = record
        try:
            with path.open("r") as f:
                records.append(json.load(f))
        except:  # noqa: E722
            failed_count += 1
            print(f"Failed to parse {record}")  # noqa: T201
    if failed_count > 0:
        print(f"Failed to parse {failed_count} entries.")  # noqa: T201
    else:
        print("All entries parsed successfully.")

    # Save manifest
    outpath = outdir / "manifest.json"
    with outpath.open("w") as f:
        json.dump(records, f)



def parse(datadir: Path, data: dict):
    data_id = data['data_id']
    data_path = datadir / data['pickle_path']
    with open(data_path, 'rb') as fp:
        data_dict = pickle.load(fp)

    conformer_data = sorted(data_dict['conformers'], key=lambda c: c['boltzmannweight'], reverse=True)[:20]

    structure_info_list = []
    boltzmann_weights_list = []
    structures = []
    rot_bond_data_list = []
    for conformer in conformer_data:
        rd_mol = conformer['rd_mol']

        structure = mol_to_struct(rd_mol, noise_ligand=False, include_h=True)
        rot_bonds, frag_a = compute_rot_bond_fragments(rd_mol)
        ring_masks = compute_ring_atom_masks(rd_mol)
        sym_groups, sym_group_sizes = compute_sym_groups(rd_mol)
        rot_bond_data_list.append({
            'rot_bonds': rot_bonds,
            'rot_frag_a': frag_a,
            'ring_masks': ring_masks,
            'sym_groups': sym_groups,
            'sym_group_sizes': sym_group_sizes,
        })

        structure_info = StructureInfo(
            deposited=None,
            revised=None,
            released=None,
            resolution=None,
            method=f"QM9:{data_dict['smiles']}",
            num_chains=1,
            num_interfaces=0,
        )

        structure_info_list.append(structure_info)
        boltzmann_weights_list.append(conformer['boltzmannweight'])
        structures.append(structure)

    total_weight = sum(boltzmann_weights_list)
    boltzmann_weights_list = [w / total_weight for w in boltzmann_weights_list]

    # Create chain metadata
    chain_info = []
    for i, chain in enumerate(structure.chains):
        chain_info.append(
            ChainInfo(
                chain_id=i,
                chain_name=chain["name"],
                msa_id="",  # FIX
                mol_type=int(chain["mol_type"]),
                cluster_id=-1,
                num_residues=int(chain["res_num"]),
            )
        )


    # Get interface metadata
    interface_info = []

    data_ids = [f"{data_id}_{i}" for i in range(len(structure_info_list))]
    # Create record
    record = ConformerRecord(
        ids=data_ids,
        structures=structure_info_list,
        chains=chain_info,
        interfaces=interface_info,
        boltzmann_weights=boltzmann_weights_list,
    )
    target = ConformerTarget(structures=structures, record=record)

    return target, rot_bond_data_list

def process_structure(
    data: tuple[str, dict],
    datadir: Path,
    outdir: Path,
    overwrite: bool = True,
) -> Optional[tuple[str, str]]:
    """Process a target. Returns (error_type, smiles) on failure, None on success."""
    smiles = data[0]
    data_dict = data[1]
    data_id = hash_sequence(smiles)
    data_dict['data_id'] = data_id
    mid = data_id[1:3]

    mid_out_dir = outdir / "structures" / mid
    mid_record_dir = outdir / "records" / mid
    if not mid_out_dir.exists():
        mid_out_dir.mkdir(parents=True, exist_ok=True)
    if not mid_record_dir.exists():
        mid_record_dir.mkdir(parents=True, exist_ok=True)

    record_path = outdir / "records" / mid / f"{data_id}.json"
    struct_path = outdir / "structures" / mid / f"{data_id}_0.npz"

    if not overwrite and struct_path.exists() and record_path.exists():
        return None

    try:
        # Parse the target
        target, rot_bond_data_list = parse(datadir, data_dict)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(f"Failed to parse {smiles}")
        return ("parse_error", smiles)

    for i, (structure, rot_bond_data) in enumerate(zip(target.structures, rot_bond_data_list)):
        struct_path = outdir / "structures" / mid / f"{data_id}_{i}.npz"
        save_dict = asdict(structure)
        save_dict['rot_bonds'] = rot_bond_data['rot_bonds']
        save_dict['rot_frag_a'] = rot_bond_data['rot_frag_a']
        save_dict['ring_masks'] = rot_bond_data['ring_masks']
        save_dict['sym_groups'] = rot_bond_data['sym_groups']
        save_dict['sym_group_sizes'] = rot_bond_data['sym_group_sizes']
        np.savez_compressed(struct_path, **save_dict)

    record = target.record
    with record_path.open("w") as f:
        json.dump(asdict(record), f)

    return None


def process(args, dataset_mode: str) -> None:
    """Run the data processing task."""
    # Create output directory
    outdir = args.outdir / dataset_mode
    outdir.mkdir(parents=True, exist_ok=True)

    # Create output directories
    records_dir = outdir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    structure_dir = outdir / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)

    datadir = Path(args.datadir)
    with open(datadir / f"{dataset_mode}_filtered_summary_{args.dataset}.json") as fp:
        metadata = json.load(fp)
        metadata = [t for t in metadata.items() if len(t[0]) > 1]

    # Set default pickle properties
    pickle_option = rdkit.Chem.PropertyPickleOptions.AllProps
    rdkit.Chem.SetDefaultPickleProperties(pickle_option)

    # Check if we can run in parallel
    max_processes = multiprocessing.cpu_count()
    num_processes = max(1, min(args.num_processes, max_processes, len(metadata)))
    parallel = num_processes > 1


    # Run processing
    print("Processing data...")
    fn = partial(process_structure, datadir=datadir, outdir=outdir, overwrite=args.overwrite)
    if parallel:
        results = p_umap(fn, metadata, num_cpus=num_processes)
    else:
        results = [fn(item) for item in tqdm(metadata)]

    # Aggregate errors and write single errors.json
    errors: dict[str, list[str]] = {}
    for result in results:
        if result is not None:
            error_type, name = result
            errors.setdefault(error_type, []).append(name)
    errors_path = outdir / "errors.json"
    with errors_path.open("w") as f:
        json.dump(errors, f, indent=2)
    total_errors = sum(len(v) for v in errors.values())
    print(f"Errors written to {errors_path} ({total_errors} total):")
    for error_type, names in errors.items():
        print(f"  {error_type}: {len(names)}")

    # Finalize
    finalize(outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MSA data.")
    parser.add_argument(
        "--datadir",
        type=Path,
        required=True,
        help="The data containing the MMCIF files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default="data",
        help="The output directory.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="The number of processes.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing output files (default: True).",
    )
    args = parser.parse_args()
    assert args.dataset in ['qm9', 'drugs']
    for dataset_mode in ['train', 'val', 'test']:
        process(args, dataset_mode)
