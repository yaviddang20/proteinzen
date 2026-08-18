"""
Compute ligand/protein composition stats for a processed Plinder dataset
(scripts/data/plinder.py or scripts/data/plinder_pocket_placer.py output), so you
can see e.g. ligand-size and protein-size distributions and the largest
ligand/protein in the dataset, without manually poking through manifest.json or
opening any .npz structure files.

Everything here is read straight off manifest.json (no npz I/O needed): the
ligand SMILES is already stored per-record as record['smiles'], and protein
residue counts are already stored per-chain as chain['num_residues'] /
chain['num_resolved_residues'].

Reads, per split:
  {outdir}/{split}/manifest.json      Record list (written by plinder.py's finalize())
  {outdir}/{split}/filter_stats.json  filter-reason counts (written by plinder.py's process())

Writes:
  {outfile} (default {outdir}/composition_stats.yaml)

Usage:
  python scripts/data/plinder_stats.py --outdir /path/to/plinder_processed
"""
import argparse
import json
import multiprocessing
import statistics
from pathlib import Path
from typing import Optional

import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors

from proteinzen.boltz.data import const

PROTEIN_ID = const.chain_type_ids["PROTEIN"]
NONPOLYMER_ID = const.chain_type_ids["NONPOLYMER"]


def _ligand_size(args) -> Optional[tuple]:
    """Return (system_id, n_heavy_atoms, mol_wt), or None if unparseable/missing."""
    system_id, smiles = args
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return system_id, smiles, mol.GetNumAtoms(), Descriptors.MolWt(mol)


def _ligand_size_stats(records: list, num_processes: int) -> dict:
    pairs = [(r.get("id"), r.get("smiles")) for r in records]
    if num_processes > 1 and len(pairs) > 1000:
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(_ligand_size, pairs, chunksize=200)
    else:
        results = [_ligand_size(p) for p in pairs]
    results = [r for r in results if r is not None]
    n_missing = len(pairs) - len(results)

    stats = {"n_ligands": len(pairs)}
    if n_missing:
        stats["n_missing_or_unparseable"] = n_missing
    if not results:
        return stats

    atom_counts = [r[2] for r in results]
    mol_wts = [r[3] for r in results]
    largest = max(results, key=lambda r: r[2])
    smallest = min(results, key=lambda r: r[2])
    stats.update({
        "atom_count": {
            "min": min(atom_counts),
            "max": max(atom_counts),
            "mean": round(statistics.mean(atom_counts), 2),
            "median": statistics.median(atom_counts),
        },
        "mol_wt": {
            "min": round(min(mol_wts), 2),
            "max": round(max(mol_wts), 2),
            "mean": round(statistics.mean(mol_wts), 2),
            "median": round(statistics.median(mol_wts), 2),
        },
        "largest_ligand": {"system_id": largest[0], "smiles": largest[1], "n_atoms": largest[2], "mol_wt": round(largest[3], 2)},
        "smallest_ligand": {"system_id": smallest[0], "smiles": smallest[1], "n_atoms": smallest[2], "mol_wt": round(smallest[3], 2)},
    })
    return stats


def _protein_size_stats(records: list) -> dict:
    counts = []  # (system_id, n_residues, n_resolved_residues, n_protein_chains)
    for r in records:
        chains = r.get("chains") or []
        protein_chains = [c for c in chains if c.get("mol_type") == PROTEIN_ID]
        if not protein_chains:
            continue
        n_res = sum(c.get("num_residues") or 0 for c in protein_chains)
        n_resolved = sum(c.get("num_resolved_residues") or 0 for c in protein_chains)
        counts.append((r.get("id"), n_res, n_resolved, len(protein_chains)))

    stats = {"n_records_with_protein": len(counts)}
    if not counts:
        return stats

    n_res_list = [c[1] for c in counts]
    largest = max(counts, key=lambda c: c[1])
    smallest = min(counts, key=lambda c: c[1])
    multi_chain = [c for c in counts if c[3] > 1]
    stats.update({
        "num_residues": {
            "min": min(n_res_list),
            "max": max(n_res_list),
            "mean": round(statistics.mean(n_res_list), 2),
            "median": statistics.median(n_res_list),
        },
        "largest_protein": {"system_id": largest[0], "num_residues": largest[1], "num_resolved_residues": largest[2]},
        "smallest_protein": {"system_id": smallest[0], "num_residues": smallest[1], "num_resolved_residues": smallest[2]},
        "n_records_with_multiple_protein_chains": len(multi_chain),
    })
    return stats


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Plinder dataset ligand/protein composition stats.")
    parser.add_argument("--outdir", type=Path, required=True,
                         help="Plinder dataset root — one subdirectory per split (train/val/test), "
                              "as produced by plinder.py / plinder_pocket_placer.py")
    parser.add_argument("--outfile", type=Path, default=None,
                         help="Where to write the yaml report (default: {outdir}/composition_stats.yaml)")
    parser.add_argument("--num-processes", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    outfile = args.outfile or (args.outdir / "composition_stats.yaml")
    report = {"splits": {}}
    all_records = []

    for split in ["train", "val", "test"]:
        split_dir = args.outdir / split
        manifest = _load_json(split_dir / "manifest.json")
        if manifest is None:
            print(f"[{split}] {split_dir / 'manifest.json'} not found, skipping")
            report["splits"][split] = None
            continue

        print(f"[{split}] {len(manifest)} records")
        all_records.extend(manifest)

        split_report = {
            "n_records": len(manifest),
            "ligand": _ligand_size_stats(manifest, args.num_processes),
            "protein": _protein_size_stats(manifest),
        }

        filter_stats = _load_json(split_dir / "filter_stats.json")
        if filter_stats is not None:
            split_report["filtered_by_reason"] = filter_stats
            split_report["n_filtered"] = sum(filter_stats.values())
            split_report["n_attempted"] = len(manifest) + split_report["n_filtered"]

        report["splits"][split] = split_report

    if all_records:
        report["combined"] = {
            "n_records": len(all_records),
            "ligand": _ligand_size_stats(all_records, args.num_processes),
            "protein": _protein_size_stats(all_records),
        }

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    print(f"\nWrote stats to {outfile}")


if __name__ == "__main__":
    main()
