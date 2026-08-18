"""
Compute pre/post-filtering composition stats for the GEOM dataset pipeline
(summary_{dataset}.json -> filter_geom.py -> scaffold_train_test.py ->
geom_conformer.py), so you can see e.g. molecule counts, atom-count/molecular-
weight distributions, and the largest molecule at each stage without manually
poking through the intermediate JSON files.

Reads (all optional; a stage whose file is missing is skipped and reported as
such rather than erroring, since you may only have run part of the pipeline):
  {rdkit_dir}/summary_{dataset}.json                 raw
  {rdkit_dir}/filtered_summary_{dataset}.json         post filter_geom.py
  {rdkit_dir}/filter_errors_{dataset}.json            filter_geom.py error breakdown
  {rdkit_dir}/{mode}_filtered_summary_{dataset}.json  post scaffold_train_test.py, per split
  {processed_dir}/{mode}/manifest.json                post geom_conformer.py, per split
  {processed_dir}/{mode}/errors.json                  geom_conformer.py error breakdown, per split

Writes:
  {outfile} (default {processed_dir}/geom_stats.yaml)

Usage:
  python scripts/data/geom_stats.py --dataset drugs
"""
import argparse
import json
import multiprocessing
import os
import statistics
from pathlib import Path
from typing import Optional

import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors


def _mol_size(smiles: str) -> Optional[tuple]:
    """Return (smiles, n_heavy_atoms, mol_wt), or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return smiles, mol.GetNumAtoms(), Descriptors.MolWt(mol)


def _size_stats(smiles_list: list, num_processes: int) -> dict:
    """Atom-count / molecular-weight distribution stats over a list of SMILES."""
    if not smiles_list:
        return {"n_molecules": 0}
    if num_processes > 1 and len(smiles_list) > 1000:
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(_mol_size, smiles_list, chunksize=200)
    else:
        results = [_mol_size(s) for s in smiles_list]
    results = [r for r in results if r is not None]
    n_unparseable = len(smiles_list) - len(results)
    stats = {"n_molecules": len(smiles_list)}
    if n_unparseable:
        stats["n_unparseable"] = n_unparseable
    if not results:
        return stats

    atom_counts = [r[1] for r in results]
    mol_wts = [r[2] for r in results]
    largest = max(results, key=lambda r: r[1])
    smallest = min(results, key=lambda r: r[1])
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
        "largest_molecule": {"smiles": largest[0], "n_atoms": largest[1], "mol_wt": round(largest[2], 2)},
        "smallest_molecule": {"smiles": smallest[0], "n_atoms": smallest[1], "mol_wt": round(smallest[2], 2)},
    })
    return stats


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _extract_smiles_from_method(method: Optional[str]) -> Optional[str]:
    """geom_conformer.py stores method=f"QM9:{smiles}" (that literal "QM9:" prefix
    is used for both the qm9 and drugs datasets — not a typo to fix here, just how
    the field is populated upstream)."""
    if not method or ":" not in method:
        return None
    return method.split(":", 1)[1]


def main():
    parser = argparse.ArgumentParser(description="GEOM dataset pre/post-filter composition stats.")
    parser.add_argument("--rdkit-dir", type=Path,
                         default=Path(os.environ.get("REPO_ROOT", ".")) / "data" / "rdkit",
                         help="Dir with summary_{dataset}.json / filtered_summary_{dataset}.json / {mode}_filtered_summary_{dataset}.json")
    parser.add_argument("--processed-dir", type=Path,
                         default=Path(os.environ.get("REPO_ROOT", ".")) / "data" / "geom_drugs_conformers",
                         help="Dir with {mode}/manifest.json + {mode}/errors.json (geom_conformer.py output)")
    parser.add_argument("--dataset", type=str, default="drugs", choices=["qm9", "drugs"])
    parser.add_argument("--outfile", type=Path, default=None,
                         help="Where to write the yaml report (default: {processed-dir}/geom_stats.yaml)")
    parser.add_argument("--num-processes", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    outfile = args.outfile or (args.processed_dir / "geom_stats.yaml")
    report = {}

    # --- Stage 1: raw ---
    raw_path = args.rdkit_dir / f"summary_{args.dataset}.json"
    raw = _load_json(raw_path)
    if raw is not None:
        smiles_list = [s for s in raw if len(s) > 1]
        print(f"[raw] {raw_path}: {len(smiles_list)} molecules")
        report["raw"] = _size_stats(smiles_list, args.num_processes)
    else:
        print(f"[raw] {raw_path} not found, skipping")
        report["raw"] = None

    # --- Stage 2: post filter_geom.py ---
    filtered_path = args.rdkit_dir / f"filtered_summary_{args.dataset}.json"
    filtered = _load_json(filtered_path)
    if filtered is not None:
        smiles_list = [s for s in filtered if len(s) > 1]
        print(f"[filter_geom] {filtered_path}: {len(smiles_list)} molecules")
        stage = _size_stats(smiles_list, args.num_processes)
        errors = _load_json(args.rdkit_dir / f"filter_errors_{args.dataset}.json")
        if errors is not None:
            stage["errors_by_type"] = {k: len(v) for k, v in errors.items()}
            stage["n_errors"] = sum(len(v) for v in errors.values())
        report["filter_geom"] = stage
    else:
        print(f"[filter_geom] {filtered_path} not found, skipping")
        report["filter_geom"] = None

    # --- Stage 3: per-split, post scaffold_train_test.py + post geom_conformer.py ---
    report["splits"] = {}
    for mode in ["train", "val", "test"]:
        split_report = {}

        split_summary_path = args.rdkit_dir / f"{mode}_filtered_summary_{args.dataset}.json"
        split_summary = _load_json(split_summary_path)
        if split_summary is not None:
            smiles_list = [s for s in split_summary if len(s) > 1]
            print(f"[{mode}/scaffold_split] {split_summary_path}: {len(smiles_list)} molecules")
            split_report["post_scaffold_split"] = _size_stats(smiles_list, args.num_processes)
        else:
            print(f"[{mode}/scaffold_split] {split_summary_path} not found, skipping")
            split_report["post_scaffold_split"] = None

        manifest_path = args.processed_dir / mode / "manifest.json"
        manifest = _load_json(manifest_path)
        if manifest is not None:
            print(f"[{mode}/processed] {manifest_path}: {len(manifest)} molecules")
            smiles_list = []
            n_conformers_list = []
            for record in manifest:
                structures = record.get("structures") or []
                method = structures[0].get("method") if structures else None
                smi = _extract_smiles_from_method(method)
                if smi:
                    smiles_list.append(smi)
                n_conformers_list.append(len(record.get("ids") or []))
            stage = _size_stats(smiles_list, args.num_processes)
            if n_conformers_list:
                stage["conformers_per_molecule"] = {
                    "min": min(n_conformers_list),
                    "max": max(n_conformers_list),
                    "mean": round(statistics.mean(n_conformers_list), 2),
                    "median": statistics.median(n_conformers_list),
                }
            errors = _load_json(args.processed_dir / mode / "errors.json")
            if errors is not None:
                stage["errors_by_type"] = {k: len(v) for k, v in errors.items()}
                stage["n_errors"] = sum(len(v) for v in errors.values())
            split_report["processed"] = stage
        else:
            print(f"[{mode}/processed] {manifest_path} not found, skipping")
            split_report["processed"] = None

        report["splits"][mode] = split_report

    # --- Overall largest molecule across all processed splits ---
    largest_per_split = [
        report["splits"][m]["processed"]["largest_molecule"]
        for m in ["train", "val", "test"]
        if report["splits"][m].get("processed") and report["splits"][m]["processed"].get("largest_molecule")
    ]
    if largest_per_split:
        report["largest_molecule_overall_processed"] = dict(max(largest_per_split, key=lambda x: x["n_atoms"]))

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    print(f"\nWrote stats to {outfile}")


if __name__ == "__main__":
    main()
