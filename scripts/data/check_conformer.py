"""
Sanity check: given a molecule's hash (data_id) or SMILES, find it across all
train/val/test splits in the geom conformers datadir, print the record SMILES,
and verify it survives a PDB round-trip via in-memory stream.
"""
import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/datastor1/dy4652/proteinzen"))
DEFAULT_DATADIR = REPO_ROOT / "data" / "geom_drugs_conformers"


def hash_sequence(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()


def find_record(datadir: Path, data_id: str) -> tuple[str, dict]:
    """Search all splits for the record matching data_id. Returns (split, record)."""
    # Strip conformer index suffix (e.g. _0, _1) if present
    data_id = re.sub(r"_\d+$", "", data_id)
    mid = data_id[1:3]
    for split in ("train", "val", "test"):
        record_path = datadir / split / "records" / mid / f"{data_id}.json"
        if record_path.exists():
            with record_path.open() as f:
                record = json.load(f)
            return split, record
    raise KeyError(f"No record found for data_id={data_id} in {datadir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check: print SMILES and do a PDB round-trip.")
    parser.add_argument("--datadir", type=Path, default=DEFAULT_DATADIR, help="geom conformers output directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-id", type=str, help="SHA256 hash of the molecule (_N suffix stripped automatically)")
    group.add_argument("--smiles", type=str, help="SMILES string (will be hashed to get data_id)")
    args = parser.parse_args()

    data_id = args.data_id if args.data_id else hash_sequence(args.smiles)
    print(f"data_id: {data_id}")

    split, record = find_record(args.datadir, data_id)
    print(f"Found in split: {split}")

    record_smiles = record["structures"][0]["method"].split(":", 1)[1]
    print(f"Record SMILES:       {record_smiles}")

    # PDB round-trip
    mol = Chem.MolFromSmiles(record_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

    pdb_block = Chem.MolToPDBBlock(mol)
    mol_from_pdb = Chem.MolFromPDBBlock(io.StringIO(pdb_block).read(), removeHs=False)
    pdb_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol_from_pdb), isomericSmiles=True, canonical=True)
    print(f"PDB round-trip SMILES: {pdb_smiles}")
