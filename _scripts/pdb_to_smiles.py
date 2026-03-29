#!/usr/bin/env python3
"""Print SMILES from a PDB file, with and without Hs/stereochemistry."""

import sys
from rdkit import Chem

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path.pdb>")
        sys.exit(1)

    pdb_path = sys.argv[1]
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
    if mol is None:
        print(f"Failed to load molecule from {pdb_path}")
        sys.exit(1)

    smiles_full = Chem.MolToSmiles(mol)
    print(f"Full SMILES:              {smiles_full}")

    mol_no_h = Chem.RemoveHs(mol)
    smiles_no_h = Chem.MolToSmiles(mol_no_h)
    print(f"No Hs SMILES:             {smiles_no_h}")

    Chem.RemoveStereochemistry(mol_no_h)
    smiles_no_stereo = Chem.MolToSmiles(mol_no_h)
    print(f"No Hs, no stereo SMILES:  {smiles_no_stereo}")

if __name__ == "__main__":
    main()
