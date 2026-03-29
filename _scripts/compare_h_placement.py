"""
Compare H placement: original rdkit pickle vs PDB roundtrip (H re-added by RDKit).

For each molecule writes two PDBs:
  {mol_id}_original.pdb  -- rd_mol from pickle (DFT H positions)
  {mol_id}_roundtrip.pdb -- write heavy atoms to PDB, reload, AddHs(addCoords=True)

Usage:
  python3 compare_h_placement.py [n_mols]
"""
import sys
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from xtb.interface import Calculator, Param

ANGSTROM_TO_BOHR   = 1.8897259886
HARTREE_TO_KCALMOL = 627.509474


def xtb_sp(mol):
    try:
        numbers   = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
        conf      = mol.GetConformer()
        positions = np.array(
            [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())],
            dtype=np.float64,
        ) * ANGSTROM_TO_BOHR
        calc = Calculator(Param.GFN2xTB, numbers, positions)
        calc.set_verbosity(0)
        return calc.singlepoint().get_energy() * HARTREE_TO_KCALMOL
    except Exception:
        return float('nan')

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RDKIT_DIR   = PROJECT_DIR / "data" / "rdkit" / "drugs"
TEST_MANIFEST = PROJECT_DIR / "data" / "geom_drugs_conformers" / "test" / "manifest.json"
REF_PDB_DIR   = PROJECT_DIR / "sampling" / "geom_conformer_test" / "conformer_mols"
OUT_DIR = PROJECT_DIR / "sampling" / "h_placement_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_MOLS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def load_pickle(smiles):
    pkl_path = RDKIT_DIR / f"{smiles.replace('/', '_')}.pickle"
    if not pkl_path.exists():
        return None
    return pickle.load(open(str(pkl_path), "rb"))


def pdb_roundtrip(mol):
    """Write mol heavy atoms to temp PDB, reload, AddHs(addCoords=True)."""
    mol_noH = Chem.RemoveHs(mol)
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        tmp = f.name
    Chem.MolToPDBFile(mol_noH, tmp)
    reloaded = Chem.MolFromPDBFile(tmp, removeHs=False, sanitize=False)
    try:
        Chem.SanitizeMol(reloaded)
    except Exception:
        pass
    return Chem.AddHs(reloaded, addCoords=True)


# ---- build id->smiles index ----
manifest = json.load(open(str(TEST_MANIFEST)))
id_to_smiles = {}
for entry in manifest:
    smiles = entry["structures"][0]["method"].replace("QM9:", "")
    for mid in entry["ids"]:
        id_to_smiles[mid] = smiles

# ---- find test mol_ids that have pickles ----
pdb_groups = {}
for p in sorted(REF_PDB_DIR.glob("*.pdb")):
    mol_id = p.stem.rsplit("_", 1)[0]
    pdb_groups.setdefault(mol_id, []).append(p)

done = 0
for mol_id in sorted(pdb_groups):
    if done >= N_MOLS:
        break
    smiles = id_to_smiles.get(mol_id + "_0")
    if smiles is None:
        continue
    d = load_pickle(smiles)
    if d is None:
        continue

    # just use first conformer
    mol_orig = d["conformers"][0]["rd_mol"]
    mol_rt   = pdb_roundtrip(mol_orig)

    e_orig = xtb_sp(mol_orig)
    e_rt   = xtb_sp(mol_rt)
    delta  = e_rt - e_orig

    orig_path = OUT_DIR / f"{mol_id[:16]}_original_E{e_orig:.1f}.pdb"
    rt_path   = OUT_DIR / f"{mol_id[:16]}_roundtrip_E{e_rt:.1f}_dE{delta:+.1f}.pdb"

    Chem.MolToPDBFile(mol_orig, str(orig_path))
    Chem.MolToPDBFile(mol_rt,   str(rt_path))

    print(f"[{mol_id[:20]}]")
    print(f"  SMILES:     {smiles[:70]}")
    print(f"  original:   {orig_path.name}")
    print(f"  roundtrip:  {rt_path.name}")
    print(f"  E_orig={e_orig:.2f}  E_rt={e_rt:.2f}  dE={delta:+.2f} kcal/mol")
    print()
    done += 1

print(f"Written to {OUT_DIR}")
