"""
Sanity check: compare xTB single-point energies across three paths:
  A) rdkit pickle (original H positions) -- ground truth
  B) PDB roundtrip: H stripped + re-added by RDKit -- current broken path
  C) PDB roundtrip + MMFF H-only optimization -- proposed fix

Usage:
  python3 sanity_xtb.py [n_mols] [n_confs]
"""
import os
import sys
import pickle
import numpy as np
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import minimize
from xtb.interface import Calculator, Param

ANGSTROM_TO_BOHR   = 1.8897259886
HARTREE_TO_KCALMOL = 627.509474
HARTREE_TO_KCALMOL_PER_BOHR = HARTREE_TO_KCALMOL / ANGSTROM_TO_BOHR  # gradient unit conversion

RDKIT_DIR   = Path(__file__).resolve().parent.parent / "data" / "rdkit" / "drugs"
REF_PDB_DIR = Path(__file__).resolve().parent.parent / "sampling" / "geom_conformer_test" / "conformer_mols"

N_MOLS  = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_CONFS = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def xtb_sp(mol):
    """xTB GFN2 single-point on mol as-is. Returns kcal/mol or nan."""
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


def optimize_h_only(mol):
    """xTB H-only optimization: freeze heavy atoms, relax H positions with L-BFGS-B."""
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
    conf = mol.GetConformer()
    all_pos = np.array(
        [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )  # Å

    h_idx = [i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() == 1]

    calc = Calculator(Param.GFN2xTB, numbers, all_pos * ANGSTROM_TO_BOHR)
    calc.set_verbosity(0)

    def energy_and_grad(h_pos_flat):
        pos = all_pos.copy()
        pos[h_idx] = h_pos_flat.reshape(-1, 3)
        calc.update(pos * ANGSTROM_TO_BOHR)
        r = calc.singlepoint()
        e = r.get_energy()  # Hartree
        grad = np.array(r.get_gradient()).reshape(-1, 3)  # Hartree/Bohr
        h_grad = grad[h_idx] / ANGSTROM_TO_BOHR  # Hartree/Å
        return e, h_grad.ravel()

    x0 = all_pos[h_idx].ravel()
    result = minimize(energy_and_grad, x0, jac=True, method="L-BFGS-B",
                      options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-5})

    opt_pos = all_pos.copy()
    opt_pos[h_idx] = result.x.reshape(-1, 3)

    rw = Chem.RWMol(mol)
    new_conf = rw.GetConformer()
    for i, pos in enumerate(opt_pos):
        new_conf.SetAtomPosition(i, pos.tolist())
    return rw.GetMol()


def load_pdb_mol(pdb_path):
    """Load PDB heavy atoms, apply SMILES template if present, add H via RDKit."""
    smiles = None
    with open(str(pdb_path)) as fh:
        for line in fh:
            if line.startswith("REMARK SMILES "):
                smiles = line.split("REMARK SMILES ", 1)[1].strip()
                break
            if line.startswith("ATOM") or line.startswith("HETATM"):
                break
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if smiles is not None:
        try:
            template = Chem.RemoveHs(AllChem.MolFromSmiles(smiles))
            mol = AllChem.AssignBondOrdersFromTemplate(template, Chem.RemoveHs(mol))
        except Exception:
            pass
    return Chem.AddHs(Chem.RemoveHs(mol), addCoords=True)


# ============================================================
# Part A: pickle ground truth
# ============================================================
print("=" * 70)
print("PART A: pickle mols (original H) -- ground truth")
print("=" * 70)

pickle_files = sorted(RDKIT_DIR.glob("*.pickle"))
for pkl_path in pickle_files[:N_MOLS]:
    d = pickle.load(open(str(pkl_path), "rb"))
    print("\nSMILES: {}".format(d["smiles"][:70]))
    energies = []
    for c in d["conformers"][:N_CONFS]:
        mol = c["rd_mol"]
        e = xtb_sp(mol)
        energies.append(e)
        print("  bw={:.4f}  rel_e_stored={:.3f}  xtb={:.2f}".format(
            c["boltzmannweight"], c["relativeenergy"], e))
    finite = [e for e in energies if np.isfinite(e)]
    if len(finite) > 1:
        print("  range: {:.3f} kcal/mol".format(max(finite) - min(finite)))

# ============================================================
# Part B vs C: PDB roundtrip, with and without H optimization
# ============================================================
print()
print("=" * 70)
print("PART B vs C: PDB mols -- H re-added (B) vs H-optimized (C)")
print("=" * 70)

pdb_groups = {}
for p in sorted(REF_PDB_DIR.glob("*.pdb")):
    mol_id = p.stem.rsplit("_", 1)[0]
    pdb_groups.setdefault(mol_id, []).append(p)

for mol_id in sorted(pdb_groups)[:N_MOLS]:
    paths = sorted(pdb_groups[mol_id])[:N_CONFS]
    print("\n[{}]".format(mol_id[:24]))
    b_energies, c_energies = [], []
    for p in paths:
        mol_b = load_pdb_mol(p)
        if mol_b is None:
            print("  {} LOAD FAILED".format(p.name[-20:]))
            continue
        mol_c = optimize_h_only(mol_b)
        e_b = xtb_sp(mol_b)
        e_c = xtb_sp(mol_c)
        b_energies.append(e_b)
        c_energies.append(e_c)
        print("  {}  B={:.2f}  C={:.2f}  diff={:.2f}".format(
            p.name[-36:], e_b, e_c, e_c - e_b))
    for label, energies in [("B (no H opt)", b_energies), ("C (H opt)", c_energies)]:
        finite = [e for e in energies if np.isfinite(e)]
        if len(finite) > 1:
            print("  {} range: {:.3f} kcal/mol".format(label, max(finite) - min(finite)))
