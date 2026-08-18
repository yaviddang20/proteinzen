import csv
import hashlib
import shutil
import traceback
import yaml
import pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count

from rdkit import Chem
from rdkit.Chem import AllChem

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

CSV_PATH = REPO_ROOT / "data" / "DRUGS" / "test_smiles.csv"
RDKIT_DIR = REPO_ROOT / "data" / "rdkit" / "drugs"
OUTPUT_DIR = REPO_ROOT / "sampling" / "geom_conformer_test_bench"


def _load_pickle_conformers(raw_smiles):
    pkl_path = RDKIT_DIR / f"{raw_smiles.replace('/', '_')}.pickle"
    if not pkl_path.exists():
        return {}
    try:
        d = pickle.load(open(str(pkl_path), "rb"))
        return {i: c["rd_mol"] for i, c in enumerate(d["conformers"])}
    except Exception:
        return {}


def _write_pdb_with_smiles_remark(mol, out_path, smiles, conf_id=0):
    """Write mol to PDB with a REMARK SMILES header line. PDB has no bond-order
    field, so eval_conformer.py's _parse_remark_smiles/_apply_smiles_template
    (and sanity_xtb.py's load_pdb_mol) rely on this line to recover correct
    bond orders from the known SMILES when RDKit's geometry-based guess on
    read-back gets them wrong."""
    block = Chem.MolToPDBBlock(mol, confId=conf_id)
    with open(out_path, "w") as f:
        f.write(f"REMARK SMILES {smiles}\n")
        f.write(block)


def _pdb_roundtrip_ok(pdb_path, smiles):
    """Confirm the PDB we just wrote can actually be reloaded — either it
    sanitizes directly, or (PDB bond-order guess is wrong but connectivity,
    preserved via CONECT records, is fine) the known SMILES template recovers
    valid bond orders. Mirrors eval_conformer.py's load_pdb/_apply_smiles_template
    recovery path, so a file that passes this check is guaranteed loadable there."""
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=False)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        pass
    try:
        template = Chem.RemoveHs(AllChem.MolFromSmiles(smiles))
        if template is None:
            return False
        Chem.SanitizeMol(mol, catchErrors=True)
        fixed = AllChem.AssignBondOrdersFromTemplate(template, mol)
        Chem.SanitizeMol(fixed)
        return True
    except Exception:
        return False


def _write_conformers_worker(args):
    raw_smiles, corrected_smiles, output_dir = args
    try:
        pkl_mols = _load_pickle_conformers(raw_smiles)
        if not pkl_mols:
            print(f"No pickle for {raw_smiles[:60]}...")
            return None, 0
        name = hashlib.sha256(corrected_smiles.encode()).hexdigest()
        first_path = None
        n_written = 0
        n_skipped = 0
        for conf_idx, mol in sorted(pkl_mols.items()):
            out = output_dir / f"{name}_{conf_idx}.pdb"
            _write_pdb_with_smiles_remark(mol, out, corrected_smiles, conf_id=0)
            if not _pdb_roundtrip_ok(out, corrected_smiles):
                print(f"Skipping unrecoverable PDB round-trip: {out}")
                out.unlink(missing_ok=True)
                n_skipped += 1
                continue
            if first_path is None:
                first_path = out
            n_written += 1
        if n_skipped:
            print(f"{corrected_smiles[:60]}...: wrote {n_written}, skipped {n_skipped} (failed PDB round-trip)")
        return first_path, n_written
    except Exception:
        traceback.print_exc()
        return None, 0


def main():
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row["smiles"].strip()
            corr = row["corrected_smiles"].strip()
            if raw and corr:
                rows.append((raw, corr))

    rows.sort(key=lambda x: x[1])  # sort by corrected_smiles

    conformer_mols_dir = OUTPUT_DIR / "conformer_mols"
    first_conformer_dir = OUTPUT_DIR / "first_conformer_mols"
    conformer_mols_dir.mkdir(parents=True, exist_ok=True)
    first_conformer_dir.mkdir(parents=True, exist_ok=True)

    worker_args = [(raw, corr, conformer_mols_dir) for raw, corr in rows]
    n_workers = cpu_count()
    print(f"Writing conformer PDBs for {len(rows)} molecules with {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        results = pool.map(_write_conformers_worker, worker_args)

    smiles_tasks = []
    mol_tasks = []
    for (raw, corr), (first_path, n_conf) in zip(rows, results):
        if first_path is None:
            continue
        shutil.copy(first_path, first_conformer_dir / first_path.name)
        name = hashlib.sha256(corr.encode()).hexdigest()
        smiles_tasks.append({
            "_target_": "proteinzen.runtime.sampling.unconditional_smiles.UnconditionalSamplingFromSMILES",
            "smiles": corr,
            "num_samples": 2 * n_conf,
            "name": name,
        })
        mol_tasks.append({
            "_target_": "proteinzen.runtime.sampling.unconditional_smiles.UnconditionalSamplingFromMol",
            "mol_pdb_path": str(first_path.resolve()),
            "num_samples": 2 * n_conf,
            "name": name,
        })

    smiles_yaml = OUTPUT_DIR / "smiles.yaml"
    mol_yaml = OUTPUT_DIR / "mol.yaml"
    for path, tasks in [(smiles_yaml, smiles_tasks), (mol_yaml, mol_tasks)]:
        with open(path, "w") as f:
            yaml.dump(tasks, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Wrote {len(smiles_tasks)} tasks to {smiles_yaml}")
    print(f"Wrote {len(mol_tasks)} tasks to {mol_yaml}")


if __name__ == "__main__":
    main()
