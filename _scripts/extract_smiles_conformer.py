import json
import pickle
import random
import hashlib
import shutil
import traceback
import yaml
import numpy as np
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol, Conformer
from rdkit.Geometry import Point3D

# Dataset path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

RDKIT_DIR = PROJECT_DIR / "data" / "rdkit" / "drugs"

TRAIN_DATASET_DIR = PROJECT_DIR / "data/geom_drugs_conformers/train"
TRAIN_MANIFEST_PATH = TRAIN_DATASET_DIR / "manifest.json"

TEST_DATASET_DIR = PROJECT_DIR / "data/geom_drugs_conformers/test"
TEST_MANIFEST_PATH = TEST_DATASET_DIR / "manifest.json"

conformer_train_dir = PROJECT_DIR / "sampling/geom_conformer_train"
conformer_test_dir = PROJECT_DIR / "sampling/geom_conformer_test"

conformer_train_dir.mkdir(parents=True, exist_ok=True)
conformer_test_dir.mkdir(parents=True, exist_ok=True)

NUM_TEST = 50
NUM_GEN_SAMPLES = 20


def extract_smiles_from_manifest(manifest):
    smiles_list = []
    failed = 0
    for entry in manifest:
        try:
            method = entry.get('structures', [])[0].get('method', '')
            if method.startswith('QM9:'):
                smiles = method[4:]
                smiles_list.append(smiles)
            else:
                failed += 1
        except Exception:
            failed += 1
    print(f"Extracted {len(smiles_list)} SMILES strings")
    print(f"Failed to extract: {failed} entries")
    print(f"\nFirst 5 SMILES:")
    for i, smiles in enumerate(smiles_list[:5], 1):
        print(f"  {i}. {smiles[:80]}..." if len(smiles) > 80 else f"  {i}. {smiles}")
    return smiles_list, failed


def print_smiles_statistics(smiles_list):
    print("SMILES Statistics:")
    print("=" * 60)
    print(f"Total unique SMILES: {len(set(smiles_list))}")
    print(f"Total SMILES (including duplicates): {len(smiles_list)}")
    print(f"Number of duplicates: {len(smiles_list) - len(set(smiles_list))}")

    smiles_counts = Counter(smiles_list)
    duplicates = {smiles: count for smiles, count in smiles_counts.items() if count > 1}
    if duplicates:
        print(f"\nFound {len(duplicates)} SMILES that appear multiple times:")
        for smiles, count in list(duplicates.items())[:5]:
            print(f"  '{smiles[:60]}...' appears {count} times")
    else:
        print("\nNo duplicate SMILES found (each SMILES appears exactly once)")


def write_smiles_yaml(smiles_sample, output_yaml_path):
    smiles_to_use = sorted([str(s) for s in smiles_sample])

    tasks = []
    for smiles in smiles_to_use:
        tasks.append({
            "task": "unconditional_smiles",
            "smiles": smiles,
            "num_samples": NUM_GEN_SAMPLES,
            "name": hashlib.sha256(smiles.encode()).hexdigest()
        })

    yaml_data = {"tasks": tasks}

    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Created YAML file with {len(tasks)} tasks from geom_drugs_sample")
    print(f"Saved to: {output_yaml_path.absolute()}")


def get_npz_paths(molecule_ids, dataset_dir):
    npz_paths = []
    for molecule_id in molecule_ids:
        subdir = molecule_id[1:3]
        npz_paths.append(dataset_dir / "structures" / subdir / f"{molecule_id}.npz")
    return npz_paths


def _load_pickle_conformers(smiles):
    """Return {conformer_index: rd_mol} from the rdkit pickle for this SMILES, or {} on failure."""
    pkl_path = RDKIT_DIR / f"{smiles.replace('/', '_')}.pickle"
    if not pkl_path.exists():
        return {}
    try:
        d = pickle.load(open(str(pkl_path), "rb"))
        return {i: c["rd_mol"] for i, c in enumerate(d["conformers"])}
    except Exception:
        return {}


def write_geom_drugs_molecule_to_pdb(npz_paths, output_dir, smiles=None):
    output_paths = []
    pkl_mols = _load_pickle_conformers(smiles) if smiles is not None else {}

    for npz_path in npz_paths:
        # conformer index is the last part of the stem: {hash}_{i}
        try:
            conf_idx = int(npz_path.stem.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            conf_idx = -1

        mol = pkl_mols.get(conf_idx)

        if mol is None:
            # fallback: build heavy-atom-only mol from NPZ
            data = np.load(npz_path)
            atoms = data['atoms']
            Z = atoms['element'].astype(int)
            q = atoms['charge'].astype(int)
            xyz = atoms['coords'].astype(float)
            n_atoms = len(Z)
            rw = RWMol()
            conf = Conformer(n_atoms)
            for i in range(n_atoms):
                atom = Chem.Atom(int(Z[i]))
                atom.SetFormalCharge(int(q[i]))
                idx = rw.AddAtom(atom)
                conf.SetAtomPosition(idx, Point3D(*xyz[i]))
            mol = rw.GetMol()
            conf.SetId(0)
            mol.AddConformer(conf, assignId=True)

        output_path = output_dir / f"{npz_path.stem}.pdb"
        output_paths.append(output_path)
        Chem.MolToPDBFile(mol, str(output_path), confId=0)

    return output_paths


def _write_conformer_worker(args):
    """Top-level worker for multiprocessing: processes one SMILES entry."""
    smiles, molecule_ids, output_dir, dataset_dir = args
    npz_paths = get_npz_paths(molecule_ids, dataset_dir)
    try:
        return write_geom_drugs_molecule_to_pdb(npz_paths=npz_paths, output_dir=output_dir, smiles=smiles)[0]
    except Exception:
        print(f"Failed for {smiles[:60]}...")
        traceback.print_exc()
        return None


def write_conformer_pdb_paths(smiles_sample, manifest, output_dir, dataset_dir, num_workers=None):
    smiles_to_use = sorted([str(s) for s in smiles_sample])

    print("Building SMILES to molecule ID mapping...")
    smiles_to_ids = {}
    for entry in manifest:
        method = entry.get('structures', {})[0].get('method', '')
        if method.startswith('QM9:'):
            entry_smiles = method[4:]
            smiles_to_ids[entry_smiles] = entry['ids']

    print(f"Mapped {len(smiles_to_ids)} SMILES to molecule IDs\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    worker_args = []
    for smiles in smiles_to_use:
        if smiles not in smiles_to_ids:
            print(f"SMILES not found in manifest: {smiles[:60]}...")
            continue
        worker_args.append((smiles, smiles_to_ids[smiles], output_dir, dataset_dir))

    n_workers = num_workers or cpu_count()
    print(f"Processing {len(worker_args)} molecules with {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        results = pool.map(_write_conformer_worker, worker_args)

    return [p for p in results if p is not None]


def write_mol_yaml(first_conformer_output_pdb_paths, output_yaml):
    tasks = []
    for pdb_path in first_conformer_output_pdb_paths:
        name = Path(pdb_path).stem
        name = name[:name.index("_")]
        tasks.append({
            "task": "unconditional_mol",
            "mol_pdb_path": str(pdb_path),
            "num_samples": NUM_GEN_SAMPLES,
            "name": name
        })

    yaml_data = {"tasks": tasks}

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Created YAML file with {len(tasks)} tasks from geom_drugs_sample")
    print(f"Saved to: {output_yaml.absolute()}")


def main():
    # Load manifests
    with open(TRAIN_MANIFEST_PATH, 'r') as f:
        train_manifest = json.load(f)
    with open(TEST_MANIFEST_PATH, 'r') as f:
        test_manifest = json.load(f)

    print(f"Total entries in train manifest: {len(train_manifest)}")
    print(f"Total entries in test manifest: {len(test_manifest)}")

    # Extract SMILES
    print("\nExtracting SMILES from train manifest...")
    train_smiles_list, _ = extract_smiles_from_manifest(train_manifest)
    print("Extracting SMILES from test manifest...")
    test_smiles_list, _ = extract_smiles_from_manifest(test_manifest)

    train_smiles_set = set(train_smiles_list)
    test_smiles_set = set(test_smiles_list)

    print_smiles_statistics(train_smiles_list)
    print_smiles_statistics(test_smiles_list)

    # Sample
    random.seed(42)
    train_smiles_sample = random.sample(sorted(train_smiles_set), NUM_TEST)
    test_smiles_sample = random.sample(sorted(test_smiles_set), NUM_TEST)

    # Write SMILES YAMLs
    write_smiles_yaml(train_smiles_sample, conformer_train_dir / "smiles.yaml")
    write_smiles_yaml(test_smiles_sample, conformer_test_dir / "smiles.yaml")

    # Write conformer PDBs
    train_first_conformer_output_pdb_paths = write_conformer_pdb_paths(
        train_smiles_sample, train_manifest, conformer_train_dir / "conformer_mols", TRAIN_DATASET_DIR
    )
    test_first_conformer_output_pdb_paths = write_conformer_pdb_paths(
        test_smiles_sample, test_manifest, conformer_test_dir / "conformer_mols", TEST_DATASET_DIR
    )

    # Copy first conformers
    train_first_conformer_out_dir = conformer_train_dir / "first_conformer_mols"
    test_first_conformer_out_dir = conformer_test_dir / "first_conformer_mols"

    train_first_conformer_out_dir.mkdir(parents=True, exist_ok=True)
    test_first_conformer_out_dir.mkdir(parents=True, exist_ok=True)

    for pdb_path in train_first_conformer_output_pdb_paths:
        shutil.copy(pdb_path, train_first_conformer_out_dir / pdb_path.name)
    for pdb_path in test_first_conformer_output_pdb_paths:
        shutil.copy(pdb_path, test_first_conformer_out_dir / pdb_path.name)

    # Write mol YAMLs
    write_mol_yaml(train_first_conformer_output_pdb_paths, conformer_train_dir / "mol.yaml")
    write_mol_yaml(test_first_conformer_output_pdb_paths, conformer_test_dir / "mol.yaml")


if __name__ == "__main__":
    main()
