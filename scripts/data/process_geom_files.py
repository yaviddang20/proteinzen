import argparse
import multiprocessing as mp
import pickle
import os
import glob
from functools import partial

import tqdm
import pandas as pd
from rdkit import Chem

from proteinzen.data.datasets.featurize.molecule import conformer_props
from proteinzen.data.datasets.featurize.coarse_grain import compute_coarse_grain_groups
from proteinzen.data.datasets.featurize.conformer_matching import optimize_rotatable_bonds, get_torsion_angles

dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')

def keep_smiles(smiles):
    # filter fragments
    if '.' in smiles:
        return False

    # filter mols rdkit can't intrinsically handle
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.AddHs(mol)
    else:
        return False
    N = mol.GetNumAtoms()

    # filter out mols model can't make predictions for
    if not mol.HasSubstructMatch(dihedral_pattern):
        return False
    if N < 4:
        return False

    return True

def process(smiles_fp, out_folder, splits=None, include_h=False):
    with open(smiles_fp, 'rb') as fp:
        data = pickle.load(fp)
    smiles = data['smiles']
    if not keep_smiles(smiles):
        return None

    try:
        boltzmann_weights = []
        conf_prop_dicts = []
        rd_mols = []
        for conf_data in data['conformers']:
            # true_mol = conf_data['rd_mol']
            # mol = Chem.Mol(true_mol)
            # rotatable_bonds = get_torsion_angles(mol)
            # opt_mol = optimize_rotatable_bonds(mol, true_mol, rotatable_bonds)
            # conformer = opt_mol.GetConformer()
            conformer = conf_data['rd_mol'].GetConformer()
            boltzmann_weights.append(conf_data['boltzmannweight'])
            conf_prop_dicts.append(conformer_props(conformer, implicit_H=(not include_h)))
            rd_mols.append(conf_data['rd_mol'])
    except Exception as e:
        print(smiles)
        raise e

    cg_groups = compute_coarse_grain_groups(rd_mols[0])

    save_data = {
        "boltzmann_weights": boltzmann_weights,
        "conformer_property_dicts": conf_prop_dicts,
        "rd_mols": rd_mols,
        "cg_groups": cg_groups
    }
    fp_name = os.path.basename(smiles_fp)
    out_path = os.path.join(out_folder, fp_name)
    with open(out_path, 'wb') as fp:
        pickle.dump(save_data, fp)

    mol = Chem.MolFromSmiles(smiles)
    metadata = {
        "smiles": data["smiles"],
        "processed_path": os.path.abspath(out_path),
        "raw_path": os.path.abspath(smiles_fp),
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds()
    }
    if splits is not None:
        if smiles in splits['train']:
            metadata['split'] = 'train'
        elif smiles in splits['val']:
            metadata['split'] = 'val'
        elif smiles in splits['test']:
            metadata['split'] = 'test'
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--out_folder")
    parser.add_argument("--num_proc", default=1, type=int)
    parser.add_argument("--include_h", default=False, action="store_true")
    parser.add_argument("--splits", default=None)
    args = parser.parse_args()

    files = list(glob.glob(os.path.join(args.data, "*.pickle")))

    process_fn = partial(process, out_folder=args.out_folder, splits=args.splits, include_h=args.include_h)
    if args.num_proc > 1:
        with mp.Pool(args.num_proc) as pool:
            all_metadata = list(tqdm.tqdm(pool.imap(process_fn, files), total=len(files)))
    else:
        all_metadata = []
        for f in tqdm.tqdm(files):
            print(f)
            all_metadata.append(process_fn(f))

    all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_path = os.path.join(args.out_folder, "filtered_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)