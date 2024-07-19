""" Compute the bbRMSD between structures """
import argparse
import os
import glob

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import tqdm
from Bio.PDB.Superimposer import Superimposer

from proteinzen.data.io.atom91 import pdb_to_structure, struct_to_atom91

def parse_esm_log(lines, fold_original=False):
    ret = {}
    samples = {}
    name = None
    for l in lines:
        l = l.strip()
        if "Reading sequences" in l:
            if name is not None:
                ret[name] = samples
            name = os.path.splitext(
                l.split("/")[-1]
            )[0]
            samples = {}
        elif fold_original and "model_name=v_48_020" in l:
            sample_dict = {}
            sample_num = 0

            other_vals = l.split(",")
            for substring in other_vals:
                substring = substring.strip()
                if "pLDDT" in substring:
                    sample_dict["plddt"] = float(substring.split(" ")[1])
                elif "pTM" in substring:
                    sample_dict["ptm"] = float(substring.split(" ")[1])
            samples[sample_num] = sample_dict

        elif "sample=" in l:
            tail = l.split("sample=")[-1]
            sample_dict = {}
            sample_num = int(tail.split(",")[0])

            other_vals = tail.split(",")
            for substring in other_vals:
                substring = substring.strip()
                if "pLDDT" in substring:
                    sample_dict["plddt"] = float(substring.split(" ")[1])
                elif "pTM" in substring:
                    sample_dict["ptm"] = float(substring.split(" ")[1])
            samples[sample_num] = sample_dict
    ret[name] = samples

    return ret

def parse_dict_to_df(d):
    keys = ["name", "sample", "plddt", "ptm"]
    pd_dict = {k: [] for k in keys}
    for name, samples in d.items():
        for sample_num, sample_dict in samples.items():
            pd_dict["name"].append(name)
            pd_dict["sample"].append(sample_num)
            pd_dict["plddt"].append(sample_dict['plddt'])
            pd_dict["ptm"].append(sample_dict['ptm'])

    return pd.DataFrame.from_dict(pd_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compute the ")
    parser.add_argument("--esmlog", help="esmfold output log")
    parser.add_argument("--folded_folders", help="emfold folded structures")
    parser.add_argument("--samples", help="gen model samples")
    parser.add_argument("--fold_original", default=True, action='store_false')
    args = parser.parse_args()

    with open(args.esmlog) as fp:
        lines = fp.readlines()
    parse_dict = parse_esm_log(lines, fold_original=args.fold_original)
    df = parse_dict_to_df(parse_dict)
    print(df)

    rmsds = []
    all_atom_rmsds = []
    superimpose = Superimposer()
    for _, row in tqdm.tqdm(list(df.iterrows())):
        sample_path = os.path.join(
            args.samples,
            row['name'] + ".pdb"
        )
        folded_path = os.path.join(
            args.folded_folders,
            row['name']
        )
        current_folded = None
        folded_paths = glob.glob(os.path.join(folded_path, "*"))
        for path in folded_paths:
            if args.fold_original and row['sample'] == 0:
                if "model_name=v_48_020" in path:
                    current_folded = path
                    break
            else:
                if f"sample={row['sample']}" in path:
                    current_folded = path
                    break

        sample = pdb_to_structure(sample_path, silent=True)
        folded = pdb_to_structure(current_folded, silent=True)
        sample_ca = [atom for atom in sample.get_atoms() if atom.get_name() == 'CA']
        folded_ca = [atom for atom in folded.get_atoms() if atom.get_name() == 'CA']
        for _ in range(5):
            superimpose.set_atoms(
                fixed=sample_ca,
                moving=folded_ca
            )
            superimpose.apply(folded_ca)
        rmsds.append(superimpose.rms)

        if args.fold_original and row['sample'] == 0:
            sample_all_atom = [atom for atom in sample.get_atoms()]
            folded_all_atom = [atom for atom in folded.get_atoms()]
            for _ in range(5):
                superimpose.set_atoms(
                    fixed=sample_all_atom,
                    moving=folded_all_atom
                )
                superimpose.apply(folded_all_atom)
            all_atom_rmsds.append(superimpose.rms)

    df['sc_rmsd'] = rmsds

    sc_df = df[df['sample'] != 0]
    sc_df.to_csv("sc_rmsd.csv")

    folding_df = df[df['sample'] == 0]
    if len(all_atom_rmsds) == len(folding_df):
        folding_df['sc_rmsd_all_atom'] = all_atom_rmsds
    folding_df.to_csv("folding_rmsd.csv")

    collapsed_df = []
    for name in parse_dict.keys():
        sub_df = sc_df[sc_df['name'] == name]
        min_rmsd = sub_df['sc_rmsd'].min()
        collapsed_df.append(sub_df[sub_df['sc_rmsd'] == min_rmsd])
    collapsed_df = pd.concat(collapsed_df)
    collapsed_df.to_csv("best_sc_rmsd.csv")
    print("Num designable:", len(collapsed_df[collapsed_df.sc_rmsd < 2]))
    print("Num folded correctly (bb):", len(folding_df[folding_df.sc_rmsd < 2]))
    print("Num folded correctly (aa):", len(folding_df[folding_df.sc_rmsd_all_atom < 2]))
