""" Compute the bbRMSD between structures """
import argparse
import os
import glob

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import tqdm
from Bio.PDB.Superimposer import Superimposer

from ligbinddiff.data.io.pdb_utils import pdb_to_structure, struct_to_atom91

def parse_esm_log(lines):
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
    args = parser.parse_args()

    with open(args.esmlog) as fp:
        lines = fp.readlines()
    parse_dict = parse_esm_log(lines)
    df = parse_dict_to_df(parse_dict)
    print(df)

    rmsds = []
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
            if f"sample={row['sample']}" in path:
                current_folded = path
                break

        sample = pdb_to_structure(sample_path, silent=True)
        sample_ca = [atom for atom in sample.get_atoms() if atom.get_name() == 'CA']
        folded = pdb_to_structure(current_folded, silent=True)
        folded_ca = [atom for atom in folded.get_atoms() if atom.get_name() == 'CA']
        for _ in range(5):
            superimpose.set_atoms(
                fixed=sample_ca,
                moving=folded_ca
            )
            superimpose.apply(folded_ca)
        rmsds.append(superimpose.rms)

    df['sc_rmsd'] = rmsds
    df.to_csv("sc_rmsd.csv")

    collapsed_df = []
    for name in parse_dict.keys():
        sub_df = df[df['name'] == name]
        min_rmsd = sub_df['sc_rmsd'].min()
        collapsed_df.append(sub_df[sub_df['sc_rmsd'] == min_rmsd])
    collapsed_df = pd.concat(collapsed_df)
    collapsed_df.to_csv("best_sc_rmsd.csv")
