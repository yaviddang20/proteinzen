import os
import glob
import tqdm
import pandas as pd
import numpy as np

import mdtraj as md

# Workaround for MDtraj not supporting mmcif in their latest release.
def compute_secondary_structure(pdb_path):
    # MDtraj
    traj = md.load(pdb_path)
    # SS calculation
    pdb_ss = md.compute_dssp(traj, simplified=True)
    return pdb_ss

if __name__ == '__main__':
    df_dict = {
        'name': [],
        'percent_helix': [],
        'percent_strand': []
    }

    for file in tqdm.tqdm(glob.glob("*_sc.pdb")):
        sample_id = file.split("_sc.pdb")[0]
        pdb_ss = compute_secondary_structure(file)
        percent_helix = np.mean(pdb_ss == 'H')
        percent_strand = np.mean(pdb_ss == 'E')
        df_dict['name'].append(sample_id)
        df_dict['percent_helix'].append(percent_helix)
        df_dict['percent_strand'].append(percent_strand)

    df = pd.DataFrame(df_dict)
    df.to_csv("dssp.csv")
