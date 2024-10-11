import numpy as np
import mdtraj as md
import os

def compute_dssp(sample_file, folded_file):
    # MDtraj
    traj = md.load(sample_file)
    # SS calculation
    sample_pdb_ss = md.compute_dssp(traj, simplified=True)
    # DG calculation
    pdb_dg = md.compute_rg(traj)

    metadata = {}
    metadata['name'] = os.path.basename(sample_file).strip(".pdb")
    metadata['seq_len'] = sample_pdb_ss.shape[1]
    metadata['sample_dssp'] = "".join(sample_pdb_ss.reshape(-1).tolist())
    # Radius of gyration
    metadata['sample_rog'] = pdb_dg[0]
    # MDtraj
    traj = md.load(folded_file)
    # SS calculation
    folded_pdb_ss = md.compute_dssp(traj, simplified=True)
    # DG calculation
    pdb_dg = md.compute_rg(traj)

    metadata['folded_dssp'] = "".join(folded_pdb_ss.reshape(-1).tolist())
    # Radius of gyration
    metadata['folded_rog'] = pdb_dg[0]

    metadata['dssp_match'] = (sample_pdb_ss == folded_pdb_ss).mean()
    metadata['sample_e_to_folded_h'] = ((sample_pdb_ss == 'E') & (folded_pdb_ss == 'H')).sum() / max((sample_pdb_ss == 'E').sum(), 1)
    metadata['sample_h_to_folded_e'] = ((sample_pdb_ss == 'H') & (folded_pdb_ss == 'E')).sum() / max((sample_pdb_ss == 'H').sum(), 1)


    return metadata

if __name__ == '__main__':
    import glob
    import pandas as pd
    from tqdm import tqdm

    results = []
    for pdb in tqdm(glob.glob("not_consistent_samples/*.pdb")):
        folded_pdb_file = os.path.basename(pdb)
        folded_pdb_file = "not_consistent_samples_folded/" + folded_pdb_file.replace(".", "_sc.")
        results.append(compute_dssp(pdb, folded_pdb_file))

    df = pd.DataFrame(results)
    df.to_csv("dssp_not_consistent_samples.csv")