import argparse
import os
import shutil
import glob

import pandas as pd
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compute the ")
    parser.add_argument("--csv")
    parser.add_argument("--folded_folders", help="emfold folded structures")
    parser.add_argument("--samples", help="gen model samples")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    designable_samples = df[df.sc_rmsd < 2]

    out_dir = os.path.join(args.samples, "designable_samples")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for _, row in tqdm.tqdm(list(designable_samples.iterrows())):
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

        shutil.copy(sample_path, out_dir)
        shutil.copy(
            current_folded,
            os.path.join(out_dir, row['name'] + "_sc.pdb")
        )
