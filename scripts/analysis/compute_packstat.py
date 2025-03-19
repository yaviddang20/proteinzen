import argparse
import glob
import os
import tqdm
import shutil

import pyrosetta
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("in_pattern")
parser.add_argument("out_path")
parser.add_argument("--reckless", default=False, action="store_true")
pyrosetta.init()

args = parser.parse_args()
out_dict = {
    "name": [],
    "path": [],
    "packstat": [],
    "holes_score": []
}

pyrosetta.rosetta.basic.options.set_file_option('holes:dalphaball', "/wynton/home/kortemme/alexjli/projects/proteinzen/scripts/analysis/DAlphaBall.gcc")

def get_residue_selector_for_residues(residues):
    '''Get a residue selector for a given set of residues.'''
    return pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(','.join(list(str(i) for i in residues)))

for path in tqdm.tqdm(sorted(glob.glob(args.in_pattern))):
    try:
        name = os.path.basename(path).split(".")[0]
        if path.endswith(".cif"):
            TMPFILE = f"/tmp/folded_intermediate-{hash(name)}.cif"
            shutil.copyfile(path, TMPFILE)
            with open(TMPFILE, 'a') as fp:
                fp.write("\n_dummy.entry AF-structure\n#")
            pose = pyrosetta.io.pose_from_pdb(TMPFILE)
            if os.path.exists(TMPFILE):
                os.remove(TMPFILE)
        else:
            pose = pyrosetta.io.pose_from_pdb(path)
        pose.conformation().detect_disulfides()
        packstat = pyrosetta.rosetta.core.scoring.packstat.compute_packing_score(pose)

        # 1-indexed
        residues = [r.seqpos() for r in pose.residues]
        hf = pyrosetta.rosetta.protocols.simple_filters.HolesFilter()
        hf.set_residue_selector(get_residue_selector_for_residues(residues))
        holes_score = hf.compute(pose) / sum(pose.residue(i).natoms() for i in residues)

        # holes_score = pyrosetta.rosetta.core.scoring.packing.compute_holes_score(pose)
        out_dict["name"].append(name)
        out_dict["path"].append(path)
        out_dict["packstat"].append(packstat)
        out_dict["holes_score"].append(holes_score)
        print(name, packstat, holes_score)
    except Exception as e:
        if args.reckless:
            continue
        else:
            raise e

df = pd.DataFrame(out_dict)
df.to_csv(args.out_path)



