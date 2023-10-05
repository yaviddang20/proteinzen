# adapted from https://github.com/jingraham/neurips19-graph-protein-design

import os, time, gzip, json
from mmtf_util import *
from collections import defaultdict
import glob

import pandas as pd

import tqdm

MAX_LENGTH = 500

df = pd.read_csv("metadata.csv")
df = df[df.modeled_seq_len <= 512]
df = df[df.modeled_seq_len  > 60]
df = df[df.oligomeric_count == "1"]
df = df[df.coil_percent < 0.5]
pdb_ids = set(df.pdb_name.tolist())

data_src = "/wynton/group/kortemme/alexjli/databases/PDB/mmtf/"

paths = []
for path in glob.glob(os.path.join(data_src, "*/*.mmtf")):
    pdb_id = os.path.basename(path).split(".")[0].lower()
    # print(pdb_id)
    if pdb_id in pdb_ids:
        paths.append(path)
        # print("henlo")
    else:
        pass
        # print("gdodbye")
        # print(path, pdb_id)
        # print(df[df.pdb_name == pdb_id])
print(len(paths))

# Build dataset
dataset = []
for chain_ix, path in tqdm.tqdm(enumerate(paths), total=len(paths)):
    try:
        # Load and parse coordinates
        # print(chain_ix, pdb, chain)
        start = time.time()
        chain_dict = mmtf_parse(
            path,
            target_atoms=[
                "N", "CA", "C", "O",
                "CB",
                "CG", "CG1", "CG2", "OG", "OG1", "SG",
                "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
                "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
                "CZ", "CZ2", "CZ3", "NZ",
                "CH2", "NH1", "NH2", "OH"
            ])
        stop = time.time() - start

        if len(chain_dict['seq']) <= MAX_LENGTH:
            pdb = os.path.basename(path).split(".")[0]
            chain = chain_dict['chain']
            chain_name = pdb + '.' + chain
            chain_dict['name'] = chain_name
            print(chain_name)
            # print(pdb, chain, chain_dict['num_chains'], chain_dict['seq'])
            if set([c for c in chain_dict['seq']]).issubset(set(['A','G','T','C','U'])):
                raise ValueError("this is a dumb way of catching if something is DNA/RNA")
            dataset.append(chain_dict)
        else:
            print('Too long')
    except Exception as e:
        print(chain_ix, path)
        print(e)

outfile = 'chain_set.jsonl'
with open(outfile, 'w') as f:
    for entry in dataset:
        f.write(json.dumps(entry) + '\n')
