# adapted from https://github.com/jingraham/neurips19-graph-protein-design

import os, time, gzip, json
from mmtf_util import *
from collections import defaultdict
import glob
import traceback

import pandas as pd

import tqdm

MAX_LENGTH = 550

df = pd.read_csv("filtered_metadata.csv")
pdb_ids = df.pdb_name.tolist()

# Build dataset
dataset = []
for chain_ix, pdb_id in tqdm.tqdm(enumerate(pdb_ids), total=len(pdb_ids)):
    try:
        # Load and parse coordinates
        # print(chain_ix, pdb, chain)
        start = time.time()
        chain_dict = mmtf_parse(
            pdb_id,
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

        chain = chain_dict['chain']
        chain_name = pdb_id + '.' + chain
        chain_dict['name'] = chain_name
        if len(chain_dict['seq']) <= MAX_LENGTH:
            dataset.append(chain_dict)
        else:
            print("Too long", chain_name, len(chain_dict['seq']))
    except Exception as e:
        print(chain_ix, pdb_id)
        print(traceback.print_exc())

outfile = 'chain_set.jsonl'
with open(outfile, 'w') as f:
    for entry in dataset:
        f.write(json.dumps(entry) + '\n')
