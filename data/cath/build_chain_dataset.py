# adapted from https://github.com/jingraham/neurips19-graph-protein-design

import os, time, gzip, json
from mmtf_util import *
from collections import defaultdict

import tqdm

MAX_LENGTH = 500

with open("chain_set_splits.json") as fp:
    chain_set_splits = json.load(fp)

chain_set = []
chain_set += [pdb_chain.split(".") for pdb_chain in chain_set_splits["train"]]
chain_set += [pdb_chain.split(".") for pdb_chain in chain_set_splits["validation"]]
chain_set += [pdb_chain.split(".") for pdb_chain in chain_set_splits["test"]]
cath_nodes = chain_set_splits["cath_nodes"]

# Build dataset
dataset = []
for chain_ix, (pdb, chain) in tqdm.tqdm(enumerate(chain_set), total=len(chain_set)):
    try:
        # Load and parse coordinates
        # print(chain_ix, pdb, chain)
        start = time.time()
        chain_dict = mmtf_parse(
            pdb,
            chain,
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
            chain_name = pdb + '.' + chain
            chain_dict['name'] = chain_name
            chain_dict['CATH'] = cath_nodes[chain_name]
            # print(pdb, chain, chain_dict['num_chains'], chain_dict['seq'])
            dataset.append(chain_dict)
        else:
            print('Too long')
    except Exception as e:
        print(chain_ix, pdb, chain)
        print(e)

outfile = 'chain_set.jsonl'
with open(outfile, 'w') as f:
    for entry in dataset:
        f.write(json.dumps(entry) + '\n')
