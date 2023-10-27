import pandas as pd
import tqdm

df = pd.read_csv("filtered_metadata.csv")
df = df[df.pdb_name != "8ayw"]
# df = df[df.modeled_seq_len <= 512]
# df = df[df.modeled_seq_len  > 60]
# df = df[df.oligomeric_count == "1"]
# df = df[df.coil_percent < 0.5]


train_split = df.sample(frac=0.9)
df = df.drop(index=train_split.index)
val_split = df.sample(frac=0.5)
test_split = df.drop(index=val_split.index)

import json

pdb_id_to_chain = {}

with open("chain_set.jsonl") as fp:
    for l in tqdm.tqdm(fp):
        data = json.loads(l.strip())
        name = data['name'].split(".")[0]
        chain = data['chain']
        pdb_id_to_chain[name] = chain


chain_set_splits = {
    "train": [pdb_id + "." + pdb_id_to_chain[pdb_id] for pdb_id in train_split.pdb_name.tolist() if pdb_id in pdb_id_to_chain.keys()],
    "validation": [pdb_id + "." + pdb_id_to_chain[pdb_id] for pdb_id in val_split.pdb_name.tolist() if pdb_id in pdb_id_to_chain.keys()],
    "test": [pdb_id + "." + pdb_id_to_chain[pdb_id] for pdb_id in test_split.pdb_name.tolist() if pdb_id in pdb_id_to_chain.keys()],
}

with open("chain_set_splits.json", 'w') as fp:
    json.dump(chain_set_splits, fp)
