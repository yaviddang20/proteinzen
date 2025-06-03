import pandas as pd
import numpy as np
import json

from proteinzen.utils.atom_reps import atom37_atom_label, atom37_to_atom14

metadata = pd.read_csv("filtered_metadata.csv")
import tqdm

not_in_data = []
too_short = []

with open("chain_set.jsonl") as fp:
    for l in tqdm.tqdm(fp.readlines()):
        data = json.loads(l.strip())
        coords = data['coords']
        data_len = len(coords['CA'])

        name = data['name'][:4]
        metadata_len = (metadata[metadata.pdb_name == name]).seq_len.tolist()
        if len(metadata_len) == 0:
            print(name, "not in filtered data")
            not_in_data.append(name)
        else:
            if data_len != metadata_len[0]:
                print(name, data_len, metadata_len[0])
                too_short.append((name, data_len, metadata_len[0]))

print(len(not_in_data))
print(len(too_short))
with open("not_in_data.json", 'w') as fp:
    json.dump(not_in_data, fp)
with open("too_short.json", 'w') as fp:
    json.dump(sorted(too_short, key=lambda x: x[1], reverse=True), fp)
