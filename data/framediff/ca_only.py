import json
import tqdm
import numpy as np

from proteinzen.utils.atom_reps import atom37_atom_label, atom37_to_atom14, alphabet

def process_entry(line):
    entry = json.loads(line)
    coords = entry['coords']
    name = entry['name']

    atom37 = list(zip(
        *[coords[atom] for atom in atom37_atom_label]
    ))

    seq = [c for c in entry['seq']]
    for c in seq:
        if c not in alphabet:
            return None

    atom14, atom14_mask = atom37_to_atom14(entry['seq'], np.array(atom37))
    entry['coords'] = atom14
    mask = atom14_mask.any(axis=-1)[:, (0, 2, 3)]

    selector = (~mask).astype(float).mean(axis=0)

    if (selector < 0.5).any():
        print(name)


with open("chain_set.jsonl") as fp:
    for l in tqdm.tqdm(fp.readlines()):
        process_entry(l.strip())
