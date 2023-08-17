""" Dataloaders for CATH fold split data

adapted from https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py"""
import json
import tqdm

import numpy as np

from ligbinddiff.utils.atom_reps import atom37_atom_label, atom37_to_atom14



class CATHDataset:
    '''
    Loader and container class for the CATH 4.2 dataset downloaded
    from http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/.

    Has attributes `self.train`, `self.val`, `self.test`, each of which are
    JSON/dictionary-type datasets as described in README.md.

    :param path: path to chain_set.jsonl
    :param splits_path: path to chain_set_splits.json or equivalent.
    '''
    def __init__(self, path, splits_path):
        with open(splits_path) as f:
            dataset_splits = json.load(f)
        train_list, val_list, test_list = dataset_splits['train'], \
            dataset_splits['validation'], dataset_splits['test']

        self.train, self.val, self.test = [], [], []

        with open(path) as f:
            lines = f.readlines()

        for line in tqdm.tqdm(lines):
            entry = json.loads(line)
            name = entry['name']

            if name in ['3k7a.M']:  # this is backbone-only somehow...
                continue

            coords = entry['coords']

            atom37 = list(zip(
                *[coords[atom] for atom in atom37_atom_label]
            ))
            atom14, atom14_mask = atom37_to_atom14(entry['seq'], np.array(atom37))
            entry['coords'] = atom14
            entry['coords_mask'] = atom14_mask

            if name in train_list:
                self.train.append(entry)
            elif name in val_list:
                self.val.append(entry)
            elif name in test_list:
                self.test.append(entry)
