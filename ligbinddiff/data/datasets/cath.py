""" Dataloaders for CATH fold split data

adapted from https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py"""
import multiprocessing as mp
import json
import tqdm

import numpy as np

from ligbinddiff.utils.atom_reps import atom37_atom_label, atom37_to_atom14, alphabet


def process_entry(line):
    entry = json.loads(line)
    name = entry['name']

    if name in ['3k7a.M']:  # this is backbone-only somehow...
        return None

    if name in [
        '1i94.H',
        '1i94.M',
        '1rws.A',
        '1jb0.K',
        '1fka.D',
        '1fka.G',
        '4dt0.A',
        '5adx.V',
        '5dn6.I',
        '5dn6.J',
    ]:  # these are ALA only somehow...
        return None

    if name in ['1din.A', '2j6v.A']:  # weird coord business (122 == 235 coord-wise?)
        return None

    if name in [
        '2qqh.A',
        '2yhx.A',
        '3jxv.A',
        '6n0w.A',
        '7f0s.A',
        '2gnx.A',
        '6n0w.A',
        '7f0s.A',
        '3jxv.A',
        '2qqh.A',
        '2yhx.A',
        '6hkq.A',
        '1pfp.A',
        '6hn3.A',
        '6un2.A',
    ]:  # these have Xs and i am too lazy to deal with that rn
        return None


    coords = entry['coords']

    atom37 = list(zip(
        *[coords[atom] for atom in atom37_atom_label]
    ))

    seq = [c for c in entry['seq']]
    for c in seq:
        if c not in alphabet:
            print(name)
            return None
    atom14, atom14_mask = atom37_to_atom14(entry['seq'], np.array(atom37))
    entry['coords'] = atom14
    if np.isnan(atom14[:, :4]).all():
        print(name)
        return None
    entry['coords_mask'] = atom14_mask
    return entry


class CATHDataset:
    '''
    Loader and container class for the CATH 4.2 dataset downloaded
    from http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/.

    Has attributes `self.train`, `self.val`, `self.test`, each of which are
    JSON/dictionary-type datasets as described in README.md.

    :param path: path to chain_set.jsonl
    :param splits_path: path to chain_set_splits.json or equivalent.
    '''
    def __init__(self, path, splits_path, num_proc=32):
        with open(splits_path) as f:
            dataset_splits = json.load(f)
        train_list, val_list, test_list = dataset_splits['train'], \
            dataset_splits['validation'], dataset_splits['test']

        self.train, self.val, self.test = [], [], []

        with open(path) as f:
            lines = f.readlines()

        # parallelize data processing
        pbar = tqdm.tqdm(total=len(lines))
        def callback(x):
            pbar.update(1)

        with mp.Pool(num_proc) as pool:
            entry_res_list = [
                pool.apply_async(process_entry, (line,), callback=callback, error_callback=callback)
                for line in lines
            ]
            pool.close()
            pool.join()
        pbar.close()
        entry_list = [res.get() for res in entry_res_list if res.get() is not None]

        for entry in entry_list:
            name = entry['name']
            if name in train_list:
                self.train.append(entry)
            elif name in val_list:
                self.val.append(entry)
            elif name in test_list:
                self.test.append(entry)
