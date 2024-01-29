""" Batch samplers for datasets """
import random

from torch.utils import data

from .dataset import PdbDataset, GEOMDataset



class BatchSampler:
    '''
    Adapted from https://github.com/jingraham/neurips19-graph-protein-design.

    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.

    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, dataset: PdbDataset, batch_size=3000, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.node_counts = dataset.csv.modeled_seq_len.tolist()
        self.idx = [i for i in range(len(self.node_counts))]

        self._form_batches()

    def _form_batches(self):
        self.batches = []
        if self.shuffle:
            random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.batch_size:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)

    def __len__(self):
        if not hasattr(self, "batches"):
            self._form_batches()
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch
        if self.shuffle:
            self._form_batches()


class LengthBatchSampler:
    '''
    Adapted from https://github.com/jingraham/neurips19-graph-protein-design.

    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.

    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, dataset: PdbDataset, batch_size=3000, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.node_counts = dataset.csv.modeled_seq_len.tolist()
        self.idx = list(range(len(self.node_counts)))

        self.len_idx_map = {}
        for node_count, idx in zip(self.node_counts, self.idx):
            if node_count not in self.len_idx_map.keys():
                self.len_idx_map[node_count] = [idx]
            else:
                self.len_idx_map[node_count].append(idx)

        self._form_batches()

    def _form_batches(self):
        self.batches = []
        for node_count, idxs in self.len_idx_map.items():
            idxs = idxs.copy()
            if self.shuffle:
                random.shuffle(idxs)

            curr_batch_size = 0
            batch = []
            while len(idxs) > 0:
                if curr_batch_size > self.batch_size:
                    self.batches.append(batch)
                    curr_batch_size = 0
                    batch = []
                batch.append(idxs.pop())
                curr_batch_size += node_count
            if len(batch) > 0:
                self.batches.append(batch)

        if self.shuffle:
            random.shuffle(self.batches)

    def __len__(self):
        if not hasattr(self, "batches"):
            self._form_batches()
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch
        if self.shuffle:
            self._form_batches()


class AtomicBatchSampler:
    '''
    Adapted from https://github.com/jingraham/neurips19-graph-protein-design.

    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.

    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, dataset: GEOMDataset, batch_size=3000, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        if isinstance(dataset, data.SequentialSampler):
            dataset = dataset.data_source

        self.node_counts = dataset.csv.num_atoms.tolist()
        self.idx = [i for i in range(len(self.node_counts))]

        self._form_batches()

    def _form_batches(self):
        self.batches = []
        if self.shuffle:
            random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.batch_size:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)

    def __len__(self):
        if not hasattr(self, "batches"):
            self._form_batches()
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch
        if self.shuffle:
            self._form_batches()