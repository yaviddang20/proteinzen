""" Batch samplers for datasets """
import random

import torch.utils.data as data


from .dataset import PdbDataset



class _BatchSampler(data.BatchSampler):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design.

    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.

    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, sampler: data.SequentialSampler, batch_size=3000, drop_last=False):
        super().__init__(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        shuffle = True

        max_nodes = batch_size
        node_counts = sampler.data_source.node_counts

        self.node_counts = node_counts
        self.idx = [i for i in range(len(node_counts))
                        if node_counts[i] <= max_nodes]
        self.shuffle = shuffle
        self.max_nodes = max_nodes

        self._form_batches()

    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)

    def __len__(self):
        if not self.batches: self._form_batches()
        return len(self.batches)

    def __iter__(self):
        self._form_batches()
        #if not self.batches: self._form_batches()
        for batch in self.batches: yield batch


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