""" Batch samplers for datasets """
import numpy as np
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


class ClusteredBatchSampler:
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
        self.clusters = self.dataset.csv.cluster.unique()
        self.dataset.csv['iloc'] = list(range(len(dataset.csv)))

        self._form_batches()

    def _form_batches(self):
        self.batches = []
        idx = []
        node_counts = []
        df = self.dataset.csv

        if self.shuffle:
            np.random.shuffle(self.clusters)

        # sample one example per cluster
        for cluster in self.clusters:
            cluster_sample = df[df['cluster'] == cluster].sample(1)
            iloc = cluster_sample['iloc'].iloc[0]
            assert df.iloc[iloc].pdb_name == cluster_sample.pdb_name.iloc[0]
            idx.append(iloc)
            node_counts.append(cluster_sample.modeled_seq_len.iloc[0])


        batch = []
        current_node_count = 0
        for i, iloc in enumerate(idx):
            sample_node_count = node_counts[i]
            if sample_node_count + current_node_count <= self.batch_size:
                batch.append(iloc)
                current_node_count += sample_node_count
            else:
                self.batches.append(batch)
                current_node_count = sample_node_count
                batch = [iloc]
        if not self.drop_last:
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