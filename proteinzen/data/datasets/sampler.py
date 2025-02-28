""" Batch samplers for datasets """
import logging

import numpy as np
import random
import torch
import copy

from torch.utils import data

from .dataset import PdbDataset, GEOMDataset

log = logging.getLogger(__name__)

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
    def __init__(self,
                 dataset: PdbDataset,
                 batch_size=3000,
                 drop_last=False,
                 shuffle=True,
                 batch_by_edge_fn=None,
                 seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.node_counts = dataset.csv.modeled_seq_len.tolist()
        self.idx = [i for i in range(len(self.node_counts))]
        self.batch_by_edge_fn = batch_by_edge_fn

        self._form_batches()

    def _form_batches(self):
        self.batches = []
        if self.shuffle:
            g = np.random.default_rng(self.seed + self.epoch)
            g.shuffle(self.idx)
        idx = np.array(self.idx)
        while len(idx) > 0:
            batch = []
            n_nodes = 0
            while len(idx) > 0 and n_nodes + self.node_counts[idx[0]] <= self.batch_size:
                next_idx, idx = idx[0], idx[1:]
                if self.batch_by_edge_fn is not None:
                    n_nodes += self.batch_by_edge_fn(self.node_counts[next_idx])
                else:
                    n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)

    def __len__(self):
        print(f"len called, current len is {len(self.batches)}")
        if not hasattr(self, "batches"):
            self._form_batches()
        return len(self.batches)

    def __iter__(self):
        print(f"__iter__ called, current len is {len(self.batches)}")
        if self.shuffle:
            self._form_batches()
        for batch in self.batches:
            yield batch

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch




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
    def __init__(self,
                 dataset: PdbDataset,
                 batch_size=3000,
                 drop_last=False,
                 shuffle=True,
                 batch_by_edge_fn=None,
                 num_replicas=1,
                 rank=0,
                 seed=0,
                 afdb_frac=0.75
                 ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.clusters = self.dataset.csv.cluster.unique()
        self.dataset.csv['iloc'] = list(range(len(dataset.csv)))
        self.batch_by_edge_fn = batch_by_edge_fn
        self.afdb_frac = afdb_frac

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.generator = np.random.default_rng(self.seed)

        self._form_batches()

    def _form_batches(self):
        self.batches = []
        idx = []
        node_counts = []
        df = self.dataset.csv

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = np.random.default_rng(self.seed + self.epoch)

            if df['is_af2_struct'].any():
                real_clusters = df[~df['is_af2_struct']].cluster.unique()
                af_clusters = df[df['is_af2_struct']].cluster.unique()
                # # sample at least 25% real data
                # num_af_clusters = min(len(real_clusters)*3, len(af_clusters))

                # subsample number of afdb samples per epoch
                num_af_clusters = min(
                    int(len(real_clusters) * (self.afdb_frac / (1 - self.afdb_frac))),
                    len(af_clusters)
                )

                subsample_af_clusters = g.choice(af_clusters, size=num_af_clusters, replace=False)
                clusters = np.concatenate([real_clusters, subsample_af_clusters])
            else:
                clusters = self.clusters
            print(len(clusters))
            g.shuffle(clusters)


        # sample one example per cluster
        for cluster in clusters:
            cluster_sample = df[df['cluster'] == cluster].sample(1, random_state=g)
            iloc = cluster_sample['iloc'].iloc[0]
            assert df.iloc[iloc].pdb_name == cluster_sample.pdb_name.iloc[0]
            idx.append(iloc)
            node_counts.append(cluster_sample.modeled_seq_len.iloc[0])


        batch = []
        current_node_count = 0
        for i, iloc in enumerate(idx):
            sample_node_count = node_counts[i]
            if self.batch_by_edge_fn is not None:
                sample_node_count = self.batch_by_edge_fn(sample_node_count)
            if sample_node_count + current_node_count <= self.batch_size:
                batch.append(iloc)
                current_node_count += sample_node_count
            else:
                if len(batch) > 0:
                    self.batches.append(batch)
                current_node_count = sample_node_count
                batch = [iloc]
        if not self.drop_last:
            self.batches.append(batch)

        split_size = len(self.batches) // self.num_replicas
        splits = []
        for i in range(self.num_replicas):
            splits.append(self.batches[split_size*i: split_size*(i+1)])
            if i == self.rank:
                log.info(f"rank {self.rank} has split {(split_size*i, split_size*(i+1))}")

        self.batches = splits[self.rank]


    def __len__(self):
        if not hasattr(self, "batches"):
            self._form_batches()
        return len(self.batches)

    def __iter__(self):
        if self.shuffle:
            self._form_batches()
        print(f"rank {self.rank}: epoch is {self.epoch}")
        log.info(f"rank {self.rank}: epoch is {self.epoch}")

        for batch in self.batches:
            yield batch

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


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
    def __init__(self,
                 dataset: PdbDataset,
                 batch_size=3000,
                 max_num_batch=None,
                 drop_last=False,
                 shuffle=True,
                 num_replicas=1,
                 rank=0,
                 seed=0,
                 batch_by_edge_fn=None,
                 ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batch_by_edge_fn = batch_by_edge_fn

        self.node_counts = dataset.csv.modeled_seq_len.tolist()
        self.idx = list(range(len(self.node_counts)))
        if max_num_batch is None:
            max_num_batch = 10000
        self.max_num_batch = max_num_batch

        self.len_idx_map = {}
        for node_count, idx in zip(self.node_counts, self.idx):
            if node_count not in self.len_idx_map.keys():
                self.len_idx_map[node_count] = [idx]
            else:
                self.len_idx_map[node_count].append(idx)

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

        self._form_batches()

    def _form_batches(self):
        self.batches = []
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = np.random.default_rng(self.seed + self.epoch)

        for node_count, idxs in self.len_idx_map.items():
            if self.batch_by_edge_fn is not None:
                node_count = self.batch_by_edge_fn(node_count)
            idxs = idxs.copy()
            if self.shuffle:
                g.shuffle(idxs)

            curr_batch_size = 0
            batch = []
            while len(idxs) > 0:
                if curr_batch_size + node_count > self.batch_size or len(batch) + 1 > self.max_num_batch:
                    self.batches.append(batch)
                    curr_batch_size = 0
                    batch = []
                batch.append(idxs.pop())
                curr_batch_size += node_count
            if len(batch) > 0:
                self.batches.append(batch)

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g.shuffle(self.batches)

        split_size = len(self.batches) // self.num_replicas
        splits = []
        for i in range(self.num_replicas):
            splits.append(self.batches[split_size*i: split_size*(i+1)])

        self.batches = splits[self.rank]

    def __len__(self):
        if not hasattr(self, "batches"):
            self._form_batches()
        return len(self.batches)

    def __iter__(self):
        if self.shuffle:
            self._form_batches()
        print(f"rank {self.rank}: epoch is {self.epoch}")

        for batch in self.batches:
            yield batch

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class ClusteredLengthBatchSampler:
    '''
    Adapted from https://github.com/jingraham/neurips19-graph-protein-design.

    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.

    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self,
                 dataset: PdbDataset,
                 batch_size=3000,
                 drop_last=False,
                 shuffle=True,
                 batch_by_edge_fn=None,
                 max_num_per_batch=None,
                 num_replicas=1,
                 rank=0,
                 seed=0,
                 afdb_frac=0.75
                 ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.clusters = self.dataset.csv.cluster.unique()
        self.dataset.csv['iloc'] = list(range(len(dataset.csv)))
        self.batch_by_edge_fn = batch_by_edge_fn
        if max_num_per_batch is None:
            max_num_per_batch = batch_size
        self.max_num_per_batch = max_num_per_batch
        self.afdb_frac = afdb_frac

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.generator = np.random.default_rng(self.seed)

        self._form_batches()

    def _form_batches(self):
        self.batches = []
        idx = []
        node_counts = []
        df = self.dataset.csv

        # deterministically shuffle based on epoch and seed
        g = np.random.default_rng(self.seed + self.epoch)

        if df['is_af2_struct'].any():
            real_clusters = df[~df['is_af2_struct']].cluster.unique()
            af_clusters = df[df['is_af2_struct']].cluster.unique()
            # subsample number of afdb samples per epoch
            num_af_clusters = min(
                int(len(real_clusters) * (self.afdb_frac / (1 - self.afdb_frac))),
                len(af_clusters)
            )
            subsample_af_clusters = g.choice(af_clusters, size=num_af_clusters, replace=False)
            clusters = np.concatenate([real_clusters, subsample_af_clusters])
        else:
            clusters = self.clusters
        print(len(clusters))

        # sample one example per cluster
        for cluster in clusters:
            cluster_sample = df[df['cluster'] == cluster].sample(1, random_state=g)
            iloc = cluster_sample['iloc'].iloc[0]
            assert df.iloc[iloc].pdb_name == cluster_sample.pdb_name.iloc[0]
            idx.append(iloc)
            node_counts.append(cluster_sample.modeled_seq_len.iloc[0])

        node_counts = np.array(node_counts)
        idx = np.array(idx)
        sort_idx = np.argsort(node_counts)
        idx = idx[sort_idx]
        node_counts = node_counts[sort_idx]

        batch = []
        current_node_count = 0
        for i, iloc in enumerate(idx):
            sample_node_count = node_counts[i]
            if i>0 and sample_node_count == node_counts[i-1] and len(batch) < self.max_num_per_batch:
                if self.batch_by_edge_fn is not None:
                    sample_node_count = self.batch_by_edge_fn(sample_node_count)
                if sample_node_count + current_node_count <= self.batch_size:
                    batch.append(iloc)
                    current_node_count += sample_node_count
                else:
                    if len(batch) > 0:
                        self.batches.append(batch)
                    current_node_count = sample_node_count
                    batch = [iloc]
            else:
                if self.batch_by_edge_fn is not None:
                    sample_node_count = self.batch_by_edge_fn(sample_node_count)
                if len(batch) > 0:
                    self.batches.append(batch)
                current_node_count = sample_node_count
                batch = [iloc]
            # print(batch, current_node_count, sample_node_count)
        if not self.drop_last and len(batch) > 0:
            self.batches.append(batch)

        if self.shuffle:
            g.shuffle(self.batches)

        split_size = len(self.batches) // self.num_replicas
        splits = []
        for i in range(self.num_replicas):
            splits.append(self.batches[split_size*i: split_size*(i+1)])

        self.batches = splits[self.rank]


    def __len__(self):
        if not hasattr(self, "batches"):
            self._form_batches()
        return len(self.batches)

    def __iter__(self):
        if self.shuffle:
            self._form_batches()
        print(f"rank {self.rank}: epoch is {self.epoch}")

        for batch in self.batches:
            yield batch

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch



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

        idx = np.array(self.idx)
        while len(idx) > 0:
            batch = []
            n_nodes = 0
            while len(idx) > 0 and n_nodes + self.node_counts[idx[0]] <= self.batch_size:
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
