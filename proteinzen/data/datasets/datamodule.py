import os
from functools import partial
from typing import List
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, HeteroData
import lightning as L
import pandas as pd

from proteinzen.tasks import TaskSampler

from .dataset import PdbDataset, LengthDataset, GEOMDataset
from .sampler import BatchSampler, LengthBatchSampler, AtomicBatchSampler, ClusteredBatchSampler, ClusteredLengthBatchSampler


def gen_collate_fn(task_sampler: TaskSampler):
    def collate_fn(data_list: List[HeteroData]):
        batch = Batch.from_data_list(data_list, exclude_keys=["mask_rotate"])
        #batch = Batch.from_data_list(data_list)
        task = task_sampler.sample_task()
        if 'ligand' in batch.node_types and 'mask_rotate' in data_list[0]['ligand', 'ligand']:
            batch['ligand', 'ligand'].mask_rotate = [d['ligand', 'ligand']['mask_rotate'] for d in data_list]
        batch = task.compile_task_inputs(batch)
        batch.task = task
        return batch
    return collate_fn


class ProteinDataModule(L.LightningDataModule):
    def __init__(self,
                 task_sampler: TaskSampler,
                 data_dir,
                 batch_size,
                 num_workers,
                 length_batch=False
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.task_sampler = task_sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.length_batch = length_batch
        csv = "filtered_metadata.csv"
        # csv = "mini_metadata.csv"
        self.train_dataset = PdbDataset(
            os.path.join(self.data_dir, csv),
            split='train',
            min_percent_ordered=0.0
        )
        self.val_dataset = PdbDataset(
            os.path.join(self.data_dir, csv),
            split='val',
            min_percent_ordered=0.0
        )

        self.test_dataset = PdbDataset(
            os.path.join(self.data_dir, csv),
            split='test',
            min_percent_ordered=0.0
        )

    def build_dataloader(self, x):
        collate_fn = gen_collate_fn(self.task_sampler)
        if self.length_batch:
            dataloader = DataLoader(
                x,
                num_workers=self.num_workers,
                batch_sampler=LengthBatchSampler(x, batch_size=self.batch_size),
                collate_fn=collate_fn,
                shuffle=False
            )
        else:
            dataloader = DataLoader(
                x,
                num_workers=self.num_workers,
                batch_sampler=BatchSampler(x, batch_size=self.batch_size),
                collate_fn=collate_fn,
                shuffle=False
            )
        return dataloader

    def train_dataloader(self):
        return self.build_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self.build_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.build_dataloader(self.test_dataset)

    def predict_dataloader(self):
        return self.build_dataloader(self.test_dataset)


class FramediffDataModule(L.LightningDataModule):
    def __init__(self,
                 task_sampler: TaskSampler,
                 data_dir,
                 batch_size,  # this is PER GPU
                 num_workers,
                 min_len=30,
                 max_len=1000,
                 max_num_batch=None,  # for backwards compatibility
                 max_num_per_batch=None,
                 sample_lengths={
                    60: 5,
                    70: 5,
                    80: 5,
                    90: 5,
                    100: 5,
                    110: 5,
                    120: 5
                 },
                 length_batch=False,
                 sample_from_clusters=False,
                 batch_by_edge_fn=None,
                 min_ordered_percent=None,
                 max_resolution=3.0,
                 use_tmpdir=False,
                 predict_on_train=False,
                 cache_dir=None,
                 normalize_cache=False,
                 use_val_split=False
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.task_sampler = task_sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.length_batch = length_batch
        self.sample_from_clusters = sample_from_clusters
        self.batch_by_edge_fn = batch_by_edge_fn
        self.max_num_per_batch = max_num_per_batch
        self.predict_on_train = predict_on_train

        self.sample_lengths = sample_lengths
        csv = "filtered_metadata.csv"
        # csv = "mini_metadata.csv"

        self.use_val_split = use_val_split
        if use_val_split:
            self.train_dataset = PdbDataset(
                csv_path=os.path.join(self.data_dir, csv),
                min_num_res=min_len,
                max_num_res=max_len,
                min_percent_ordered=min_ordered_percent,
                max_resolution=max_resolution,
                use_tmpdir=use_tmpdir,
                cache_dir=cache_dir,
                normalize_cache=normalize_cache,
                split='train'
            )
            self.val_dataset = PdbDataset(
                csv_path=os.path.join(self.data_dir, csv),
                min_num_res=min_len,
                max_num_res=max_len,
                min_percent_ordered=min_ordered_percent,
                max_resolution=max_resolution,
                use_tmpdir=use_tmpdir,
                cache_dir=cache_dir,
                normalize_cache=normalize_cache,
                split='val'
            )
            self.val_dataloader = lambda: self.build_dataloader(self.val_dataset)

        else:
            self.train_dataset = PdbDataset(
                csv_path=os.path.join(self.data_dir, csv),
                min_num_res=min_len,
                max_num_res=max_len,
                min_percent_ordered=min_ordered_percent,
                max_resolution=max_resolution,
                use_tmpdir=use_tmpdir,
                cache_dir=cache_dir,
                normalize_cache=normalize_cache
            )
        self.predict_dataset = LengthDataset(
            self.sample_lengths,
            batch_size=batch_size,
            same_length_per_batch=length_batch,
            normalize_cache_path=(
                None if not normalize_cache
                else os.path.join(cache_dir, "latent_stats.pt")
            )
        )

    def build_dataloader(self, x):
        collate_fn = gen_collate_fn(self.task_sampler)
        if self.length_batch and self.sample_from_clusters:
            rank = self.trainer.local_rank
            num_replicas = self.trainer.num_devices
            dataloader = DataLoader(
                x,
                num_workers=self.num_workers,
                batch_sampler=ClusteredLengthBatchSampler(
                    x,
                    batch_size=self.batch_size,
                    rank=rank,
                    num_replicas=num_replicas,
                    batch_by_edge_fn=self.batch_by_edge_fn,
                    max_num_per_batch=self.max_num_per_batch
                ),
                collate_fn=collate_fn,
                shuffle=False
            )
        elif self.length_batch:
            rank = self.trainer.local_rank
            num_replicas = self.trainer.num_devices
            dataloader = DataLoader(
                x,
                num_workers=self.num_workers,
                batch_sampler=LengthBatchSampler(
                    x,
                    batch_size=self.batch_size,
                    rank=rank,
                    num_replicas=num_replicas,
                    batch_by_edge_fn=self.batch_by_edge_fn,
                    max_num_batch=self.max_num_per_batch
                ),
                collate_fn=collate_fn,
                shuffle=False
            )
        elif self.sample_from_clusters:
            rank = self.trainer.local_rank
            num_replicas = self.trainer.num_devices
            dataloader = DataLoader(
                x,
                num_workers=self.num_workers,
                batch_sampler=ClusteredBatchSampler(
                    x,
                    batch_size=self.batch_size,
                    batch_by_edge_fn=self.batch_by_edge_fn,
                    rank=rank,
                    num_replicas=num_replicas
                ),
                collate_fn=collate_fn,
                shuffle=False
            )
        else:
            dataloader = DataLoader(
                x,
                num_workers=self.num_workers,
                batch_sampler=BatchSampler(x, batch_size=self.batch_size, batch_by_edge_fn=self.batch_by_edge_fn),
                collate_fn=collate_fn,
                shuffle=False
            )
        return dataloader

    def train_dataloader(self):
        return self.build_dataloader(self.train_dataset)

    # def val_dataloader(self):
    #     if self.use_val_split:
    #         return self.build_dataloader(self.val_dataset)
    #     else:
    #         return super().val_dataloader()

    def predict_dataloader(self):
        if self.predict_on_train:
            print(self.train_dataloader())
            return self.train_dataloader()
        else:
            def collate_fn(batch):
                batch = batch[0]
                batch['task'] = self.task_sampler.sample_task()
                return batch

            return DataLoader(
                self.predict_dataset,
                shuffle=False,
                collate_fn=collate_fn,
                batch_size=1
            )


class GeomDataModule(L.LightningDataModule):
    def __init__(self,
                 task_sampler: TaskSampler,
                 data_dir,
                 batch_size,
                 num_workers,
                 max_len=1000,
                 length_batch=False
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.task_sampler = task_sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.length_batch = length_batch

        csv = "filtered_metadata.csv"
        df = pd.read_csv(os.path.join(self.data_dir, csv))
        # num_train = len(df) // 10 * 9
        # num_test = len(df) // 10
        num_train = 1 # int(3e5)
        num_test = -1 # -100
        # csv = "mini_metadata.csv"
        self.train_dataset = GEOMDataset(
            os.path.join(self.data_dir, csv),
            subset=num_train
        )
        self.val_dataset = GEOMDataset(
            os.path.join(self.data_dir, csv),
            subset=-num_test
        )
        self.predict_dataset = GEOMDataset(
            os.path.join(self.data_dir, csv),
            subset=-num_test
        )


    def build_dataloader(self, x):
        collate_fn = gen_collate_fn(self.task_sampler)
        dataloader = DataLoader(
            x,
            num_workers=self.num_workers,
            batch_sampler=AtomicBatchSampler(x, batch_size=self.batch_size),
            collate_fn=collate_fn,
            shuffle=False
        )
        return dataloader

    def train_dataloader(self):
        return self.build_dataloader(self.train_dataset)

    def val_dataloader(self):
        collate_fn = gen_collate_fn(self.task_sampler)
        dataloader = DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_sampler=AtomicBatchSampler(self.val_dataset, batch_size=self.batch_size, shuffle=False),
            collate_fn=collate_fn,
            shuffle=False
        )
        return dataloader

    def predict_dataloader(self):
        collate_fn = gen_collate_fn(self.task_sampler)
        dataloader = DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_sampler=AtomicBatchSampler(self.predict_dataset, batch_size=self.batch_size, shuffle=False),
            collate_fn=collate_fn,
            shuffle=False
        )
        return dataloader
