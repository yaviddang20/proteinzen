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

from proteinzen.harness.harness import TrainingHarness
from proteinzen.runtime.sampling.dispatcher import TaskDispatcher

from .dataset import PdbDataset, LengthDataset
from .sampler import BatchSampler, LengthBatchSampler, ClusteredBatchSampler, ClusteredLengthBatchSampler
from .collate import featurize_input, collate


def gen_collate_fn(training_harness: TrainingHarness):
    def collate_fn(data_list: List[HeteroData]):
        batch = Batch.from_data_list(data_list, exclude_keys=["mask_rotate"])
        #batch = Batch.from_data_list(data_list)
        if 'ligand' in batch.node_types and 'mask_rotate' in data_list[0]['ligand', 'ligand']:
            batch['ligand', 'ligand'].mask_rotate = [d['ligand', 'ligand']['mask_rotate'] for d in data_list]
        batch = training_harness.process_input(batch)
        return batch
    return collate_fn


class FramediffDataModule(L.LightningDataModule):
    def __init__(self,
                 training_harness: TrainingHarness,
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
                 sample_by_length_bucket=False,
                 batch_by_edge_fn=None,
                 min_ordered_percent=None,
                 max_resolution=3.0,
                 use_tmpdir=False,
                 predict_on_train=False,
                 use_val_split=False,
                 afdb_frac=0.75,
                 use_collate_v2=False
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.training_harness = training_harness
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.length_batch = length_batch
        self.sample_from_clusters = sample_from_clusters
        self.batch_by_edge_fn = batch_by_edge_fn
        self.max_num_per_batch = max_num_per_batch
        self.predict_on_train = predict_on_train
        self.afdb_frac = afdb_frac
        self.sample_by_length_bucket = sample_by_length_bucket
        self.use_collate_v2 = use_collate_v2

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
                split='train'
            )
            self.val_dataset = PdbDataset(
                csv_path=os.path.join(self.data_dir, csv),
                min_num_res=min_len,
                max_num_res=max_len,
                min_percent_ordered=min_ordered_percent,
                max_resolution=max_resolution,
                use_tmpdir=use_tmpdir,
                split='val'
            )

            def val_collate_fn(data_list: List[HeteroData]):
                batch = Batch.from_data_list(data_list, exclude_keys=["mask_rotate"])
                #batch = Batch.from_data_list(data_list)
                if 'ligand' in batch.node_types and 'mask_rotate' in data_list[0]['ligand', 'ligand']:
                    batch['ligand', 'ligand'].mask_rotate = [d['ligand', 'ligand']['mask_rotate'] for d in data_list]
                batch = training_harness.process_input(batch) # , force_t0=True)
                return batch
            self.val_dataloader = lambda: self.build_dataloader(self.val_dataset, val_collate_fn)

        else:
            self.train_dataset = PdbDataset(
                csv_path=os.path.join(self.data_dir, csv),
                min_num_res=min_len,
                max_num_res=max_len,
                min_percent_ordered=min_ordered_percent,
                max_resolution=max_resolution,
                use_tmpdir=use_tmpdir,
            )
        self.predict_dataset = LengthDataset(
            self.sample_lengths,
            batch_size=batch_size,
            same_length_per_batch=length_batch,
            batch_by_edge_fn=batch_by_edge_fn
        )

    def build_dataloader(self, x, collate_fn):
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
                    max_num_per_batch=self.max_num_per_batch,
                    afdb_frac=self.afdb_frac,
                    sample_by_length_bucket=self.sample_by_length_bucket
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
                    num_replicas=num_replicas,
                    afdb_frac=self.afdb_frac,
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
        if self.use_collate_v2:
            def collate_fn(data_list):
                task = self.training_harness.task_sampler.sample_task()
                corrupter = self.training_harness.frame_noiser
                data_list = [featurize_input(task, d) for d in data_list]
                batch = collate(data_list)
                batch = corrupter.corrupt_dense_batch(batch)
                batch['task'] = task
                return batch
        else:
            collate_fn = gen_collate_fn(self.training_harness)
        return self.build_dataloader(self.train_dataset, collate_fn)

    def predict_dataloader(self):
        if self.predict_on_train:
            print(self.train_dataloader())
            return self.train_dataloader()
        else:
            def collate_fn(batch):
                batch = batch[0]
                return batch
            rank = self.trainer.local_rank
            num_replicas = self.trainer.num_devices

            return DataLoader(
                self.predict_dataset,
                shuffle=False,
                sampler=torch.utils.data.distributed.DistributedSampler(
                    self.predict_dataset,
                    num_replicas=num_replicas,
                    rank=rank,
                    shuffle=True, #False,
                    seed=0,
                    drop_last=False
                ),
                collate_fn=collate_fn,
                batch_size=1
            )


class SamplingDataModule(L.LightningDataModule):
    def __init__(self,
                 tasks_yaml,
                 batch_size,
                 batching_mode="optimal",
                 use_collate_for_pad=False
    ):
        super().__init__()
        self.batching_mode = batching_mode

        self.task_dispatcher = TaskDispatcher(
            tasks_yaml,
            batch_size,
            batching_mode,
            use_collate_for_pad
        )

    def predict_dataloader(self):
        if self.batching_mode == "optimal":
            rank = self.trainer.local_rank
            num_replicas = self.trainer.num_devices
            return DataLoader(
                self.task_dispatcher,
                sampler=torch.utils.data.distributed.DistributedSampler(
                    self.task_dispatcher,
                    num_replicas=num_replicas,
                    rank=rank,
                    shuffle=False,
                    seed=0,
                    drop_last=False
                ),
                shuffle=False,
                collate_fn=lambda x: x[0],
                batch_size=1
            )

        elif self.batching_mode == "lazy":
            return DataLoader(
                self.task_dispatcher,
                shuffle=False,
                collate_fn=lambda x: x[0],
                batch_size=1
            )
