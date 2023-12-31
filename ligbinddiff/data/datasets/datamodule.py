import os
from functools import partial
from typing import List

from torch.utils.data import DataLoader
from torch_geometric.data import Batch, HeteroData
import lightning as L

from ligbinddiff.tasks import TaskSampler

from .dataset import PdbDataset
from .sampler import BatchSampler


def gen_collate_fn(task_sampler: TaskSampler):
    def collate_fn(data_list: List[HeteroData]):
        batch = Batch.from_data_list(data_list)
        task = task_sampler.sample_task()
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
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.task_sampler = task_sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        csv = "filtered_metadata.csv"
        self.train_dataset = PdbDataset(
            os.path.join(self.data_dir, csv),
            split='train'
        )
        self.val_dataset = PdbDataset(
            os.path.join(self.data_dir, csv),
            split='val'
        )

        self.test_dataset = PdbDataset(
            os.path.join(self.data_dir, csv),
            split='test'
        )

    def build_dataloader(self, x):
        collate_fn = gen_collate_fn(self.task_sampler)
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