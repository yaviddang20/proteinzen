import os
from functools import partial
from typing import List
import sys
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, HeteroData
import lightning as L
import pandas as pd

from proteinzen.boltz.data.types import (
    Structure,
    Connection,
    Record
)
from proteinzen.boltz.data import const
from proteinzen.boltz.data.sample.sampler import Sample

from proteinzen.data.datasets.featurize.cropper import Cropper
from proteinzen.data.datasets.featurize.tokenize import tokenize_structure, Tokenized
from proteinzen.data.datasets.featurize.assembler import featurize_training, collate as boltz_collate

from proteinzen.harness.harness import TrainingHarness
from proteinzen.runtime.sampling.dispatcher import TaskDispatcher, BiomoleculeTaskDispatcher

from .dataset import PdbDataset, MMCIFDataset, LengthDataset
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
                # print(task)
                # corrupter = self.training_harness.frame_noiser
                data_list = [featurize_input(task, d) for d in data_list]
                batch = collate(data_list)
                # batch = corrupter.corrupt_dense_batch(batch)
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


def load_input(record, data_dir):
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to msa directory.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    try:
        # find the subdirectory
        if "AF-" in record.id:
            mid = record.id[6:8]
        elif "af-" in record.id:
            mid = record.id[6:8]
        else:
            mid = record.id[1:3]
        structure = np.load(data_dir / "structures" / mid / f"{record.id}.npz")
    except:
        # original boltz format
        structure = np.load(data_dir / "structures" / f"{record.id}.npz")

    # In order to add cyclic_period to chains if it does not exist
    # Extract the chains array
    chains = structure["chains"]
    # Check if the field exists
    if "cyclic_period" not in chains.dtype.names:
        # Create a new dtype with the additional field
        new_dtype = chains.dtype.descr + [("cyclic_period", "i4")]
        # Create a new array with the new dtype
        new_chains = np.empty(chains.shape, dtype=new_dtype)
        # Copy over existing fields
        for name in chains.dtype.names:
            new_chains[name] = chains[name]
        # Set the new field to 0
        new_chains["cyclic_period"] = 0
        # Replace old chains array with new one
        chains = new_chains

    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=chains, # chains var accounting for missing cyclic_period
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    return structure


def mask_nonstandard_residues(struct: Structure):
    residues = struct.residues
    atoms = struct.atoms

    residues_copy = residues.copy()
    residues_copy['is_present'] = residues['is_present'] & residues['is_standard']
    atoms_copy = atoms.copy()
    for residue in residues:
        atom_idx = residue['atom_idx']
        atom_num = residue['atom_num']
        res_atoms = atoms_copy[atom_idx:atom_idx+atom_num]
        if not residue['is_standard']:
            res_atoms['is_present'] = False

    return replace(struct, atoms=atoms_copy, residues=residues_copy)


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets,
        max_crop_residues,
        max_crop_rigids,
        use_cropper=True,
        samples_per_epoch=1000,  # this is PER GPU
        crop_min_neighbors=0,
        crop_max_neighbors=40,
        dataset_probs=None,
        remove_mol_types=None,
        mask_nonstandard=False
    ):
        super().__init__()
        self.datasets = datasets
        self.max_crop_residues = max_crop_residues
        self.max_crop_rigids = max_crop_rigids
        self.samples_per_epoch = samples_per_epoch
        if dataset_probs is None:
            self.dataset_probs = [1/len(datasets) for _ in datasets]
        else:
            self.dataset_probs = dataset_probs
        self.samples = []
        if use_cropper:
            self.cropper = Cropper(
                min_neighborhood=crop_min_neighbors,
                max_neighborhood=crop_max_neighbors
            )
        else:
            self.cropper = None

        if remove_mol_types is None:
            self.remove_mol_types = []
        else:
            print("Removing chains of types:", remove_mol_types)
            self.remove_mol_types = [const.chain_types.index(s) for s in remove_mol_types]
        self.mask_nonstandard = mask_nonstandard

        for dataset in datasets:
            records = dataset.manifest
            iterator = dataset.data_sampler.sample(records, np.random)
            self.samples.append(iterator)

    def __getitem__(self, idx):
        dataset_idx = np.random.choice(
            len(self.datasets),
            p=self.dataset_probs,
        )
        dataset = self.datasets[dataset_idx]
        task_sampler = dataset.task_sampler
        sample: Sample = next(self.samples[dataset_idx])
        task = task_sampler.sample_task()

        struct = load_input(sample.record, Path(dataset.data_dir))
        chain_mask = struct.mask
        remove_chain_masks = [struct.chains['mol_type'] == i for i in self.remove_mol_types]
        for remove_mask in remove_chain_masks:
            chain_mask[remove_mask] = False

        if self.mask_nonstandard:
            struct = mask_nonstandard_residues(struct)

        task_data = task.sample_t_and_mask(struct)

        token_data, rigid_data, token_bonds = tokenize_structure(
            struct,
            task_data
        )
        tokenized_data = Tokenized(
            tokens=token_data,
            rigids=rigid_data,
            bonds=token_bonds,
            structure=struct
        )

        if self.cropper is not None:
            crop_size = self.max_crop_residues - task.max_added_tokens(token_data.shape[0])
            if len(tokenized_data.tokens) > crop_size:
                tokenized_data = self.cropper.crop(
                    tokenized_data,
                    max_tokens=crop_size,
                    random=np.random,
                    chain_id=sample.chain_id,
                    interface_id=sample.interface_id
                )

        features = featurize_training(
            tokenized_data,
            task_data,
            max_tokens=self.max_crop_residues,
            max_rigids=self.max_crop_rigids
        )
        features['task'] = task

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self.samples_per_epoch


class BiomoleculeDataModule(L.LightningDataModule):
    def __init__(self,
                 train_dataset: TrainingDataset,
                 batch_size,  # this is PER GPU
                 num_workers,
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = train_dataset

    def build_dataloader(self, x, collate_fn):
        dataloader = DataLoader(
            x,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=False
        )
        return dataloader

    def train_dataloader(self):
        return self.build_dataloader(self.train_dataset, boltz_collate)


class BiomoleculeSamplingDataModule(L.LightningDataModule):
    def __init__(self,
                 tasks_yaml,
                 batch_size,
                 batching_mode="optimal",
                 use_collate_for_pad=False
    ):
        super().__init__()
        self.batching_mode = batching_mode
        self.batch_size = batch_size

        self.task_dispatcher = BiomoleculeTaskDispatcher(
            tasks_yaml,
            1, #batch_size,
            batching_mode,
            use_collate_for_pad
        )

    def predict_dataloader(self):
        dataloader = DataLoader(
            self.task_dispatcher,
            batch_size=self.batch_size,
            collate_fn=boltz_collate,
            shuffle=False
        )
        return dataloader



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
