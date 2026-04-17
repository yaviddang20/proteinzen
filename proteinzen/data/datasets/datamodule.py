import os
from functools import partial
from typing import List
import sys
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L
import pandas as pd

from proteinzen.boltz.data.types import (
    Structure,
    Connection,
    Record,
    ConformerRecord,
)
from proteinzen.boltz.data import const
from proteinzen.boltz.data.sample.sampler import Sample

from proteinzen.data.featurize.cropper import Cropper
from proteinzen.data.featurize.tokenize import tokenize_structure, Tokenized
# from proteinzen.data.featurize.assembler import featurize_training, collate
from proteinzen.data.featurize.assembler import featurize, collate

from proteinzen.runtime.sampling.dispatcher import BiomoleculeTaskDispatcher


def compute_lap_pe(adj: np.ndarray, k: int) -> np.ndarray:
    """k smallest non-trivial eigenvectors of normalized graph Laplacian.

    L = I - D^{-1/2} A D^{-1/2}

    Returns [n, k] float32 array. Columns are zero-padded if the graph has
    fewer than k non-trivial eigenvectors (e.g. disconnected or tiny graphs).
    Sign ambiguity is left to the model to handle (random flip at train time
    is an option but not applied here).
    """
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    L = np.eye(n) - d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    eigvals, eigvecs = np.linalg.eigh(L)
    # skip the trivial eigenvector(s) — take indices 1..k
    pe = eigvecs[:, 1:k + 1]
    if pe.shape[1] < k:
        pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
    return pe.astype(np.float32)


def _add_lap_pe_to_features(features: dict, k: int) -> None:
    """Compute LapPE from token_bonds + rigid→token mapping, store in features['rigids']."""
    rigids = features['rigids']
    n_rigids = rigids['rigids_mask'].shape[0]
    token_bonds = features['token']['token_bonds']  # [n_tokens, n_tokens]
    rigid_to_token = rigids['rigids_to_token']      # [n_rigids]
    is_atom = rigids['rigids_is_atom_mask']          # [n_rigids] bool

    lap_pe = torch.zeros(n_rigids, k, dtype=torch.float32)
    atom_idx = torch.where(is_atom)[0]
    if len(atom_idx) > 1:
        # sub-adjacency for atomized rigids only
        tok_idx = rigid_to_token[atom_idx]
        sub_bonds = token_bonds[tok_idx][:, tok_idx]  # [n_atoms, n_atoms]
        adj = (sub_bonds > 0).float().numpy().astype(np.float32)
        pe = compute_lap_pe(adj, k)
        lap_pe[atom_idx] = torch.from_numpy(pe)

    rigids['rigids_lap_pe'] = lap_pe


def strip_h_from_structure(struct: Structure) -> Structure:
    """Remove hydrogen atoms (element=1) from a structure, reindexing bonds/residues/chains."""
    atoms = struct.atoms
    heavy_mask = atoms['element'] != 1
    heavy_indices = np.where(heavy_mask)[0]
    old_to_new = np.full(len(atoms), -1, dtype=np.int32)
    old_to_new[heavy_indices] = np.arange(len(heavy_indices), dtype=np.int32)

    new_atoms = atoms[heavy_mask]

    bonds = struct.bonds
    if len(bonds) > 0:
        bond_heavy = (old_to_new[bonds['atom_1']] >= 0) & (old_to_new[bonds['atom_2']] >= 0)
        new_bonds = bonds[bond_heavy].copy()
        new_bonds['atom_1'] = old_to_new[new_bonds['atom_1']]
        new_bonds['atom_2'] = old_to_new[new_bonds['atom_2']]
    else:
        new_bonds = bonds.copy()

    cumulative_heavy = np.concatenate([[0], np.cumsum(heavy_mask)])

    new_residues = struct.residues.copy()
    for i, res in enumerate(new_residues):
        start = res['atom_idx']
        end = start + res['atom_num']
        new_residues[i]['atom_idx'] = cumulative_heavy[start]
        new_residues[i]['atom_num'] = cumulative_heavy[end] - cumulative_heavy[start]

    new_chains = struct.chains.copy()
    for i, chain in enumerate(new_chains):
        start = chain['atom_idx']
        end = start + chain['atom_num']
        new_chains[i]['atom_idx'] = cumulative_heavy[start]
        new_chains[i]['atom_num'] = cumulative_heavy[end] - cumulative_heavy[start]

    return replace(struct, atoms=new_atoms, bonds=new_bonds, residues=new_residues, chains=new_chains)


def load_input(record: Record, data_dir, include_h: bool = False):
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

    atoms = structure["atoms"]
    if "chirality" not in atoms.dtype.names:
        new_dtype = atoms.dtype.descr + [("chirality", "i1")]
        new_atoms = np.empty(atoms.shape, dtype=new_dtype)
        for name in atoms.dtype.names:
            new_atoms[name] = atoms[name]
        new_atoms["chirality"] = 0
        atoms = new_atoms

    struct = Structure(
        atoms=atoms,
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=chains, # chains var accounting for missing cyclic_period
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    if not include_h:
        struct = strip_h_from_structure(struct)

    rot_bond_data = None
    if 'rot_bonds' in structure:
        n_atoms = struct.atoms.shape[0]
        rot_bond_data = {
            'rot_bonds': structure['rot_bonds'],
            'rot_frag_a': structure['rot_frag_a'],
            'ring_masks': structure['ring_masks'] if 'ring_masks' in structure
                          else np.zeros((0, n_atoms), dtype=bool),
            'sym_groups': structure['sym_groups'] if 'sym_groups' in structure
                          else np.zeros((0, 1), dtype=np.int32),
            'sym_group_sizes': structure['sym_group_sizes'] if 'sym_group_sizes' in structure
                               else np.zeros(0, dtype=np.int32),
        }

    return struct, rot_bond_data


def mirror_structure(struct: Structure) -> Structure:
    """Reflect all atom coordinates across the x-axis and swap CW/CCW chirality tags.

    Produces the enantiomer of the structure. Valid augmentation since enantiomers
    have identical energies.
    """
    atoms = struct.atoms.copy()
    atoms['coords'][:, 0] *= -1
    cw  = const.chirality_type_ids['CHI_TETRAHEDRAL_CW']
    ccw = const.chirality_type_ids['CHI_TETRAHEDRAL_CCW']
    cw_mask  = atoms['chirality'] == cw
    ccw_mask = atoms['chirality'] == ccw
    atoms['chirality'][cw_mask]  = ccw
    atoms['chirality'][ccw_mask] = cw
    return replace(struct, atoms=atoms)


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
        mask_nonstandard=False,
        include_h=False,
        use_identity_rot=True,
        lap_pe_k=0,
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
        self.include_h = include_h
        self.use_identity_rot = use_identity_rot
        self.lap_pe_k = lap_pe_k

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

        struct, rot_bond_data = load_input(sample.record, Path(dataset.data_dir), include_h=self.include_h)
        chain_mask = struct.mask
        remove_chain_masks = [struct.chains['mol_type'] == i for i in self.remove_mol_types]
        for remove_mask in remove_chain_masks:
            chain_mask[remove_mask] = False

        if np.random.random() < 0.5:
            struct = mirror_structure(struct)

        if self.mask_nonstandard:
            struct = mask_nonstandard_residues(struct)

        task_data = task.sample_t_and_mask(struct)

        token_data, rigid_data, token_bonds = tokenize_structure(
            struct,
            task_data,
            use_identity_rot=self.use_identity_rot,
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

        if len(tokenized_data.tokens) == 0:
            return self.__getitem__(idx)

        features = featurize(
            tokenized_data,
            task_data,
            max_tokens=self.max_crop_residues,
            max_rigids=self.max_crop_rigids
        )
        features['task'] = task
        features['structure'] = struct

        if rot_bond_data is not None and rot_bond_data['rot_bonds'].shape[0] > 0:
            features['rot_bonds'] = torch.from_numpy(rot_bond_data['rot_bonds']).long()
            features['rot_frag_a'] = torch.from_numpy(rot_bond_data['rot_frag_a'])
        else:
            features['rot_bonds'] = torch.zeros((0, 2), dtype=torch.long)
            features['rot_frag_a'] = torch.zeros((0, 0), dtype=torch.bool)

        if rot_bond_data is not None:
            features['ring_masks'] = torch.from_numpy(rot_bond_data['ring_masks'])
            features['sym_groups'] = torch.from_numpy(rot_bond_data['sym_groups']).long()
            features['sym_group_sizes'] = torch.from_numpy(rot_bond_data['sym_group_sizes']).long()
        else:
            features['ring_masks'] = torch.zeros((0, 0), dtype=torch.bool)
            features['sym_groups'] = torch.zeros((0, 1), dtype=torch.long)
            features['sym_group_sizes'] = torch.zeros(0, dtype=torch.long)

        e_min = sample.e_min
        features['e_min'] = torch.tensor(e_min, dtype=torch.float32) if e_min is not None else torch.tensor(float('nan'), dtype=torch.float32)

        # Sequential two-group task: store per-rigid group 1 mask.
        # Always present so collate sees a consistent key across all tasks.
        n_rigids = features['rigids']['rigids_mask'].shape[0]
        group1_rigid_mask = torch.zeros(n_rigids, dtype=torch.bool)
        if 'group1_atom_mask' in task_data:
            group1_atom_mask = task_data['group1_atom_mask']  # [n_atoms]
            n = min(len(group1_atom_mask), n_rigids)
            group1_rigid_mask[:n] = torch.from_numpy(group1_atom_mask[:n])
        features['rigids']['group1_rigid_mask'] = group1_rigid_mask

        if self.lap_pe_k > 0:
            _add_lap_pe_to_features(features, self.lap_pe_k)

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self.samples_per_epoch

class ValidationDataset(torch.utils.data.Dataset):
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
        mask_nonstandard=False,
        include_h=False,
        use_identity_rot=True,
        lap_pe_k=0,
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
        self.include_h = include_h
        self.use_identity_rot = use_identity_rot
        self.lap_pe_k = lap_pe_k

        for dataset in datasets:
            for entry in dataset.manifest:
                if isinstance(entry, ConformerRecord):
                    boltzmann_weights = entry.boltzmann_weights
                    conformer_index = np.argmax(boltzmann_weights)
                    record = Record(
                        id=entry.ids[conformer_index],
                        structure=entry.structures[conformer_index],
                        chains=entry.chains,
                        interfaces=entry.interfaces,
                        inference_options=entry.inference_options,
                        templates=entry.templates,
                        md=entry.md,
                        affinity=entry.affinity,
                    )
                    self.samples.append(Sample(record=record, e_min=entry.e_min))
                else:
                    self.samples.append(Sample(record=entry))

    def __getitem__(self, idx):
        dataset_idx = np.random.choice(
            len(self.datasets),
            p=self.dataset_probs,
        )
        dataset = self.datasets[dataset_idx]
        task_sampler = dataset.task_sampler
        sample = self.samples[idx]
        task = task_sampler.sample_task()

        struct, rot_bond_data = load_input(sample.record, Path(dataset.data_dir), include_h=self.include_h)
        chain_mask = struct.mask
        remove_chain_masks = [struct.chains['mol_type'] == i for i in self.remove_mol_types]
        for remove_mask in remove_chain_masks:
            chain_mask[remove_mask] = False

        if self.mask_nonstandard:
            struct = mask_nonstandard_residues(struct)

        task_data = task.sample_t_and_mask(struct)

        token_data, rigid_data, token_bonds = tokenize_structure(
            struct,
            task_data,
            use_identity_rot=self.use_identity_rot,
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

        features = featurize(
            tokenized_data,
        if len(tokenized_data.tokens) == 0:
            return self.__getitem__(idx)

            task_data,
            max_tokens=self.max_crop_residues,
            max_rigids=self.max_crop_rigids
        )
        features['task'] = task
        features['structure'] = struct

        if rot_bond_data is not None and rot_bond_data['rot_bonds'].shape[0] > 0:
            features['rot_bonds'] = torch.from_numpy(rot_bond_data['rot_bonds']).long()
            features['rot_frag_a'] = torch.from_numpy(rot_bond_data['rot_frag_a'])
        else:
            features['rot_bonds'] = torch.zeros((0, 2), dtype=torch.long)
            features['rot_frag_a'] = torch.zeros((0, 0), dtype=torch.bool)

        if rot_bond_data is not None:
            features['ring_masks'] = torch.from_numpy(rot_bond_data['ring_masks'])
            features['sym_groups'] = torch.from_numpy(rot_bond_data['sym_groups']).long()
            features['sym_group_sizes'] = torch.from_numpy(rot_bond_data['sym_group_sizes']).long()
        else:
            features['ring_masks'] = torch.zeros((0, 0), dtype=torch.bool)
            features['sym_groups'] = torch.zeros((0, 1), dtype=torch.long)
            features['sym_group_sizes'] = torch.zeros(0, dtype=torch.long)

        e_min = sample.e_min
        features['e_min'] = torch.tensor(e_min, dtype=torch.float32) if e_min is not None else torch.tensor(float('nan'), dtype=torch.float32)

        # Sequential two-group task: store per-rigid group 1 mask.
        # Always present so collate sees a consistent key across all tasks.
        n_rigids = features['rigids']['rigids_mask'].shape[0]
        group1_rigid_mask = torch.zeros(n_rigids, dtype=torch.bool)
        if 'group1_atom_mask' in task_data:
            group1_atom_mask = task_data['group1_atom_mask']  # [n_atoms]
            n = min(len(group1_atom_mask), n_rigids)
            group1_rigid_mask[:n] = torch.from_numpy(group1_atom_mask[:n])
        features['rigids']['group1_rigid_mask'] = group1_rigid_mask

        if self.lap_pe_k > 0:
            _add_lap_pe_to_features(features, self.lap_pe_k)

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.samples)


class BiomoleculeDataModule(L.LightningDataModule):
    def __init__(self,
                 train_dataset: TrainingDataset,
                 val_dataset: ValidationDataset,
                 batch_size,  # this is PER GPU
                 num_workers,
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def build_dataloader(self, x, collate_fn):
        dataloader = DataLoader(
            x,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )
        return dataloader

    def train_dataloader(self):
        return self.build_dataloader(self.train_dataset, collate)

    def val_dataloader(self):
        return self.build_dataloader(self.val_dataset, collate)


class BiomoleculeSamplingDataModule(L.LightningDataModule):
    def __init__(self,
                 tasks_yaml,
                 batch_size,
                 batching_mode="optimal",
                 use_collate_for_pad=False,
                 trans_std: float = 3,
                 include_h: bool = False,
    ):
        super().__init__()
        self.batching_mode = batching_mode
        self.batch_size = batch_size

        self.task_dispatcher = BiomoleculeTaskDispatcher(
            tasks_yaml,
            1, #batch_size,
            batching_mode,
            use_collate_for_pad,
            trans_std=trans_std,
            include_h=include_h,
        )

    def predict_dataloader(self):
        dataloader = DataLoader(
            self.task_dispatcher,
            batch_size=self.batch_size,
            collate_fn=collate,
            shuffle=False
        )
        return dataloader