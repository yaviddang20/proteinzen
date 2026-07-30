import os
from functools import partial
from typing import List
import sys
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader
import lightning as L
import pandas as pd

from proteinzen.boltz.data.types import (
    Structure,
    Connection,
    Interface,
    Record,
    ConformerRecord,
)
from proteinzen.boltz.data import const
from proteinzen.boltz.data.sample.sampler import Sample

from proteinzen.data.featurize.cropper import Cropper
from proteinzen.data.featurize.tokenize import tokenize_structure, Tokenized
# from proteinzen.data.featurize.assembler import featurize_training, collate
from proteinzen.data.featurize.assembler import featurize, collate

from proteinzen.runtime.sampling.dispatcher import BiomoleculeTaskDispatcher, TaskBatchSampler
from typing import Optional

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


def _smiles_etkdg_pos(smiles, n_rigids, rigids_noising_mask):
    """Generate a random ETKDG conformer from SMILES and return positions as a
    (n_rigids, 3) tensor (zeros for protein rigids, or on failure)."""
    result = torch.zeros(n_rigids, 3, dtype=torch.float32)
    if not smiles:
        return result

    lig_rigid_indices = rigids_noising_mask.nonzero(as_tuple=True)[0]

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return result
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = -1
        if AllChem.EmbedMolecule(mol, params) != 0:
            return result
        mol = Chem.RemoveHs(mol)
        positions = mol.GetConformer().GetPositions()  # (n_lig_atoms, 3)
        n = min(len(positions), len(lig_rigid_indices))
        result[lig_rigid_indices[:n]] = torch.from_numpy(positions[:n]).float()
    except Exception:
        pass

    return result


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

    interfaces = structure["interfaces"]
    if interfaces.dtype.names is None or 'chain_1' not in interfaces.dtype.names:
        interfaces = np.array([], dtype=np.dtype(Interface))
    elif 'chain_1_num_res' not in interfaces.dtype.names:
        new_interfaces = np.zeros(len(interfaces), dtype=np.dtype(Interface))
        for name in interfaces.dtype.names:
            new_interfaces[name] = interfaces[name]
        interfaces = new_interfaces

    struct = Structure(
        atoms=atoms,
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=chains, # chains var accounting for missing cyclic_period
        connections=structure["connections"].astype(Connection),
        interfaces=interfaces,
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

    interaction_residue_mask = structure['interaction_residue_mask'] if 'interaction_residue_mask' in structure else None
    pocket_residue_mask = structure['pocket_residue_mask'] if 'pocket_residue_mask' in structure else None

    return struct, rot_bond_data, interaction_residue_mask, pocket_residue_mask


def _build_priority_token_mask(token_data, struct, interaction_residue_mask):
    """Map per-residue interaction mask to per-token priority mask."""
    if interaction_residue_mask is None:
        return None
    res_lookup = {}  # (asym_id, seqres_j) -> residue_array_idx
    for chain in struct.chains:
        asym_id = int(chain['asym_id'])
        res_start = int(chain['res_idx'])
        for k in range(int(chain['res_num'])):
            j = int(struct.residues[res_start + k]['res_idx'])
            res_lookup[(asym_id, j)] = res_start + k
    mask = np.zeros(len(token_data), dtype=bool)
    for i, token in enumerate(token_data):
        r = res_lookup.get((int(token['asym_id']), int(token['res_idx'])))
        if r is not None and r < len(interaction_residue_mask):
            mask[i] = bool(interaction_residue_mask[r])
    return mask


def _apply_rot_bond_data_to_features(features: dict, rot_bond_data) -> None:
    """Remap rot_bond data from local ligand atom space to global padded rigid space.

    rot_bonds/rot_frag_a/ring_masks are stored in local (0-indexed) ligand atom space.
    After cropping, rigids_is_atom_mask identifies which rigid positions are atomized
    ligand atoms.  We scatter the local arrays to those positions so indices are correct
    in the cropped+padded rigid tensor, regardless of how many protein residues were
    cropped out.

    Works for pure-ligand data (GEOM) where atom_positions = [0..n-1] — the remap is
    then an identity operation.
    """
    L_pad = features['rigids']['rigids_mask'].shape[0]
    atom_positions = torch.where(features['rigids']['rigids_is_atom_mask'])[0]
    n_atom_pos = atom_positions.shape[0]

    if rot_bond_data is not None and rot_bond_data['rot_bonds'].shape[0] > 0:
        rb_local = torch.from_numpy(rot_bond_data['rot_bonds']).long()  # (B, 2)
        n_lig_local = rot_bond_data['rot_frag_a'].shape[1]
        n_map = min(n_lig_local, n_atom_pos)
        target = atom_positions[:n_map]
        features['rot_bonds'] = target[rb_local.clamp(0, max(n_map - 1, 0))]
        fa_local = torch.from_numpy(rot_bond_data['rot_frag_a'])  # (B, n_lig)
        fa_global = torch.zeros(fa_local.shape[0], L_pad, dtype=torch.bool)
        fa_global[:, target] = fa_local[:, :n_map]
        features['rot_frag_a'] = fa_global
    else:
        features['rot_bonds'] = torch.zeros((0, 2), dtype=torch.long)
        features['rot_frag_a'] = torch.zeros((0, L_pad), dtype=torch.bool)

    if rot_bond_data is not None:
        rm_local = torch.from_numpy(rot_bond_data['ring_masks'])  # (R, n_lig)
        R = rm_local.shape[0]
        if R > 0:
            n_lig_local = rm_local.shape[1]
            n_map = min(n_lig_local, n_atom_pos)
            target = atom_positions[:n_map]
            rm_global = torch.zeros(R, L_pad, dtype=torch.bool)
            rm_global[:, target] = rm_local[:, :n_map]
            features['ring_masks'] = rm_global
        else:
            features['ring_masks'] = torch.zeros((0, L_pad), dtype=torch.bool)
        features['sym_groups'] = torch.from_numpy(rot_bond_data['sym_groups']).long()
        features['sym_group_sizes'] = torch.from_numpy(rot_bond_data['sym_group_sizes']).long()
    else:
        features['ring_masks'] = torch.zeros((0, L_pad), dtype=torch.bool)
        features['sym_groups'] = torch.zeros((0, 1), dtype=torch.long)
        features['sym_group_sizes'] = torch.zeros(0, dtype=torch.long)


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
        crop_max_protein_residues=None,
        dataset_probs=None,
        remove_mol_types=None,
        mask_nonstandard=False,
        include_h=False,
        use_identity_rot=True,
        lap_pe_k=0,
        use_pocket_priority=False,
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
                max_neighborhood=crop_max_neighbors,
                max_protein_residues=crop_max_protein_residues,
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
        self.mask_nonstandard = mask_nonstandard
        self.use_pocket_priority = use_pocket_priority

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
        if dataset.interface_crop:
            sample.interface_id = 0
            sample.chain_id = None
        task = task_sampler.sample_task()

        struct, rot_bond_data, interaction_residue_mask, pocket_residue_mask = load_input(sample.record, Path(dataset.data_dir), include_h=self.include_h)

        # Skip structures where a PROTEIN chain contains nucleotide residues
        _nuc_names = {'DA', 'DC', 'DG', 'DT', 'A', 'G', 'C', 'U'}
        _protein_id = const.chain_type_ids["PROTEIN"]
        _has_nuc_in_protein = False
        for _chain in struct.chains:
            if int(_chain['mol_type']) != _protein_id:
                continue
            rs, re = int(_chain['res_idx']), int(_chain['res_idx']) + int(_chain['res_num'])
            if set(struct.residues['name'][rs:re].tolist()) & _nuc_names:
                _has_nuc_in_protein = True
                break
        if _has_nuc_in_protein:
            return self.__getitem__(idx)

        new_struct_mask = struct.mask.copy()
        for i, chain in enumerate(sample.record.chains):
            new_struct_mask[i] = chain.valid

        remove_chain_masks = [struct.chains['mol_type'] == i for i in self.remove_mol_types]
        for remove_mask in remove_chain_masks:
            new_struct_mask[remove_mask] = False

        # if np.random.random() < 0.5:
        #     struct = mirror_structure(struct)

        struct = replace(struct, mask=new_struct_mask)
        # update interface data with record information
        interface_ids = [(c1, c2) for c1, c2 in zip(struct.interfaces['chain_1'], struct.interfaces['chain_2'])]
        interfaces_to_retain = set(
            [
                (interface.chain_1, interface.chain_2)
                for interface in sample.record.interfaces if interface.valid
            ]
        )
        keep_interface = [interface in interfaces_to_retain for interface in interface_ids]
        struct = replace(struct, interfaces=struct.interfaces[keep_interface])


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

        if "seed_interface" in task_data:
            interface_tuples = [tuple(sorted((i.chain_1, i.chain_2))) for i in sample.record.interfaces if i.valid]
            try:
                interface_id = interface_tuples.index(task_data['seed_interface'])
                sample = replace(sample, interface_id=interface_id, chain_id=None)
            except ValueError as e:
                valid_interfaces = struct.interfaces
                valid_interfaces = valid_interfaces[struct.mask[valid_interfaces["chain_1"]]]
                valid_interfaces = valid_interfaces[struct.mask[valid_interfaces["chain_2"]]]
                print(sample.record.id, "interface id context", valid_interfaces, interface_tuples, task_data['seed_interface'], struct.chains[struct.mask])
                raise e

        if self.use_pocket_priority and pocket_residue_mask is not None:
            if interaction_residue_mask is not None:
                interaction_residue_mask = interaction_residue_mask | pocket_residue_mask
            else:
                interaction_residue_mask = pocket_residue_mask

        priority_token_mask = _build_priority_token_mask(token_data, struct, interaction_residue_mask)

        if self.cropper is not None:
            crop_size = self.max_crop_residues - task.max_added_tokens(token_data.shape[0])
            if len(tokenized_data.tokens) > crop_size:
                tokenized_data = self.cropper.crop(
                    tokenized_data,
                    max_tokens=crop_size,
                    random=np.random,
                    chain_id=sample.chain_id,
                    interface_id=sample.interface_id,
                    priority_token_mask=priority_token_mask,
                )

        # if len(tokenized_data.tokens) == 0:
        #     return self.__getitem__(idx)

        features = featurize(
            tokenized_data,
            task_data,
            max_tokens=self.max_crop_residues,
            max_rigids=self.max_crop_rigids
        )
        features['task'] = task
        features['structure'] = struct
        features['record_id'] = sample.record.id
        task_sig_perturb = getattr(task, 'sig_perturb', None)
        features['sig_perturb'] = torch.tensor(
            task_sig_perturb if task_sig_perturb is not None else float('nan'),
            dtype=torch.float32,
        )
        task_trans_prior_std = getattr(task, 'trans_prior_std', None)
        features['trans_prior_std'] = torch.tensor(
            task_trans_prior_std if task_trans_prior_std is not None else float('nan'),
            dtype=torch.float32,
        )

        _apply_rot_bond_data_to_features(features, rot_bond_data)

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

        features['rigids']['etkdg_pos'] = _smiles_etkdg_pos(
            sample.record.smiles, n_rigids, features['rigids']['rigids_noising_mask'].bool()
        )

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
        crop_max_protein_residues=None,
        dataset_probs=None,
        remove_mol_types=None,
        mask_nonstandard=False,
        include_h=False,
        use_identity_rot=True,
        lap_pe_k=0,
        use_pocket_priority=False,
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
                max_neighborhood=crop_max_neighbors,
                max_protein_residues=crop_max_protein_residues,
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
        self.use_pocket_priority = use_pocket_priority

        for dataset in datasets:
            interface_id = 0 if dataset.interface_crop else None
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
                    self.samples.append(Sample(record=record, e_min=entry.e_min, interface_id=interface_id))
                else:
                    self.samples.append(Sample(record=entry, interface_id=interface_id))

    def __getitem__(self, idx):
        dataset_idx = np.random.choice(
            len(self.datasets),
            p=self.dataset_probs,
        )
        dataset = self.datasets[dataset_idx]
        task_sampler = dataset.task_sampler
        sample = self.samples[idx]
        task = task_sampler.sample_task()

        struct, rot_bond_data, interaction_residue_mask, pocket_residue_mask = load_input(sample.record, Path(dataset.data_dir), include_h=self.include_h)

        _nuc_names = {'DA', 'DC', 'DG', 'DT', 'A', 'G', 'C', 'U'}
        _protein_id = const.chain_type_ids["PROTEIN"]
        _has_nuc_in_protein = False
        for _chain in struct.chains:
            if int(_chain['mol_type']) != _protein_id:
                continue
            rs, re = int(_chain['res_idx']), int(_chain['res_idx']) + int(_chain['res_num'])
            if set(struct.residues['name'][rs:re].tolist()) & _nuc_names:
                _has_nuc_in_protein = True
                break
        if _has_nuc_in_protein:
            return self.__getitem__(idx)

        new_struct_mask = struct.mask.copy()
        for i, chain in enumerate(sample.record.chains):
            new_struct_mask[i] = chain.valid

        remove_chain_masks = [struct.chains['mol_type'] == i for i in self.remove_mol_types]
        for remove_mask in remove_chain_masks:
            new_struct_mask[remove_mask] = False

        struct = replace(struct, mask=new_struct_mask)


        interface_ids = [(c1, c2) for c1, c2 in zip(struct.interfaces['chain_1'], struct.interfaces['chain_2'])]
        interfaces_to_retain = set(
            [
                (interface.chain_1, interface.chain_2)
                for interface in sample.record.interfaces if interface.valid
            ]
        )
        keep_interface = [interface in interfaces_to_retain for interface in interface_ids]
        struct = replace(struct, interfaces=struct.interfaces[keep_interface])

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

        if "seed_interface" in task_data:
            interface_tuples = [tuple(sorted((i.chain_1, i.chain_2))) for i in sample.record.interfaces if i.valid]
            try:
                interface_id = interface_tuples.index(task_data['seed_interface'])
                sample = replace(sample, interface_id=interface_id, chain_id=None)
            except ValueError as e:
                valid_interfaces = struct.interfaces
                valid_interfaces = valid_interfaces[struct.mask[valid_interfaces["chain_1"]]]
                valid_interfaces = valid_interfaces[struct.mask[valid_interfaces["chain_2"]]]
                print(sample.record.id, "interface id context", valid_interfaces, interface_tuples, task_data['seed_interface'], struct.chains[struct.mask])
                raise e

        if self.use_pocket_priority and pocket_residue_mask is not None:
            if interaction_residue_mask is not None:
                interaction_residue_mask = interaction_residue_mask | pocket_residue_mask
            else:
                interaction_residue_mask = pocket_residue_mask

        priority_token_mask = _build_priority_token_mask(token_data, struct, interaction_residue_mask)

        if self.cropper is not None:
            crop_size = self.max_crop_residues - task.max_added_tokens(token_data.shape[0])
            if len(tokenized_data.tokens) > crop_size:
                tokenized_data = self.cropper.crop(
                    tokenized_data,
                    max_tokens=crop_size,
                    random=np.random,
                    chain_id=sample.chain_id,
                    interface_id=sample.interface_id,
                    priority_token_mask=priority_token_mask,
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
        features['record_id'] = sample.record.id
        task_sig_perturb = getattr(task, 'sig_perturb', None)
        features['sig_perturb'] = torch.tensor(
            task_sig_perturb if task_sig_perturb is not None else float('nan'),
            dtype=torch.float32,
        )
        task_trans_prior_std = getattr(task, 'trans_prior_std', None)
        features['trans_prior_std'] = torch.tensor(
            task_trans_prior_std if task_trans_prior_std is not None else float('nan'),
            dtype=torch.float32,
        )

        _apply_rot_bond_data_to_features(features, rot_bond_data)

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

        features['rigids']['etkdg_pos'] = _smiles_etkdg_pos(
            sample.record.smiles, n_rigids, features['rigids']['rigids_noising_mask'].bool()
        )

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
                 batch_size,  # this is PER GPU
                 num_workers,
                 val_dataset: Optional[ValidationDataset] = None,
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def build_dataloader(self, x, collate_fn, shuffle=False):
        dataloader = DataLoader(
            x,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
        )
        return dataloader

    def train_dataloader(self):
        return self.build_dataloader(self.train_dataset, collate)

    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.val_dataset, shuffle=True
        )
        return DataLoader(
            self.val_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            collate_fn=collate,
            num_workers=self.num_workers,
        )


class BiomoleculeSamplingDataModule(L.LightningDataModule):
    def __init__(self,
                 tasks_yaml,
                 batch_size,
                 batching_mode="optimal",
                 use_collate_for_pad=False,
                 trans_std: float = 3,
                 include_h: bool = False,
                 batch_same_task_only=False,
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


        if batch_same_task_only:
            self.batch_sampler = TaskBatchSampler(
                self.task_dispatcher,
                self.batch_size
            )
        else:
            self.batch_sampler = None

    def predict_dataloader(self):
        if self.batch_sampler is not None:
            dataloader = DataLoader(
                self.task_dispatcher,
                batch_sampler=self.batch_sampler,
                collate_fn=collate,
                shuffle=False
            )
        else:
            dataloader = DataLoader(
                self.task_dispatcher,
                batch_size=self.batch_size,
                collate_fn=collate,
                shuffle=False
            )
        return dataloader