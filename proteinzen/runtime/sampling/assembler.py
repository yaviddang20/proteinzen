import copy
import string
import functools as fn
from dataclasses import astuple, fields
from typing import List

from rdkit import Chem

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure as BPStructure

from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import tree
from torch_geometric.data import HeteroData
import numpy as np

from proteinzen.boltz.data.types import Structure, Atom, Residue, Chain, SamplingResidue
from proteinzen.boltz.data import const

from proteinzen.openfold.data import residue_constants
from proteinzen.data.constants import coarse_grain as cg
from proteinzen.openfold.data import data_transforms
from proteinzen.openfold.utils import rigid_utils as ru
from proteinzen.data.featurize.tokenize import Tokenized, convert_atom_str_to_tuple
from proteinzen.data.featurize.sampling import (
    sample_noise_from_struct_template,
)
from proteinzen.data.featurize.assembler import featurize as featurize_inference

from proteinzen.runtime.sampling.protein import PartiallyNoisedChain

from .biomolecule import Biomolecule


class ChainAssembly:
    task_name = "universal_sampling_task"

    def __init__(
        self,
        chains: List[Biomolecule],
        num_samples: int,
        center_noise_on_hotspots: bool=False,
        center_on_hotspots: bool=False,
        estimate_binder_com: bool=False,
        task_name=None
    ):
        super().__init__()
        self.chains = chains
        self.num_samples = num_samples
        self.center_noise_on_hotspots = center_noise_on_hotspots
        self.center_on_hotspots = center_on_hotspots
        self.estimate_binder_com = estimate_binder_com

        if task_name is not None:
            self.task_name = task_name

        partial_noise_ts = []
        for chain in self.chains:
            if isinstance(chain, PartiallyNoisedChain):
                partial_noise_ts.append(chain.t_noise)
        if len(partial_noise_ts) > 0:
            partial_noise_ts = np.array(partial_noise_ts)
            assert (partial_noise_ts == partial_noise_ts[0]).all(), f"we have different levels of partial noising in the same sample! {partial_noise_ts}"
            self.t_noise = float(partial_noise_ts[0])
        else:
            self.t_noise = 0


    def generate_sample(self):
        struct_keys = [f.name for f in fields(Structure)]
        struct_dict = {
            key: []
            for key in struct_keys
        }
        chain_offset = 0
        chain_mapping = {}
        res_index_offset = 0
        res_entity_id_offset = 0
        atom_index_offset = 0
        task_masks = {
            "t": self.t_noise,
            "atom_noising_mask": [],
            "res_type_noising_mask": [],
            "res_is_unindexed_mask": [],
            "res_hotspot_type": [],
            "residue_entity_ids": []
        }

        for entity_id, chain in enumerate(self.chains):
            chain_data = chain.sample()
            struct = chain_data['struct']
            atom_noising_mask = chain_data['atom_noising_mask']
            res_type_noising_mask = chain_data['res_type_noising_mask']
            res_is_unindexed_mask = chain_data['res_is_unindexed_mask']
            res_entity_ids = chain_data['residue_entity_ids']
            res_is_copy_mask = struct.residues['is_copy']
            update_res_index = ~res_is_copy_mask | res_is_unindexed_mask

            chain_id = struct.chains['name'].item()
            if chain_id not in chain_mapping:
                chain_mapping[chain_id] = chain_offset
                chain_offset += 1
            chain_idx = chain_mapping[chain_id]

            # offset indicies in current chain
            struct.chains['asym_id'] = chain_idx
            struct.chains['entity_id'] = entity_id
            # TODO: add an option to handle symmetric chains
            struct.chains['sym_id'] = chain_idx
            struct.chains['res_idx'] += res_index_offset
            struct.chains['atom_idx'] += atom_index_offset
            # struct.residues['res_idx'][update_res_index] += res_index_offset
            struct.residues['atom_idx'] += atom_index_offset
            struct.bonds['atom_1'] += atom_index_offset
            struct.bonds['atom_2'] += atom_index_offset

            # update index offset
            res_index_offset += struct.chains['res_num'].item()
            atom_index_offset += struct.chains['atom_num'].item()

            for key in struct_keys:
                struct_dict[key].append(getattr(struct, key))

            task_masks['atom_noising_mask'].append(atom_noising_mask)
            task_masks['res_type_noising_mask'].append(res_type_noising_mask)
            task_masks['res_is_unindexed_mask'].append(res_is_unindexed_mask)
            task_masks['residue_entity_ids'].append(res_entity_ids + res_entity_id_offset)
            res_entity_id_offset += max(res_entity_ids) + 1

            if "res_hotspot_type" in chain_data:
                res_hotspot_type = chain_data['res_hotspot_type']
                task_masks['res_hotspot_type'].append(res_hotspot_type)
            else:
                task_masks['res_hotspot_type'].append(np.zeros(res_is_unindexed_mask.shape))

        task_masks['atom_noising_mask'] = np.concatenate(task_masks['atom_noising_mask'], axis=-1)
        task_masks['res_type_noising_mask'] = np.concatenate(task_masks['res_type_noising_mask'], axis=-1)
        task_masks['residue_is_unindexed_mask'] = np.concatenate(task_masks['res_is_unindexed_mask'], axis=-1)
        task_masks['res_hotspot_type'] = np.concatenate(task_masks['res_hotspot_type'], axis=-1)
        task_masks['residue_entity_ids'] = np.concatenate(task_masks['residue_entity_ids'], axis=-1)

        for key in struct_keys:
            struct_dict[key] = np.concatenate(struct_dict[key], axis=-1)

        struct = Structure(**struct_dict)
        return struct, task_masks

    def sample_data(self):
        for _ in range(self.num_samples):
            struct, task_masks = self.generate_sample()
            token_data, rigids_data, token_bonds, fixed_rigids_com, hotspot_rigids_com = sample_noise_from_struct_template(
                struct,
                task_masks,
            )

            if self.center_on_hotspots:
                struct.atoms['coords'] -= hotspot_rigids_com[None]
            else:
                struct.atoms['coords'] -= fixed_rigids_com[None]
            tokenized = Tokenized(
                token_data,
                rigids_data,
                token_bonds,
                struct
            )
            yield featurize_inference(
                tokenized,
                task_masks,
                task_name=self.task_name
            )
