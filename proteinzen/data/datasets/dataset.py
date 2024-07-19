""" Generic dataset for all input data """
from typing import Dict
import logging

import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import tree


from torch_geometric.data import HeteroData

from proteinzen.data.io import utils as du
from proteinzen.data.openfold.residue_constants import restype_order_with_x

from proteinzen.data.datasets.featurize.molecule import featurize_props
from proteinzen.utils.torsion import get_transformation_mask


class PdbDataset(data.Dataset):
    def __init__(
            self,
            csv_path,
            min_num_res=30,
            max_num_res=500,
            min_percent_ordered=None,
            subset=None,
            split=None
        ):
        self._log = logging.getLogger(__name__)
        self.csv_path = csv_path
        self.min_num_res = min_num_res
        self.max_num_res = max_num_res
        self.subset = subset
        self.split = split
        self.min_percent_ordered = min_percent_ordered
        self._init_metadata()

    def _init_metadata(self):
        """Initialize metadata."""

        # Process CSV with different filtering criterions.
        pdb_csv = pd.read_csv(self.csv_path)
        self.raw_csv = pdb_csv
        pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= self.max_num_res]
        pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= self.min_num_res]
        if self.subset is not None:
            pdb_csv = pdb_csv.iloc[:self.subset]
        pdb_csv = pdb_csv.sort_values('modeled_seq_len', ascending=False)

        if self.split is not None:
            pdb_csv = pdb_csv[pdb_csv.splits == self.split]

        # this is kinda hacky but im doing this for backwards compatibility
        # TODO: get rid of the backward compatible stuff
        if self.min_percent_ordered is None:
            if "frameflow" in self.csv_path:
                self.min_percent_ordered = 0
            elif "framediff" in self.csv_path:
                self.min_percent_ordered = 0.5

        pdb_csv = pdb_csv[pdb_csv["coil_percent"] < (1 - self.min_percent_ordered)]

        self.csv = pdb_csv


    def _process_csv_row(self, processed_file_path):
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_chain_feats(processed_feats)

        # Only take modeled residues.
        modeled_idx = processed_feats['modeled_idx']
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        del processed_feats['modeled_idx']
        processed_feats = tree.map_structure(
            lambda x: x[min_idx:(max_idx+1)], processed_feats)
        res_idx = processed_feats['residue_index']
        processed_feats['residue_index'] = res_idx - np.min(res_idx) + 1
        num_nodes = max_idx - min_idx + 1

        res_data = {
            "atom37": processed_feats["atom_positions"],
            "atom37_mask": processed_feats["atom_mask"],
            "seq": processed_feats["aatype"],
            "seq_mask": (processed_feats["aatype"] != restype_order_with_x.get("X")),
            "chain_idx": processed_feats["chain_index"],
            "res_idx": processed_feats["residue_index"],
            "res_ca": processed_feats["bb_positions"],
            "res_mask": processed_feats['bb_mask'].astype(bool),
            "num_nodes": num_nodes,
            "data_lens": [num_nodes],
            "dssp_helix": processed_feats["ss"] == "H",
            "dssp_sheet": processed_feats["ss"] == "E",
            "dssp_coil": processed_feats["ss"] == "C",
        }
        # print({k: v.shape if hasattr(v, "shape") else processed_feats["ss"] for k, v in res_data.items() })
        # print(processed_file_path)

        graph = HeteroData(
            residue={
                k: torch.as_tensor(t)
                for k, t in res_data.items()
            }
        )
        return graph

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # Sample data example.
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        processed_file_path = csv_row['processed_path']
        chain_feats = self._process_csv_row(processed_file_path)
        chain_feats['name'] = csv_row['pdb_name']
        chain_feats['csv_idx'] = torch.ones(1, dtype=torch.long) * idx
        return chain_feats


class LengthDataset(data.Dataset):
    def __init__(
            self,
            lengths: Dict[int, int],
            batch_size: int = 3000,
            same_length_per_batch=False,
        ):
        self._log = logging.getLogger(__name__)
        self.lengths = lengths
        self.batch_size = batch_size

        self.batches = []
        self.sample_ids = []
        current_batch = []
        current_sample_ids = []
        sample_id = 0
        for sample_l, num_samples in self.lengths.items():
            for _ in range(num_samples):
                if sum(current_batch) >= self.batch_size:
                    self.batches.append(current_batch)
                    self.sample_ids.append(current_sample_ids)
                    current_batch = []
                    current_sample_ids = []
                current_batch.append(sample_l)
                current_sample_ids.append(sample_id)
                sample_id += 1
            if same_length_per_batch:
                self.batches.append(current_batch)
                self.sample_ids.append(current_sample_ids)
                current_batch = []
                current_sample_ids = []

        if len(current_batch) > 0:
            self.batches.append(current_batch)
        if len(current_sample_ids) > 0:
            self.sample_ids.append(current_sample_ids)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return {
            "num_res": self.batches[idx],
            "sample_id": self.sample_ids[idx]
        }


class GEOMDataset(data.Dataset):
    def __init__(
            self,
            csv_path,
            subset=None,
            split=None
        ):
        self._log = logging.getLogger(__name__)
        self.csv_path = csv_path
        self.subset = subset
        self.split = split
        self._init_metadata()

    def _init_metadata(self):
        """Initialize metadata."""

        # Process CSV with different filtering criterions.
        pdb_csv = pd.read_csv(self.csv_path)
        self.raw_csv = pdb_csv

        pdb_csv = pdb_csv[~pdb_csv.smiles.str.contains('.', regex=False)]
        if self.subset is not None:
            if self.subset > 0:
                pdb_csv = pdb_csv.iloc[:self.subset]
            elif self.subset < 0:
                pdb_csv = pdb_csv.iloc[self.subset:]

        if self.split is not None:
            pdb_csv = pdb_csv[pdb_csv.splits == self.split]

        self.csv = pdb_csv


    def _process_csv_row(self, processed_file_path):
        data_dict = du.read_pkl(processed_file_path)
        weights = data_dict['boltzmann_weights']
        rng = np.random.default_rng()
        conformer_id = rng.choice(
            a=np.arange(len(weights)),
            p=np.array(weights) / np.sum(weights)
        )
        conformer_id = 0
        unprocessed_feats = data_dict['conformer_property_dicts'][int(conformer_id)]
        graph = featurize_props(unprocessed_feats)
        graph['rd_mol'] = data_dict['rd_mols']
        graph['path'] = processed_file_path
        mask_edges, mask_rotate = get_transformation_mask(graph)
        graph['ligand', 'ligand']['rotatable_bonds'] = torch.as_tensor(mask_edges)
        graph['ligand', 'ligand']['mask_rotate'] = torch.as_tensor(mask_rotate)

        return graph

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # Sample data example.
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        processed_file_path = csv_row['processed_path']
        mol_feats = self._process_csv_row(processed_file_path)
        mol_feats['csv_idx'] = torch.ones(1, dtype=torch.long) * idx
        return mol_feats