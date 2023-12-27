""" Generic dataset for all input data """
import logging

import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import tree

from torch_geometric.data import HeteroData

from ligbinddiff.utils.atom_reps import atom91_atom_masks, letter_to_num
from ligbinddiff.utils.zernike import ZernikeTransform
from ligbinddiff.data.io import utils as du

from ligbinddiff.data.datasets.featurize.sidechain import featurize_atomic, featurize_density, featurize_cross_scale_atomic
from ligbinddiff.data.openfold.residue_constants import restype_order_with_x


class ProteinGraphDataset(data.Dataset):
    '''
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the
    manuscript.

    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6]
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are excluded from message passing

    Portions from https://github.com/jingraham/neurips19-graph-protein-design.

    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param num_positional_embeddings: number of positional embeddings
    :param top_k: number of edges to draw per node (as destination node)
    :param device: if "cuda", will do preprocessing on the GPU
    '''

    FEATURE_MODES = ['sidechain.atomic', 'sidechain.density', 'sidechain.cross_scale']

    def __init__(self, data_list,
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, device="cpu",
                 feature_mode="sidechain.atomic",
                 density_nmax=5,
                 density_rmax=8,
                 channel_atoms=False,
                 bb_density=True,
                 diffuser=None):

        super(ProteinGraphDataset, self).__init__()
        assert feature_mode in self.__class__.FEATURE_MODES, f"feature_mode must be in {self.__class__.FEATURE_MODES}"
        self.feature_mode = feature_mode

        self.data_list = data_list
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(e['seq']) for e in data_list]

        self.letter_to_num = letter_to_num
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}

        if 'density' in self.feature_mode:
            self.density_nmax = density_nmax
            self.density_rmax = density_rmax
            self.channel_atoms = channel_atoms
            self.bb_density = bb_density
            self.zernike_transform = ZernikeTransform(density_nmax, density_rmax)
            self.channel_atoms = channel_atoms
            if channel_atoms:
                self.channel_values = np.array([
                    atom91_atom_masks[atom] for atom in ['C', 'N', 'O', 'S']
                ])
                self.num_channels = len(atom91_atom_masks)
            else:
                self.channel_values = None
                self.num_channels = 1
        elif 'atomic' in self.feature_mode:
            self.diffuser = diffuser

    def __len__(self): return len(self.data_list)

    def __getitem__(self, i):
        if self.feature_mode == 'sidechain.atomic':
            return featurize_atomic(self.data_list[i],
                                              letter_to_num=self.letter_to_num,
                                              num_rbf=self.num_rbf,
                                              top_k=self.top_k,
                                              diffuser=self.diffuser,
                                              device=self.device)
        elif self.feature_mode == 'sidechain.density':
            return featurize_density(self.data_list[i],
                                               zernike_transform=self.zernike_transform,
                                               channel_atoms=self.channel_atoms,
                                               channel_values=self.channel_values,
                                               letter_to_num=self.letter_to_num,
                                               num_rbf=self.num_rbf,
                                               top_k=self.top_k,
                                               device=self.device)
        elif self.feature_mode == 'sidechain.cross_scale':
            return featurize_cross_scale_atomic(
                self.data_list[i],
                letter_to_num=self.letter_to_num,
                num_rbf=self.num_rbf,
                top_k=self.top_k,
                device=self.device)


class PdbDataset(data.Dataset):
    def __init__(
            self,
            csv_path,
            min_num_res=30,
            max_num_res=500,
            subset=None,
            split='train'
        ):
        self._log = logging.getLogger(__name__)
        self.csv_path = csv_path
        self.min_num_res = min_num_res
        self.max_num_res = max_num_res
        self.subset = subset
        self.split = split
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

        pdb_csv = pdb_csv[pdb_csv.splits == self.split]

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
        }

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
        chain_feats['csv_idx'] = torch.ones(1, dtype=torch.long) * idx
        return chain_feats
