""" Generic dataset for all input data """

import numpy as np
import torch.utils.data as data


from ligbinddiff.utils.atom_reps import atom14_to_atom91, atom91_atom_masks, letter_to_num
from ligbinddiff.utils.zernike import ZernikeTransform

from .featurize import sidechain


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

    FEATURE_MODES = ['sidechain.atomic', 'sidechain.density']

    def __init__(self, data_list,
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, device="cpu",
                 feature_mode="sidechain.atomic",
                 density_nmax=5,
                 density_rmax=8,
                 channel_atoms=False,
                 bb_density=True):

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

    def __len__(self): return len(self.data_list)

    def __getitem__(self, i):
        if self.feature_mode == 'sidechain.atomic':
            return sidechain.featurize_atomic(self.data_list[i],
                                              letter_to_num=self.letter_to_num,
                                              num_rbf=self.num_rbf,
                                              top_k=self.top_k,
                                              device=self.device)
        elif self.feature_mode == 'sidechain.density':
            return sidechain.featurize_density(self.data_list[i],
                                               zernike_transform=self.zernike_transform,
                                               channel_atoms=self.channel_atoms,
                                               channel_values=self.channel_values,
                                               letter_to_num=self.letter_to_num,
                                               num_rbf=self.num_rbf,
                                               top_k=self.top_k,
                                               device=self.device)
