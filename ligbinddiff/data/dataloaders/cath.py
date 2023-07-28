""" Dataloaders for CATH fold split data

adapted from https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py"""
import json
import tqdm, random

import dgl
import numpy as np
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster

from ligbinddiff.utils.atom_reps import atom37_atom_label, atom37_to_atom14, atom14_to_atom91, letter_to_num, atom91_atom_masks
from ligbinddiff.utils.fiber import nl_to_fiber
from ligbinddiff.utils.zernike import ZernikeTransform

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


class CATHDataset:
    '''
    Loader and container class for the CATH 4.2 dataset downloaded
    from http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/.

    Has attributes `self.train`, `self.val`, `self.test`, each of which are
    JSON/dictionary-type datasets as described in README.md.

    :param path: path to chain_set.jsonl
    :param splits_path: path to chain_set_splits.json or equivalent.
    '''
    def __init__(self, path, splits_path):
        with open(splits_path) as f:
            dataset_splits = json.load(f)
        train_list, val_list, test_list = dataset_splits['train'], \
            dataset_splits['validation'], dataset_splits['test']

        self.train, self.val, self.test = [], [], []

        with open(path) as f:
            lines = f.readlines()

        for line in tqdm.tqdm(lines):
            entry = json.loads(line)
            name = entry['name']

            if name in ['3k7a.M']:  # this is backbone-only somehow...
                continue

            coords = entry['coords']

            atom37 = list(zip(
                *[coords[atom] for atom in atom37_atom_label]
            ))
            atom14, atom14_mask = atom37_to_atom14(entry['seq'], np.array(atom37))
            entry['coords'] = atom14
            entry['coords_mask'] = atom14_mask

            if name in train_list:
                self.train.append(entry)
            elif name in val_list:
                self.val.append(entry)
            elif name in test_list:
                self.test.append(entry)


class BatchSampler(data.BatchSampler):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design.

    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.

    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, sampler, batch_size=3000, drop_last=False):
        super().__init__(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        shuffle = True

        max_nodes = batch_size
        node_counts = sampler.data_source.node_counts

        self.node_counts = node_counts
        self.idx = [i for i in range(len(node_counts))
                        if node_counts[i] <= max_nodes]
        self.shuffle = shuffle
        self.max_nodes = max_nodes

        self._form_batches()

    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)

    def __len__(self):
        if not self.batches: self._form_batches()
        return len(self.batches)

    def __iter__(self):
        self._form_batches()
        #if not self.batches: self._form_batches()
        for batch in self.batches: yield batch



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
    def __init__(self, data_list,
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, device="cpu",
                 density_nmax=5,
                 density_rmax=8,
                 channel_atoms=False,
                 bb_density=True):

        super(ProteinGraphDataset, self).__init__()

        self.data_list = data_list
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(e['seq']) for e in data_list]

        self.density_nmax = density_nmax
        self.density_rmax = density_rmax
        self.channel_atoms = channel_atoms
        self.bb_density = bb_density
        self.zernike_transform = ZernikeTransform(density_nmax, density_rmax)

        self.letter_to_num = letter_to_num
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}

        if channel_atoms:
            self.channel_values = np.array([
                atom91_atom_masks[atom] for atom in ['C', 'N', 'O', 'S']
            ])
            self.num_channels = len(atom91_atom_masks)
        else:
            self.channel_values = None
            self.num_channels = 1


    def __len__(self): return len(self.data_list)

    def __getitem__(self, i): return self._featurize_as_graph(self.data_list[i])

    def _featurize_as_graph(self, protein):
        name = protein['name']
        with torch.no_grad():
            atom14 = protein['coords']
            atom14_mask = protein['coords_mask']
            atom91, atom91_mask = atom14_to_atom91(protein['seq'], atom14)
            atom14 = torch.as_tensor(atom14,
                                     device=self.device, dtype=torch.float32)
            atom14_mask = torch.as_tensor(atom14_mask,
                                          device=self.device, dtype=torch.bool)
            atom91 = torch.as_tensor(atom91,
                                     device=self.device, dtype=torch.float32)
            atom91_mask = torch.as_tensor(atom91_mask,
                                          device=self.device, dtype=torch.bool)
            seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']],
                                  device=self.device, dtype=torch.long)
            # backbone coords
            coords = atom14[:, :4]
            X_cb = self._ideal_virtual_Cb(coords)

            x_mask = torch.isfinite(coords.sum(dim=(1,2)))
            x_mask = ~x_mask
            coords[x_mask] = np.inf
            atom14_mask[x_mask] = True
            atom91_mask[x_mask] = True

            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)

            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                    (node_s, node_v, edge_s, edge_v))

            graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=len(protein['seq']))
            graph.name = name
            graph.ndata['x'] = X_ca
            graph.ndata['x_mask'] = x_mask
            graph.ndata['x_cb'] = X_cb
            graph.ndata['seq'] = seq
            graph.ndata['backbone_s'] = node_s
            graph.ndata['backbone_v'] = node_v
            graph.ndata['atom14'] = atom14
            graph.ndata['atom91'] = atom91
            graph.ndata['atom91_centered'] = atom91 - X_ca.unsqueeze(-2)
            graph.ndata['atom91_mask'] = atom91_mask
            graph.edata['edge_s'] = edge_s
            graph.edata['edge_v'] = edge_v
            graph.edata['rel_pos'] = torch.nan_to_num(E_vectors)

            # densities should be computed centered at C_beta
            if self.channel_atoms:
                # we could do this using atom14 but it's not clear to me that's any faster
                # since we can't take advantage of double masking to avoid taking sequence as input
                atom_mask = atom91_mask.any(dim=-1)
                channel_values = torch.as_tensor(self.channel_values, device=self.device, dtype=torch.float)
                if not self.bb_density:
                    atom91 = atom91[..., 4:, :]
                    atom_mask = atom_mask[..., 4:]
                    channel_values = channel_values[..., 4:]
                Z = self.zernike_transform.forward_transform(atom91, X_cb, atom_mask, point_value=channel_values)
            else:
                atom_mask = atom14_mask.any(dim=-1)
                Z = self.zernike_transform.forward_transform(atom14, X_cb, atom_mask)
            fiber_dict = nl_to_fiber(Z)

            for l, coeffs in fiber_dict.items():
                graph.ndata[l] = coeffs

        return graph

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design

        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features


    def _positional_embeddings(self, edge_index,
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec

    def _ideal_virtual_Cb(self, X_bb):
        # from ProteinMPNN paper computation of Cb
        N, Ca, C = [X_bb[..., i, :] for i in range(3)]
        b = Ca - N
        c = C- Ca
        a = torch.cross(b, c)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
        return Cb
