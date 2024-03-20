""" Molecule denoiser """

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_cluster import knn, knn_graph, radius_graph
from torch_scatter import scatter_mean

from e3nn import o3
from e3nn.nn import BatchNorm, FullyConnectedNet


from ligbinddiff.model.modules.common import GaussianRandomFourierBasis
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf
from ligbinddiff.model.modules.openfold.frames import Linear


class EdgeUpdate(nn.Module):
    def __init__(self,
                 feat_irreps,
                 h_edge):
        super().__init__()
        self.feat_irreps = feat_irreps
        self.out_irreps = o3.Irreps([(h_edge, (0, 1))])

        self.lin = o3.Linear(
            feat_irreps,
            self.out_irreps
        )
        self.fc = nn.Sequential(
            nn.Linear(h_edge * 3, h_edge),
            nn.ReLU(),
            nn.Linear(h_edge, h_edge),
            nn.ReLU(),
            Linear(h_edge, h_edge, init='final'),
        )
        self.ln = nn.LayerNorm(h_edge)

    def forward(self,
                atom_features: torch.Tensor,
                edge_features: torch.Tensor,
                edge_index: torch.Tensor):

        atom_scalars = self.lin(atom_features)

        edge_dst, edge_src = edge_index
        edge_in = torch.cat([
            atom_scalars[edge_dst],
            atom_scalars[edge_src],
            edge_features
        ], dim=-1)
        edge_features = self.ln(edge_features + self.fc(edge_in))
        return edge_features


class TensorConvLayer(nn.Module):
    def __init__(self,
                 feat_irreps,
                 sh_irreps,
                 h_edge,
                 update_edge=False):
        super().__init__()
        self.feat_irreps = feat_irreps
        self.sh_irreps = sh_irreps

        self.tp = o3.FullyConnectedTensorProduct(
            feat_irreps,
            sh_irreps,
            feat_irreps,
            shared_weights=False
        )
        self.fc = nn.Sequential(
            nn.Linear(h_edge, h_edge),
            nn.ReLU(),
            Linear(h_edge, self.tp.weight_numel)#, init='final')
        )
        self.norm = BatchNorm(feat_irreps)

        self.update_edge = update_edge
        if update_edge:
            self.edge_update = EdgeUpdate(
                feat_irreps=feat_irreps,
                h_edge=h_edge
            )

    def forward(self,
                atom_features: torch.Tensor,
                edge_features: torch.Tensor,
                edge_sh,
                edge_index: torch.Tensor):

        edge_dst, edge_src = edge_index
        # print(atom_features.shape, edge_features.shape, edge_index.shape)
        # print(self.fc(edge_features).shape, self.tp.weight_numel)
        tp = self.tp(
            atom_features[edge_dst],
            edge_sh,
            self.fc(edge_features)
        )

        out = scatter_mean(tp, edge_src, dim=0, dim_size=atom_features.shape[0])
        padded = F.pad(atom_features, (0, out.shape[-1] - atom_features.shape[-1]))
        out = out + padded
        out = self.norm(out)

        if self.update_edge:
            edge_features = self.edge_update(
                out,
                edge_features,
                edge_index
            )

        return out, edge_features

class RefineLayer(nn.Module):
    def __init__(self,
                 feat_irreps,
                 sh_irreps,
                 h_edge,
                 n_layers=3):
        super().__init__()
        self.feat_irreps = feat_irreps
        self.sh_irreps = sh_irreps

        self.spatial_tpc = nn.ModuleList(
            [
                TensorConvLayer(
                    feat_irreps, sh_irreps, h_edge
                )
            for _ in range(n_layers)
            ]
        )
        self.bond_tpc = nn.ModuleList(
            [
                TensorConvLayer(
                    feat_irreps, sh_irreps, h_edge, update_edge=True
                )
            for _ in range(n_layers)
            ]
        )

    def forward(self,
                atom_features,
                bond_features,
                bond_sh,
                bond_edge_index,
                spatial_features,
                spatial_sh,
                spatial_edge_index):
        for bond_tpc, spatial_tpc in zip(self.bond_tpc, self.spatial_tpc):
            atom_features, bond_features = bond_tpc(atom_features, bond_features, bond_sh, bond_edge_index)
            atom_features, _ = spatial_tpc(atom_features, spatial_features, spatial_sh, spatial_edge_index)

        return atom_features, bond_features



class MoleculeDenoiser(nn.Module):
    def __init__(self,
                 c_s=32,
                 c_v=8,
                 feat_lmax=1,
                 sh_lmax=1,
                 h_time=64,
                 edge_hidden=128,
                 n_rbf=64,
                 n_atom_in=43,
                 n_bond_in=7,
                 n_layers=4,
                 k=10,
                 self_conditioning=True):
        super().__init__()
        self.feat_irreps = o3.Irreps([(c_s if l == 0 else c_v, (l, 1)) for l in range(feat_lmax + 1)])
        self.sh_irreps = o3.Irreps([(1, (l, 1)) for l in range(sh_lmax + 1)])
        self.self_conditioning = self_conditioning
        self.n_rbf = n_rbf
        self.k = k

        self.time_rbf = GaussianRandomFourierBasis(n_basis=h_time//2)

        self.embed_atoms = nn.Sequential(
            nn.Linear(n_atom_in + h_time, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.LayerNorm(c_s),
        )
        self.embed_bonds = nn.Sequential(
            nn.Linear(n_bond_in + n_rbf, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.LayerNorm(edge_hidden),
        )
        self.refine_layers = nn.ModuleList(
            [
                RefineLayer(
                    feat_irreps=self.feat_irreps,
                    sh_irreps=self.sh_irreps,
                    h_edge=edge_hidden,
                )
                for _ in range(n_layers)
            ]
        )
        self.bond_linears = nn.ModuleList(
            [
                nn.Linear(n_rbf + n_rbf * self_conditioning + edge_hidden, edge_hidden)
                for _ in range(n_layers)
            ]
        )
        self.embed_spatial_edge = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_rbf + n_rbf * self_conditioning, edge_hidden),
                    nn.ReLU(),
                    nn.Linear(edge_hidden, edge_hidden),
                    nn.ReLU(),
                    nn.Linear(edge_hidden, edge_hidden),
                    nn.LayerNorm(edge_hidden),
                )
                for _ in range(n_layers)
            ]
        )
        self.pos_update = nn.ModuleList(
            [
                o3.Linear(self.feat_irreps, "1x1o")
                for _ in range(n_layers)
            ]
        )
        self.pos_gate = nn.ModuleList(
            [
                nn.Sequential(
                    o3.Linear(self.feat_irreps, "1x0e"),
                    nn.Sigmoid()
                )
                for _ in range(n_layers)
            ]
        )

        with torch.no_grad():
            for o3_lin in self.pos_update:
                o3_lin.weight.fill_(0.)
        # for o3_lin in self.pos_update:
        #     o3_lin.weight.requires_grad = True

        self.k = k


    @torch.no_grad()
    def _prep_features(self, graph, eps=1e-8):
        atom_data = graph['ligand']
        bond_data = graph['ligand', 'bonds', 'ligand']

        atom_pos = atom_data['noised_atom_pos']

        atom_features = [
            atom_data[key].view(atom_data.num_nodes, -1)
            for key in (
                "atom_period",                 # N, categorical
                "atom_row",                       # N, categorical
                "atom_chirality",           # N, ordinal
                "atom_hybridization",   # N, categorical
                "atom_implicit_hs",       # N, ordinal
                "atom_formal_charge",   # N, ordinal
                "atom_aromatic",             # N, bool
                "atom_ring_props",         # N x 7, 1-6 bool, 7 ordinal
                "atom_degree",                 # N, ordinal
        )]

        atom_features = torch.cat(atom_features, dim=-1).float()

        bond_edge_index = bond_data.edge_index
        bond_features = [
            bond_data[key].view(bond_edge_index.shape[-1], -1) for key in (
                "bond_order",                   # E, ordinal
                "bond_aromatic",             # E, bool
                "bond_type",                     # E, categorical
                "bond_conjugated",         # E, bool
        )]
        bond_len = torch.linalg.vector_norm(
            atom_pos[bond_edge_index[0]] - atom_pos[bond_edge_index[1]] + eps,
            dim=-1
        )
        bond_len_rbf = _rbf(bond_len, D_min=0., D_max=10., D_count=self.n_rbf, device=bond_len.device)
        bond_features.append(bond_len_rbf)

        bond_features = torch.cat(bond_features, dim=-1).float()


        return atom_pos.float(), atom_features.float(), bond_features.float(), bond_edge_index

    def _gen_spatial_edges(self, atom_pos, batch, eps=1e-8, self_condition=None):
        # spatial_edge_index = knn_graph(atom_pos, self.k, batch=batch)
        spatial_edge_index = radius_graph(atom_pos, r=6.0, batch=batch, max_num_neighbors=1000)
        # print(atom_pos)
        # print(self.k, atom_pos.shape, spatial_edge_index.shape)
        edge_vecs = atom_pos[spatial_edge_index[0]] - atom_pos[spatial_edge_index[1]]

        spatial_edge_len = torch.linalg.vector_norm(edge_vecs + eps, dim=-1)
        spatial_edge_features = _rbf(spatial_edge_len, D_min=0., D_max=10., D_count=self.n_rbf, device=spatial_edge_index.device)

        if self.self_conditioning and self_condition is not None:
            sc_atom_pos = self_condition['pred_atom_pos']
            sc_edge_vecs = sc_atom_pos[spatial_edge_index[0]] - sc_atom_pos[spatial_edge_index[1]]
            sc_len = torch.linalg.vector_norm(
                sc_edge_vecs + eps,
                dim=-1
            )
            sc_len_rbf = _rbf(sc_len, D_min=0., D_max=10., D_count=self.n_rbf, device=sc_atom_pos.device)
            spatial_edge_features = torch.cat([spatial_edge_features, sc_len_rbf], dim=-1)
        elif self.self_conditioning:
            spatial_edge_features = F.pad(spatial_edge_features, (0, self.n_rbf))

        return spatial_edge_features, spatial_edge_index

    def forward(self, data, self_condition=None, eps=1e-6):
        atom_pos, atom_features, bond_features, bond_edge_index = self._prep_features(data)
        t = data['t']
        time_embed = self.time_rbf(t[..., None])
        nodewise_t_embed = time_embed[data['ligand'].batch]

        atom_features = self.embed_atoms(
            torch.cat([atom_features, nodewise_t_embed], dim=-1)
        )
        atom_features = F.pad(atom_features, (0, self.feat_irreps.dim - atom_features.shape[-1]))
        bond_features = self.embed_bonds(bond_features)

        atom_traj = []

        for refine_layer, bond_lin, embed_spatial_edge, pos_update, pos_gate in zip(
            self.refine_layers, self.bond_linears, self.embed_spatial_edge, self.pos_update, self.pos_gate
        ):
            bond_edge_vecs = atom_pos[bond_edge_index[0]] - atom_pos[bond_edge_index[1]]
            bond_sh = o3.spherical_harmonics(self.sh_irreps, bond_edge_vecs, normalize=True, normalization='component')

            bond_len = torch.linalg.vector_norm(
                bond_edge_vecs + eps,
                dim=-1
            )
            bond_len_rbf = _rbf(bond_len, D_min=0., D_max=10., D_count=self.n_rbf, device=bond_len.device)
            bond_features = torch.cat([bond_features, bond_len_rbf], dim=-1)
            if self.self_conditioning and self_condition is not None:
                sc_atom_pos = self_condition['pred_atom_pos']
                sc_bond_edge_vecs = sc_atom_pos[bond_edge_index[0]] - sc_atom_pos[bond_edge_index[1]]
                sc_bond_len = torch.linalg.vector_norm(
                    sc_bond_edge_vecs + eps,
                    dim=-1
                )
                sc_bond_len_rbf = _rbf(sc_bond_len, D_min=0., D_max=10., D_count=self.n_rbf, device=bond_len.device)
                bond_features = torch.cat([bond_features, sc_bond_len_rbf], dim=-1)
            elif self.self_conditioning:
                bond_features = F.pad(bond_features, (0, self.n_rbf))

            bond_features = bond_lin(bond_features)

            spatial_edges, spatial_edge_index = self._gen_spatial_edges(atom_pos, data['ligand'].batch, self_condition=self_condition)
            spatial_edge_vecs = atom_pos[spatial_edge_index[0]] - atom_pos[spatial_edge_index[1]]
            spatial_sh = o3.spherical_harmonics(self.sh_irreps, spatial_edge_vecs, normalize=True, normalization='component')
            spatial_edge_features = embed_spatial_edge(spatial_edges)

            atom_features, bond_features = refine_layer(
                atom_features,
                bond_features,
                bond_sh,
                bond_edge_index,
                spatial_edge_features,
                spatial_sh,
                spatial_edge_index)

            update = pos_update(atom_features)
            update_gate = pos_gate(atom_features)
            atom_pos = atom_pos + update * update_gate
            atom_traj.append(atom_pos)

        return {
            "pred_atom_pos": atom_pos,
            "atom_traj": atom_traj
        }
