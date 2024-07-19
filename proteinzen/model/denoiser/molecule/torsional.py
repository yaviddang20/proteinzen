import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_cluster import knn, knn_graph, radius, radius_graph
from torch_scatter import scatter_mean
import torch_geometric.utils as pygu

from e3nn import o3
from e3nn.nn import NormActivation, Gate, BatchNorm, S2Activation

from proteinzen.model.modules.common import GaussianRandomFourierBasis
from proteinzen.model.modules.layers.edge.tfn import FasterTensorProduct
from proteinzen.data.datasets.featurize.common import _edge_positional_embeddings, _rbf
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.model.modules.openfold.layers import Linear
from proteinzen.utils.torsion import modify_conformer_torsion_angles_batch_cached

def get_irrep_seq(ns, nv, reduce_pseudoscalars):
    irrep_seq = [
        f'{ns}x0e',
        f'{ns}x0e + {nv}x1o',
        f'{ns}x0e + {nv}x1o + {nv}x1e',
        f'{ns}x0e + {nv}x1o + {nv}x1e + {nv if reduce_pseudoscalars else ns}x0o'
    ]
    return irrep_seq


class SeparableS2Activation(nn.Module):
    def __init__(self,
                 irreps,
                 grid_res=16):
        super().__init__()
        self.num_scalars = irreps.count("0e")
        self.scalar_act = torch.nn.SiLU()
        self.s2_act     = S2Activation(irreps, act=self.scalar_act, res=grid_res)

    def forward(self, input_tensors):
        input_scalars = input_tensors[..., :self.num_scalars]
        output_scalars = self.scalar_act(input_scalars)
        output_tensors = self.s2_act(input_tensors)
        outputs = torch.cat(
            (output_scalars, output_tensors[..., self.num_scalars:]),
            dim=-1
        )
        return outputs


class NodeTransition(nn.Module):
    def __init__(self,
                 feat_irreps):
        super().__init__()
        self.feat_irreps = feat_irreps

        self.fc = nn.Sequential(
            o3.Linear(feat_irreps, feat_irreps),
            NormActivation(feat_irreps, torch.relu),
            #SeparableS2Activation(feat_irreps),
            o3.Linear(feat_irreps, feat_irreps),
            NormActivation(feat_irreps, torch.relu),
            #SeparableS2Activation(feat_irreps),
            o3.Linear(feat_irreps, feat_irreps)
        )
        self.bn = BatchNorm(feat_irreps)

    def forward(self, node_features):
        return self.bn(node_features + self.fc(node_features))


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
            nn.Linear(h_edge, h_edge),
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
                 in_irreps,
                 sh_irreps,
                 out_irreps,
                 h_edge,
                 update_edge=False,
                 residual=True,
                 slower=False):
        super().__init__()
        self.sh_irreps = sh_irreps

        if slower:
            self.tp = o3.FullyConnectedTensorProduct(
                in_irreps,
                sh_irreps,
                out_irreps,
                shared_weights=False
            )
        else:
            self.tp = FasterTensorProduct(
                in_irreps,
                sh_irreps,
                out_irreps,
            )

        self.fc = nn.Sequential(
            nn.Linear(h_edge, h_edge),
            nn.ReLU(),
            nn.Linear(h_edge, self.tp.weight_numel)
        )
        self.norm = BatchNorm(out_irreps)

        self.update_edge = update_edge
        self.residual = residual
        if update_edge:
            self.edge_update = EdgeUpdate(
                feat_irreps=out_irreps,
                h_edge=h_edge
            )

    def forward(self,
                atom_features: torch.Tensor,
                edge_features: torch.Tensor,
                edge_sh,
                edge_index: torch.Tensor,
                enforce_dim_size=True):

        edge_dst, edge_src = edge_index
        tp = self.tp(
            atom_features[edge_dst],
            edge_sh,
            self.fc(edge_features)
        )
        if enforce_dim_size:
            out = scatter_mean(tp, edge_src, dim=0, dim_size=atom_features.shape[0])
        else:
            out = scatter_mean(tp, edge_src, dim=0)

        if self.residual:
            padded = F.pad(atom_features, (0, out.shape[-1] - atom_features.shape[-1]))
            out = out + padded
        # print(tp.shape, edge_src, out.shape)
        out = self.norm(out)

        if self.update_edge:
            edge_features = self.edge_update(
                out,
                edge_features,
                edge_index
            )

        return out, edge_features


class EmbeddingLayer(nn.Module):
    def __init__(self,
                 irreps_seq,
                 sh_irreps,
                 h_edge):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TensorConvLayer(
                    irreps_seq[i],
                    sh_irreps,
                    irreps_seq[i+1],
                    h_edge,
                    update_edge=False)
                for i in range(len(irreps_seq)-1)
            ]
        )

    def forward(self,
                node_features,
                edge_features,
                edge_sh,
                edge_index):
        for layer in self.layers:
            node_features, _ = layer(
                node_features,
                edge_features,
                edge_sh,
                edge_index
            )
        return node_features


class UpdateLayer(nn.Module):
    def __init__(self,
                 feat_irreps,
                 sh_irreps,
                 h_edge,
                 tor_neighbor_atom_radius=5,
                 num_rbf=32):
        super().__init__()
        self.feat_irreps = feat_irreps
        self.sh_irreps = sh_irreps
        self.tor_neighbor_atom_radius = tor_neighbor_atom_radius
        self.num_rbf = num_rbf
        ns = feat_irreps.count("0e")
        self.ns = ns

        self.bond_tpc = nn.ModuleList([
            TensorConvLayer(
                feat_irreps,
                sh_irreps,
                feat_irreps,
                h_edge,
                update_edge=True
            )
            for _ in range(5)]
        )
        self.spatial_tpc = nn.ModuleList([
            TensorConvLayer(
                feat_irreps,
                sh_irreps,
                feat_irreps,
                h_edge,
            )
            for _ in range(5)]
        )
        self.node_transition = NodeTransition(feat_irreps)

        self.tor_edge_embedding = nn.Sequential(
            nn.Linear(num_rbf, h_edge),
            nn.ReLU(),
            nn.Linear(h_edge, h_edge)
        )
        self.tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
        self.tor_bond_conv = TensorConvLayer(
            feat_irreps,
            self.tp_tor.irreps_out,
            o3.Irreps(f"{ns}x0e+{ns}x0o"),
            h_edge + 2*ns,
            residual=False,
            slower=True
        )
        def init_fn(weight, bias):
            nn.init.normal_(weight, std=0.01)
            # nn.init.normal_(bias, std=0.01)

        self.tor_final_layer = nn.Sequential(
            nn.Linear(2 * ns, ns, bias=False),
            nn.ReLU(),
            # nn.Tanh(),
            Linear(ns, 1, bias=False, init_fn=init_fn),
            nn.Tanh()
        )

    def build_bond_conv_graph(self, atom_pos, rot_bond_edge_index, atom_batch):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bond_pos = (atom_pos[rot_bond_edge_index[0]] + atom_pos[rot_bond_edge_index[1]]) / 2
        bond_batch = atom_batch[rot_bond_edge_index[0]]
        edge_index = radius(atom_pos, bond_pos, self.tor_neighbor_atom_radius, batch_x=atom_batch, batch_y=bond_batch)

        edge_vec = atom_pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = _rbf(edge_vec.norm(dim=-1), D_max=self.tor_neighbor_atom_radius, D_count=self.num_rbf, device=edge_vec.device)

        edge_attr = self.tor_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh

    def forward(self,
                atom_features,
                atom_pos,
                bond_features,
                bond_sh,
                bond_edge_index,
                spatial_features,
                spatial_sh,
                spatial_edge_index,
                rotatable_bonds,
                batch,
                update_instructs):

        # print("bond")
        for bond_tpc, spatial_tpc in zip(self.bond_tpc, self.spatial_tpc):
            atom_features, bond_features = bond_tpc(
                atom_features,
                bond_features,
                bond_sh,
                bond_edge_index)
            # print("spatial")
            atom_features, _ = spatial_tpc(
                atom_features,
                spatial_features,
                spatial_sh,
                spatial_edge_index)
        # atom_features = self.node_transition(atom_features)

        rot_bond_edge_index = bond_edge_index[:, rotatable_bonds]
        # print(rot_bond_edge_index, rotatable_bonds.sum())
        tor_bond_edge_index, tor_bond_edge_features, tor_bond_edge_sh = self.build_bond_conv_graph(atom_pos, rot_bond_edge_index, batch)
        # print(tor_bond_edge_index)

        tor_bond_vec = atom_pos[tor_bond_edge_index[1]] - atom_pos[tor_bond_edge_index[0]]
        tor_bond_attr = atom_features[tor_bond_edge_index[0]] + atom_features[tor_bond_edge_index[1]]
        tor_edge_attr = torch.cat([
            tor_bond_edge_features,
            atom_features[tor_bond_edge_index[1], :self.ns],
            tor_bond_attr[tor_bond_edge_index[0], :self.ns]
        ], dim=-1)
        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        # print("torsion")
        tor_edge_sh = self.tp_tor(tor_bond_edge_sh, tor_bonds_sh[tor_bond_edge_index[0]])

        # we flip cuz my code is mainly "source_to_target" but above is "target_to_src"
        tor_update, _ = self.tor_bond_conv(atom_features, tor_edge_attr, tor_edge_sh, tor_bond_edge_index.flip(0), enforce_dim_size=False)
        tor_update = self.tor_final_layer(tor_update).squeeze(-1) * torch.pi + 1e-4
        # print(tor_update)

        new_atom_pos = modify_conformer_torsion_angles_batch_cached(
            atom_pos,
            tor_update,
            update_instructs
        )

        return new_atom_pos, atom_features, bond_features, tor_update


class MoleculeTorsionDenoiser(nn.Module):
    def __init__(self,
                 c_s=32,
                 c_v=8,
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
        self.irreps_seq = get_irrep_seq(c_s, c_v, reduce_pseudoscalars=False)
        self.feat_irreps = o3.Irreps(self.irreps_seq[-1])
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
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
        self.embedding = EmbeddingLayer(
            self.irreps_seq,
            self.sh_irreps,
            edge_hidden
        )
        self.update_layers = nn.ModuleList(
            [
                UpdateLayer(
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

        self.k = k


    @torch.no_grad()
    def _prep_features(self, graph, eps=1e-8):
        atom_data = graph['ligand']
        bond_data = graph['ligand', 'ligand']

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
        bond_len_rbf = _rbf(bond_len, D_min=0., D_max=5., D_count=self.n_rbf, device=bond_len.device)
        bond_features.append(bond_len_rbf)

        bond_features = torch.cat(bond_features, dim=-1).float()


        return atom_pos.float(), atom_features.float(), bond_features.float(), bond_edge_index

    def _gen_spatial_edges(self, atom_pos, batch, eps=1e-8, self_condition=None):
        # spatial_edge_index = knn_graph(atom_pos, self.k, batch=batch)
        spatial_edge_index = radius_graph(atom_pos, r=100.0, batch=batch, max_num_neighbors=1000)
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
        lig_data = data['ligand']
        bond_data = data['ligand', 'ligand']
        atom_pos, atom_features, bond_features, bond_edge_index = self._prep_features(data)
        t = data['t']
        time_embed = self.time_rbf(t[..., None])
        nodewise_t_embed = time_embed[lig_data.batch]

        atom_features = self.embed_atoms(
            torch.cat([atom_features, nodewise_t_embed], dim=-1)
        )
        bond_features = self.embed_bonds(bond_features)

        bond_edge_vecs = atom_pos[bond_edge_index[0]] - atom_pos[bond_edge_index[1]]
        bond_sh = o3.spherical_harmonics(self.sh_irreps, bond_edge_vecs, normalize=True, normalization='component')
        atom_features = self.embedding(
            atom_features,
            bond_features,
            bond_sh,
            bond_edge_index
        )

        atom_traj = []
        total_tor_update = 0

        for update_layer, bond_lin, embed_spatial_edge in zip(
            self.update_layers, self.bond_linears, self.embed_spatial_edge
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

            atom_pos, atom_features, bond_features, tor_update = update_layer(
                atom_features,
                atom_pos,
                bond_features,
                bond_sh,
                bond_edge_index,
                spatial_edge_features,
                spatial_sh,
                spatial_edge_index,
                bond_data.rotatable_bonds,
                lig_data.batch,
                data['conformer_update_instructions'])

            atom_traj.append(atom_pos)
            total_tor_update = total_tor_update + tor_update
            # print(total_tor_update)

        total_tor_update = torch.fmod(total_tor_update + np.pi, 2 * np.pi) - np.pi

        return {
            "pred_atom_pos": atom_pos,
            "atom_traj": atom_traj,
            "pred_torsion_update": total_tor_update
        }
