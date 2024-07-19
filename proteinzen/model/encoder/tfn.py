from typing import Union, List

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_cluster import knn, knn_graph, radius_graph
from torch_scatter import scatter_mean
import torch_geometric.utils as pygu

import e3nn
from e3nn import o3


from proteinzen.model.modules.layers.node.tfn import TensorProductConvLayer, TensorProductAggLayer
from proteinzen.model.modules.common import GaussianRandomFourierBasis
from proteinzen.model.utils.atomic import gen_bond_graph
from proteinzen.data.constants.atom14 import atom14_atom_props, atom14_bond_props
from proteinzen.data.datasets.featurize.common import _edge_positional_embeddings, _rbf
from proteinzen.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from proteinzen.utils.openfold import rigid_utils as ru


def get_irrep_seq(ns, nv, reduce_pseudoscalars):
    irrep_seq = [
        f'{ns}x0e',
        f'{ns}x0e + {nv}x1o',
        f'{ns}x0e + {nv}x1o + {nv}x1e',
        f'{ns}x0e + {nv}x1o + {nv}x1e + {nv if reduce_pseudoscalars else ns}x0o'
    ]
    return irrep_seq


class JointConvLayer(nn.Module):
    def __init__(self,
                 atom_in_irreps,
                 res_in_irreps,
                 sh_irreps,
                 atom_out_irreps,
                 res_out_irreps,
                 h_edge,
                 res_h_mult_factor):
        super().__init__()
        res_h_edge = h_edge * res_h_mult_factor

        self.atom_conv = TensorProductConvLayer(
            atom_in_irreps,
            sh_irreps,
            atom_out_irreps,
            h_edge,
            faster=True,
            edge_groups=["bond_features", "radius_edge_features"]
        )
        self.agg_conv = TensorProductAggLayer(
            atom_in_irreps,
            sh_irreps,
            res_out_irreps,
            h_edge,
            faster=True
        )
        self.res_conv = TensorProductConvLayer(
            res_in_irreps,
            sh_irreps,
            res_out_irreps,
            res_h_edge,
            faster=True
        )

    def forward(self, data_dict):
        atom_features = self.atom_conv(
            node_attr=data_dict["atom_features"],
            edge_index=data_dict['atom_edge_index'],
            edge_attr={
                "bond_features": data_dict["bond_features"],
                "radius_edge_features": data_dict["radius_edge_features"]
            },
            edge_sh=data_dict['atom_edge_sh']
        )
        res_features = self.agg_conv(
            dst_node_attr=data_dict["res_features"],
            agg_node_attr=atom_features,
            agg_index=data_dict['atom_res_batch'],
            edge_attr=data_dict['agg_edge_features'],
            edge_sh=data_dict['agg_edge_sh']
        )
        res_features = self.res_conv(
            node_attr=res_features,
            edge_index=data_dict['res_edge_index'],
            edge_attr=data_dict['res_edge_features'],
            edge_sh=data_dict['res_edge_sh']
        )
        return atom_features, res_features




class ProteinAtomicEmbedder(nn.Module):
    def __init__(
        self,
        ns=16,
        nv=4,
        h_edge=16,
        atom_in=atom14_atom_props.shape[-1],
        bond_in=atom14_bond_props.shape[-1],
        res_h_mult_factor=2,
        h_frame=None,
        atomic_num_rbf=16,
        res_num_rbf=32,
        num_pos_embed=32,
        num_aa=21,
        num_timestep=0,
        n_layers=8,
        atomic_r=5,
        agg_r=10,
        res_D_max=22,
        knn_k=30,
        reduce_pseudoscalars=False,
        dropout=0.0
    ):
        super().__init__()
        assert n_layers > 4
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=1)
        atom_irrep_seq = get_irrep_seq(ns, nv, reduce_pseudoscalars=reduce_pseudoscalars)
        atom_irrep_seq = atom_irrep_seq + [atom_irrep_seq[-1] for _ in range(n_layers - len(atom_irrep_seq) + 1)]
        self.atom_irrep_seq = atom_irrep_seq
        res_irrep_seq = get_irrep_seq(ns * res_h_mult_factor, nv * res_h_mult_factor, reduce_pseudoscalars=reduce_pseudoscalars)
        res_irrep_seq = res_irrep_seq + [res_irrep_seq[-1] for _ in range(n_layers - len(res_irrep_seq) + 1)]
        self.res_irrep_seq = res_irrep_seq

        self.final_irrep = o3.Irreps(res_irrep_seq[-1])
        self.atomic_num_rbf = atomic_num_rbf
        self.res_num_rbf = res_num_rbf
        self.num_timestep = num_timestep
        self.num_pos_embed = num_pos_embed
        self.num_aa = num_aa

        self.atomic_r = atomic_r
        self.agg_r = agg_r
        self.knn_k = knn_k
        self.res_D_max = res_D_max

        self.time_embed = GaussianRandomFourierBasis(n_basis=num_timestep//2)

        self.atom_embed = nn.Sequential(
            nn.Linear(atom_in, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
            nn.LayerNorm(ns)
        )

        self.radius_edge_embed = nn.Sequential(
            nn.Linear(atom_in*2 + atomic_num_rbf, h_edge),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_edge, h_edge),
            nn.LayerNorm(h_edge)
        )

        self.bond_edge_embed = nn.Sequential(
            nn.Linear(atom_in*2 + atomic_num_rbf + bond_in, h_edge),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_edge, h_edge),
            nn.LayerNorm(h_edge)
        )

        self.agg_edge_embed = nn.Sequential(
            nn.Linear(atom_in + atomic_num_rbf, h_edge),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_edge, h_edge),
            nn.LayerNorm(h_edge)
        )


        res_h_edge = res_h_mult_factor * h_edge
        res_ns = res_h_mult_factor * ns
        res_nv = res_h_mult_factor * nv
        self.res_h_edge = res_h_edge
        self.res_ns = res_ns
        self.res_nv = res_nv

        res_in = num_aa + num_timestep
        self.res_embed = nn.Sequential(
            nn.Linear(res_in, res_ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(res_ns, res_ns),
            nn.LayerNorm(res_ns)
        )

        self.res_edge_embed = nn.Sequential(
            nn.Linear(res_in*2 + res_num_rbf + num_pos_embed, res_h_edge),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(res_h_edge, res_h_edge),
            nn.LayerNorm(res_h_edge)
        )

        self.embedding_layers = nn.ModuleList(
            [
                JointConvLayer(
                    atom_in_irreps=atom_irrep_seq[i],
                    res_in_irreps=res_irrep_seq[i],
                    sh_irreps=self.sh_irreps,
                    atom_out_irreps=atom_irrep_seq[i+1],
                    res_out_irreps=res_irrep_seq[i+1],
                    h_edge=h_edge,
                    res_h_mult_factor=res_h_mult_factor
                )
                for i in range(len(atom_irrep_seq)-1)
            ]
        )

        if h_frame is not None:
            self.to_frame_scalars = nn.Linear(
                self.final_irrep.dim,
                h_frame
            )
            self.out_ln = nn.LayerNorm(h_frame)
        else:
            self.to_frame_scalars = None


    def _edge_rbf_features(self, coords, edge_index, D_max, num_rbf, eps=1e-12):
        edge_dst, edge_src = edge_index
        edge_vecs = coords[edge_dst] - coords[edge_src]
        edge_dist = torch.linalg.vector_norm(edge_vecs + eps, dim=-1)
        edge_rbf = _rbf(edge_dist, D_max=D_max, D_count=num_rbf, device=edge_dist.device)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vecs, normalize=True, normalization='component')
        return edge_rbf, edge_sh

    def _agg_edge_rbf_features(self, atom_coords, res_coords, atom_res_batch, eps=1e-12):
        edge_vecs = atom_coords - res_coords[atom_res_batch]
        edge_dist = torch.linalg.vector_norm(edge_vecs + eps, dim=-1)
        edge_rbf = _rbf(edge_dist, D_max=self.agg_r, D_count=self.atomic_num_rbf, device=edge_dist.device)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vecs, normalize=True, normalization='component')
        return edge_rbf, edge_sh

    def _gen_initial_features(self, data, ablate_timestep=False, zero_timestep=False):
        assert not (ablate_timestep and zero_timestep)

        res_data = data['residue']
        atom14 = res_data['atom14_gt_positions'].float()
        atom14_mask = res_data['atom14_gt_exists'].bool()
        atom_coords = atom14[atom14_mask]
        seq = res_data['seq']
        t = data['t']
        reswise_t = t[res_data.batch]
        if zero_timestep:
            reswise_t = reswise_t * 0.0

        X_ca = atom14[:, 1]
        X_ca_copy = X_ca.clone()
        X_ca_copy[~res_data['res_mask']] = torch.inf

        res_features = [
            F.one_hot(seq, num_classes=self.num_aa) * res_data['seq_mask'][..., None],
            self.time_embed(reswise_t[..., None]) * ablate_timestep
        ]
        res_features = torch.cat(res_features, dim=-1)

        res_edge_index = knn_graph(X_ca_copy, self.knn_k, res_data.batch)
        res_rbf_features, res_edge_sh = self._edge_rbf_features(
            X_ca,
            res_edge_index,
            D_max=self.res_D_max,
            num_rbf=self.res_num_rbf
        )
        res_pos_embed_features = _edge_positional_embeddings(res_edge_index, self.num_pos_embed, device=X_ca.device)
        res_edge_features = self.res_edge_embed(
            torch.cat([
                res_rbf_features,
                res_pos_embed_features,
                res_features[res_edge_index[0]],
                res_features[res_edge_index[1]],
            ], dim=-1)
        )
        res_features = self.res_embed(res_features)


        bond_graph = gen_bond_graph(seq, atom14_mask, res_data.batch)
        atom_features = bond_graph['atom_props'].float()
        bond_features = bond_graph['bond_props'].float()
        bond_edge_index = bond_graph["bond_edge_index"]
        radius_edge_index = radius_graph(atom_coords, self.atomic_r, bond_graph['atom_batch'])
        atom_res_batch = bond_graph['atom_res_batch']

        agg_rbf_features, agg_edge_sh = self._agg_edge_rbf_features(atom_coords, X_ca, atom_res_batch)
        radius_rbf_features, radius_edge_sh = self._edge_rbf_features(
            atom_coords,
            radius_edge_index,
            D_max=self.atomic_r,
            num_rbf=self.atomic_num_rbf
        )
        bond_rbf_features, bond_sh = self._edge_rbf_features(
            atom_coords,
            bond_edge_index,
            D_max=self.atomic_r,
            num_rbf=self.atomic_num_rbf
        )

        agg_edge_features = self.agg_edge_embed(
            torch.cat([
                agg_rbf_features,
                atom_features,
            ], dim=-1)
        )
        bond_features = self.bond_edge_embed(
            torch.cat([
                bond_features,
                bond_rbf_features,
                atom_features[bond_edge_index[0]],
                atom_features[bond_edge_index[0]]
            ], dim=-1)
        )
        radius_edge_features = self.radius_edge_embed(
            torch.cat([
                radius_rbf_features,
                atom_features[radius_edge_index[0]],
                atom_features[radius_edge_index[0]]
            ], dim=-1)
        )
        atom_features = self.atom_embed(atom_features)

        atom_edge_index = torch.cat([
            bond_edge_index, radius_edge_index
        ], dim=-1)
        atom_edge_sh = torch.cat([
            bond_sh, radius_edge_sh
        ], dim=0)

        # print(atom_edge_index.shape, atom_edge_sh.shape)

        return {
            "res_coords": X_ca,
            "atom_coords": atom_coords,
            "atom_features": atom_features,
            "res_features": res_features,
            "bond_features": bond_features,
            "radius_edge_features": radius_edge_features,
            "agg_edge_features": agg_edge_features,
            "res_edge_features": res_edge_features,
            "bond_edge_index": bond_edge_index,
            "radius_edge_index": radius_edge_index,
            "atom_edge_index": atom_edge_index,
            "res_edge_index": res_edge_index,
            "atom_res_batch": atom_res_batch,
            "agg_edge_sh": agg_edge_sh,
            "radius_edge_sh": radius_edge_sh,
            "bond_sh": bond_sh,
            "atom_edge_sh": atom_edge_sh,
            "res_edge_sh": res_edge_sh,
        }


    def forward(self, data, ablate_timestep=False, zero_timestep=False):
        data_dict = self._gen_initial_features(
            data,
            ablate_timestep=ablate_timestep,
            zero_timestep=zero_timestep,
        )
        for layer in self.embedding_layers:
            atom_features, res_features = layer(data_dict)
            data_dict["atom_features"] = atom_features
            data_dict["res_features"] = res_features

        final_res_features = data_dict["res_features"]

        if self.to_frame_scalars is not None:
            final_res_features = self.out_ln(self.to_frame_scalars(final_res_features))


        return final_res_features
