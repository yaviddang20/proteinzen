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

from . import wigner


from proteinzen.model.modules.layers.node.tfn import TensorProductConvLayer, TensorProductAggLayer, TensorProductBroadcastLayer
from proteinzen.model.modules.layers.edge.tfn import FasterTensorProduct
from proteinzen.model.modules.common import GaussianRandomFourierBasis
from proteinzen.model.modules.openfold.layers import StructureModuleTransition, Linear
from proteinzen.model.utils.graph import sample_inv_cubic_edges
from proteinzen.model.utils.atomic import gen_bond_graph
from proteinzen.data.openfold.residue_constants import resname_to_idx
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


class EdgeUpdate(nn.Module):
    def __init__(self,
                 res_irreps,
                 sh_irreps,
                 res_h_edge):
        super().__init__()
        if not isinstance(res_irreps, o3.Irreps):
            res_irreps = o3.Irreps(res_irreps)
        self.res_irreps = res_irreps

        self.res_irreps_slice_dict = {
            ir: sl for (_, ir), sl in zip(self.res_irreps, self.res_irreps.slices())
        }

        self.tp_in_irreps = (res_irreps * 2).simplify()
        self.sh_irreps = sh_irreps
        self.out_irreps = o3.Irreps(f"{res_h_edge}x0e")

        self.tp = FasterTensorProduct(
            self.tp_in_irreps,
            self.sh_irreps,
            self.out_irreps
        )
        self.ffn = nn.Sequential(
            nn.Linear(res_h_edge, 2*res_h_edge),
            nn.ReLU(),
            nn.Linear(2*res_h_edge, self.tp.weight_numel),
        )
        self.ln = nn.LayerNorm(res_h_edge)


    def forward(self,
                res_features,
                edge_features,
                edge_sh,
                edge_index
    ):
        res_src = res_features[edge_index[1]]
        res_dst = res_features[edge_index[0]]

        cat_list = []
        for _, ir in self.tp_in_irreps:
            sl = self.res_irreps_slice_dict[ir]
            cat_list.append(res_src[..., sl])
            cat_list.append(res_dst[..., sl])

        tp_input = torch.cat(cat_list, dim=-1)
        tp_update = self.tp(
            tp_input,
            edge_sh,
            self.ffn(edge_features)
        )
        edge_features = self.ln(edge_features + tp_update)
        return edge_features


class JointConvLayer(nn.Module):
    def __init__(self,
                 atom_in_irreps,
                 res_in_irreps,
                 sh_irreps,
                 atom_out_irreps,
                 res_out_irreps,
                 h_edge,
                 res_h_mult_factor,
                 res_edge_mult_factor,
                 broadcast=False,
                 res_edge_update=False,
                 compatibility_mode=True):
        super().__init__()
        res_h_edge = h_edge * res_h_mult_factor * res_edge_mult_factor

        self.atom_conv = TensorProductConvLayer(
            atom_in_irreps,
            sh_irreps,
            atom_out_irreps,
            h_edge,
            faster=True,
            edge_groups=["bond_features", "radius_edge_features"]
        )
        self.agg_conv = TensorProductAggLayer(
            atom_in_irreps if compatibility_mode else atom_out_irreps,
            sh_irreps,
            res_out_irreps if compatibility_mode else res_in_irreps,
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
        self.broadcast = broadcast
        if self.broadcast:
            self.bcast_conv = TensorProductBroadcastLayer(
                res_out_irreps,
                sh_irreps,
                atom_out_irreps,
                h_edge,
                faster=True
            )
        if res_edge_update:
            self.res_edge_update = EdgeUpdate(
                res_out_irreps,
                sh_irreps,
                res_h_edge
            )
        else:
            self.res_edge_update = None


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
        if self.broadcast:
            atom_features = self.bcast_conv(
                src_node_attr=res_features,
                bcast_node_attr=atom_features,
                bcast_index=data_dict['atom_res_batch'],
                edge_attr=data_dict['bcast_edge_features'],
                edge_sh=data_dict['bcast_edge_sh']
            )
        if self.res_edge_update is not None:
            res_edge_features = self.res_edge_update(
                res_features=res_features,
                edge_features=data_dict['res_edge_features'],
                edge_sh=data_dict['res_edge_sh'],
                edge_index=data_dict['res_edge_index'],
            )
        else:
            res_edge_features=data_dict['res_edge_features']

        return atom_features, res_features, res_edge_features


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
        dropout=0.0,
        res_edge_mult_factor=1,
        lrange_graph=False,
        broadcast_to_atoms=False,
        smooth_nodewise=False,
        use_masking_features=False,
        res_edge_update=False,
        compatibility_mode=True
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

        self.final_irreps = o3.Irreps(res_irrep_seq[-1])
        self.atomic_num_rbf = atomic_num_rbf
        self.res_num_rbf = res_num_rbf
        self.num_timestep = num_timestep
        self.num_pos_embed = num_pos_embed
        self.num_aa = num_aa
        self.lrange_graph = lrange_graph

        self.atomic_r = atomic_r
        self.agg_r = agg_r
        self.knn_k = knn_k
        self.res_D_max = res_D_max
        self.use_masking_features = use_masking_features
        if use_masking_features:
            atom_in = atom_in + 1

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


        res_h_edge = res_h_mult_factor * h_edge * res_edge_mult_factor
        res_ns = res_h_mult_factor * ns
        res_nv = res_h_mult_factor * nv
        self.res_h_edge = res_h_edge
        self.res_ns = res_ns
        self.res_nv = res_nv

        res_in = num_aa + num_timestep
        if use_masking_features:
            res_in = res_in + 1
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
                    res_h_mult_factor=res_h_mult_factor,
                    res_edge_mult_factor=res_edge_mult_factor,
                    broadcast=broadcast_to_atoms,
                    res_edge_update=res_edge_update,
                    compatibility_mode=compatibility_mode
                )
                for i in range(len(atom_irrep_seq)-1)
            ]
        )

        if smooth_nodewise:
            self.smooth = TensorProductConvLayer(
                in_irreps=res_irrep_seq[-1],
                sh_irreps=self.sh_irreps,
                out_irreps=res_irrep_seq[-1],
                n_edge_features=res_h_mult_factor * h_edge,
                faster=True,
                residual=False
            )
        else:
            self.smooth = None

        if h_frame is not None:
            self.to_frame_scalars = nn.Linear(
                self.final_irreps.dim,
                h_frame
            )
            self.out_ln = nn.LayerNorm(h_frame)
            self.transition = StructureModuleTransition(h_frame)
            self.latent_mu = Linear(h_frame, h_frame, init='final')
            self.latent_logvar = Linear(h_frame, h_frame)
        else:
            self.to_frame_scalars = None

        self.compatibility_mode = compatibility_mode


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
        rev_edge_sh = o3.spherical_harmonics(self.sh_irreps, -edge_vecs, normalize=True, normalization='component')
        return edge_rbf, edge_sh, rev_edge_sh

    def _gen_initial_features(self,
                              data,
                              ablate_timestep=False,
                              zero_timestep=False,
                              apply_noising_masks=False):
        assert not (ablate_timestep and zero_timestep)

        res_data = data['residue']
        atom14 = res_data['atom14_gt_positions'].float()
        atom14 = atom14.clone()
        # atom14[:, 4:] = 0
        atom14_mask = res_data['atom14_gt_exists'].bool()
        seq_mask = res_data['seq_mask'].bool()
        atom14_mask[seq_mask, 4:] = False
        if apply_noising_masks:
            atom14_noising_mask = res_data['atom14_noising_mask'].clone().bool()
            atom14_noising_mask[:, 5:] &= res_data['seq_noising_mask'][..., None]
            atom14_mask = atom14_mask & atom14_noising_mask & res_data['res_noising_mask'][..., None]
        atom_coords = atom14[atom14_mask]
        seq = res_data['seq']
        t = data['t']
        reswise_t = t[res_data.batch]
        if zero_timestep:
            reswise_t = reswise_t * 0.0

        X_ca = atom14[:, 1]
        X_ca_copy = X_ca.clone()
        X_ca_copy[~res_data['res_mask']] = torch.inf

        seq_one_hot = F.one_hot(seq, num_classes=self.num_aa) * res_data['seq_mask'][..., None]
        if apply_noising_masks:
            seq_one_hot = seq_one_hot * res_data['seq_noising_mask'][..., None]
            # print(seq_one_hot.sum(dim=-1))

        res_features = [
            seq_one_hot,
            self.time_embed(reswise_t[..., None]) * ablate_timestep
        ]
        if self.use_masking_features:
            res_features.append(res_data['seq_noising_mask'][..., None])
        res_features = torch.cat(res_features, dim=-1)

        if self.lrange_graph:
            res_edge_index, _, _ = sample_inv_cubic_edges(
                batched_X_ca=X_ca_copy,
                batched_x_mask=~res_data['res_mask'],
                batch=res_data.batch,
                knn_k=self.knn_k,
                inv_cube_k=self.knn_k*2,
            )
        else:
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

        # we do this so we don't give away
        # whether or not the masked residue is a gly or not
        # (the C-alpha atomic embedding changes which can be picked up on by the network)
        all_ala = torch.full_like(seq, resname_to_idx['ALA'])
        ala_masked_seq = all_ala * (~res_data['seq_noising_mask']) + seq * res_data['seq_noising_mask']

        bond_graph = gen_bond_graph(ala_masked_seq, atom14_mask, res_data.batch)
        atom_features = bond_graph['atom_props'].float()
        if self.use_masking_features:
            seq_noising_mask = res_data['seq_noising_mask']
            atom_noising_mask = seq_noising_mask[bond_graph['atom_res_batch']]
            atom_features = torch.cat([
                atom_features, atom_noising_mask[..., None]
            ], dim=-1)
        bond_features = bond_graph['bond_props'].float()
        bond_edge_index = bond_graph["bond_edge_index"]
        radius_edge_index = radius_graph(atom_coords, self.atomic_r, bond_graph['atom_batch'])
        atom_res_batch = bond_graph['atom_res_batch']

        agg_rbf_features, agg_edge_sh, bcast_edge_sh = self._agg_edge_rbf_features(atom_coords, X_ca, atom_res_batch)
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
            "bcast_edge_features": agg_edge_features,
            "res_edge_features": res_edge_features,
            "bond_edge_index": bond_edge_index,
            "radius_edge_index": radius_edge_index,
            "atom_edge_index": atom_edge_index,
            "res_edge_index": res_edge_index,
            "atom_res_batch": atom_res_batch,
            "agg_edge_sh": agg_edge_sh,
            "bcast_edge_sh": bcast_edge_sh,
            "radius_edge_sh": radius_edge_sh,
            "bond_sh": bond_sh,
            "atom_edge_sh": atom_edge_sh,
            "res_edge_sh": res_edge_sh,
        }


    def forward(self,
                data,
                ablate_timestep=False,
                zero_timestep=False,
                apply_noising_masks=False,
    ):
        if apply_noising_masks:
            assert self.use_masking_features
        data_dict = self._gen_initial_features(
            data,
            ablate_timestep=ablate_timestep,
            zero_timestep=zero_timestep,
            apply_noising_masks=apply_noising_masks,
        )
        # print(
        #     data_dict['atom_coords'].shape,
        #     data_dict['atom_features'].shape,
        #     data_dict['res_features'].shape,
        #     data_dict['bond_features'].shape,
        #     data['residue']['seq_noising_mask'].float().mean()
        # )
        for layer in self.embedding_layers:
            atom_features, res_features, res_edge_features = layer(data_dict)
            data_dict["atom_features"] = atom_features
            data_dict["res_features"] = res_features
            data_dict['res_edge_features'] = res_edge_features

        if self.smooth is not None:
            res_features = self.smooth(
                node_attr=data_dict["res_features"],
                edge_index=data_dict['res_edge_index'],
                edge_attr=data_dict['res_edge_features'],
                edge_sh=data_dict['res_edge_sh']
            )
            data_dict["res_features"] = res_features

        final_res_features = data_dict["res_features"]

        if self.to_frame_scalars is not None:
            rigids_1 = ru.Rigid.from_tensor_7(data['residue']['rigids_1'])
            rigids_inv_quat = rigids_1.get_rots().invert().get_quats()

            if self.compatibility_mode:
                inv_wigner_D = self.final_irreps.D_from_quaternion(rigids_inv_quat.cpu()).to(rigids_inv_quat.device)
                final_res_features = torch.einsum("...ij,...j->...i", inv_wigner_D, final_res_features)
            else:
                final_res_features = wigner.fast_wigner_D_rotation(self.final_irreps, rigids_inv_quat, final_res_features)
            final_res_features = self.out_ln(self.to_frame_scalars(final_res_features))
            final_res_features = self.transition(final_res_features)
            final_res_features = {
                'latent_mu': self.latent_mu(final_res_features),
                'latent_logvar': self.latent_logvar(final_res_features),
            }

        return final_res_features
