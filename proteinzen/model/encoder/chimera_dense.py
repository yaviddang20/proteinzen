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
from .quantize import Quantizer


from proteinzen.model.modules.layers.node.conv import SequenceDownscaler
from proteinzen.model.modules.layers.edge.conv import EdgeDownscaler
from proteinzen.model.modules.layers.node.tfn import TensorProductConvLayer, TensorProductAggLayer, TensorProductBroadcastLayer
from proteinzen.model.modules.layers.edge.sitewise import DenseEdgeTransition as EdgeTransition
from proteinzen.model.modules.common import GaussianRandomFourierBasis, FeedForward
from proteinzen.model.modules.openfold.layers import StructureModuleTransition, Linear, InvariantPointAttention
from proteinzen.model.utils.graph import sample_inv_cubic_edges
from proteinzen.model.utils.atomic import gen_bond_graph
from proteinzen.data.openfold.residue_constants import resname_to_idx
from proteinzen.data.constants.atom14 import atom14_atom_props, atom14_bond_props
from proteinzen.data.datasets.featurize.common import _edge_positional_embeddings, _rbf, _node_positional_embeddings
from proteinzen.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from proteinzen.utils.openfold import rigid_utils as ru


def get_irrep_seq(ns, nv):
    irrep_seq = [
        f'{ns}x0e',
        f'{ns}x0e + {nv}x1o',
        f'{ns}x0e + {nv}x1o + {nv}x1e',
        f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
    ]
    return irrep_seq



class JointConvLayer(nn.Module):
    def __init__(
        self,
        atom_in_irreps,
        sh_irreps,
        atom_out_irreps,
        atom_h_edge,
        res_irreps,
        c_s,
        c_z,
        c_hidden=16,
        num_qk_points=8,
        num_v_points=12,
        num_heads=16,
        broadcast=False,
        dropout=0.0,
        pre_ln=False,
        lin_bias=True,
    ):
        super().__init__()

        self.pre_ln = pre_ln

        self.atom_conv = TensorProductConvLayer(
            atom_in_irreps,
            sh_irreps,
            atom_out_irreps,
            atom_h_edge,
            faster=True,
            edge_groups=["bond_features", "radius_edge_features"]
        )
        self.agg_conv = TensorProductAggLayer(
            atom_out_irreps,
            sh_irreps,
            res_irreps,
            atom_h_edge,
            faster=True,
            residual=False
        )

        if not isinstance(res_irreps, o3.Irreps):
            res_irreps = o3.Irreps(res_irreps)
        self.res_irreps = res_irreps
        self.irreps_to_scalar = nn.Sequential(
            nn.Linear(res_irreps.dim, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )
        if not self.pre_ln:
            self.atom_to_res_ln = nn.LayerNorm(c_s)
            self.ipa_ln = nn.LayerNorm(c_s)
        else:
            self.res_to_atom_ln = nn.LayerNorm(c_s)

        self.ipa = InvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            pre_ln=pre_ln,
            lin_bias=lin_bias,
            # final_init="default"
        )

        self.ffn = FeedForward(
            c_in=c_s,
            c_hidden=c_s,
            c_out=c_s,
            dropout=dropout,
            pre_ln=pre_ln,
            lin_bias=lin_bias,
            # final_init='default'
        )

        self.edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z,
            pre_ln=pre_ln,
            lin_bias=lin_bias
        )

        self.broadcast = broadcast
        if self.broadcast:
            if not isinstance(atom_out_irreps, o3.Irreps):
                atom_out_irreps = o3.Irreps(atom_out_irreps)

            self.scalar_to_0e = Linear(c_s, atom_out_irreps.count("0e"), bias=lin_bias)
            self.scalar_to_1o = Linear(c_s, atom_out_irreps.count("1o") * 3, bias=lin_bias)

            bcast_irreps = f"{atom_out_irreps.count('0e')}x0e + {atom_out_irreps.count('1o')}x1o"

            self.bcast_conv = TensorProductBroadcastLayer(
                bcast_irreps,
                sh_irreps,
                bcast_irreps,
                atom_h_edge,
                faster=True,
                residual=False
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

        res_update = self.agg_conv(
            dst_node_attr=data_dict["res_features"],  # TODO: this is only used for shape?
            agg_node_attr=atom_features,
            agg_index=data_dict['atom_res_batch'],
            edge_attr=data_dict['agg_edge_features'],
            edge_sh=data_dict['agg_edge_sh']
        )
        rigids: ru.Rigid = data_dict['rigids']
        rigids_inv_quat = rigids.get_rots().invert().get_quats()
        res_update = wigner.fast_wigner_D_rotation(self.res_irreps, rigids_inv_quat, res_update)
        res_update = self.irreps_to_scalar(res_update)

        res_features = data_dict['res_features']
        res_edge_features = data_dict['res_edge_features']

        res_features = res_features + res_update
        if not self.pre_ln:
            res_features = self.atom_to_res_ln(res_features)

        dense_res_features = res_features.view(*res_edge_features.shape[:2], -1)

        # print(dense_res_features.shape, res_edge_features.shape, rigids.shape, data_dict['res_mask'].shape)
        ipa_update = self.ipa(
            s=dense_res_features,
            z=res_edge_features,
            r=rigids.view(dense_res_features.shape[:2]),
            mask=data_dict['res_mask'].float()
        )
        dense_res_features = dense_res_features + ipa_update
        if not self.pre_ln:
            dense_res_features = self.ipa_ln(dense_res_features)

        res_edge_features = self.edge_transition(
            dense_res_features,
            res_edge_features
        )

        res_features = dense_res_features.view(res_features.shape)

        if self.broadcast:
            if not self.pre_ln:
                ln_res_features = self.res_to_atom_ln(res_features)
            else:
                ln_res_features = res_features
            res_0e = self.scalar_to_0e(ln_res_features)
            res_1o = self.scalar_to_1o(ln_res_features)
            res_1o = res_1o.view(list(res_1o.shape[:-1]) + [-1, 3])
            res_1o = rigids[..., None].get_rots().apply(res_1o)

            res_bcast_features = torch.cat(
                [res_0e, res_1o.view(list(res_1o.shape[:-2]) + [-1])],
                dim=-1
            )
            atom_update = self.bcast_conv(
                src_node_attr=res_bcast_features,
                bcast_node_attr=atom_features,
                bcast_index=data_dict['atom_res_batch'],
                edge_attr=data_dict['bcast_edge_features'],
                edge_sh=data_dict['bcast_edge_sh']
            )
            padded = F.pad(atom_update, (0, atom_features.shape[-1] - atom_update.shape[-1]))
            atom_features = atom_features + padded

        return atom_features, res_features, res_edge_features


class ProteinAtomicChimeraDenseEmbedder(nn.Module):
    def __init__(
        self,
        ns=16,
        nv=4,
        atom_h_edge=16,
        c_s=128,
        c_z=128,
        c_s_out=32,
        c_z_out=32,
        atom_in=atom14_atom_props.shape[-1],
        bond_in=atom14_bond_props.shape[-1],
        res_h_mult_factor=2,
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
        dropout=0.0,
        lrange_graph=False,
        broadcast_to_atoms=False,
        use_masking_features=False,
        pre_ln=False,
        lin_bias=True,
        quantize_codebook_size=None,
        conv_downsample_factor=0,
    ):
        super().__init__()
        assert n_layers >= 4
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=1)
        atom_irrep_seq = get_irrep_seq(ns, nv)
        atom_irrep_seq = atom_irrep_seq + [atom_irrep_seq[-1] for _ in range(n_layers - len(atom_irrep_seq) + 1)]
        self.atom_irrep_seq = atom_irrep_seq
        res_irrep_seq = get_irrep_seq(ns * res_h_mult_factor, nv * res_h_mult_factor)
        res_irrep_seq = res_irrep_seq + [res_irrep_seq[-1] for _ in range(n_layers - len(res_irrep_seq) + 1)]
        self.res_irrep_seq = res_irrep_seq

        self.c_s = c_s

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
            nn.Linear(atom_in*2 + atomic_num_rbf, atom_h_edge),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(atom_h_edge, atom_h_edge),
            nn.LayerNorm(atom_h_edge)
        )

        self.bond_edge_embed = nn.Sequential(
            nn.Linear(atom_in*2 + atomic_num_rbf + bond_in, atom_h_edge),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(atom_h_edge, atom_h_edge),
            nn.LayerNorm(atom_h_edge)
        )

        self.agg_edge_embed = nn.Sequential(
            nn.Linear(atom_in + atomic_num_rbf, atom_h_edge),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(atom_h_edge, atom_h_edge),
            nn.LayerNorm(atom_h_edge)
        )


        res_h_edge = res_h_mult_factor * atom_h_edge
        res_ns = res_h_mult_factor * ns
        res_nv = res_h_mult_factor * nv
        self.res_h_edge = res_h_edge
        self.res_ns = res_ns
        self.res_nv = res_nv

        res_in = num_aa + num_timestep
        if use_masking_features:
            res_in = res_in + 1
        self.res_embed = nn.Sequential(
            nn.Linear(res_in, c_s),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(c_s, c_s),
            nn.LayerNorm(c_s)
        )

        self.res_edge_embed = nn.Sequential(
            nn.Linear(res_in*2 + res_num_rbf + num_pos_embed, c_z),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(c_z, c_z),
            nn.LayerNorm(c_z)
        )

        self.embedding_layers = nn.ModuleList(
            [
                JointConvLayer(
                    atom_in_irreps=atom_irrep_seq[i],
                    sh_irreps=self.sh_irreps,
                    atom_out_irreps=atom_irrep_seq[i+1],
                    res_irreps=res_irrep_seq[i+1],
                    atom_h_edge=atom_h_edge,
                    c_s=c_s,
                    c_z=c_z,
                    broadcast=broadcast_to_atoms,
                    pre_ln=pre_ln,
                    lin_bias=lin_bias
                )
                for i in range(len(atom_irrep_seq)-1)
            ]
        )

        self.pre_ln = pre_ln
        if pre_ln:
            self.ln_s = nn.LayerNorm(c_s)
            self.ln_z = nn.LayerNorm(c_z)

        self.conv_downsample_factor = conv_downsample_factor
        c_s_interim = c_s // (2**conv_downsample_factor)
        c_z_interim = c_z // (2**conv_downsample_factor)

        if conv_downsample_factor > 0:
            self.node_downsample = SequenceDownscaler(
                n_layers=conv_downsample_factor,
                c_in=c_s,
                c_out=c_s_interim,
                dropout=dropout
            )
            self.edge_downsample = EdgeDownscaler(
                n_layers=conv_downsample_factor,
                c_in=c_z,
                c_out=c_z_interim,
                dropout=dropout
            )
            self.latent_node_mu = nn.Sequential(
                nn.LayerNorm(c_s_interim),
                nn.Linear(c_s_interim, c_s_out, bias=False)
            )
            self.latent_node_logvar = nn.Sequential(
                nn.LayerNorm(c_s_interim),
                nn.Linear(c_s_interim, c_s_out, bias=False)
            )
            self.latent_edge_mu = nn.Sequential(
                nn.LayerNorm(c_s_interim),
                nn.Linear(c_s_interim, c_s_out, bias=False)
            )
            self.latent_edge_logvar = nn.Sequential(
                nn.LayerNorm(c_z_interim),
                nn.Linear(c_z_interim, c_s_out, bias=False)
            )

        else:
            self.latent_node_mu = nn.Linear(c_s, c_s_out)
            self.latent_node_logvar = nn.Linear(c_s, c_s_out)
            self.latent_edge_mu = nn.Linear(c_z, c_z_out)
            self.latent_edge_logvar = nn.Linear(c_z, c_z_out)


        if quantize_codebook_size is None:
            self.node_quantizer = None
            self.edge_quantizer = None
        else:
            self.node_quantizer = Quantizer(h_dim=c_s_out, n_codebook=quantize_codebook_size)
            self.edge_quantizer = Quantizer(h_dim=c_z_out, n_codebook=quantize_codebook_size)


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

    def _res_edge_features(self, coords):
        assert coords.dim() == 3
        n_batch = coords.shape[0]
        edge_dists = torch.cdist(coords, coords)
        edge_rbf = _rbf(edge_dists, D_max=self.res_D_max, D_count=self.res_num_rbf, device=edge_dists.device)
        node_pos = torch.arange(coords.shape[-2], device=coords.device)
        edge_rel_pos = node_pos[..., None] - node_pos[None]
        edge_rel_pos_embed = _node_positional_embeddings(edge_rel_pos, num_embeddings=self.num_pos_embed, device=edge_dists.device)
        edge_features = torch.cat([
            edge_rbf,
            edge_rel_pos_embed[None].expand(n_batch, -1, -1, -1)
        ], dim=-1)
        return edge_features


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

        res_features = [
            seq_one_hot,
            self.time_embed(reswise_t[..., None]) * ablate_timestep
        ]
        if self.use_masking_features:
            res_features.append(res_data['seq_noising_mask'][..., None])
        res_features = torch.cat(res_features, dim=-1)

        n_batch = data.num_graphs
        dense_res_features = res_features.view(n_batch, -1, res_features.shape[-1])
        dense_X_ca = X_ca.view(n_batch, -1, 3)
        n_res = dense_X_ca.shape[1]

        res_edge_features = self._res_edge_features(dense_X_ca)

        res_edge_features = self.res_edge_embed(
            torch.cat([
                res_edge_features,
                torch.tile(dense_res_features[:, None], (1, n_res, 1, 1)),
                torch.tile(dense_res_features[:, :, None], (1, 1, n_res, 1)),
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

        rigids = ru.Rigid.from_tensor_7(data['residue']['rigids_1'])

        return {
            "rigids": rigids,
            "res_coords": X_ca,
            "atom_coords": atom_coords,
            "atom_features": atom_features,
            "res_features": res_features,
            "res_mask": data['residue']['res_mask'].view(n_batch, -1),
            "bond_features": bond_features,
            "radius_edge_features": radius_edge_features,
            "agg_edge_features": agg_edge_features,
            "bcast_edge_features": agg_edge_features,
            "res_edge_features": res_edge_features,
            "bond_edge_index": bond_edge_index,
            "radius_edge_index": radius_edge_index,
            "atom_edge_index": atom_edge_index,
            "atom_res_batch": atom_res_batch,
            "agg_edge_sh": agg_edge_sh,
            "bcast_edge_sh": bcast_edge_sh,
            "radius_edge_sh": radius_edge_sh,
            "bond_sh": bond_sh,
            "atom_edge_sh": atom_edge_sh,
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
        for layer in self.embedding_layers:
            atom_features, res_features, res_edge_features = layer(data_dict)
            data_dict["atom_features"] = atom_features
            data_dict["res_features"] = res_features
            data_dict["res_edge_features"] = res_edge_features

        final_res_features = data_dict["res_features"].view(data.num_graphs, -1, self.c_s)
        final_res_edge_features = data_dict["res_edge_features"]
        if self.pre_ln:
            final_res_features = self.ln_s(final_res_features)
            final_res_edge_features = self.ln_z(final_res_edge_features)

        if self.conv_downsample_factor > 0:
            final_res_features = self.node_downsample(final_res_features * data_dict["res_mask"][..., None])
            edge_mask = data_dict["res_mask"][..., None] & data_dict["res_mask"][..., None, :]
            final_res_edge_features = self.edge_downsample(final_res_edge_features * edge_mask[..., None])

        out_dict = {
            'latent_node_mu': self.latent_node_mu(final_res_features),
            'latent_edge_mu': self.latent_edge_mu(final_res_edge_features),
        }
        # print(out_dict['latent_edge_mu'].shape, out_dict['latent_edge_logvar'].shape)
        if self.node_quantizer is not None:
            quantized_node, node_quant_loss, node_commit_loss = self.node_quantizer(out_dict['latent_node_mu'], data_dict['res_mask'])
            out_dict['quantized_node'] = quantized_node
            out_dict.update({
                'quantized_node': quantized_node,
                'node_quant_loss': node_quant_loss,
                'node_commit_loss': node_commit_loss,
            })
        else:
            out_dict.update({
                'latent_node_logvar': self.latent_node_logvar(final_res_features),
            })

        if self.edge_quantizer is not None:
            edge_mask = data_dict['res_mask'][..., None] & data_dict['res_mask'][..., None, :]
            quantized_edge, edge_quant_loss, edge_commit_loss = self.edge_quantizer(out_dict['latent_edge_mu'], edge_mask)
            out_dict.update({
                'quantized_edge': quantized_edge,
                'edge_quant_loss': edge_quant_loss,
                'edge_commit_loss': edge_commit_loss,
            })
        else:
            out_dict.update({
                'latent_edge_logvar': self.latent_edge_logvar(final_res_edge_features),
            })


        return out_dict
