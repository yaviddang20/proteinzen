""" Autoencoder module """

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_cluster import knn, knn_graph, radius_graph
from torch_scatter import scatter_mean

from e3nn import o3
from e3nn.nn import Gate, BatchNorm

from ligbinddiff.model.modules.common import GaussianRandomFourierBasis
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf
from ligbinddiff.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from ligbinddiff.utils.openfold import rigid_utils as ru


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
                 node_irreps,
                 latent_irreps,
                 sh_irreps,
                 h_edge,
                 update_edge=False):
        super().__init__()
        self.node_irreps = node_irreps
        self.latent_irreps = latent_irreps
        self.sh_irreps = sh_irreps

        self.node_update = o3.Linear(
            latent_irreps,
            node_irreps
        )

        self.tp = o3.FullyConnectedTensorProduct(
            node_irreps,
            sh_irreps,
            node_irreps,
            shared_weights=False
        )
        self.fc = nn.Sequential(
            nn.Linear(h_edge, h_edge),
            nn.ReLU(),
            nn.Linear(h_edge, self.tp.weight_numel)
        )
        self.norm = BatchNorm(node_irreps)

        self.latent_update = o3.Linear(
            node_irreps,
            latent_irreps
        )

        self.update_edge = update_edge
        if update_edge:
            self.edge_update = EdgeUpdate(
                feat_irreps=node_irreps,
                h_edge=h_edge
            )

    def forward(self,
                node_features: torch.Tensor,
                edge_features: torch.Tensor,
                edge_sh,
                edge_index: torch.Tensor):

        edge_dst, edge_src = edge_index
        tp = self.tp(
            node_features[edge_dst],
            edge_sh,
            self.fc(edge_features)
        )

        out = scatter_mean(tp, edge_src, dim=0, dim_size=node_features.shape[0])
        padded = F.pad(node_features, (0, out.shape[-1] - node_features.shape[-1]))
        out = out + padded
        out = self.norm(out)

        if self.update_edge:
            edge_features = self.edge_update(
                out,
                edge_features,
                edge_index
            )

        return out, edge_features


class Atom37Encoder(nn.Module):
    def __init__(self,
                 c_s=32,
                 c_v=8,
                 c_z=128,
                 c_out=128,
                 feat_lmax=1,
                 sh_lmax=1,
                 in_v=37,
                 num_rbf=64,
                 num_pos_embed=64,
                 num_layers=4,
                 k=30,
                 n_aa=20,
                 scalarize_wrt_frames=True
                 ):
        super().__init__()
        self.n_aa = n_aa + 1
        self.c_s = c_s
        self.c_v = c_v
        in_s = (
            self.n_aa
            + 1  # mask bit
            + 6  # bb torsion angles
        )
        in_z = (
            num_rbf * (5 * 5)
            + num_pos_embed
            # + 4  # rel orientation quat
        )
        self.in_irreps = o3.Irreps([
            (in_s, (0, 1)),
            (in_v, (1, 1))
        ])
        self.feat_irreps = o3.Irreps([(c_s if l == 0 else c_v, (l, 1)) for l in range(feat_lmax + 1)])
        self.sh_irreps = o3.Irreps([(1, (l, 1)) for l in range(sh_lmax + 1)])
        self.num_rbf = num_rbf
        self.num_pos_embed = num_pos_embed
        self.k = k

        self.edge_embed = nn.Sequential(
            nn.Linear(in_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.LayerNorm(c_z),
        )
        self.node_embed = EmbedNode(self.in_irreps, self.feat_irreps, self.sh_irreps, c_z)

        self.tc = nn.ModuleList(
            [
                TensorConvLayer(
                    self.feat_irreps,
                    self.sh_irreps,
                    c_z,
                    update_edge=True
                )
                for _ in range(num_layers)
            ]
        )

        self.scalarize_wrt_frames = scalarize_wrt_frames
        if scalarize_wrt_frames:
            self.output_mu = nn.Linear(self.feat_irreps.dim, c_out)
            self.output_logvar = nn.Linear(self.feat_irreps.dim, c_out)
        else:
            self.output_mu = o3.Linear(self.feat_irreps, o3.Irreps((c_out, (0, 1))))
            self.output_logvar = o3.Linear(self.feat_irreps, o3.Irreps((c_out, (0, 1))))



    @torch.no_grad
    def _prep_features(self, graph, eps=1e-8):
        res_data = graph['residue']

        # node features
        X_ca = res_data['x'].float()
        bb = res_data['bb'].float()
        dihedrals = _dihedrals(bb)

        mask = res_data['mlm_mask']

        seq = F.one_hot(res_data['seq'], num_classes=self.n_aa).to(mask.device)
        masked_seq = seq * mask[..., None]

        node_scalars = torch.cat(
            [
                dihedrals,
                mask.float()[..., None],
                masked_seq
            ],
        dim=-1)
        node_vectors = res_data['atom37'] - X_ca[..., None, :]
        node_vectors[..., 4:, :] = node_vectors[..., 4:, :] * mask[..., None, None]
        node_features = torch.cat(
            [node_scalars, node_vectors.view([node_scalars.shape[0], -1])],
            dim=-1
        ).float()

        # edge graph
        res_mask = res_data['res_mask']
        masked_X_ca = X_ca.clone()
        masked_X_ca[~res_mask] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph['residue'].batch)

        # edge features
        edge_features = []

        virtual_Cb = _ideal_virtual_Cb(bb)
        bb = torch.cat([bb, virtual_Cb[..., None, :]], dim=-2)
        src = edge_index[1]
        dst = edge_index[0]

        ## edge distances
        edge_bb_src = bb[src]
        edge_bb_dst = bb[dst]
        edge_bb_dists = torch.linalg.vector_norm(
            edge_bb_src[..., None, :] - edge_bb_dst[..., None, :, :] + eps,
            dim=-1)
        edge_bb_dists = edge_bb_dists.view(edge_index.shape[1], -1, 1)
        edge_rbf = _rbf(edge_bb_dists, D_min=2.0, D_max=22.0, D_count=self.num_rbf, device=edge_index.device)
        edge_rbf = edge_rbf.view(edge_index.shape[1], -1)
        edge_features.append(edge_rbf)
        ## edge rel pos embedding
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=self.num_pos_embed, device=edge_index.device)
        edge_features.append(edge_dist_rel_pos)
        edge_features = torch.cat(edge_features, dim=-1)

        edge_vecs = X_ca[dst] - X_ca[src]
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vecs, normalize=True, normalization='component')

        return node_features, edge_features, edge_sh, edge_index


    def forward(self, graph):
        node_features, edge_features, edge_sh, edge_index = self._prep_features(graph)

        edge_features = self.edge_embed(edge_features)
        node_features = self.node_embed(node_features, edge_features, edge_sh, edge_index)

        for layer in self.tc:
            node_features, edge_features = layer(node_features, edge_features, edge_sh, edge_index)

        out_dict = {}

        if self.scalarize_wrt_frames:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)
            node_vecs = node_features[..., self.c_s:].view(-1, self.c_v, 3)
            node_vecs = rigids[..., None].get_rots().invert_apply(node_vecs)
            num_nodes = node_features.shape[0]
            node_features = torch.cat(
                [node_features[..., :self.c_s], node_vecs.view(num_nodes, -1)],
                dim=-1
            )

        latent_mu = self.output_mu(node_features)
        latent_logvar = self.output_logvar(node_features)

        out_dict['latent_mu'] = latent_mu
        out_dict['latent_logvar'] = latent_logvar
        return out_dict
