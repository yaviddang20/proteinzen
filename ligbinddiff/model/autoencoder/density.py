""" Autoencoder module """

import torch
from torch import nn
from torch_cluster import knn_graph

from ligbinddiff.data.openfold.residue_constants import atom_types, van_der_waals_radius

from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf
from ligbinddiff.model.modules.common import ProjectLayer

from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid
from ligbinddiff.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from ligbinddiff.model.modules.equiformer_v2.transformer_block import TransBlockV2
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.so3_embedding import density_to_so3
from ligbinddiff.utils.fiber import nl_to_fiber
from ligbinddiff.utils.zernike import ZernikeTransform
from ligbinddiff.data.datasets.featurize.sidechain import _ideal_virtual_Cb

from ligbinddiff.model.modules.layers.edge.sitewise import EdgeTransition
from ligbinddiff.model.modules.layers.node.mpnn import IPMP

from ligbinddiff.model.modules.common import EdgeUpdate



class DensityEncoder(nn.Module):
    def __init__(self,
                 density_nmax=5,
                 node_lmax_list=[1],
                 c_s=[32, 16, 8],
                 c_z=128,
                 c_output=32,
                 num_heads=4,
                 num_qk=4,
                 num_v=6,
                 num_layers=4,
                 num_rbf=16,
                 num_pos_embed=16,
                 zernike_scale=8,
                 k=30,
                 ):
        super().__init__()
        self.atom37_channel_mask = torch.tensor(
            [
                [atom[0] == element for atom in atom_types]
                for element in van_der_waals_radius.keys()
            ]
        ).float()
        self.atom37_channels = self.atom37_channel_mask.shape[0]
        self.density_lmax_list = list(range(density_nmax, -1, -1))
        self.node_lmax_list = node_lmax_list
        self.num_rbf = num_rbf
        self.num_pos_embed = num_pos_embed
        self.k = k

        ## setup SO3 helpers
        self.super_lmax_list = [max(l1, l2) for l1, l2 in zip(self.density_lmax_list, node_lmax_list)]
        self.node_SO3_rotation = nn.ModuleList()
        self.super_SO3_rotation = nn.ModuleList()
        for lmax in node_lmax_list:
            self.node_SO3_rotation.append(
                SO3_Rotation(lmax)
            )
        for lmax in self.super_lmax_list:
            self.super_SO3_rotation.append(
                SO3_Rotation(lmax)
            )

        self.super_SO3_grid = nn.ModuleList()
        self.node_SO3_grid = nn.ModuleList()
        for l in range(max(self.super_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.super_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            self.super_SO3_grid.append(SO3_m_grid)
        for l in range(max(self.node_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.node_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            self.node_SO3_grid.append(SO3_m_grid)

        self.mappingReduced_nodes = CoefficientMappingModule(node_lmax_list, node_lmax_list)
        self.mappingReduced_super = CoefficientMappingModule(self.super_lmax_list, self.super_lmax_list)

        self.zernike = ZernikeTransform(density_nmax, r_scale=zernike_scale)

        self.edge_channels_list = [c_z, c_z, c_z]
        self.embed_node = ProjectLayer(
            in_lmax_list=self.density_lmax_list,
            in_channels=self.atom37_channels,
            out_lmax_list=node_lmax_list,
            out_channels=c_s[0],
            edge_channels_list=self.edge_channels_list,
            mappingReduced_super=self.mappingReduced_super,
            super_SO3_rotation=self.super_SO3_rotation,
            super_SO3_grid=self.super_SO3_grid
        )
        bb_atoms_per_res = 5
        num_edge_in = (
            num_rbf * (bb_atoms_per_res ** 2)  # bb x bb distances
            + num_pos_embed  # rel pos embed
        )
        self.embed_edge = nn.Sequential(
            nn.Linear(num_edge_in, 2*c_z),
            nn.ReLU(),
            nn.Linear(2*c_z, 2*c_z),
            nn.ReLU(),
            nn.Linear(2*c_z, c_z),
            nn.LayerNorm(c_z),
        )

        self.transformer = nn.ModuleList(
            [
                TransBlockV2(
                    sphere_channels=c_s_i,
                    attn_hidden_channels=c_z,
                    num_heads=num_heads,
                    attn_alpha_channels=num_qk,
                    attn_value_channels=num_v,
                    ffn_hidden_channels=c_s_ip1*2,
                    output_channels=c_s_ip1,
                    lmax_list=node_lmax_list[0:1],
                    mmax_list=node_lmax_list[0:1],
                    SO3_rotation=self.node_SO3_rotation,
                    SO3_grid=self.node_SO3_grid,
                    edge_channels_list=self.edge_channels_list,
                    mappingReduced=self.mappingReduced_nodes
                )
                for c_s_i, c_s_ip1 in zip(c_s[:-1], c_s[1:])
            ]
        )

        self.edge_update = nn.ModuleList(
            [
                EdgeUpdate(
                    node_lmax_list=node_lmax_list,
                    edge_channels_list=self.edge_channels_list,
                    h_channels=c_s_ip1
                )
                for c_s_ip1 in c_s[1:]
            ]
        )

        self.node_norm = NormSO3(
            lmax_list=node_lmax_list,
            num_channels=c_s[-1]
        )
        # self.scalarize = ProjectLayer(
        #     in_lmax_list=node_lmax_list,
        #     in_channels=c_s,
        #     out_lmax_list=[0],
        #     out_channels=c_output,
        #     edge_channels_list=self.edge_channels_list,
        #     mappingReduced_super=self.mappingReduced_nodes,
        #     super_SO3_rotation=self.node_SO3_rotation,
        #     super_SO3_grid=self.node_SO3_grid
        # )
        # self.ln = nn.LayerNorm(c_output)
        self.frame_feats = nn.Linear(
            c_s[-1] * 5,c_output
        )


        self.latent_mu = nn.Linear(c_output, c_output)
        self.latent_logvar = nn.Linear(c_output, c_output)


    @torch.no_grad()
    def _prep_features(self, graph, eps=1e-8):
        res_data = graph['residue']
        X_ca = res_data['x'].float()
        noising_mask = res_data['noising_mask']
        bb = res_data['bb'].float()
        virtual_Cb = _ideal_virtual_Cb(bb)

        # node features
        atom37 = res_data['atom37']
        atom37_mask = res_data['atom37_mask'].clone()
        # mask atoms which are noised
        # we have to do CB separately
        # atom37_mask[:, 3] = atom37_mask[:, 3] * (~noising_mask)
        # atom37_mask[:, 5:] = atom37_mask[:, 5:] * (~noising_mask)[..., None]

        density_dict = self.zernike.forward_transform(
            points=atom37,
            center=virtual_Cb,
            points_mask=atom37_mask,
            point_value=self.atom37_channel_mask.to(atom37.device))
        # TODO: make this less hacky
        density_embedding = density_to_so3(nl_to_fiber(density_dict), num_channels=self.atom37_channels)
        node_features = density_embedding
        node_features.embedding = node_features.embedding * res_data['res_mask'][..., None, None]
        node_features.set_embedding(node_features.embedding.float())
        node_features.dtype = node_features.embedding.dtype

        # edge graph
        res_mask = res_data['res_mask']
        masked_X_ca = X_ca.clone()
        masked_X_ca[~res_mask] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph['residue'].batch)
        src = edge_index[1]
        dst = edge_index[0]

        # filter for bad edges
        edge_distance_vec = X_ca[dst] - X_ca[src]
        edge_dist = torch.linalg.vector_norm(edge_distance_vec, dim=-1)
        # TODO: figure out why edges are bad
        edge_select = edge_dist > 0.0001
        if not edge_select.all():
            edge_index = edge_index[:, edge_select]
            edge_distance_vec = edge_distance_vec[edge_select]
            src = edge_index[1]
            dst = edge_index[0]

        ## edge rot mats
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)

        # edge features
        bb = torch.cat([bb, virtual_Cb[..., None, :]], dim=-2)

        ## edge distances
        edge_bb_src = bb[src]
        edge_bb_dst = bb[dst]
        edge_bb_dists = torch.linalg.vector_norm(
            edge_bb_src[..., None, :] - edge_bb_dst[..., None, :, :] + eps,
            dim=-1)
        edge_bb_dists = edge_bb_dists.view(edge_index.shape[1], -1, 1)
        edge_rbf = _rbf(edge_bb_dists, D_min=2.0, D_max=22.0, D_count=self.num_rbf, device=edge_index.device)
        edge_rbf = edge_rbf.view(edge_index.shape[1], -1)
        ## edge rel pos embedding
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=self.num_pos_embed, device=edge_index.device)

        edge_features = torch.cat([edge_rbf, edge_dist_rel_pos], dim=-1)

        return node_features, edge_features, edge_index, edge_rot_mat

    def forward(self, graph):
        res_data = graph['residue']
        res_mask = res_data['res_mask']
        node_features, edge_features, edge_index, edge_rot_mat = self._prep_features(graph)

        for rot in self.node_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        for rot in self.super_SO3_rotation:
            rot.set_wigner(edge_rot_mat)

        edge_features = self.embed_edge(edge_features)
        node_features = self.embed_node(
            node_features,
            edge_features,
            edge_index
        )
        node_features.embedding = node_features.embedding * res_mask[..., None, None]

        for node_layer, edge_layer in zip(self.transformer, self.edge_update):
            node_features = node_layer(node_features, edge_features, edge_index)
            node_features.embedding = node_features.embedding * res_mask[..., None, None]
            edge_features = edge_layer(node_features, edge_features, edge_index)

        node_features.embedding = self.node_norm(node_features.embedding)
        node_scalars = self._scalarize(
            node_features,
            ru.Rigid.from_tensor_7(res_data['rigids_0'])
        )
        node_scalars = node_scalars * res_mask[..., None]
        # node_scalars = self.scalarize(node_features, edge_features, edge_index)
        # node_scalars = self.ln(node_scalars.get_invariant_features(flat=True))

        latent_mu = self.latent_mu(node_scalars)
        latent_logvar = self.latent_logvar(node_scalars)

        out_dict = {}

        out_dict['latent_mu'] = latent_mu
        out_dict['latent_logvar'] = latent_logvar
        return out_dict

    def _scalarize(self, node_features: SO3_Embedding, rigids: ru.Rigid, eps=1e-8):
        node_scalars = node_features.embedding[..., 0, :].squeeze(-1)
        node_vecs = node_features.embedding[..., 1:4, :].transpose(-1, -2)
        node_vec_mags = torch.linalg.vector_norm(node_vecs + eps, dim=-1)
        node_vecs_local_frame = rigids[..., None].get_rots().apply(node_vecs)

        frame_feats = torch.cat(
            [node_scalars, node_vec_mags, node_vecs_local_frame.flatten(-2, -1)],
            dim=-1
        )
        return self.frame_feats(frame_feats)


