from ligbinddiff.data.datasets.featurize.sidechain import _rbf, _positional_embeddings
from ligbinddiff.model.modules.common import ProjectLayer, EdgeUpdate
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import sanitize_edge_index, init_edge_rot_mat
from ligbinddiff.model.modules.equiformer_v2.so3 import SO3_Embedding, SO3_LinearV2
from ligbinddiff.model.modules.equiformer_v2.transformer_block import TransBlockV2


import torch
from torch import nn
from torch_geometric.utils import bipartite_subgraph, sort_edge_index
from torch_cluster import knn_graph


class AtomicSidechainEncoder(nn.Module):
    def __init__(self,
                 atom_lmax_list,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_atoms,
                 mappingReduced_super_atoms,
                 mappingReduced_nodes,
                 node_SO3_rotation,
                 node_SO3_grid,
                 atom_SO3_rotation,
                 atom_SO3_grid,
                 atom_super_SO3_rotation,
                 atom_super_SO3_grid,
                 atom_channels=18+1+5+1,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4,
                 knn_k=30,
                 ):
        super().__init__()
        self.atom_lmax_list = atom_lmax_list
        self.node_lmax_list = node_lmax_list
        self.atom_channels = atom_channels
        self.h_channels = h_channels

        self.atom_SO3_rotation = atom_SO3_rotation
        self.node_SO3_rotation = node_SO3_rotation
        self.atom_super_SO3_rotation = atom_super_SO3_rotation

        self.embed_atoms = SO3_LinearV2(
            in_features=atom_channels,
            out_features=h_channels,
            lmax=max(atom_lmax_list)
        )

        self.atom_radius_interactions = nn.ModuleList(
            [
                TransBlockV2(
                    sphere_channels=h_channels,
                    attn_hidden_channels=h_channels,
                    num_heads=num_heads,
                    attn_alpha_channels=h_channels // 2,
                    attn_value_channels=h_channels // 4,
                    ffn_hidden_channels=h_channels,
                    output_channels=h_channels,
                    lmax_list=atom_lmax_list[0:1],
                    mmax_list=atom_lmax_list[0:1],
                    SO3_rotation=atom_SO3_rotation,
                    SO3_grid=atom_SO3_grid,
                    edge_channels_list=edge_channels_list,
                    mappingReduced=mappingReduced_atoms
                )
                for _ in range(num_layers)
            ]
        )

        self.atomic_radius_edge_update = nn.ModuleList(
            [
                EdgeUpdate(
                    node_lmax_list=atom_lmax_list,
                    edge_channels_list=edge_channels_list,
                    h_channels=h_channels
                )
                for _ in range(num_layers)
            ]
        )

        self.residue_aggregation = TransBlockV2(
            sphere_channels=h_channels,
            attn_hidden_channels=h_channels,
            num_heads=num_heads,
            attn_alpha_channels=h_channels // 2,
            attn_value_channels=h_channels // 4,
            ffn_hidden_channels=h_channels,
            output_channels=h_channels,
            lmax_list=atom_lmax_list[0:1],
            mmax_list=atom_lmax_list[0:1],
            SO3_rotation=atom_SO3_rotation,
            SO3_grid=atom_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_atoms
        )

        self.project_to_node = ProjectLayer(
            in_lmax_list=atom_lmax_list,
            in_channels=h_channels,
            out_lmax_list=node_lmax_list,
            out_channels=h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_atoms,
            super_SO3_rotation=atom_super_SO3_rotation,
            super_SO3_grid=atom_super_SO3_grid
        )

        self.output_mu = TransBlockV2(
            sphere_channels=h_channels,
            attn_hidden_channels=h_channels,
            num_heads=num_heads,
            attn_alpha_channels=h_channels // 2,
            attn_value_channels=h_channels // 4,
            ffn_hidden_channels=h_channels,
            output_channels=h_channels,
            lmax_list=node_lmax_list[0:1],
            mmax_list=node_lmax_list[0:1],
            SO3_rotation=node_SO3_rotation,
            SO3_grid=node_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_nodes
        )

        self.output_logvar = TransBlockV2(
            sphere_channels=h_channels,
            attn_hidden_channels=h_channels,
            num_heads=num_heads,
            attn_alpha_channels=h_channels // 2,
            attn_value_channels=h_channels // 4,
            ffn_hidden_channels=h_channels,
            output_channels=h_channels,
            lmax_list=node_lmax_list[0:1],
            mmax_list=node_lmax_list[0:1],
            SO3_rotation=node_SO3_rotation,
            SO3_grid=node_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_nodes
        )

        self.k = knn_k


    def forward(self, graph, residue_noising_select=None):
        ## prep features
        num_atoms = graph['atomic'].num_nodes
        num_res = graph['residue'].num_nodes
        device = graph['atomic']['x'].device
        if residue_noising_select is None:
            residue_noising_select = torch.ones(
                num_res,
                device=device
            ).bool()

        atom_features = SO3_Embedding(
            num_atoms,
            lmax_list=self.atom_lmax_list,
            num_channels=self.atom_channels,
            device=graph['atomic']['x'].device,
            dtype=torch.float
        )
        atom_features.set_invariant_features(graph['atomic'].atom_embedding)
        atom_features = self.embed_atoms(atom_features)

        # update atomic embeddings using atomic radius-based graph
        radius_edge_index = graph['atomic', 'radius_interact', 'atomic'].edge_index
        with torch.no_grad():
            # i'm pretty sure this is already no grad but either way
            if not (residue_noising_select.all() or not residue_noising_select.any()):
                atom_to_res_map = graph['atomic'].atom_to_residue_map
                # noising more than half the residues
                if residue_noising_select.sum() > len(residue_noising_select) // 2:
                    unnoised_residx = torch.arange(num_res, device=device)[~residue_noising_select]
                    unnoised_atoms_select = (atom_to_res_map[:, None] == unnoised_residx[None, :]).any(dim=-1)
                    noised_atoms_select = ~unnoised_atoms_select
                else:
                    noised_residx = torch.arange(num_res, device=device)[residue_noising_select]
                    noised_atoms_select = (atom_to_res_map[:, None] == noised_residx[None, :]).any(dim=-1)
                    unnoised_atoms_select = ~noised_atoms_select

                all_atoms = torch.arange(num_atoms, device=device)
                unnoised_atoms = all_atoms[unnoised_atoms_select]
                noised_atoms = all_atoms[noised_atoms_select]
                unnoised_edge_index, _ = bipartite_subgraph((unnoised_atoms, unnoised_atoms), radius_edge_index)
                noised_edge_index, _ = bipartite_subgraph((all_atoms, noised_atoms), radius_edge_index)
                radius_edge_index = sort_edge_index(
                    torch.cat([unnoised_edge_index, noised_edge_index], dim=-1),
                    sort_by_row=False
                )

        atom_x = graph['atomic'].x
        edge_distance_vec = atom_x[radius_edge_index[1]] - atom_x[radius_edge_index[0]]
        edge_dist = torch.linalg.vector_norm(edge_distance_vec, dim=-1)
        radius_edge_index, edge_distance_vec, edge_dist = sanitize_edge_index(radius_edge_index, edge_distance_vec)

        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for rot in self.atom_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device, D_count=32)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf], dim=-1)

        for node_layer, edge_layer in zip(self.atom_radius_interactions, self.atomic_radius_edge_update):
            atom_features = node_layer(atom_features, edge_features, radius_edge_index)
            edge_features = edge_layer(atom_features, edge_features, radius_edge_index)

        # aggregate atom features to residue level
        agg_edge_index = graph['atomic', 'to_ca', 'atomic'].edge_index
        # torch.set_printoptions(threshold=100000)
        # print(agg_edge_index)

        atom_x = graph['atomic'].x
        edge_distance_vec = atom_x[agg_edge_index[1]] - atom_x[agg_edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for rot in self.atom_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        edge_dist = torch.linalg.vector_norm(edge_distance_vec, dim=-1)
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device, D_count=32)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf], dim=-1)

        atom_agg_features = self.residue_aggregation(atom_features, edge_features, agg_edge_index)
        # take the CA embedding
        atom_agg_bb_embedding = atom_agg_features.embedding[graph['atomic'].backbone_atoms_select]
        embedding_CA = atom_agg_bb_embedding.view(
            -1, 4,
            atom_agg_features.num_coefficients, atom_agg_features.num_channels
        )[:, 1]

        res_features = SO3_Embedding(
            num_res,
            lmax_list=self.atom_lmax_list,
            num_channels=self.h_channels,
            device=atom_features.device,
            dtype=torch.float
        )
        x_mask = graph['residue']['x_mask']
        res_features.embedding[~x_mask] = embedding_CA

        X_ca = graph['residue']['x']
        res_edge_index = graph['residue', 'knn', 'residue'].edge_index
        edge_distance_vec = X_ca[res_edge_index[1]] - X_ca[res_edge_index[0]]
        edge_dist = torch.linalg.vector_norm(edge_distance_vec, dim=-1)
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for rot in self.node_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        for rot in self.atom_super_SO3_rotation:
            rot.set_wigner(edge_rot_mat)
        edge_dist_rbf = _rbf(edge_dist, device=edge_dist.device, D_count=16)  # edge_channels_list
        edge_dist_rel_pos = _positional_embeddings(res_edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)

        res_features = self.project_to_node(res_features, edge_features, res_edge_index)

        latent_mu = self.output_mu(res_features, edge_features, res_edge_index)
        latent_logvar = self.output_logvar(res_features, edge_features, res_edge_index)

        out_dict = {}
        out_dict['latent_mu'] = latent_mu
        out_dict['latent_logvar'] = latent_logvar

        return out_dict
