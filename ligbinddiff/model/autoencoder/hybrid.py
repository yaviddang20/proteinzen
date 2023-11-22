import torch
from torch import nn
from torch_geometric.utils import bipartite_subgraph, sort_edge_index
from torch_cluster import knn_graph

from ligbinddiff.model.modules.common import ProjectLayer, EdgeUpdate
from ligbinddiff.model.modules.equiformer_v2.layer_norm import MultiResEquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from ligbinddiff.model.modules.equiformer_v2.edge_rot_mat import sanitize_edge_index, init_edge_rot_mat
from ligbinddiff.model.modules.equiformer_v2.so3 import SO3_Embedding, SO3_LinearV2
from ligbinddiff.model.modules.equiformer_v2.transformer_block import TransBlockV2
from ligbinddiff.data.datasets.featurize.sidechain import _orientations, _ideal_virtual_Cb, _dihedrals

from ligbinddiff.model.modules.layers.node.mpnn import IPMP
from ligbinddiff.model.modules.openfold.frames import  StructureModuleTransition
from ligbinddiff.model.modules.layers.edge.sitewise import EdgeTransition
from ligbinddiff.data.datasets.featurize.sidechain import _dihedrals
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.framediff.all_atom import compute_all_atom14
from ligbinddiff.utils.atom_reps import restype_1to3, atom91_start_end
from ligbinddiff.data.openfold.residue_constants import restypes


class Atom2Residue(nn.Module):
    def __init__(self,
                 atom_lmax_list,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_atom,
                 atom_SO3_rotation,
                 atom_SO3_grid,
                 mappingReduced_node,
                 node_SO3_rotation,
                 node_SO3_grid,
                 num_heads=8,
                 atom_h_channels=16,
                 node_h_channels=32,
                 ):
        super().__init__()
        self.atom_h_channels = atom_h_channels
        self.node_h_channels = node_h_channels
        self.atom_lmax_list = atom_lmax_list
        self.node_lmax_list = node_lmax_list

        self.agg = TransBlockV2(
            sphere_channels=atom_h_channels,
            attn_hidden_channels=atom_h_channels,
            num_heads=num_heads,
            attn_alpha_channels=atom_h_channels // 2,
            attn_value_channels=atom_h_channels // 2,
            ffn_hidden_channels=atom_h_channels,
            output_channels=atom_h_channels,
            lmax_list=atom_lmax_list,
            mmax_list=atom_lmax_list,
            SO3_rotation=atom_SO3_rotation,
            SO3_grid=atom_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_atom,
            alpha_drop=0.1,
            proj_drop=0.1
        )
        # self.ca_proj = ProjectLayer(
        #     in_channels=node_h_channels + atom_h_channels,
        #     in_lmax_list=atom_lmax_list,
        #     out_channels=node_h_channels,
        #     out_lmax_list=node_lmax_list,
        #     edge_channels_list=edge_channels_list,
        #     mappingReduced_super=mappingReduced_node,
        #     super_SO3_rotation=node_SO3_rotation,
        #     super_SO3_grid=node_SO3_grid
        # )
        self.ca_proj = SO3_LinearV2(
            node_h_channels + atom_h_channels,
            node_h_channels,
            lmax=node_lmax_list[0]
        )


    def forward(self, 
                atom_features, 
                a2r_edge_features, 
                res_features, 
                graph):
        # this assumes res_features has a lower resolution than atom_features
        res_features = res_features.to_resolutions(atom_features.lmax_list, atom_features.lmax_list)

        num_res = graph['residue'].num_nodes
        agg_edge_index = graph['atomic', 'to_ca', 'atomic'].edge_index

        atom_agg_features = self.agg(atom_features, a2r_edge_features, agg_edge_index)
        # take the CA embedding
        atom_agg_bb_embedding = atom_agg_features.embedding[graph['atomic'].backbone_atoms_select]
        embedding_CA = atom_agg_bb_embedding.view(
            -1, 4,
            atom_agg_features.num_coefficients, self.atom_h_channels 
        )[:, 1]

        container = torch.zeros(
            (num_res, atom_agg_features.num_coefficients, self.atom_h_channels), 
            device=res_features.device)
        container[~graph['residue'].x_mask] = embedding_CA

        fuse_embedding = torch.cat(
            [container, res_features.embedding], dim=-1
        )
        update_res_features = self.ca_proj(
            SO3_Embedding(
                0,
                self.atom_lmax_list,
                fuse_embedding.shape[-1],
                None,
                None,
                embedding=fuse_embedding
            ),
        )
        return update_res_features 


class Residue2Atom(nn.Module):
    def __init__(self,
                 atom_lmax_list,
                 node_lmax_list,
                 edge_channels_list,
                 mappingReduced_atom,
                 atom_SO3_rotation,
                 atom_SO3_grid,
                 mappingReduced_node,
                 node_SO3_rotation,
                 node_SO3_grid,
                 num_heads=8,
                 atom_h_channels=16,
                 node_h_channels=32,
                 ):
        super().__init__()
        self.atom_lmax_list = atom_lmax_list
        self.node_lmax_list = node_lmax_list
        self.atom_h_channels = atom_h_channels
        self.node_h_channels = node_h_channels

        # self.ca_proj = ProjectLayer(
        #     in_channels=node_h_channels + atom_h_channels,
        #     in_lmax_list=atom_lmax_list,
        #     out_channels=atom_h_channels,
        #     out_lmax_list=atom_lmax_list,
        #     edge_channels_list=edge_channels_list,
        #     mappingReduced_super=mappingReduced_node,
        #     super_SO3_rotation=node_SO3_rotation,
        #     super_SO3_grid=node_SO3_grid
        # )
        self.ca_proj = SO3_LinearV2(
            node_h_channels + atom_h_channels,
            atom_h_channels,
            lmax=node_lmax_list[0]
        )

        self.scatter = TransBlockV2(
            sphere_channels=atom_h_channels,
            attn_hidden_channels=atom_h_channels,
            num_heads=num_heads,
            attn_alpha_channels=atom_h_channels // 2,
            attn_value_channels=atom_h_channels // 2,
            ffn_hidden_channels=atom_h_channels,
            output_channels=atom_h_channels,
            lmax_list=atom_lmax_list,
            mmax_list=atom_lmax_list,
            SO3_rotation=atom_SO3_rotation,
            SO3_grid=atom_SO3_grid,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced_atom,
            alpha_drop=0.1,
            proj_drop=0.1
        )


    def forward(self, 
                atom_features: SO3_Embedding, 
                edge_features, 
                res_features: SO3_Embedding,
                graph):
        # this assumes res_features has a lower resolution than atom_features
        res_features = res_features.to_resolutions(atom_features.lmax_list, atom_features.lmax_list)

        agg_edge_index = graph['atomic', 'to_ca', 'atomic'].edge_index
        scatter_edge_index = torch.stack([agg_edge_index[1], agg_edge_index[0]], dim=0)
        x_mask = graph['residue'].x_mask

        atom_idx = torch.arange(graph['atomic'].num_nodes, device=x_mask.device)
        ca_idx_select = atom_idx[graph['atomic'].backbone_atoms_select].view(-1, 4)[:, 1].view(-1)

        # take the CA embedding
        ca_embedding = atom_features.embedding[ca_idx_select]

        # proj embedding with res-level embedding
        fuse_embedding = torch.cat(
            [ca_embedding, res_features.embedding[~x_mask]], dim=-1
        )
        ca_impute = self.ca_proj(
            SO3_Embedding(
                0,
                self.atom_lmax_list,
                fuse_embedding.shape[-1],
                None,
                None,
                embedding=fuse_embedding
            )
        )

        input_features = atom_features.clone()
        input_features.embedding[ca_idx_select] = ca_impute.embedding

        update_atom_features = self.scatter(input_features, edge_features, scatter_edge_index)
        return update_atom_features

class Res2Res(nn.Module):
    def __init__(self, c_s, c_z, c_hidden):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden

        self.ipmp = IPMP(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
        )
        self.ln = nn.LayerNorm(c_s)
        self.node_transition = StructureModuleTransition(
            c_s
        ) 
        self.edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        ) 
    
    def forward(self, 
                node_features: SO3_Embedding, 
                edge_features, 
                edge_index, 
                rigids,
                node_mask):
        node_scalars = node_features.get_invariant_features(flat=True)
        node_update = self.ipmp(
            s=node_scalars, 
            z=edge_features, 
            edge_index=edge_index, 
            r=rigids)
        node_scalars = self.ln(node_scalars + node_update * node_mask[..., None])
        node_scalars = self.node_transition(node_scalars) * node_mask[..., None]
        edge_features = self.edge_transition(node_scalars, edge_features, edge_index)

        updated_node_features = node_features.clone() 
        updated_node_features.set_invariant_features(node_scalars)

        return updated_node_features, edge_features


class MultiscaleSidechainEncoder(nn.Module):
    def __init__(self,
                 node_lmax_list,
                 latent_lmax_list,
                 edge_channels_list,
                 mappingReduced_node,
                 node_SO3_rotation,
                 node_SO3_grid,
                 mappingReduced_super_latent,
                 latent_super_SO3_rotation,
                 latent_super_SO3_grid,
                 atom_channels=18+1+5+1,
                 atom_lmax_list=[1],
                 num_heads=8,
                 atom_h_channels=16,
                 node_h_channels=32,
                 num_layers=4,
                 knn_k=30,
                 ):
        super().__init__()
        self.node_lmax_list = node_lmax_list
        self.latent_lmax_list = latent_lmax_list
        self.atom_lmax_list = node_lmax_list
        self.atom_channels = atom_channels
        self.atom_h_channels = atom_h_channels
        self.node_h_channels = node_h_channels
        self.num_layers = num_layers

        self.SO3_rotation = node_SO3_rotation
        self.latent_super_SO3_rotation = latent_super_SO3_rotation

        self.embed_atoms = ProjectLayer(
            in_lmax_list=atom_lmax_list,
            in_channels=atom_channels,
            out_lmax_list=node_lmax_list,
            out_channels=atom_h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_node,
            super_SO3_rotation=node_SO3_rotation,
            super_SO3_grid=node_SO3_grid
        )

        self.atom2atom = nn.ModuleList(
            [
                TransBlockV2(
                    sphere_channels=atom_h_channels,
                    attn_hidden_channels=atom_h_channels,
                    num_heads=num_heads,
                    attn_alpha_channels=atom_h_channels // 2,
                    attn_value_channels=atom_h_channels // 2,
                    ffn_hidden_channels=atom_h_channels,
                    output_channels=atom_h_channels,
                    lmax_list=node_lmax_list[0:1],
                    mmax_list=node_lmax_list[0:1],
                    SO3_rotation=node_SO3_rotation,
                    SO3_grid=node_SO3_grid,
                    edge_channels_list=edge_channels_list,
                    mappingReduced=mappingReduced_node,
                    alpha_drop=0.1,
                    proj_drop=0.1
                )
                for _ in range(num_layers)
            ]
        )

        self.atom2res = nn.ModuleList(
            [
                Atom2Residue(
                    atom_lmax_list=atom_lmax_list,
                    node_lmax_list=node_lmax_list,
                    edge_channels_list=edge_channels_list,
                    mappingReduced_node=mappingReduced_node,
                    node_SO3_rotation=node_SO3_rotation,
                    node_SO3_grid=node_SO3_grid,
                    mappingReduced_atom=mappingReduced_node,
                    atom_SO3_rotation=node_SO3_rotation,
                    atom_SO3_grid=node_SO3_grid,
                    num_heads=num_heads,
                    atom_h_channels=atom_h_channels,
                    node_h_channels=node_h_channels
                )
                for _ in range(num_layers)
            ]
        )

        self.res2res = nn.ModuleList(
            [
                Res2Res(
                    c_s=node_h_channels,
                    c_z=edge_channels_list[0],
                    c_hidden=node_h_channels,
                )
                for _ in range(num_layers)
            ]
        )

        self.res2atom = nn.ModuleList(
            [
                Residue2Atom(
                    atom_lmax_list=atom_lmax_list,
                    node_lmax_list=node_lmax_list,
                    edge_channels_list=edge_channels_list,
                    mappingReduced_node=mappingReduced_node,
                    node_SO3_rotation=node_SO3_rotation,
                    node_SO3_grid=node_SO3_grid,
                    mappingReduced_atom=mappingReduced_node,
                    atom_SO3_rotation=node_SO3_rotation,
                    atom_SO3_grid=node_SO3_grid,
                    num_heads=num_heads,
                    atom_h_channels=atom_h_channels,
                    node_h_channels=node_h_channels
                )
                for _ in range(num_layers-1)
            ]
        )
        self.res_ln = NormSO3(
            lmax_list=node_lmax_list,
            num_channels=node_h_channels
        )

        self.mu_ln = NormSO3(
            lmax_list=latent_lmax_list,
            num_channels=node_h_channels
        )
        self.output_mu = ProjectLayer(
            in_lmax_list=node_lmax_list,
            in_channels=node_h_channels,
            out_lmax_list=latent_lmax_list,
            out_channels=node_h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_latent,
            super_SO3_rotation=latent_super_SO3_rotation,
            super_SO3_grid=latent_super_SO3_grid
        )

        self.logvar_ln = NormSO3(
            lmax_list=latent_lmax_list,
            num_channels=node_h_channels
        )
        self.output_logvar = ProjectLayer(
            in_lmax_list=node_lmax_list,
            in_channels=node_h_channels,
            out_lmax_list=latent_lmax_list,
            out_channels=node_h_channels,
            edge_channels_list=edge_channels_list,
            mappingReduced_super=mappingReduced_super_latent,
            super_SO3_rotation=latent_super_SO3_rotation,
            super_SO3_grid=latent_super_SO3_grid
        )

        self.k = knn_k
    
    
    def clean_radius_edge_index(self, graph, residue_noising_select=None):
        num_atoms = graph['atomic'].num_nodes
        num_res = graph['residue'].num_nodes
        device = graph['atomic']['x'].device
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
        return radius_edge_index
    
    def gen_res_features(self, graph):
        num_nodes = graph['residue'].num_nodes
        X_ca = graph['residue']['x']
        bb = graph['residue']['bb']
        orientations = _orientations(X_ca)
        virtual_Cb = _ideal_virtual_Cb(bb) - X_ca
        dihedrals = _dihedrals(bb)
        bb_rel = bb - X_ca.unsqueeze(-2)
        l1 = torch.nan_to_num(
                torch.cat([
                    bb_rel,  # 4
                    orientations,  # 2
                    virtual_Cb.unsqueeze(-2)
                ], dim=-2) #1
            )  # total 7
        res_features = SO3_Embedding(
            num_nodes,
            lmax_list=self.node_lmax_list,
            num_channels=self.node_h_channels,
            device=graph['atomic']['x'].device,
            dtype=torch.float
        )
        res_features.embedding[..., 0, :dihedrals.shape[-1]] = dihedrals
        res_features.embedding[..., 1:4, :l1.shape[-2]] = l1.transpose(-1, -2)
        return res_features

    def gen_a2a_edge_features(self, graph, radius_edge_index):
        atom_x = graph['atomic'].x
        edge_distance_vec = atom_x[radius_edge_index[1]] - atom_x[radius_edge_index[0]]
        a2a_edge_dist = torch.linalg.vector_norm(edge_distance_vec, dim=-1)
        a2a_edge_index, a2a_edge_distance_vec, a2a_edge_dist = sanitize_edge_index(radius_edge_index, edge_distance_vec)

        filter = a2a_edge_dist > 0.001
        a2a_edge_index = a2a_edge_index[:, filter]
        a2a_edge_distance_vec = a2a_edge_distance_vec[filter]
        a2a_edge_dist = a2a_edge_dist[filter]

        a2a_edge_rot_mat = init_edge_rot_mat(a2a_edge_distance_vec)
        a2a_edge_dist_rbf = _rbf(a2a_edge_dist, device=a2a_edge_dist.device, D_count=32)  # edge_channels_list
        a2a_edge_features = torch.cat([a2a_edge_dist_rbf], dim=-1)
        return a2a_edge_index, a2a_edge_features, a2a_edge_rot_mat

    def gen_a2r_edge_features(self, graph):
        # aggregate atom features to residue level
        a2r_edge_index = graph['atomic', 'to_ca', 'atomic'].edge_index
        atom_x = graph['atomic'].x
        a2r_edge_distance_vec = atom_x[a2r_edge_index[1]] - atom_x[a2r_edge_index[0]]
        a2r_edge_dist = torch.linalg.vector_norm(a2r_edge_distance_vec, dim=-1)

        filter = a2r_edge_dist > 0.001
        a2r_edge_index = a2r_edge_index[:, filter]
        a2r_edge_distance_vec = a2r_edge_distance_vec[filter]
        a2r_edge_dist = a2r_edge_dist[filter]

        a2r_edge_rot_mat = init_edge_rot_mat(a2r_edge_distance_vec)
        a2r_edge_dist_rbf = _rbf(a2r_edge_dist, device=a2r_edge_dist.device, D_count=32)  # edge_channels_list
        a2r_edge_features = torch.cat([a2r_edge_dist_rbf], dim=-1)
        return a2r_edge_index, a2r_edge_features, a2r_edge_rot_mat

    def gen_r2a_edge_features(self, graph):
        # aggregate atom features to residue level
        a2r_edge_index = graph['atomic', 'to_ca', 'atomic'].edge_index
        r2a_edge_index = torch.stack([a2r_edge_index[1], a2r_edge_index[0]], dim=0)
        atom_x = graph['atomic'].x
        r2a_edge_distance_vec = atom_x[r2a_edge_index[1]] - atom_x[r2a_edge_index[0]]
        r2a_edge_dist = torch.linalg.vector_norm(r2a_edge_distance_vec, dim=-1)

        filter = r2a_edge_dist > 0.001
        r2a_edge_index = r2a_edge_index[:, filter]
        r2a_edge_distance_vec = r2a_edge_distance_vec[filter]
        r2a_edge_dist = r2a_edge_dist[filter]

        r2a_edge_rot_mat = init_edge_rot_mat(r2a_edge_distance_vec)
        r2a_edge_dist_rbf = _rbf(r2a_edge_dist, device=r2a_edge_dist.device, D_count=32)  # edge_channels_list
        r2a_edge_features = torch.cat([r2a_edge_dist_rbf], dim=-1)
        return r2a_edge_index, r2a_edge_features, r2a_edge_rot_mat
        
    def gen_r2r_edge_features(self, graph, D_scale=10):
        X_ca = graph['residue']['x']
        res_edge_index = graph['residue', 'knn', 'residue'].edge_index
        edge_distance_vec = X_ca[res_edge_index[1]] - X_ca[res_edge_index[0]]
        edge_dist = torch.linalg.vector_norm(edge_distance_vec, dim=-1)

        filter = edge_dist > 0.001
        res_edge_index = res_edge_index[:, filter]
        edge_distance_vec = edge_distance_vec[filter]
        edge_dist = edge_dist[filter]

        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        edge_dist_rbf = _rbf(edge_dist / D_scale, device=edge_dist.device, D_count=16)  # edge_channels_list
        edge_dist_rel_pos = _edge_positional_embeddings(res_edge_index, num_embeddings=16, device=edge_dist.device)  # edge_channels_list
        edge_features = torch.cat([edge_dist_rbf, edge_dist_rel_pos], dim=-1)
        return res_edge_index, edge_features, edge_rot_mat

    def forward(self, graph, residue_noising_select=None):
        ## prep features
        num_atoms = graph['atomic'].num_nodes
        num_res = graph['residue'].num_nodes
        device = graph['atomic']['x'].device
        rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)
        x_mask = graph['residue'].x_mask
        if residue_noising_select is None:
            residue_noising_select = torch.ones(
                num_res,
                device=device
            ).bool()


        res_features = self.gen_res_features(graph) 

        radius_edge_index = self.clean_radius_edge_index(graph, residue_noising_select)
        a2a_edge_index, a2a_edge_features, a2a_edge_rot_mat = self.gen_a2a_edge_features(graph, radius_edge_index)
        a2r_edge_index, a2r_edge_features, a2r_edge_rot_mat = self.gen_a2r_edge_features(graph)
        r2r_edge_index, r2r_edge_features, r2r_edge_rot_mat = self.gen_r2r_edge_features(graph)
        r2a_edge_index, r2a_edge_features, r2a_edge_rot_mat = self.gen_r2a_edge_features(graph)

        for rot in self.SO3_rotation:
            rot.set_wigner(a2a_edge_rot_mat)
        atom_features = SO3_Embedding(
            num_atoms,
            lmax_list=self.atom_lmax_list,
            num_channels=self.atom_channels,
            device=graph['atomic']['x'].device,
            dtype=torch.float
        )
        atom_features.set_invariant_features(graph['atomic'].atom_embedding)
        atom_features = self.embed_atoms(atom_features, a2a_edge_features, a2a_edge_index)

        for i in range(self.num_layers):
            for rot in self.SO3_rotation:
                rot.set_wigner(a2a_edge_rot_mat)
            atom_features = self.atom2atom[i](atom_features, a2a_edge_features, a2a_edge_index)
            for rot in self.SO3_rotation:
                rot.set_wigner(a2r_edge_rot_mat)
            res_features = self.atom2res[i](atom_features, a2r_edge_features, res_features, graph)

            for rot in self.SO3_rotation:
                rot.set_wigner(r2r_edge_rot_mat)
            
            res_features, r2r_edge_features = self.res2res[i](res_features, r2r_edge_features, r2r_edge_index, rigids, ~x_mask)

            if i < self.num_layers - 1:
                for rot in self.SO3_rotation:
                    rot.set_wigner(r2a_edge_rot_mat)
                atom_features = self.res2atom[i](atom_features, r2a_edge_features, res_features, graph)


        for rot in self.latent_super_SO3_rotation:
            rot.set_wigner(r2r_edge_rot_mat)
        res_features.embedding = self.res_ln(res_features.embedding)
        latent_mu = self.output_mu(res_features, r2r_edge_features, r2r_edge_index)
        latent_mu.embedding = self.mu_ln(latent_mu.embedding)
        latent_logvar = self.output_logvar(res_features, r2r_edge_features, r2r_edge_index)
        latent_logvar.embedding = self.logvar_ln(latent_logvar.embedding)

        out_dict = {}
        out_dict['latent_mu'] = latent_mu.get_invariant_features(flat=True)
        out_dict['latent_logvar'] = latent_logvar.get_invariant_features(flat=True)

        return out_dict
