import torch
from torch import nn
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.utils import dropout_edge

from ligbinddiff.model.modules.layers.node.mpnn import IPMP
from ligbinddiff.model.modules.common import GaussianRandomFourierBasis
from ligbinddiff.model.modules.openfold.frames import  StructureModuleTransition
from ligbinddiff.model.modules.layers.edge.sitewise import EdgeTransition
from ligbinddiff.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.framediff.all_atom import compute_all_atom14, compute_atom14
from ligbinddiff.utils.atom_reps import restype_1to3, atom91_start_end
from ligbinddiff.data.openfold.residue_constants import restypes
from ligbinddiff.data.openfold.data_transforms import make_atom14_masks


class IPMPUpdateLayer(nn.Module):
    def __init__(self, c_s, c_z, c_hidden, dropout=0.):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.p_dropout = dropout

        self.ipmp = IPMP(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
        )
        # self.ln = nn.LayerNorm(c_s)
        # self.dropout = nn.Dropout(p=dropout)
        # self.node_transition = StructureModuleTransition(
        #     c_s,
        #     dropout=dropout
        # )
        # self.edge_transition = EdgeTransition(
        #     node_embed_size=c_s,
        #     edge_embed_in=c_z,
        #     edge_embed_out=c_z,
        #     dropout=dropout
        # )

    def forward(self,
                node_features,
                edge_features,
                edge_index,
                rigids,
                node_mask):
        # dropout_edge_index, dropout_edge_mask = dropout_edge(edge_index, p=self.p_dropout, training=self.training)
        # dropout_edge_features = edge_features[dropout_edge_mask]
        # node_update = self.ipmp(
        #     s=node_features,
        #     z=dropout_edge_features,
        #     edge_index=dropout_edge_index,
        #     r=rigids)
        # node_features = self.ln(node_features + self.dropout(node_update) * node_mask[..., None])
        # # node_features = node_features + node_update * node_mask[..., None]
        # node_features = self.node_transition(node_features) * node_mask[..., None]
        # edge_features = self.edge_transition(node_features, edge_features, edge_index)
        node_features, edge_features = self.ipmp(
            s=node_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=node_mask)
        # edge_features = self.edge_transition(node_features, edge_features, edge_index)
        return node_features, edge_features



class IPMPEncoder(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_hidden=256,
                 c_s_in=6,
                 num_rbf=16,
                 num_pos_embed=16,
                 num_layers=4,
                 k=30,
                 dropout=0.1):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_rbf = num_rbf
        self.num_pos_embed = num_pos_embed
        atoms_per_res = 5
        self.c_z_in = (
            num_rbf * (atoms_per_res ** 2)  # bb x bb distances
            # + atoms_per_res * 3  # src CA to dst bb direction unit vectors
            + num_pos_embed  # rel pos embed
        )

        self.embed_node = nn.Sequential(
            nn.Linear(c_s_in, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, c_s),
            nn.LayerNorm(c_s),
        )
        self.embed_edge = nn.Sequential(
            nn.Linear(self.c_z_in, 2*c_z),
            nn.ReLU(),
            nn.Linear(2*c_z, 2*c_z),
            nn.ReLU(),
            nn.Linear(2*c_z, c_z),
            nn.LayerNorm(c_z),
        )
        self.update = nn.ModuleList([
            IPMPUpdateLayer(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.output_mu = nn.Linear(c_s, c_s)
        self.output_logvar = nn.Linear(c_s, c_s)

        self.k = k

    @torch.no_grad()
    def _prep_features(self, graph, eps=1e-8):
        res_data = graph['residue']

        # node features
        X_ca = res_data['x'].float()
        bb = res_data['bb'].float()
        dihedrals = _dihedrals(bb)
        node_features = torch.cat([dihedrals], dim=-1)

        # edge graph
        res_mask = res_data['res_mask']
        masked_X_ca = X_ca.clone()
        masked_X_ca[~res_mask] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph['residue'].batch)

        # edge features
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
        ## edge rel pos embedding
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=self.num_pos_embed, device=edge_index.device)
        # ## direction vecs from src CA to dst bb
        # rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)
        # src_rigids = rigids[src]
        # src_X_ca = X_ca[src]
        # edge_dist_vecs = edge_bb_dst - src_X_ca[..., None, :]
        # edge_dir_vecs = F.normalize(edge_dist_vecs, dim=-1)
        # local_edge_dir_vecs = src_rigids[...,  None].invert_apply(edge_dir_vecs)
        # local_edge_dir_feats = local_edge_dir_vecs.view(edge_index.shape[1], -1)

        # edge_features = torch.cat([edge_rbf, edge_dist_rel_pos, local_edge_dir_feats], dim=-1)
        edge_features = torch.cat([edge_rbf, edge_dist_rel_pos], dim=-1)

        return node_features, edge_features, edge_index


    def forward(self, graph, eps=1e-8):
        ## prep features
        res_data = graph['residue']
        res_mask = res_data['res_mask']
        # TODO: fix this inconsistency
        if "rigids_1" in res_data:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_1)
        else:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)


        node_features, edge_features, edge_index = self._prep_features(graph, eps=eps)

        node_features = self.embed_node(node_features)
        edge_features = self.embed_edge(edge_features)

        for layer in self.update:
            node_features, edge_features = layer(
                node_features,
                edge_features,
                edge_index,
                rigids,
                res_mask.float()
            )

        latent_mu = self.output_mu(node_features)
        latent_logvar = self.output_logvar(node_features)

        out_dict = {}
        out_dict['latent_mu'] = latent_mu
        out_dict['latent_logvar'] = latent_logvar

        return out_dict


class IPMPDecoder(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_hidden=256,
                 c_s_in=6,
                 num_rbf=16,
                 num_pos_embed=16,
                 num_layers=4,
                 h_time=64,
                 k=30,
                 dropout=0.1):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_rbf = num_rbf
        self.num_pos_embed = num_pos_embed
        atoms_per_res = 5
        self.c_z_in = (
            num_rbf * (atoms_per_res ** 2)  # bb x bb distances
            # + atoms_per_res * 3  # src CA to dst bb direction unit vectors
            + num_pos_embed  # rel pos embed
        )

        self.embed_time = GaussianRandomFourierBasis(h_time)

        self.embed_node = nn.Sequential(
            nn.Linear(c_s + h_time*2, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, c_s),
        )
        self.node_ln = nn.LayerNorm(c_s)

        self.embed_edge = nn.Sequential(
            nn.Linear(self.c_z_in, 2*c_z),
            nn.ReLU(),
            nn.Linear(2*c_z, 2*c_z),
            nn.ReLU(),
            nn.Linear(2*c_z, c_z),
            nn.LayerNorm(c_z),
        )
        self.update = nn.ModuleList([
            IPMPUpdateLayer(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.seq_head = nn.Linear(c_s, 20)
        self.seq_embed = nn.Embedding(21, 20)
        self.torsion_pred = nn.Linear(c_s + 20, (4 + 1) * 2)

        self.k = k


    # @torch.no_grad()
    def _prep_features(self, graph, eps=1e-8):
        res_data = graph['residue']
        res_mask = res_data['res_mask']

        # node features
        X_ca = res_data['x'].float()
        bb = res_data['bb'].float()
        dihedrals = _dihedrals(bb)
        node_features = torch.cat([dihedrals], dim=-1)
        node_features = node_features * res_mask[..., None]

        # edge graph
        masked_X_ca = X_ca.clone()
        masked_X_ca[~res_mask] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph['residue'].batch)

        # edge features
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
        ## edge rel pos embedding
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=self.num_pos_embed, device=edge_index.device)
        # ## direction vecs from src CA to dst bb
        # rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)
        # src_rigids = rigids[src]
        # src_X_ca = X_ca[src]
        # edge_dist_vecs = edge_bb_dst - src_X_ca[..., None, :]
        # edge_dir_vecs = F.normalize(edge_dist_vecs, dim=-1)
        # local_edge_dir_vecs = src_rigids[...,  None].invert_apply(edge_dir_vecs)
        # local_edge_dir_feats = local_edge_dir_vecs.view(edge_index.shape[1], -1)

        # edge_features = torch.cat([edge_rbf, edge_dist_rel_pos, local_edge_dir_feats], dim=-1)
        edge_features = torch.cat([edge_rbf, edge_dist_rel_pos], dim=-1)
        # technically this shouldn't be necessary but just to be safe
        edge_mask = res_mask[src] & res_mask[dst]
        edge_features = edge_features * edge_mask[..., None]

        return node_features, edge_features, edge_index

    def forward(self,
                graph,
                intermediates,
                t=None,
                eps=1e-8,
                use_gt_seq=True):
        res_data = graph['residue']
        num_nodes = res_data.num_nodes
        res_mask = res_data['res_mask']
        # TODO: fix this inconsistency
        if "rigids_1" in res_data:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_1)
        else:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)

        _, edge_features, edge_index = self._prep_features(graph, eps=eps)

        if t is None:
            if "rigids_1" in res_data:
                t = torch.ones(graph.num_graphs, device=edge_features.device)
            else:
                t = torch.zeros(graph.num_graphs, device=edge_features.device)

        timestep_embed = self.embed_time(t[..., None])
        timestep_embed = timestep_embed[res_data.batch]

        node_features = intermediates['latent_sidechain']
        node_update = self.embed_node(
            torch.cat([timestep_embed, node_features], dim=-1)
        )
        node_features = self.node_ln(node_features + node_update * res_mask[..., None])
        edge_features = self.embed_edge(edge_features)

        for layer in self.update:
            node_features, edge_features = layer(
                node_features,
                edge_features,
                edge_index,
                rigids,
                res_mask.float()
            )

        seq_logits = self.seq_head(node_features)

        seq = seq_logits.argmax(dim=-1)
        seq = seq.to(node_features.device)
        atom14_mask_dict = make_atom14_masks({"aatype": seq})

        unnorm_torsions = self.torsion_pred(
            torch.cat([
                node_features,
                self.seq_embed(seq)
            ], dim=-1)
        )
        chi_per_aatype = unnorm_torsions.view(-1, 5, 2)
        chi_per_aatype = F.normalize(chi_per_aatype, dim=-1)
        psi_torsions, chi_per_aatype = chi_per_aatype.split([1, 4], dim=-2)

        # chi_per_aatype = chi_per_aatype.view(-1, 4, 2)
        output_atom14 = compute_atom14(rigids, psi_torsions, chi_per_aatype, seq)


        out_dict = {}
        # out_dict['decoded_all_atom14'] = all_atom14
        # out_dict['decoded_all_chis'] = chi_per_aatype
        out_dict['decoded_atom14'] = output_atom14
        out_dict['decoded_atom14_mask'] = atom14_mask_dict['atom14_atom_exists']
        out_dict['decoded_chis'] = chi_per_aatype
        out_dict['decoded_seq_logits'] = seq_logits

        if use_gt_seq:
            gt_seq = res_data['seq']
            unnorm_torsions = self.torsion_pred(
                torch.cat([
                    node_features,
                    self.seq_embed(gt_seq)
                ], dim=-1)
            )
            chi_per_aatype = unnorm_torsions.view(-1, 5, 2)
            chi_per_aatype = F.normalize(chi_per_aatype, dim=-1)
            psi_torsions, chi_per_aatype = chi_per_aatype.split([1, 4], dim=-2)

            # chi_per_aatype = chi_per_aatype.view(-1, 4, 2)
            output_atom14 = compute_atom14(rigids, psi_torsions, chi_per_aatype, gt_seq)
            out_dict['decoded_atom14_gt_seq'] = output_atom14
            out_dict['decoded_chis_gt_seq'] = chi_per_aatype

        return out_dict
