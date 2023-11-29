import torch
from torch import nn
from torch_cluster import knn_graph

from ligbinddiff.model.modules.layers.node.mpnn import IPMP
from ligbinddiff.model.modules.openfold.frames import  StructureModuleTransition
from ligbinddiff.model.modules.layers.edge.sitewise import EdgeTransition
from ligbinddiff.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.framediff.all_atom import compute_all_atom14
from ligbinddiff.utils.atom_reps import restype_1to3, atom91_start_end
from ligbinddiff.data.openfold.residue_constants import restypes


class IPMPUpdateLayer(nn.Module):
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
                node_features,
                edge_features,
                edge_index,
                rigids,
                node_mask):
        node_update = self.ipmp(
            s=node_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids)
        node_features = self.ln(node_features + node_update * node_mask[..., None])
        node_features = self.node_transition(node_features) * node_mask[..., None]
        edge_features = self.edge_transition(node_features, edge_features, edge_index)
        return node_features, edge_features



class IPMPEncoder(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 c_s_in=6,
                 num_rbf=16,
                 num_layers=4,
                 k=30):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_rbf = num_rbf
        self.c_z_in = num_rbf * (25 + 1)

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
                c_hidden=c_hidden
            )
            for _ in range(num_layers)
        ])
        self.output_mu = nn.Linear(c_s, c_s)
        self.output_logvar = nn.Linear(c_s, c_s)

        self.k = k

    def forward(self, graph, eps=1e-8):
        ## prep features
        X_ca = graph['residue']['x']
        bb = graph['residue']['bb']
        virtual_Cb = _ideal_virtual_Cb(bb)
        bb = torch.cat([bb, virtual_Cb[..., None, :]], dim=-2)
        x_mask = graph['residue'].x_mask
        rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)
        dihedrals = _dihedrals(bb)

        masked_X_ca = X_ca.clone()
        masked_X_ca[graph['residue']['x_mask']] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph['residue'].batch)

        edge_bb_src = bb[edge_index[1]]
        edge_bb_dst = bb[edge_index[0]]
        edge_bb_dists = torch.linalg.vector_norm(
            edge_bb_src[..., None, :] - edge_bb_dst[..., None, :, :] + eps,
            dim=-1)
        edge_bb_dists = edge_bb_dists.view(edge_index.shape[1], -1, 1)
        edge_rbf = _rbf(edge_bb_dists, D_min=2.0, D_max=22.0, D_count=16, device=edge_index.device).view(edge_index.shape[1], -1)
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=16, device=edge_index.device)
        edge_features = torch.cat([edge_rbf, edge_dist_rel_pos], dim=-1)

        node_features = self.embed_node(dihedrals)
        edge_features = self.embed_edge(edge_features)

        for layer in self.update:
            node_features, edge_features = layer(
                node_features,
                edge_features,
                edge_index,
                rigids,
                (~x_mask).float()
            )

        latent_mu = self.output_mu(node_features)
        latent_logvar = self.output_logvar(node_features)

        out_dict = {}
        out_dict['latent_mu'] = latent_mu
        out_dict['latent_logvar'] = latent_logvar

        return out_dict


class IPMPDecoder(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 c_s_in=6,
                 num_rbf=16,
                 num_layers=4,
                 k=30):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_rbf = num_rbf
        self.c_z_in = num_rbf * (25 + 1)

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
                c_hidden=c_hidden
            )
            for _ in range(num_layers)
        ])
        self.torsion_pred = nn.Linear(c_s, (20 * 4 + 1) * 2)
        self.seq_head = nn.Linear(c_s, 20)

        self.k = k

    def forward(self, graph, intermediates, eps=1e-8):
        ## prep features
        num_nodes = graph['residue'].num_nodes
        X_ca = graph['residue']['x']
        bb = graph['residue']['bb']
        virtual_Cb = _ideal_virtual_Cb(bb)
        bb = torch.cat([bb, virtual_Cb[..., None, :]], dim=-2)
        x_mask = graph['residue'].x_mask
        rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)
        dihedrals = _dihedrals(bb)

        masked_X_ca = X_ca.clone()
        masked_X_ca[graph['residue']['x_mask']] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph['residue'].batch)

        edge_bb_src = bb[edge_index[1]]
        edge_bb_dst = bb[edge_index[0]]
        edge_bb_dists = torch.linalg.vector_norm(
            edge_bb_src[..., None, :] - edge_bb_dst[..., None, :, :] + eps,
            dim=-1)
        edge_bb_dists = edge_bb_dists.view(edge_index.shape[1], -1, 1)
        edge_rbf = _rbf(edge_bb_dists, D_min=2.0, D_max=22.0, device=edge_index.device).view(edge_index.shape[1], -1)
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=16, device=edge_index.device)
        edge_features = torch.cat([edge_rbf, edge_dist_rel_pos], dim=-1)

        node_features = intermediates['latent_sidechain']
        edge_features = self.embed_edge(edge_features)

        for layer in self.update:
            node_features, edge_features = layer(
                node_features,
                edge_features,
                edge_index,
                rigids,
                (~x_mask).float()
            )

        unnorm_torsions = self.torsion_pred(node_features)
        chi_per_aatype = unnorm_torsions.view(-1, 81, 2)
        chi_per_aatype = chi_per_aatype / torch.linalg.vector_norm(chi_per_aatype + 1e-8, dim=-1)[..., None]
        psi_torsions, chi_per_aatype = chi_per_aatype.split([1, 80], dim=-2)

        chi_per_aatype = chi_per_aatype.view(-1, 20, 4, 2)
        all_atom14 = compute_all_atom14(rigids, psi_torsions, chi_per_aatype)
        atom91 = torch.zeros((num_nodes, 91, 3), device=all_atom14.device)
        atom91[..., :4, :] = all_atom14[..., 0, :4, :]
        for i in range(20):
            aa = restype_1to3[restypes[i]]
            start, end = atom91_start_end[aa]
            atom91[..., start:end, :] = all_atom14[..., i, 4:4+(end-start), :]
        atom91 = atom91 - rigids.get_trans()[..., None, :]

        seq_logits = self.seq_head(node_features)

        out_dict = {}
        out_dict['decoded_latent'] = atom91
        out_dict['decoded_seq_logits'] = seq_logits

        return out_dict
