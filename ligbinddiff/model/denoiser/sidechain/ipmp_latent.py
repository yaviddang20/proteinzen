import torch
from torch import nn
from torch_cluster import knn_graph

from ligbinddiff.model.modules.layers.node.mpnn import IPMP
from ligbinddiff.model.modules.common import RBF
from ligbinddiff.model.modules.openfold.frames import  StructureModuleTransition
from ligbinddiff.model.modules.layers.edge.sitewise import EdgeTransition
from ligbinddiff.data.datasets.featurize.sidechain import _dihedrals
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.model.utils.graph import batchwise_to_nodewise


class IPMPUpdateLayer(nn.Module):
    def __init__(self, c_s, c_z, c_hidden):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden

        self.fuse_inputs = nn.Linear(c_s*2, c_s)

        self.ipmp = IPMP(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
        )
        # self.node_ln = nn.LayerNorm(c_s)
        # self.node_transition = StructureModuleTransition(
        #     c_s
        # )

        self.latent_update = nn.Linear(c_s, c_s)
        self.latent_ln = nn.LayerNorm(c_s)

        # self.edge_transition = EdgeTransition(
        #     node_embed_size=c_s,
        #     edge_embed_in=c_z,
        #     edge_embed_out=c_z
        # )

    def forward(self,
                latent_features,
                node_features,
                edge_features,
                edge_index,
                rigids,
                node_mask):

        input_features = self.fuse_inputs(
            torch.cat([node_features, latent_features], dim=-1)
        )
        # node_update = self.ipmp(
        node_features, edge_features = self.ipmp(
            s=input_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=node_mask)
        # node_features = self.node_ln(node_features + node_update * node_mask[..., None])
        # node_features = self.node_transition(node_features) * node_mask[..., None]

        latent_update = self.latent_update(node_features)
        latent_features = latent_features + self.latent_ln(latent_update)

        # edge_features = self.edge_transition(node_features, edge_features, edge_index)
        return latent_features, node_features, edge_features


class IPMPDenoiser(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_hidden=256,
                 c_s_in=6,
                 c_time=64,
                 num_rbf=16,
                 num_layers=4,
                 k=30):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_rbf = num_rbf
        self.c_z_in = num_rbf * (16 + 1)

        self.time_rbf = RBF(n_basis=c_time//2)

        self.embed_node = nn.Sequential(
            nn.Linear(c_s_in + c_time, 2*c_s),
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

        self.k = k

    def forward(self, graph, intermediates, self_condition=None, eps=1e-8):
        latent_features = intermediates['noised_latent_sidechain']

        ## prep features
        X_ca = graph['residue']['x'].float()
        bb = graph['residue']['bb'].float()
        x_mask = ~graph['residue'].res_mask
        rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)
        dihedrals = _dihedrals(bb)

        ## create time embedding
        ts = graph['residue']['t']
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (B x h_time,)
        fourier_time = batchwise_to_nodewise(fourier_time, graph['residue'].batch)
        node_features = torch.cat([dihedrals, fourier_time], dim=-1)
        node_features = self.embed_node(node_features)

        masked_X_ca = X_ca.clone()
        masked_X_ca[x_mask] = torch.inf
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

        edge_features = self.embed_edge(edge_features)

        for layer in self.update:
            latent_features, node_features, edge_features = layer(
                latent_features,
                node_features,
                edge_features,
                edge_index,
                rigids,
                (~x_mask).float()
            )

        intermediates['pred_latent_sidechain'] = latent_features
        return intermediates