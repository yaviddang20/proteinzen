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
    def __init__(self,
                 c_s,
                 c_latent,
                 c_z,
                 c_hidden,
                 self_conditioning=True):
        super().__init__()
        self.c_s = c_s
        self.c_latent = c_latent
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.self_conditioning = self_conditioning

        h_dim = c_s + c_latent + c_latent*self_conditioning

        self.ipmp = IPMP(
            c_s=h_dim,
            c_z=c_z,
            c_hidden=c_hidden,
            dropout=0.,
            edge_dropout=0.,
        )

        self.latent_update = nn.Linear(h_dim, c_latent)
        self.latent_gate = nn.Sequential(
            nn.Linear(h_dim, c_latent),
            nn.Sigmoid(),
        )

        self.node_update = nn.Linear(
            h_dim,
            c_s)
        self.node_ln = nn.LayerNorm(c_s)

    def forward(self,
                latent_features,
                node_features,
                edge_features,
                edge_index,
                rigids,
                node_mask,
                self_condition=None):

        input_features = [node_features, latent_features]
        if self.self_conditioning and self_condition is not None:
            input_features.append(self_condition['pred_latent_sidechain'])
        elif self.self_conditioning:
            input_features.append(torch.zeros_like(latent_features))

        input_features = torch.cat(input_features, dim=-1)

        # node_update = self.ipmp(
        joint_features, edge_features = self.ipmp(
            s=input_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=node_mask)
        # node_features = self.node_ln(node_features + node_update * node_mask[..., None])
        # node_features = self.node_transition(node_features) * node_mask[..., None]

        latent_features = latent_features + self.latent_update(joint_features) * self.latent_gate(joint_features)
        node_features = self.node_ln(node_features + self.node_update(joint_features))

        return latent_features, node_features, edge_features


class IPMPDenoiser(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_latent=128,
                 c_z=128,
                 c_hidden=256,
                 c_s_in=6,
                 c_time=64,
                 num_rbf=16,
                 num_layers=4,
                 k=30,
                 self_conditioning=True):
        super().__init__()
        self.c_s = c_s
        self.c_latent = c_latent
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_rbf = num_rbf
        self.c_z_in = num_rbf * (16 + 1)
        self.self_conditioning = self_conditioning

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
                c_latent=c_latent,
                c_z=c_z,
                c_hidden=c_hidden,
                self_conditioning=self_conditioning
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
        if self.training:
            rigids = rigids.translate(torch.randn((*rigids.shape, 3), device=rigids.device) * 0.1)
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
                latent_features=latent_features,
                node_features=node_features,
                edge_features=edge_features,
                edge_index=edge_index,
                rigids=rigids,
                node_mask=(~x_mask).float(),
                self_condition=self_condition
            )

        intermediates['pred_latent_sidechain'] = latent_features
        return intermediates