import numpy as np
import torch
from torch import nn

import torch_geometric.utils as pygu

from ligbinddiff.data.datasets.featurize.sidechain import _rbf
from ligbinddiff.model.modules.openfold.frames import Linear
from ligbinddiff.model.utils.graph import sparse_to_knn_graph, knn_to_sparse_graph


# inspired by proteus
class SparseTriangleMultiplicativeUpdate(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_gate_s=16,
                 num_rbf=64,
                 assume_sorted=False):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_gate_s = c_gate_s
        self.num_rbf = num_rbf

        self.node_left = nn.Linear(c_s, c_gate_s)
        self.node_right = nn.Linear(c_s, c_gate_s)

        self.edge_left = nn.Linear(c_s, c_s)
        self.edge_left_gate = Linear(c_s, c_s)
        self.edge_right = nn.Linear(c_s, c_s)
        self.edge_right_gate = Linear(c_s, c_s)

        self.dist_bias_gate = nn.Sequential(
            nn.Linear(2*c_s, c_z),
            nn.Sigmoid()
        )
        self.dist_bias = nn.Linear(self.num_rbf, c_z)

        self.lin_edge = nn.Linear(c_z, c_z)
        self.ln = nn.LayerNorm(c_z)
        self.lin_out = Linear(c_z, c_z, init='final')
        self.assume_sorted = assume_sorted

    def forward(self,
                node_features,
                rigids,
                edge_features,
                edge_index,
                eps=1e-8):
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[-1]

        if not self.assume_sorted:
            sorted_edge_index, edge_map = pygu.sort_edge_index(
                edge_index,
                edge_attr=torch.arange(num_edges, device=edge_index.device),
                sort_by_row=False
            )
            sorted_edge_features = edge_features[edge_map]
        else:
            edge_map = torch.arange(edge_index.shape[1], device=edge_index.device)
            sorted_edge_index = edge_index
            sorted_edge_features = edge_features

        knn_edge_features, knn_edge_index, knn_edge_mask = sparse_to_knn_graph(sorted_edge_features, sorted_edge_index)
        k = knn_edge_index.shape[-1]
        # print(knn_edge_index.shape)

        edge1 = knn_edge_index[..., None]
        edge2 = knn_edge_index[..., None, :]

        edge3_node1 = node_features[edge1]
        edge3_node2 = node_features[edge2]
        edge3_gate = self.dist_bias_gate(
            torch.cat([edge3_node1, edge3_node2], dim=-1)
        )

        node_trans = rigids.get_trans()
        edge3_node1_trans = node_trans[edge1]
        edge3_node2_trans = node_trans[edge2]
        edge3_dist = torch.linalg.vector_norm(
            edge3_node1_trans - edge3_node2_trans + eps,
            dim=-1
        )

        edge3_dist_features = _rbf(edge3_dist, D_count=self.num_rbf, device=edge3_dist.device)
        edge3_dist_bias = edge3_gate * self.dist_bias(edge3_dist_features)  # n x k x k x ch

        knn_edge_features = self.lin_edge(knn_edge_features)  # n x k x ch

        # mult_update = edge3_dist_bias * knn_edge_features[..., None, :, :]  # n x k x k x ch
        # edge_update = mult_update.sum(dim=-2)
        edge_update = torch.einsum("nijh,nih->njh", edge3_dist_bias, knn_edge_features)

        edge_update = self.ln(edge_update)
        edge_update, _ = knn_to_sparse_graph(edge_update, knn_edge_index, knn_edge_mask)
        edge_update = self.lin_out(edge_update)

        if not self.assume_sorted:
            edge_update = pygu.scatter(
                edge_update,
                index=edge_map,
                dim=0,
                dim_size=num_edges
            )

        return edge_update


class FusedSparseTriangleMultiplicativeTransition(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 num_rbf=64,
                 use_ffn=False,
                 assume_sorted=False):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.num_rbf = num_rbf

        self.incoming = SparseTriangleMultiplicativeUpdate(
            c_s=c_s,
            c_z=c_z,
            num_rbf=num_rbf,
            assume_sorted=False
        )
        self.incoming_ln = nn.LayerNorm(c_z)
        self.outgoing = SparseTriangleMultiplicativeUpdate(
            c_s=c_s,
            c_z=c_z,
            num_rbf=num_rbf,
            assume_sorted=assume_sorted
        )
        self.outgoing_ln = nn.LayerNorm(c_z)

        self.use_ffn = use_ffn
        if use_ffn:
            self.transition = nn.Sequential(
                nn.Linear(c_z, c_z),
                nn.ReLU(),
                nn.Linear(c_z, c_z),
                nn.ReLU(),
                nn.Linear(c_z, c_z),
            )
            self.transition_ln = nn.LayerNorm(c_z)

    def forward(self,
                node_features,
                rigids,
                edge_features,
                edge_index,
                eps=1e-8):

        # edge_update = self.incoming(node_features, rigids, edge_features, edge_index, eps)
        # edge_features = self.incoming_ln(edge_features + edge_update)
        edge_update = self.outgoing(node_features, rigids, edge_features, edge_index, eps)
        edge_features = self.outgoing_ln(edge_features + edge_update)

        if self.use_ffn:
            edge_update = self.transition(edge_features)
            edge_features = self.transition_ln(edge_features + edge_update)

        return edge_features
