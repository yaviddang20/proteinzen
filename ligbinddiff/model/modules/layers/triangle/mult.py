import numpy as np
import torch
from torch import nn

import torch_geometric.utils as pygu
from torch_cluster import knn_graph

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
                 vpa=False,
                 gate_out=True,
                 dropout=0,
                 dtype=torch.bfloat16,
                 assume_sorted=False):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_gate_s = c_gate_s
        self.num_rbf = num_rbf
        self.vpa = vpa
        self.layer_norm = nn.LayerNorm(c_z, dtype=dtype)
        self.gate_out = gate_out
        self.dropout_rate = dropout
        self.dtype = dtype

        self.node_left = nn.Linear(c_s, c_gate_s, dtype=dtype)
        self.node_right = nn.Linear(c_s, c_gate_s, dtype=dtype)

        self.edge_proj = Linear(c_z, c_z, dtype=dtype)
        self.edge_gate = Linear(c_z, c_z, dtype=dtype)
        self.dist_gate = Linear(c_gate_s*c_gate_s, c_z, init='gating', dtype=dtype)
        self.dist_proj = Linear(self.num_rbf, c_z, dtype=dtype)
        self.ln_out = nn.LayerNorm(c_z, dtype=dtype)
        self.lin_out = Linear(c_z, c_z, init='final', dtype=dtype)
        if self.gate_out:
            self.out_gate = Linear(c_z, c_z, init='gating', dtype=dtype)

        self.assume_sorted = assume_sorted

    def forward(self,
                node_features,
                rigids,
                edge_features,
                edge_index,
                k=None,
                eps=1e-8):
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[-1]
        initial_dtype = edge_features.dtype
        node_features = node_features.to(self.dtype)
        edge_features = edge_features.to(self.dtype)
        node_trans = rigids.get_trans()

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

        if k is None:
            # max for pygu knn_graph with loop=False
            k = 99
        edge_edge_index = knn_graph(
            node_trans[sorted_edge_index[0]],
            k=k,
            batch=sorted_edge_index[1]
        )
        node_trans = node_trans.to(self.dtype)
        edge_edge_index = pygu.coalesce(edge_edge_index, sort_by_row=False)
        edge_edge_index, _ = pygu.dropout_edge(edge_index=edge_edge_index, p=self.dropout_rate, training=self.training)
        assert (sorted_edge_index[1, edge_edge_index[1]] == sorted_edge_index[1, edge_edge_index[0]]).all()
        edge3_node1_idx = sorted_edge_index[0, edge_edge_index[1]]
        edge3_node2_idx = sorted_edge_index[0, edge_edge_index[0]]

        sorted_edge_features = self.layer_norm(sorted_edge_features)

        node_left = self.node_left(node_features)
        node_right = self.node_right(node_features)

        edge3_node1 = node_left[edge3_node1_idx]
        edge3_node2 = node_right[edge3_node2_idx]
        edge3 = torch.einsum("bi,bj->bij", edge3_node1, edge3_node2)
        edge3_gate = self.dist_gate(
            edge3.view(-1, self.c_gate_s * self.c_gate_s)
        )

        edge3_node1_trans = node_trans[edge3_node1_idx]
        edge3_node2_trans = node_trans[edge3_node2_idx]
        edge3_dist = torch.linalg.vector_norm(
            edge3_node1_trans - edge3_node2_trans + eps,
            dim=-1
        )

        edge3_dist_features = _rbf(edge3_dist, D_count=self.num_rbf, device=edge3_dist.device, dtype=self.dtype)
        edge3_dist_features = self.dist_proj(edge3_dist_features)  # n x k x k x h
        edge3_features = torch.sigmoid(edge3_gate) * edge3_dist_features

        edge2_features = self.edge_proj(sorted_edge_features)
        edge2_gate = self.edge_gate(sorted_edge_features)
        edge2_features = torch.sigmoid(edge2_gate) * edge2_features
        edge2_features = edge2_features[edge_edge_index[0]]
        edge_update = edge2_features * edge3_features

        edge_update = pygu.scatter(
            edge_update,
            edge_edge_index[0],
            reduce='sum'
        )
        if self.vpa:
            reduce_factor = pygu.scatter(
                torch.ones_like(edge_edge_index[0]),
                edge_edge_index[0]
            )
            reduce_factor[reduce_factor <= 0] = 1
            edge_update = edge_update / torch.sqrt(reduce_factor)

        edge_update = self.lin_out(self.ln_out(edge_update))
        if self.gate_out:
            edge_update = edge_update * torch.sigmoid(self.out_gate(sorted_edge_features))

        if not self.assume_sorted:
            edge_update = pygu.scatter(
                edge_update,
                index=edge_map,
                dim=0,
                dim_size=num_edges
            )
        
        edge_update = edge_update.to(dtype=initial_dtype)

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
