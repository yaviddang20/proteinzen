import numpy as np
import torch
from torch import nn

import torch_geometric.utils as pygu

from ligbinddiff.data.datasets.featurize.sidechain import _rbf
from ligbinddiff.model.modules.openfold.frames import Linear
from ligbinddiff.model.utils.graph import sparse_to_knn_graph, knn_to_sparse_graph

# inspired by proteus
class SparseTriangleAttention(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 num_heads=4,
                 num_rbf=64,
                 inf=1e4,
                 assume_sorted=False):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.num_rbf = num_rbf
        self.num_heads = num_heads

        self.dist_bias_gate = nn.Sequential(
            nn.Linear(2*c_s, num_heads),
            nn.Sigmoid()
        )
        self.dist_bias = nn.Linear(self.num_rbf, num_heads)

        self.lin_qk = nn.Linear(c_z, 2*c_z)
        self.lin_v = nn.Linear(c_z, c_z)
        self.lin_out = Linear(c_z, c_z, init='final')
        self.assume_sorted = assume_sorted

        self.inf = inf

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

        edge1 = knn_edge_index[..., None].expand(-1, -1, k),
        edge2 = knn_edge_index[..., None, :].expand(-1, k, -1)

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
        edge3_dist_bias = edge3_gate * self.dist_bias(edge3_dist_features)  # n x k x k x h

        edge_qk = self.lin_qk(knn_edge_features)
        edge_q, edge_k = edge_qk.split([self.c_z, self.c_z], dim=-1)
        edge_q = edge_q.view(num_nodes, k, self.num_heads, -1)
        edge_k = edge_k.view(num_nodes, k, -1, self.num_heads)

        edge_attn = edge_q.transpose(-2, -3) @ edge_k.transpose(-1, -3)  # n x h x k x k
        edge_attn = edge_attn.transpose(-2, -3).transpose(-1, -2)  # n x k x k x h
        edge_attn = 1 / np.sqrt(self.c_z) * edge_attn
        edge_attn = edge_attn + edge3_dist_bias

        edge_edge_mask = knn_edge_mask[..., None] & knn_edge_mask[..., None, :]  # n x k x k
        edge_attn = edge_attn - self.inf * (~edge_edge_mask[..., None]).float()
        edge_attn = torch.softmax(edge_attn, dim=-2) # n x k x k x h

        edge_v = self.lin_v(knn_edge_features).view(num_nodes, k, self.num_heads, -1)  # n x k x h x c_z//h
        edge_update = torch.sum(
            edge_v[..., None, :, :]  # n x k x 1 x h x c_z//h
            * edge_attn[..., None]
            * edge_edge_mask[..., None, None],  # n x k x k x h x 1
        dim=-3)  # n x k x h x c_z//h
        edge_update = self.lin_out(edge_update.view(-1, k, self.c_z))
        edge_update, _ = knn_to_sparse_graph(edge_update, knn_edge_index, knn_edge_mask)

        if not self.assume_sorted:
            edge_update = pygu.scatter(
                edge_update,
                index=edge_map,
                dim=0,
                dim_size=num_edges
            )

        return edge_update


# inspired by proteus
class SparseSubsampledTriangleAttention(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 k=60,
                 subsample_k=20,
                 num_heads=4,
                 num_rbf=64,
                 assume_sorted=False):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.k = k
        self.subsample_k = subsample_k
        self.num_rbf = num_rbf
        self.num_heads = num_heads

        self.dist_bias_gate = nn.Sequential(
            nn.Linear(2*c_s, num_heads),
            nn.Sigmoid()
        )
        self.dist_bias = nn.Linear(self.num_rbf, num_heads)

        self.lin_qk = nn.Linear(c_z, 2*c_z)
        self.lin_v = nn.Linear(c_z, c_z)
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

        semidense_dst = sorted_edge_index[0].view(-1, self.k)  # n x k

        edge1 = semidense_dst[..., None].expand(-1, -1, self.subsample_k)  # n x k x subsample_k
        # select a random subsample_k of remaining edges
        # randperm = torch.empty((num_nodes, self.k, self.k), dtype=torch.long, device=edge1.device)
        # for i in range(num_nodes):
        #     for j in range(self.k):
        #         randperm[i, j] = torch.randperm(self.k, device=randperm.device)
        randperm = torch.stack([torch.randperm(self.k, device=edge1.device) for _ in range(num_nodes * self.k)])
        randperm = randperm.view(num_nodes, self.k, self.k)

        subsample_idx = randperm[..., :self.subsample_k]
        edge2 = semidense_dst[..., None, :].expand(-1, self.k, -1)
        edge2 = torch.gather(
            edge2,
            -1,
            subsample_idx
        )  # n x k x subsample_k

        def _gather(nodes, triangles):
            return nodes[triangles.reshape(-1)].view(
                list(triangles.shape) + [-1])

        edge3_node1 = _gather(node_features, edge1)
        edge3_node2 = _gather(node_features, edge2)
        edge3_gate = self.dist_bias_gate(
            torch.cat([edge3_node1, edge3_node2], dim=-1)
        )

        node_trans = rigids.get_trans()
        edge3_node1_trans = _gather(node_trans, edge1)
        edge3_node2_trans = _gather(node_trans, edge2)
        edge3_dist = torch.linalg.vector_norm(
            edge3_node1_trans - edge3_node2_trans + eps,
            dim=-1
        )

        edge3_dist_features = _rbf(edge3_dist, D_count=self.num_rbf, device=edge3_dist.device)
        edge3_dist_bias = edge3_gate * self.dist_bias(edge3_dist_features)  # n x k x subsample_k x h

        edges = sorted_edge_features.view(num_nodes, self.k, -1)
        edge_qk = self.lin_qk(edges)
        edge_q, edge_k = edge_qk.split([self.c_z, self.c_z], dim=-1)
        edge_q = edge_q.view(num_nodes, self.k, self.num_heads, -1)
        edge_k = edge_k.view(num_nodes, self.k, self.num_heads, -1)

        edge_k = torch.gather(
            edge_k[:, None].expand(-1, self.k, -1, -1, -1),  # n x k x k x c_z//h x h
            1,
            subsample_idx[..., None, None].expand(-1, -1, -1, self.num_heads, self.c_z//self.num_heads)  # n x k x subsample_k x c_z//h x h
        )  # n x k x subsample_k x c_z//h x h

        edge_attn = torch.sum(
            edge_q[:, :, None]  # n x k x 1 x h x c_z//h
            * edge_k,  # n x k x subsample_k x h x c_z//h
        dim=-1)  # n x k x subsample_k x h
        edge_attn = 1 / np.sqrt(self.c_z) * edge_attn
        edge_attn = edge_attn + edge3_dist_bias
        edge_attn = torch.softmax(edge_attn, dim=-2) # n x k x subsample_k x h

        edge_v = self.lin_v(edges).view(num_nodes, self.k, self.num_heads, -1)  # n x k x h x c_z//h
        edge_v = torch.gather(
            edge_v[:, None].expand(-1, self.k, -1, -1, -1),  # n x k x k x c_z//h x h
            1,
            subsample_idx[..., None, None].expand(-1, -1, -1, self.num_heads, self.c_z//self.num_heads)  # n x k x subsample_k x c_z//h x h
        )  # n x k x subsample_k x c_z//h x h

        edge_update = torch.sum(
            edge_v  # n x k x subsample_k x h x c_z//h
            * edge_attn[..., None],  # n x k x subsample_k x h x 1
        dim=-3)  # n x k x h x c_z//h
        edge_update = self.lin_out(edge_update.view(num_edges, self.c_z))

        if not self.assume_sorted:
            edge_update = pygu.scatter(
                edge_update,
                index=edge_map,
                dim=0,
                dim_size=num_edges
            )

        return edge_update