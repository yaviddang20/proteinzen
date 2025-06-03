import numpy as np
import torch
from torch import nn

import torch_geometric.utils as pygu
from torch_cluster import knn, knn_graph

from proteinzen.data.datasets.featurize.common import _rbf
from proteinzen.model.modules.openfold.layers import Linear
from proteinzen.model.utils.graph import sparse_to_nested, sparse_to_knn_graph, knn_to_sparse_graph


# inspired by proteus
class SparseTriangleMultiplicativeUpdate(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_gate_s=16,
                 num_rbf=64,
                 vpa=True,
                 gate_out=True,
                 dropout=0,
                 # dtype=torch.bfloat16):
                 dtype=torch.float32):
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

        self._gen_edge_update = torch.compile(self._gen_edge_update, dynamic=True)

    # @torch.compile(dynamic=True)
    def _gen_edge_update(self, node_features, edge_features, node_trans, edge_index, edge_edge_index, eps=1e-8):
        edge3_node1_idx = edge_index[0, edge_edge_index[1]]
        edge3_node2_idx = edge_index[0, edge_edge_index[0]]

        edge_features = self.layer_norm(edge_features)

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

        edge2_features = self.edge_proj(edge_features)
        edge2_gate = self.edge_gate(edge_features)
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
                edge_edge_index[0],
                dim_size=edge_update.shape[0]
            )
            reduce_factor[reduce_factor <= 0] = 1
            edge_update = edge_update / torch.sqrt(reduce_factor)[..., None]
        else:
            edge_update = self.ln_out(edge_update)

        edge_update = self.lin_out(edge_update)
        if self.gate_out:
            edge_update = edge_update * torch.sigmoid(self.out_gate(edge_features))
        return edge_update

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

        if k is None:
            # max for pygu knn_graph with loop=False
            k = 99
        edge_edge_index = knn_graph(
            node_trans[edge_index[0]],
            k=k,
            batch=edge_index[1]
        )
        node_trans = node_trans.to(self.dtype)
        edge_edge_index = pygu.coalesce(edge_edge_index, sort_by_row=False)
        edge_edge_index, _ = pygu.dropout_edge(edge_index=edge_edge_index, p=self.dropout_rate, training=self.training)
        assert (edge_index[1, edge_edge_index[1]] == edge_index[1, edge_edge_index[0]]).all()
        # edge3_node1_idx = edge_index[0, edge_edge_index[1]]
        # edge3_node2_idx = edge_index[0, edge_edge_index[0]]

        # edge_features = self.layer_norm(edge_features)

        # node_left = self.node_left(node_features)
        # node_right = self.node_right(node_features)

        # edge3_node1 = node_left[edge3_node1_idx]
        # edge3_node2 = node_right[edge3_node2_idx]
        # edge3 = torch.einsum("bi,bj->bij", edge3_node1, edge3_node2)
        # edge3_gate = self.dist_gate(
        #     edge3.view(-1, self.c_gate_s * self.c_gate_s)
        # )

        # edge3_node1_trans = node_trans[edge3_node1_idx]
        # edge3_node2_trans = node_trans[edge3_node2_idx]
        # edge3_dist = torch.linalg.vector_norm(
        #     edge3_node1_trans - edge3_node2_trans + eps,
        #     dim=-1
        # )

        # edge3_dist_features = _rbf(edge3_dist, D_count=self.num_rbf, device=edge3_dist.device, dtype=self.dtype)
        # edge3_dist_features = self.dist_proj(edge3_dist_features)  # n x k x k x h
        # edge3_features = torch.sigmoid(edge3_gate) * edge3_dist_features

        # edge2_features = self.edge_proj(edge_features)
        # edge2_gate = self.edge_gate(edge_features)
        # edge2_features = torch.sigmoid(edge2_gate) * edge2_features
        # edge2_features = edge2_features[edge_edge_index[0]]
        # edge_update = edge2_features * edge3_features

        # edge_update = pygu.scatter(
        #     edge_update,
        #     edge_edge_index[0],
        #     reduce='sum'
        # )
        edge_update = self._gen_edge_update(node_features, edge_features, node_trans, edge_index, edge_edge_index)

        edge_update = edge_update.to(dtype=initial_dtype)

        return edge_update


# inspired by proteus
class TriangleMultiplicativeUpdate(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_gate_s=16,
                 num_rbf=64,
                 vpa=False,
                 gate_out=True,
                 dropout=0,
                 # dtype=torch.bfloat16):
                 dtype=torch.float32):
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

    # @torch.compile(dynamic=True)
    def forward(self,
                node_features,
                rigids,
                edge_features,
                edge_index,
                batch,
                res_mask,
                eps=1e-8):
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[-1]
        initial_dtype = edge_features.dtype
        node_features = node_features.to(self.dtype)
        edge_features = edge_features.to(self.dtype)
        node_trans = rigids.get_trans().to(self.dtype)

        knn_edge_features, knn_edge_index, knn_edge_mask = sparse_to_knn_graph(
            edge_features,
            edge_index,
            num_nodes=num_nodes,
            batch=batch,
            res_mask=res_mask
        )
        knn_edge_features = self.layer_norm(knn_edge_features)

        node_left = self.node_left(node_features)
        node_right = self.node_right(node_features)
        edge3_node1 = node_left[knn_edge_index[0]]
        edge3_node2 = node_right[knn_edge_index[0]]

        # n_node x n_edge x 1 x hdim x 1 * n_node x 1 x n_edge x 1 x h_dim
        edge3 = edge3_node1[..., None, :, None] * edge3_node2[..., None, :, None, :]
        edge3_gate = self.dist_gate(edge3.flatten(-2, -1))

        edge3_trans = node_trans[knn_edge_index]
        edge3_dist = torch.linalg.vector_norm(
            edge3_trans[..., None, :, :] - edge3_trans[..., None, :] + eps,
            dim=-1
        )

        edge3_dist_features = _rbf(edge3_dist, D_count=self.num_rbf, device=edge3_dist.device, dtype=self.dtype)
        edge3_dist_features = self.dist_proj(edge3_dist_features)  # n x k x k x h
        edge3_features = torch.sigmoid(edge3_gate) * edge3_dist_features
        edge_mask = knn_edge_mask[..., None] & knn_edge_mask[..., None, :]
        edge3_features = edge3_features * edge_mask[..., None]

        edge2_features = self.edge_proj(knn_edge_features)
        edge2_gate = self.edge_gate(knn_edge_features)
        edge2_features = torch.sigmoid(edge2_gate) * edge2_features
        edge_update = edge3_features.permute(0, 3, 1, 2) @ edge2_features.permute(0, 2, 1)[..., None]
        edge_update = edge_update.squeeze(-1).permute(0, 2, 1)

        edge_update = self.lin_out(self.ln_out(edge_update))
        if self.gate_out:
            edge_update = edge_update * torch.sigmoid(self.out_gate(knn_edge_features))

        edge_update, _ = knn_to_sparse_graph(knn_edge_features, knn_edge_index, knn_edge_mask)
        edge_update = edge_update.to(dtype=initial_dtype)

        return edge_update


# inspired by proteus
class NestedTriangleMultiplicativeUpdate(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_gate_s=16,
                 num_rbf=64,
                 vpa=False,
                 gate_out=True,
                 dropout=0,
                 dtype=torch.bfloat16):
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

    def forward(self,
                node_features,
                rigids,
                edge_features,
                edge_index,
                eps=1e-8):
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[-1]
        initial_dtype = edge_features.dtype
        node_features = node_features.to(self.dtype)
        edge_features = edge_features.to(self.dtype)
        node_trans = rigids.get_trans().to(self.dtype)

        # we do this first since we can't index with nested tensors
        # TODO: doing this might take up all our memory gains
        node_left = self.node_left(node_features)
        node_right = self.node_right(node_features)
        edge3_node1 = node_left[edge_index[0]]
        edge3_node2 = node_right[edge_index[0]]
        print(edge3_node1.shape, edge3_node2.shape)
        # n_node x n_edge x hdim
        edge3_node1, _ = sparse_to_nested(edge3_node1[..., None, :, None], edge_index)
        edge3_node2, _ = sparse_to_nested(edge3_node2[..., None, None, :], edge_index)

        edge_features, edge_index = sparse_to_nested(edge_features, edge_index)
        edge_features = self.layer_norm(edge_features)

        # n_node x n_edge x 1 x hdim x 1 * n_node x 1 x n_edge x 1 x h_dim
        edge3 = edge3_node1 * edge3_node2.transpose(1, 2)
        edge3_gate = self.dist_gate(edge3.flatten(-2, -1))

        edge3_trans = node_trans[edge_index]
        edge3_dist = torch.linalg.vector_norm(
            edge3_trans[..., None, :, :] - edge3_trans[..., None, :] + eps,
            dim=-1
        )

        edge3_dist_features = _rbf(edge3_dist, D_count=self.num_rbf, device=edge3_dist.device, dtype=self.dtype)
        edge3_dist_features = self.dist_proj(edge3_dist_features)  # n x k x k x h
        edge3_features = torch.sigmoid(edge3_gate) * edge3_dist_features

        edge2_features = self.edge_proj(edge_features)
        edge2_gate = self.edge_gate(edge_features)
        edge2_features = torch.sigmoid(edge2_gate) * edge2_features
        edge_update = edge3_features.permute(0, 3, 1, 2) @ edge2_features.permute(0, 2, 1)[..., None]
        edge_update = edge_update.squeeze(-1).permute(0, 2, 1)

        edge_update = self.lin_out(self.ln_out(edge_update))
        if self.gate_out:
            edge_update = edge_update * torch.sigmoid(self.out_gate(edge_features))

        print(edge_update.isnested)
        if edge_update.isnested:
            edge_update = torch.cat(edge_update.unbind(), dim=0)
        else:
            edge_update = edge_update.flatten(0, 1)
        edge_update = edge_update.to(dtype=initial_dtype)

        return edge_update


class TriangleCrossMultiplicativeUpdate(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_gate_s=16,
                 num_rbf=64,
                 dropout=0,
                 vpa=False,
                 # dtype=torch.bfloat16,
                 dtype=torch.float32,
                 gate_out=True):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_gate_s = c_gate_s
        self.num_rbf = num_rbf
        self.vpa = vpa
        self.src_layer_norm = nn.LayerNorm(c_z, dtype=dtype)
        self.dst_layer_norm = nn.LayerNorm(c_z, dtype=dtype)
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

    # @torch.compile(dynamic=True)
    def forward(self,
                node_features,
                rigids,
                src_edge_features,
                src_edge_index,
                dst_edge_features,
                dst_edge_index,
                batch,
                res_mask,
                eps=1e-8):
        num_nodes = node_features.shape[0]

        knn_src_edge_features, knn_src_edge_index, knn_src_edge_mask = sparse_to_knn_graph(
            src_edge_features,
            src_edge_index,
            num_nodes=num_nodes,
            batch=batch,
            res_mask=res_mask
        )
        knn_dst_edge_features, knn_dst_edge_index, knn_dst_edge_mask = sparse_to_knn_graph(
            dst_edge_features,
            dst_edge_index,
            num_nodes=num_nodes,
            batch=batch,
            res_mask=res_mask
        )

        if self.dropout_rate > 0 and self.training:
            num_drop_edges = round(k * self.dropout_rate)
            num_keep_edges = k - num_drop_edges
            idxs = np.repeat(np.arange(k)[None], knn_src_edge_features.shape[0], 0)
            # this is a kinda jenk way of doing this
            # but i think this will allow checkpointing to control the seed used for permutation
            torch_rng = torch.Generator()
            rng = np.random.default_rng(torch_rng.initial_seed())
            shuffled_idxs = rng.permutation(idxs, axis=1)
            keep_mask = torch.as_tensor(shuffled_idxs < num_keep_edges, device=knn_src_edge_features.device)
            knn_src_edge_features = knn_src_edge_features[keep_mask].reshape(-1, num_keep_edges, self.c_z)
            knn_src_edge_index = knn_src_edge_index[keep_mask].reshape(-1, num_keep_edges)
            knn_src_edge_mask = knn_src_edge_mask[keep_mask].reshape(-1, num_keep_edges)
            shuffled_idxs = rng.permutation(idxs, axis=1)
            keep_mask = torch.as_tensor(shuffled_idxs < num_keep_edges, device=knn_src_edge_features.device)
            knn_dst_edge_features = knn_dst_edge_features[keep_mask].reshape(-1, num_keep_edges, self.c_z)
            knn_dst_edge_index = knn_dst_edge_index[keep_mask].reshape(-1, num_keep_edges)
            knn_dst_edge_mask = knn_dst_edge_mask[keep_mask].reshape(-1, num_keep_edges)
            k = num_keep_edges

        knn_src_edge_features = self.src_layer_norm(knn_src_edge_features)
        knn_dst_edge_features = self.dst_layer_norm(knn_dst_edge_features)

        node_left = self.node_left(node_features)
        node_right = self.node_right(node_features)

        # n_node x
        edge3_node1 = node_left[knn_src_edge_index[0]]
        edge3_node2 = node_right[knn_dst_edge_index[0]]

        # n_node x n_edge x 1 x hdim x 1 * n_node x 1 x n_edge x 1 x h_dim
        edge3 = edge3_node1[..., None, :, None] * edge3_node2[..., None, :, None, :]
        edge3_gate = self.dist_gate(edge3.flatten(-2, -1))

        node_trans = rigids.get_trans()
        edge3_node1_trans = node_trans[knn_src_edge_index[0]]
        edge3_node2_trans = node_trans[knn_dst_edge_index[0]]
        edge3_dist = torch.linalg.vector_norm(
            edge3_node1_trans[..., None, :] - edge3_node2_trans[..., None, :, :] + eps,
            dim=-1
        )

        edge3_dist_features = _rbf(edge3_dist, D_count=self.num_rbf, device=edge3_dist.device, dtype=self.dtype)
        edge3_dist_features = self.dist_proj(edge3_dist_features)  # n x k x k x h
        edge3_features = torch.sigmoid(edge3_gate) * edge3_dist_features
        edge_mask = knn_src_edge_mask[..., None] & knn_dst_edge_mask[..., None, :]
        edge3_features = edge3_features * edge_mask[..., None]

        edge2_features = self.edge_proj(knn_src_edge_features)
        edge2_gate = self.edge_gate(knn_src_edge_features)
        edge2_features = torch.sigmoid(edge2_gate) * edge2_features
        edge_update = edge3_features.permute(0, 3, 1, 2) @ edge2_features.permute(0, 2, 1)[..., None]
        edge_update = edge_update.squeeze(-1).permute(0, 2, 1)

        edge_update = self.lin_out(self.ln_out(edge_update))
        if self.gate_out:
            edge_update = edge_update * torch.sigmoid(self.out_gate(knn_dst_edge_features))

        edge_update, _ = knn_to_sparse_graph(knn_dst_edge_features, knn_dst_edge_index, knn_dst_edge_mask)
        # edge_update = edge_update.to(dtype=initial_dtype)

        return edge_update


# heavily inspired by proteus
class SparseTriangleCrossMultiplicativeUpdate(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_gate_s=16,
                 num_rbf=64,
                 dtype=torch.float32,
                 vpa=False,
                 gate_out=True,
                 dropout=0):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_gate_s = c_gate_s
        self.num_rbf = num_rbf
        self.gate_out = gate_out
        self.dropout_rate = dropout
        self.vpa = vpa

        self.node_left = Linear(c_s, c_gate_s, dtype=dtype)
        self.node_right = Linear(c_s, c_gate_s, dtype=dtype)
        self.bias_gate = Linear(c_gate_s*c_gate_s, c_z, init='gating', dtype=dtype)
        self.dist_bias = nn.Linear(self.num_rbf, c_z, dtype=dtype)

        self.dst_layer_norm = nn.LayerNorm(c_z, dtype=dtype)
        self.src_layer_norm = nn.LayerNorm(c_z, dtype=dtype)

        self.edge_proj = Linear(c_z, c_z, dtype=dtype)
        self.edge_gate = Linear(c_z, c_z, dtype=dtype)
        self.dist_gate = Linear(c_gate_s*c_gate_s, c_z, init='gating', dtype=dtype)
        self.dist_proj = Linear(self.num_rbf, c_z, dtype=dtype)
        self.ln_out = nn.LayerNorm(c_z, dtype=dtype)
        self.lin_out = Linear(c_z, c_z, init='final', dtype=dtype)
        if self.gate_out:
            self.out_gate = Linear(c_z, c_z, init='gating', dtype=dtype)

    # @torch.compile(dynamic=True)
    def _gen_edge_update(self,
                         node_features,
                         dst_edge_features,
                         dst_edge_index,
                         src_edge_features,
                         src_edge_index,
                         node_trans,
                         edge_edge_index, eps=1e-8):
        edge3_node1_idx = src_edge_index[0, edge_edge_index[1]]
        edge3_node2_idx = dst_edge_index[0, edge_edge_index[0]]

        dst_edge_features = self.dst_layer_norm(dst_edge_features)
        src_edge_features = self.src_layer_norm(src_edge_features)

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

        edge3_dist_features = _rbf(edge3_dist, D_count=self.num_rbf, device=edge3_dist.device)#, dtype=self.dtype)
        edge3_dist_features = self.dist_proj(edge3_dist_features)  # n x k x k x h
        edge3_features = torch.sigmoid(edge3_gate) * edge3_dist_features

        edge2_features = self.edge_proj(src_edge_features)
        edge2_gate = self.edge_gate(src_edge_features)
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
                edge_edge_index[0],
                dim_size=edge_update.shape[0]
            )
            reduce_factor[reduce_factor <= 0] = 1
            edge_update = edge_update / torch.sqrt(reduce_factor)[..., None]
        else:
            edge_update = self.ln_out(edge_update)

        edge_update = self.lin_out(edge_update)
        if self.gate_out:
            edge_update = edge_update * torch.sigmoid(self.out_gate(dst_edge_features))
        return edge_update


    def forward(self,
                node_features,
                rigids,
                dst_edge_features,
                dst_edge_index,
                src_edge_features,
                src_edge_index,
                k=None,
                eps=1e-8):
        num_nodes = node_features.shape[0]
        node_trans = rigids.get_trans()

        if k is None:
            # max for pygu knn
            k = 100

        edge_edge_index = knn(
            x=node_trans[dst_edge_index[0]],
            y=node_trans[src_edge_index[0]],
            k=k,
            batch_x=dst_edge_index[1],
            batch_y=src_edge_index[1]
        )
        edge_edge_index = edge_edge_index.flip(0)

        edge_edge_index = pygu.coalesce(edge_edge_index, sort_by_row=False)
        edge_edge_index, _ = pygu.dropout_edge(edge_index=edge_edge_index, p=self.dropout_rate, training=self.training)
        assert (src_edge_index[1, edge_edge_index[1]] == dst_edge_index[1, edge_edge_index[0]]).all()
        edge_update = self._gen_edge_update(
            node_features,
            dst_edge_features,
            dst_edge_index,
            src_edge_features,
            src_edge_index,
            node_trans,
            edge_edge_index
        )

        return edge_update