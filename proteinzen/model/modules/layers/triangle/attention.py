import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric.utils as pygu

from proteinzen.data.datasets.featurize.common import _rbf
from proteinzen.model.modules.openfold.layers import Linear
from proteinzen.model.utils.graph import sparse_to_knn_graph, knn_to_sparse_graph

from torch_cluster import knn, knn_graph

# heavily inspired by proteus
class TriangleSelfAttention(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_gate_s=16,
                 num_heads=4,
                 num_rbf=64,
                 inf=1e4,
                 gate_out=True,
                 dropout=0,
                 dtype=torch.bfloat16):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_gate_s = c_gate_s
        self.num_rbf = num_rbf
        self.num_heads = num_heads
        self.gate_out = gate_out
        self.dropout_rate = dropout
        self.dtype = dtype

        self.node_left = Linear(c_s, c_gate_s, dtype=dtype)
        self.node_right = Linear(c_s, c_gate_s, dtype=dtype)
        self.bias_gate = Linear(c_gate_s*c_gate_s, c_z, init='gating', dtype=dtype)
        self.dist_bias = nn.Linear(self.num_rbf, c_z, dtype=dtype)
        self.to_bias = Linear(c_z, num_heads, bias=False, init="normal", dtype=dtype)

        self.layer_norm = nn.LayerNorm(c_z, dtype=dtype)
        self.lin_q = Linear(c_z, c_z, init='glorot', dtype=dtype)
        self.lin_kv = Linear(c_z, 2*c_z, init='glorot', dtype=dtype)
        self.lin_out = Linear(c_z, c_z, init='final', dtype=dtype)
        if self.gate_out:
            self.out_gate = Linear(c_z, c_z, init='gating', dtype=dtype)

        self.inf = inf

    def _gen_qkv(self):
        pass

    def forward(self,
                node_features,
                rigids,
                edge_features,
                edge_index,
                flash=True,
                assume_sorted=False,
                eps=1e-8):
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[-1]

        init_dtype = edge_features.dtype

        node_features = node_features.to(self.dtype)
        edge_features = edge_features.to(self.dtype)

        if not assume_sorted:
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

        knn_edge_features, knn_edge_index, knn_edge_mask = sparse_to_knn_graph(sorted_edge_features, sorted_edge_index, num_nodes=num_nodes)
        k = knn_edge_index.shape[-1]
        if self.dropout_rate > 0 and self.training:
            num_drop_edges = round(k * self.dropout_rate)
            num_keep_edges = k - num_drop_edges
            idxs = np.repeat(np.arange(k)[None], knn_edge_features.shape[0], 0)
            # this is a kinda jenk way of doing this
            # but i think this will allow checkpointing to control the seed used for permutation
            torch_rng = torch.Generator()
            rng = np.random.default_rng(torch_rng.initial_seed())
            shuffled_idxs = rng.permutation(idxs, axis=1)
            keep_mask = torch.as_tensor(shuffled_idxs < num_keep_edges, device=knn_edge_features.device)
            knn_edge_features = knn_edge_features[keep_mask].reshape(-1, num_keep_edges, self.c_z)
            knn_edge_index = knn_edge_index[keep_mask].reshape(-1, num_keep_edges)
            knn_edge_mask = knn_edge_mask[keep_mask].reshape(-1, num_keep_edges)
            k = num_keep_edges

        knn_edge_features = self.layer_norm(knn_edge_features)

        edge1 = knn_edge_index[..., None]
        edge2 = knn_edge_index[..., None, :]

        node_left = self.node_left(node_features)
        node_right = self.node_right(node_features)

        edge3_node1 = node_left[knn_edge_index]
        edge3_node2 = node_right[knn_edge_index]
        edge3 = torch.einsum("bik,bjl->bikjl", edge3_node1, edge3_node2)
        edge3_gate = self.bias_gate(
            edge3.view(-1, k, k, self.c_gate_s * self.c_gate_s)
        )

        node_trans = rigids.get_trans().to(self.dtype)
        edge3_node1_trans = node_trans[edge1]
        edge3_node2_trans = node_trans[edge2]
        edge3_dist = torch.linalg.vector_norm(
            edge3_node1_trans - edge3_node2_trans + eps,
            dim=-1
        )

        edge3_dist_features = _rbf(edge3_dist, D_count=self.num_rbf, device=edge3_dist.device, dtype=self.dtype)
        edge3_dist_bias = self.dist_bias(edge3_dist_features)  # n x k x k x h
        edge3_dist_bias = self.to_bias(torch.sigmoid(edge3_gate) * edge3_dist_bias)

        edge_q = self.lin_q(knn_edge_features)
        edge_kv = self.lin_kv(knn_edge_features)
        edge_k, edge_v = edge_kv.split([self.c_z, self.c_z], dim=-1)
        if flash:
            edge_q = edge_q.view(num_nodes, k, self.num_heads, -1).transpose(-2, -3)
            edge_k = edge_k.view(num_nodes, k, self.num_heads, -1).transpose(-2, -3)
            edge_v = edge_v.view(num_nodes, k, self.num_heads, -1).transpose(-2, -3)

            edge_edge_mask = knn_edge_mask[..., None] & knn_edge_mask[..., None, :]  # n x k x k
            attn_bias = edge3_dist_bias.permute(0, 3, 1, 2) - self.inf * (~edge_edge_mask[:, None]).to(self.dtype)
            edge_update = F.scaled_dot_product_attention(
                query=edge_q,
                key=edge_k,
                value=edge_v,
                attn_mask=attn_bias
            )

        else:
            edge_q = edge_q.view(num_nodes, k, self.num_heads, -1)
            edge_k = edge_k.view(num_nodes, k, -1, self.num_heads)

            edge_attn = edge_q.transpose(-2, -3) @ edge_k.transpose(-1, -3)  # n x h x k x k
            edge_attn = edge_attn.transpose(-2, -3).transpose(-1, -2)  # n x k x k x h
            edge_attn = 1 / np.sqrt(self.c_z) * edge_attn
            edge_attn = edge_attn + edge3_dist_bias

            edge_edge_mask = knn_edge_mask[..., None] & knn_edge_mask[..., None, :]  # n x k x k
            edge_attn = edge_attn - self.inf * (~edge_edge_mask[..., None]).float()
            edge_attn = torch.softmax(edge_attn, dim=-2) # n x k x k x h

            edge_v = edge_v.view(num_nodes, k, self.num_heads, -1)  # n x k x h x c_z//h
            edge_update = torch.sum(
                edge_v[..., None, :, :]  # n x k x 1 x h x c_z//h
                * edge_attn[..., None]
                * edge_edge_mask[..., None, None],  # n x k x k x h x 1
            dim=-3)  # n x k x h x c_z//h

        edge_update = edge_update.view(-1, k, self.c_z)
        if self.gate_out:
            edge_update = edge_update * torch.sigmoid(self.out_gate(knn_edge_features))
        edge_update = self.lin_out(edge_update)
        edge_update, _ = knn_to_sparse_graph(edge_update, knn_edge_index, knn_edge_mask)

        if not assume_sorted:
            edge_update = pygu.scatter(
                edge_update,
                index=edge_map,
                dim=0,
                dim_size=num_edges
            )

        edge_update = edge_update.to(init_dtype)

        return edge_update


# heavily inspired by proteus
class SparseTriangleSelfAttention(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_gate_s=16,
                 num_heads=4,
                 num_rbf=64,
                 inf=1e4,
                 gate_out=True,
                 dropout=0,
                 # dtype=torch.bfloat16):
                 dtype=torch.float32):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_gate_s = c_gate_s
        self.num_rbf = num_rbf
        self.num_heads = num_heads
        self.gate_out = gate_out
        self.dropout_rate = dropout
        self.dtype = dtype

        self.node_left = Linear(c_s, c_gate_s, dtype=dtype)
        self.node_right = Linear(c_s, c_gate_s, dtype=dtype)
        self.bias_gate = Linear(c_gate_s*c_gate_s, c_z, init='gating', dtype=dtype)
        self.dist_bias = nn.Linear(self.num_rbf, c_z, dtype=dtype)
        self.to_bias = Linear(c_z, num_heads, bias=False, init="normal", dtype=dtype)

        self.layer_norm = nn.LayerNorm(c_z, dtype=dtype)
        self.lin_q = Linear(c_z, c_z, init='glorot', dtype=dtype)
        self.lin_kv = Linear(c_z, 2*c_z, init='glorot', dtype=dtype)
        self.lin_out = Linear(c_z, c_z, init='final', dtype=dtype)
        if self.gate_out:
            self.out_gate = Linear(c_z, c_z, init='gating', dtype=dtype)

        self.inf = inf

    # @torch.compile(dynamic=True)
    def _gen_edge_update(self, node_features, node_trans, edge_features, edge_index, edge_edge_index, eps=1e-8):
        edge3_node1_idx = edge_index[0, edge_edge_index[1]]
        edge3_node2_idx = edge_index[0, edge_edge_index[0]]

        edge_features = self.layer_norm(edge_features)
        node_left = self.node_left(node_features)
        node_right = self.node_right(node_features)

        edge3_node1 = node_left[edge3_node1_idx]
        edge3_node2 = node_right[edge3_node2_idx]
        edge3 = torch.einsum("bi,bj->bij", edge3_node1, edge3_node2)
        edge3_gate = self.bias_gate(
            edge3.view(-1, self.c_gate_s * self.c_gate_s)
        )

        edge3_node1_trans = node_trans[edge3_node1_idx]
        edge3_node2_trans = node_trans[edge3_node2_idx]
        edge3_dist = torch.linalg.vector_norm(
            edge3_node1_trans - edge3_node2_trans + eps,
            dim=-1
        )

        edge3_dist_features = _rbf(edge3_dist, D_count=self.num_rbf, device=edge3_dist.device, dtype=self.dtype)
        edge3_dist_bias = self.dist_bias(edge3_dist_features)  # n x k x k x h
        edge3_dist_bias = self.to_bias(torch.sigmoid(edge3_gate) * edge3_dist_bias)

        edge_q = self.lin_q(edge_features)
        edge_kv = self.lin_kv(edge_features)
        edge_k, edge_v = edge_kv.split([self.c_z, self.c_z], dim=-1)
        edge_q = edge_q.view(-1, self.num_heads, self.c_z//self.num_heads)
        edge_k = edge_k.view(-1, self.num_heads, self.c_z//self.num_heads)

        edge_edge_q = edge_q[edge_edge_index[1]]
        edge_edge_k = edge_k[edge_edge_index[0]]

        attn = torch.einsum("bhi,bhi->bh", edge_edge_q, edge_edge_k)  # n_e_e x h
        attn = 1 / np.sqrt(self.c_z) * attn
        attn = attn + edge3_dist_bias
        attn = pygu.softmax(
            attn,
            index=edge_edge_index[1],
            dim=0
        )

        edge_v = edge_v.view(-1, self.num_heads, self.c_z//self.num_heads)  # n x h x c_z//h
        edge_edge_v = edge_v[edge_edge_index[0]]
        edge_update = pygu.scatter(
            edge_edge_v * attn[..., None],
            index=edge_edge_index[1],
            dim=0,
            dim_size=edge_features.shape[0]
        )
        edge_update = edge_update.view(-1, self.c_z)
        if self.gate_out:
            edge_update = edge_update * torch.sigmoid(self.out_gate(edge_features))
        edge_update = self.lin_out(edge_update)
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
        input_dtype = edge_features.dtype

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
        edge_edge_index = pygu.coalesce(edge_edge_index, sort_by_row=False)
        edge_edge_index, _ = pygu.dropout_edge(edge_index=edge_edge_index, p=self.dropout_rate, training=self.training)
        assert (edge_index[1, edge_edge_index[1]] == edge_index[1, edge_edge_index[0]]).all()
        node_trans = node_trans.to(self.dtype)
        edge_update = self._gen_edge_update(node_features, node_trans, edge_features, edge_index, edge_edge_index)
        # edge3_node1_idx = edge_index[0, edge_edge_index[1]]
        # edge3_node2_idx = edge_index[0, edge_edge_index[0]]

        # edge_features = self.layer_norm(edge_features)
        # node_left = self.node_left(node_features)
        # node_right = self.node_right(node_features)

        # edge3_node1 = node_left[edge3_node1_idx]
        # edge3_node2 = node_right[edge3_node2_idx]
        # edge3 = torch.einsum("bi,bj->bij", edge3_node1, edge3_node2)
        # edge3_gate = self.bias_gate(
        #     edge3.view(-1, self.c_gate_s * self.c_gate_s)
        # )

        # node_trans = node_trans.to(self.dtype)
        # edge3_node1_trans = node_trans[edge3_node1_idx]
        # edge3_node2_trans = node_trans[edge3_node2_idx]
        # edge3_dist = torch.linalg.vector_norm(
        #     edge3_node1_trans - edge3_node2_trans + eps,
        #     dim=-1
        # )

        # edge3_dist_features = _rbf(edge3_dist, D_count=self.num_rbf, device=edge3_dist.device, dtype=self.dtype)
        # edge3_dist_bias = self.dist_bias(edge3_dist_features)  # n x k x k x h
        # edge3_dist_bias = self.to_bias(torch.sigmoid(edge3_gate) * edge3_dist_bias)

        # edge_q = self.lin_q(edge_features)
        # edge_kv = self.lin_kv(edge_features)
        # edge_k, edge_v = edge_kv.split([self.c_z, self.c_z], dim=-1)
        # edge_q = edge_q.view(-1, self.num_heads, self.c_z//self.num_heads)
        # edge_k = edge_k.view(-1, self.num_heads, self.c_z//self.num_heads)

        # edge_edge_q = edge_q[edge_edge_index[1]]
        # edge_edge_k = edge_k[edge_edge_index[0]]

        # attn = torch.einsum("bhi,bhi->bh", edge_edge_q, edge_edge_k)  # n_e_e x h
        # attn = 1 / np.sqrt(self.c_z) * attn
        # attn = attn + edge3_dist_bias
        # attn = pygu.softmax(
        #     attn,
        #     index=edge_edge_index[1],
        #     dim=0
        # )

        # edge_v = edge_v.view(-1, self.num_heads, self.c_z//self.num_heads)  # n x h x c_z//h
        # edge_edge_v = edge_v[edge_edge_index[0]]
        # edge_update = pygu.scatter(
        #     edge_edge_v * attn[..., None],
        #     index=edge_edge_index[1],
        #     dim=0,
        #     dim_size=edge_features.shape[0]
        # )
        # edge_update = edge_update.view(-1, self.c_z)
        # if self.gate_out:
        #     edge_update = edge_update * torch.sigmoid(self.out_gate(edge_features))
        # edge_update = self.lin_out(edge_update)

        edge_update = edge_update.to(input_dtype)
        return edge_update


class TriangleCrossAttention(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_gate_s=16,
                 num_heads=4,
                 num_rbf=64,
                 inf=1e4,
                 dropout=0,
                 gate_out=True):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_gate_s = c_gate_s
        self.num_rbf = num_rbf
        self.num_heads = num_heads
        self.gate_out = gate_out
        self.dropout_rate = dropout

        self.node_left = Linear(c_s, c_gate_s)
        self.node_right = Linear(c_s, c_gate_s)
        self.bias_gate = Linear(c_gate_s*c_gate_s, c_z, init='gating')
        self.dist_bias = nn.Linear(self.num_rbf, c_z)
        self.to_bias = Linear(c_z, num_heads, bias=False, init="normal")

        self.dst_layer_norm = nn.LayerNorm(c_z)
        self.src_layer_norm = nn.LayerNorm(c_z)
        self.lin_q = Linear(c_z, c_z, init='glorot')
        self.lin_kv = Linear(c_z, 2*c_z, init='glorot')
        self.lin_out = Linear(c_z, c_z, init='final')
        if self.gate_out:
            self.out_gate = Linear(c_z, c_z, init='gating')

        self.inf = inf

    # @torch.compile(dynamic=True)
    def forward(self,
                node_features,
                rigids,
                transfer_index,
                transfer_k,
                old_edge_features,
                old_edge_index,
                # knn_src_edge_features,
                # knn_src_edge_index,
                dst_edge_features,
                dst_edge_index,
                flash=True,
                knn_edge_mask=None,
                eps=1e-8):
        num_nodes = node_features.shape[0]
        num_edges = dst_edge_features.shape[0]
        transfer_edge_features = old_edge_features[transfer_index[0]]
        transfer_edge_index = old_edge_index[:, transfer_index[0]]

        # this isn't resistant to bugs but it should be much faster
        num_edges = dst_edge_features.shape[0]
        knn_src_edge_features = transfer_edge_features.view(num_edges, transfer_k, -1)
        knn_src_edge_index = transfer_edge_index.view(2, num_edges, transfer_k)
        k = knn_src_edge_index.shape[-1]

        if knn_edge_mask is None:
            knn_edge_mask = torch.ones_like(knn_src_edge_index[0]).bool()
        elif len(knn_edge_mask.shape) == 4:
            knn_edge_mask = knn_edge_mask[0]

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
            knn_edge_mask = knn_edge_mask[keep_mask].reshape(-1, num_keep_edges)
            k = num_keep_edges

        # knn_src_edge_features = self.src_layer_norm(knn_src_edge_features)
        dst_edge_features = self.dst_layer_norm(dst_edge_features)

        edge1 = dst_edge_index[..., None]
        edge2 = knn_src_edge_index

        node_left = self.node_left(node_features)
        node_right = self.node_right(node_features)

        edge3_node1 = node_left[edge1[0]]
        edge3_node2 = node_right[edge2[0]]
        edge3 = torch.einsum("bik,bjl->bijkl", edge3_node1, edge3_node2)
        edge3_gate = self.bias_gate(
            edge3.view(-1, k, self.c_gate_s * self.c_gate_s)
        )

        node_trans = rigids.get_trans()
        edge3_node1_trans = node_trans[edge1[0]]
        edge3_node2_trans = node_trans[edge2[0]]
        edge3_dist = torch.linalg.vector_norm(
            edge3_node1_trans - edge3_node2_trans + eps,
            dim=-1
        )

        edge3_dist_features = _rbf(edge3_dist, D_count=self.num_rbf, device=edge3_dist.device)
        edge3_dist_bias = self.dist_bias(edge3_dist_features)  # n x k x k x h
        edge3_dist_bias = self.to_bias(torch.sigmoid(edge3_gate) * edge3_dist_bias)

        edge_q = self.lin_q(dst_edge_features)
        edge_kv = self.lin_kv(knn_src_edge_features)
        edge_q = edge_q.view(num_edges, 1, self.num_heads, -1)  # n_edge x 1 x h x c_z//h
        edge_k, edge_v = edge_kv.split([self.c_z, self.c_z], dim=-1)
        edge_k = edge_k.view(num_edges, k, -1, self.num_heads)  # n_edge x k x c_z//h x h
        edge_v = edge_v.view(num_edges, k, self.num_heads, -1)  # n_edge x k x h x c_z//h

        if flash:
            attn_bias = edge3_dist_bias - self.inf * (1 - knn_edge_mask.float()[..., None])
            attn_bias = attn_bias.transpose(-1, -2)[..., None, :]
            # n_node x h x 1 x c
            edge_update = F.scaled_dot_product_attention(
                edge_q.transpose(1, 2),
                edge_k.permute(0, 3, 1, 2),
                edge_v.transpose(1, 2),
                attn_mask=attn_bias
            )
            edge_update = edge_update.squeeze(-2).transpose(-1, -2).contiguous()
        else:
            edge_attn = edge_q.transpose(-2, -3) @ edge_k.transpose(-1, -3)  # n_edge x h x 1 x k
            edge_attn = edge_attn.squeeze(-2).transpose(-1, -2)  # n_edge x k x h
            edge_attn = 1 / np.sqrt(self.c_z) * edge_attn
            edge_attn = edge_attn + edge3_dist_bias - self.inf * (1 - knn_edge_mask.float()[..., None])
            edge_attn = torch.softmax(edge_attn, dim=-2) # n_edge x k x h

            edge_update = torch.sum(
                edge_v * edge_attn[..., None] * knn_edge_mask.float()[..., None, None],
                dim=-3
            )  # n_edge x h x c_z//h
            # print(edge_update.shape)
        edge_update = edge_update.view(num_edges, self.c_z)
        if self.gate_out:
            edge_update = edge_update * torch.sigmoid(self.out_gate(dst_edge_features))
        edge_update = self.lin_out(edge_update)

        return edge_update


# heavily inspired by proteus
class SparseTriangleCrossAttention(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_gate_s=16,
                 num_heads=4,
                 num_rbf=64,
                 inf=1e4,
                 gate_out=True,
                 dropout=0):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_gate_s = c_gate_s
        self.num_rbf = num_rbf
        self.num_heads = num_heads
        self.gate_out = gate_out
        self.dropout_rate = dropout

        self.node_left = Linear(c_s, c_gate_s)
        self.node_right = Linear(c_s, c_gate_s)
        self.bias_gate = Linear(c_gate_s*c_gate_s, c_z, init='gating')
        self.dist_bias = nn.Linear(self.num_rbf, c_z)
        self.to_bias = Linear(c_z, num_heads, bias=False, init="normal")

        self.dst_layer_norm = nn.LayerNorm(c_z)
        self.src_layer_norm = nn.LayerNorm(c_z)
        self.lin_q = Linear(c_z, c_z, init='glorot')
        self.lin_kv = Linear(c_z, 2*c_z, init='glorot')
        self.lin_out = Linear(c_z, c_z, init='final')
        if self.gate_out:
            self.out_gate = Linear(c_z, c_z, init='gating')

        self.inf = inf

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
        edge3_node1_idx = src_edge_index[0, edge_edge_index[1]]
        edge3_node2_idx = dst_edge_index[0, edge_edge_index[0]]

        dst_edge_features = self.dst_layer_norm(dst_edge_features)
        src_edge_features = self.src_layer_norm(src_edge_features)

        node_left = self.node_left(node_features)
        node_right = self.node_right(node_features)

        edge3_node1 = node_left[edge3_node1_idx]
        edge3_node2 = node_right[edge3_node2_idx]
        edge3 = torch.einsum("bi,bj->bij", edge3_node1, edge3_node2)
        edge3_gate = self.bias_gate(
            edge3.view(-1, self.c_gate_s * self.c_gate_s)
        )

        edge3_node1_trans = node_trans[edge3_node1_idx]
        edge3_node2_trans = node_trans[edge3_node2_idx]
        edge3_dist = torch.linalg.vector_norm(
            edge3_node1_trans - edge3_node2_trans + eps,
            dim=-1
        )

        edge3_dist_features = _rbf(edge3_dist, D_count=self.num_rbf, device=edge3_dist.device)
        edge3_dist_bias = self.dist_bias(edge3_dist_features)  # n x k x k x h
        edge3_dist_bias = self.to_bias(torch.sigmoid(edge3_gate) * edge3_dist_bias)

        edge_q = self.lin_q(src_edge_features)
        edge_kv = self.lin_kv(dst_edge_features)
        edge_k, edge_v = edge_kv.split([self.c_z, self.c_z], dim=-1)
        edge_q = edge_q.view(-1, self.num_heads, self.c_z//self.num_heads)
        edge_k = edge_k.view(-1, self.num_heads, self.c_z//self.num_heads)

        edge_edge_q = edge_q[edge_edge_index[1]]
        edge_edge_k = edge_k[edge_edge_index[0]]

        attn = torch.einsum("bhi,bhi->bh", edge_edge_q, edge_edge_k)  # n_e_e x h
        attn = 1 / np.sqrt(self.c_z) * attn
        attn = attn + edge3_dist_bias
        attn = pygu.softmax(
            attn,
            index=edge_edge_index[1],
            dim=0
        )

        edge_v = edge_v.view(-1, self.num_heads, self.c_z//self.num_heads)  # n x h x c_z//h
        edge_edge_v = edge_v[edge_edge_index[0]]
        edge_update = pygu.scatter(
            edge_edge_v * attn[..., None],
            index=edge_edge_index[1],
            dim=0,
            dim_size=src_edge_features.shape[0]
        )
        edge_update = edge_update.view(-1, self.c_z)
        if self.gate_out:
            edge_update = edge_update * torch.sigmoid(self.out_gate(src_edge_features))
        edge_update = self.lin_out(edge_update)

        return edge_update
