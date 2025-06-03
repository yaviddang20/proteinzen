import abc

import torch
from torch import nn

from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.nn import nearest
from torch_geometric.nn.models import GAT
import torch_geometric.utils as pygu

from proteinzen.model.modules.openfold.layers import Linear
from proteinzen.data.datasets.featurize.common import _rbf


class PoolUpdate(nn.Module, abc.ABC):
    @abc.abstractmethod
    def select(self, node_x, node_features, edge_index, batch):
        pass

    @abc.abstractmethod
    def connect(self,
                anchor_x,
                anchor_features,
                node_x,
                node_features,
                edge_index,
                batch, anchor_batch):
        pass

    @abc.abstractmethod
    def node_to_anchor(self,
                anchor_x,
                anchor_features,
                node_x,
                node_features,
                n2a_edge_index):
        pass

    @abc.abstractmethod
    def anchor_update(self,
                anchor_x,
                anchor_features):
        pass

    @abc.abstractmethod
    def anchor_to_node(self,
                       anchor_x,
                       anchor_features,
                       node_x,
                       node_features,
                       a2n_edge_index):
        pass

    def forward(self, node_x, node_features, edge_index, batch):
        anchor_x, anchor_features, anchor_batch = self.select(node_x, node_features, edge_index, batch)
        a2n_edge_index, a2n_edge_features, a2a_edge_index, a2a_edge_features, anchor_proj_kl, node_proj_kl = self.connect(anchor_x, anchor_features, node_x, node_features, edge_index, batch, anchor_batch)
        n2a_edge_index = torch.flip(a2n_edge_index, dims=(0,))
        n2a_edge_features = -a2n_edge_features
        anchor_features = self.node_to_anchor(anchor_x, anchor_features, node_x, node_features, a2n_edge_index, a2n_edge_features)
        anchor_features = self.anchor_update(anchor_x, anchor_features, a2a_edge_index, a2a_edge_features, anchor_batch)
        node_features = self.anchor_to_node(anchor_x, anchor_features, node_x, node_features, n2a_edge_index, n2a_edge_features)
        # print(anchor_proj_kl, node_proj_kl)
        return node_features, anchor_proj_kl, node_proj_kl


class Node2AnchorAttention(nn.Module):
    def __init__(self, h_node, h_edge):
        super().__init__()
        self.h_node = h_node
        self.q_anchor = nn.Linear(h_node, h_node)
        self.kv_node = nn.Linear(h_node + h_edge, 2 * h_node)

    def forward(self, anchor_features, node_features, a2n_edge_index, a2n_edge_features):
        anchor_q = self.q_anchor(anchor_features)
        node_dst = torch.cat([node_features[a2n_edge_index[0]], a2n_edge_features], dim=-1)

        anchor_q_src = anchor_q[a2n_edge_index[1]]
        node_kv_dst = self.kv_node(node_dst)
        node_k_dst, node_v_dst = node_kv_dst.split(self.h_node, dim=-1)

        attn = torch.sum(anchor_q_src * node_k_dst, dim=-1)
        attn = pygu.softmax(
            attn,
            a2n_edge_index[1]
        )

        anchor_update = pygu.scatter(
            attn[..., None] * node_v_dst,
            a2n_edge_index[1],
            dim=0,
            dim_size=anchor_features.shape[0]
        )
        return anchor_update


class ProjectivePoolUpdate(PoolUpdate):
    def __init__(self,
                 h_node,
                 ratio,
                 connect_dim):
        super().__init__()
        self.h_node = h_node
        self.ratio = ratio

        # select
        self.score_layer = nn.Sequential(
            nn.Linear(h_node, h_node),
            nn.ReLU(),
            nn.Linear(h_node, h_node),
            nn.ReLU(),
            # nn.Linear(h_node, 1)
        )
        self.topk = SelectTopK(h_node, ratio=ratio)

        # connect
        self.project_anchor = nn.Linear(h_node, connect_dim)
        self.project_node = nn.Linear(h_node, connect_dim)

        # n2a
        self.n2a = Node2AnchorAttention(h_node, connect_dim)
        self.n2a_ln = nn.LayerNorm(h_node)

        # a2a
        self.a2a = GAT(
            in_channels=h_node,
            hidden_channels=h_node*2,
            out_channels=h_node,
            num_layers=2,
            edge_dim=connect_dim
        )

        # a2n
        self.a2n = nn.Sequential(
            nn.Linear(2 * h_node + connect_dim, h_node),
            nn.ReLU(),
            nn.Linear(h_node, h_node),
            nn.ReLU(),
            Linear(h_node, h_node, init="final")
        )
        self.a2n_ln = nn.LayerNorm(h_node)


    def select(self, node_x, node_features, edge_index, batch):
        score_vec = self.score_layer(node_features)
        select_out = self.topk(score_vec, batch)

        perm = select_out.node_index
        score = select_out.weight
        assert score is not None

        out = torch.arange(select_out.num_clusters, device=batch.device)
        anchor_batch = out.scatter_(0, select_out.cluster_index,
                            batch[select_out.node_index])

        anchor_x = node_x[perm]
        anchor_features = score_vec[perm] * score.view(-1, 1)
        return anchor_x, anchor_features, anchor_batch

    def connect(self,
                anchor_x,
                anchor_features,
                node_x,
                node_features,
                edge_index,
                batch, anchor_batch):
        anchor_proj = self.project_anchor(anchor_features)
        node_proj = self.project_node(node_features)

        anchor_kl = []
        for i in range(anchor_batch.max().item() + 1):
            select = (anchor_batch == i)
            anchors = anchor_proj[select]
            mu = anchors.mean(dim=0)
            var = anchors.var(dim=0)
            kl = 0.5 * (var + mu.square() - 1 - var.log()).sum()
            anchor_kl.append(kl)
        anchor_kl = torch.stack(anchor_kl)

        node_kl = []
        for i in range(batch.max().item() + 1):
            select = (batch == i)
            nodes = node_proj[select]
            mu = nodes.mean(dim=0)
            var = nodes.var(dim=0)
            kl = 0.5 * (var + mu.square() - 1 - var.log()).sum()
            node_kl.append(kl)
        node_kl = torch.stack(node_kl)

        a2n_anchors = nearest(node_proj, anchor_proj, batch, anchor_batch)
        a2n_nodes = torch.arange(node_features.shape[0], device=node_features.device)
        a2n_edge_index = torch.stack([a2n_nodes, a2n_anchors])

        a2a_edge_index = []
        a2n_dist_score = []
        offset = 0
        for i in range(anchor_batch.max().item() + 1):
            # gen a2a_edge_index per graph
            select = (anchor_batch == i)
            num_node = select.sum().long().item()
            dst = torch.arange(num_node, device=select.device)
            dst = torch.tile(dst, dims=(num_node, 1))
            src = dst.T
            anchor_edge_index = torch.stack(
                [dst.reshape(-1), src.reshape(-1)]
            )
            a2a_edge_index.append(anchor_edge_index + offset)

            node_select = (batch == i)
            node_subset = node_proj[node_select]  # n_node x h_dim
            anchor_subset = anchor_proj[select]  # n_anchor x h_dim
            proj_dist_mat = torch.linalg.vector_norm(node_subset[:, None] - anchor_subset[None], dim=-1)
            proj_dist_score = torch.softmax(-proj_dist_mat, dim=-1)
            proj_dist_score = torch.gather(
                proj_dist_score,
                1,
                (a2n_anchors[node_select] - offset)[..., None]
            ).squeeze(-1)
            a2n_dist_score.append(
                proj_dist_score
            )

            offset += num_node


        a2a_edge_index = torch.cat(a2a_edge_index, dim=-1)
        a2n_dist_score = torch.cat(a2n_dist_score, dim=-1)

        a2a_edge_features = anchor_proj[a2a_edge_index[0]] - anchor_proj[a2a_edge_index[1]]
        a2n_edge_features = node_proj[a2n_edge_index[0]] - anchor_proj[a2n_edge_index[1]]
        a2n_edge_features = a2n_edge_features * a2n_dist_score[..., None]

        return a2n_edge_index, a2n_edge_features, a2a_edge_index, a2a_edge_features, anchor_kl, node_kl

    def node_to_anchor(self,
                anchor_x,
                anchor_features,
                node_x,
                node_features,
                a2n_edge_index,
                a2n_edge_features):
        anchor_update = self.n2a(anchor_features, node_features, a2n_edge_index, a2n_edge_features)
        anchor_features = self.n2a_ln(anchor_features + anchor_update)
        return anchor_features

    def anchor_update(self,
                anchor_x,
                anchor_features,
                a2a_edge_index,
                a2a_edge_features,
                anchor_batch):

        return self.a2a(anchor_features, a2a_edge_index, edge_attr=a2a_edge_features)

    def anchor_to_node(self,
                       anchor_x,
                       anchor_features,
                       node_x,
                       node_features,
                       n2a_edge_index,
                       n2a_edge_features):
        node_expand = node_features[n2a_edge_index[1]]
        anchor_expand = anchor_features[n2a_edge_index[0]]

        node_update = self.a2n(
            torch.cat(
                [node_expand, anchor_expand, n2a_edge_features], dim=-1
            )
        )
        # this is probably redundant but i'll just keep it here
        node_update = pygu.scatter(
            node_update,
            n2a_edge_index[1],
            dim=0,
            dim_size=node_features.shape[0]
        )

        node_features = self.n2a_ln(node_features + node_update)
        return node_features


class Node2AnchorSetAttentionUpdate(nn.Module):
    def __init__(self, h_node, h_edge, inf=1e6):
        super().__init__()
        self.h_node = h_node
        self.h_edge = h_edge
        self.q_anchor = nn.Linear(h_node, h_node)
        self.kv_node = nn.Linear(h_node + h_edge, 2 * h_node)
        self.ln1 = nn.LayerNorm(h_node)
        self.inf = inf

        self.mlp = nn.Sequential(
            nn.Linear(h_node, h_node * 2),
            nn.ReLU(),
            nn.Linear(h_node * 2, h_node*2),
            nn.ReLU(),
            Linear(h_node*2, h_node, init='final')
        )
        self.ln2 = nn.LayerNorm(h_node)

    def forward(self, anchor_x, node_x, anchor_features, node_features, batch, anchor_batch, node_mask, eps=1e-8):
        anchor_q = self.q_anchor(anchor_features)

        anchor_update = []
        for i in range(batch.max().item() + 1):
            select_node = (batch == i)
            select_anchor = (anchor_batch == i)
            anchor_q_src = anchor_q[select_anchor] # n_anchor x h_node
            mask_subset = node_mask[select_node]

            a2n_dist_mat = torch.linalg.vector_norm(
                anchor_x[select_anchor][:, None] - node_x[select_node][None] + eps, dim=-1
            )  # n_anchor x n_node
            edge_embed = _rbf(a2n_dist_mat / 10, D_count=self.h_edge, device=a2n_dist_mat.device)  # n_anchor x n_node x h_edge

            node_dst = node_features[select_node]
            node_dst_expand = node_dst[None].expand(edge_embed.shape[0], -1, -1)
            node_dst_expand = torch.cat([node_dst_expand, edge_embed], dim=-1)

            node_kv_dst = self.kv_node(node_dst_expand)  # n_node x h_node
            node_k_dst, node_v_dst = node_kv_dst.split(self.h_node, dim=-1)
            attn = (anchor_q_src[..., None, :] @ node_k_dst.transpose(-1, -2)).squeeze(-2)  # n_anchor x n_node
            attn = attn * ((mask_subset[None] - 1) * self.inf)
            attn = torch.softmax(attn, dim=-1)
            anchor_update.append(
                (attn[..., None] * node_v_dst).sum(dim=-2)
            )

        anchor_update = torch.cat(anchor_update, dim=0)
        anchor_features = self.ln1(anchor_features + anchor_update)

        anchor_update = self.mlp(anchor_features)
        anchor_features = self.ln2(anchor_features + anchor_update)

        return anchor_features


class AnchorSet2NodeMPNN(nn.Module):
    def __init__(self, h_node, h_edge):
        super().__init__()
        self.h_node = h_node
        self.h_edge = h_edge
        self.mlp1 = nn.Sequential(
            nn.Linear(h_node*2 + h_edge, h_node * 2),
            nn.ReLU(),
            nn.Linear(h_node * 2, h_node*2),
            nn.ReLU(),
            Linear(h_node*2, h_node, init='final')
        )
        self.ln1 = nn.LayerNorm(h_node)

        self.mlp2 = nn.Sequential(
            nn.Linear(h_node, h_node * 2),
            nn.ReLU(),
            nn.Linear(h_node * 2, h_node*2),
            nn.ReLU(),
            Linear(h_node*2, h_node, init='final')
        )
        self.ln2 = nn.LayerNorm(h_node)

    def forward(self, anchor_x, node_x, anchor_features, node_features, batch, anchor_batch, node_mask, eps=1e-8):
        node_update = []
        for i in range(batch.max().item() + 1):
            select_node = (batch == i)
            select_anchor = (anchor_batch == i)
            node_src = node_features[select_node]
            anchor_dst = anchor_features[select_anchor]

            n2a_dist_mat = torch.linalg.vector_norm(
                node_x[select_node][:, None] - anchor_x[select_anchor][None] + eps, dim=-1
            )  # n_anchor x n_node
            edge_embed = _rbf(n2a_dist_mat / 10, D_count=self.h_edge, device=n2a_dist_mat.device)  # n_node x n_anchor x h_edge

            node_src_expand = node_src[:, None].expand(-1, anchor_dst.shape[0], -1)
            anchor_dst_expand = anchor_dst[None].expand(node_src.shape[0], -1, -1)
            update_input = torch.cat([node_src_expand, anchor_dst_expand, edge_embed], dim=-1)
            node_msg = self.mlp1(update_input)
            node_update.append(
                node_msg.mean(dim=-2)
            )
        node_update = torch.cat(node_update, dim=0)
        node_features = self.ln1(node_features + node_update * node_mask[..., None])

        node_update = self.mlp2(node_features)
        node_features = self.ln2(node_features + node_update * node_mask[..., None])

        return node_features


class AnchorUpdate(nn.Module):
    def __init__(self,
                 h_node,
                 h_edge,
                 ratio):
        super().__init__()
        self.h_node = h_node
        self.ratio = ratio

        # select
        self.score_layer = nn.Sequential(
            nn.Linear(h_node, h_node),
            nn.ReLU(),
            nn.Linear(h_node, h_node),
            nn.ReLU(),
            # nn.Linear(h_node, 1)
        )
        self.topk = SelectTopK(h_node, ratio=ratio)

        # n2a
        self.n2a = Node2AnchorSetAttentionUpdate(h_node, h_edge)
        self.n2a_ln = nn.LayerNorm(h_node)

        # a2a
        self.a2a = nn.ModuleList(
            [
                Node2AnchorSetAttentionUpdate(h_node, h_edge)
                for _ in range(2)
            ]
        )

        # a2n
        self.a2n = AnchorSet2NodeMPNN(h_node, h_edge)

    def select(self, node_x, node_features, edge_index, batch, node_mask):
        score_vec = self.score_layer(node_features * node_mask[..., None])
        select_out = self.topk(score_vec, batch)

        perm = select_out.node_index
        score = select_out.weight
        assert score is not None

        out = torch.arange(select_out.num_clusters, device=batch.device)
        anchor_batch = out.scatter_(0, select_out.cluster_index,
                            batch[select_out.node_index])

        anchor_x = node_x[perm]
        anchor_features = score_vec[perm] * score.view(-1, 1)
        return anchor_x, anchor_features, anchor_batch

    def forward(self, node_x, node_features, edge_index, batch, node_mask, eps=1e-8):
        anchor_x, anchor_features, anchor_batch = self.select(node_x, node_features, edge_index, batch, node_mask)

        anchor_mask = torch.ones(anchor_x.shape[0], device=anchor_x.device).float()
        anchor_features = self.n2a(anchor_x, node_x, anchor_features, node_features, batch, anchor_batch, node_mask)
        for layer in self.a2a:
            anchor_features = layer(anchor_x, anchor_x, anchor_features, anchor_features, anchor_batch, anchor_batch, anchor_mask)
        node_features = self.a2n(anchor_x, node_x, anchor_features, node_features, batch, anchor_batch, node_mask)
        anchor_proj_kl = torch.zeros(batch.max().item() + 1, device=batch.device)
        node_proj_kl = torch.zeros(batch.max().item() + 1, device=batch.device)

        return node_features, anchor_proj_kl, node_proj_kl
