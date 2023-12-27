from ligbinddiff.model.modules.openfold.frames import Linear


import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeTransition(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2,
            dropout=0.
        ):
        super().__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, node_embed, edge_embed, edge_index):
        node_embed = self.initial_embed(node_embed)

        node_dst = node_embed[edge_index[0]]
        node_src = node_embed[edge_index[1]]

        edge_bias = torch.cat([
            node_dst,
            node_src
        ], dim=-1)

        edge_embed = torch.cat([edge_embed, edge_bias], dim=-1)
        edge_update = self.trunk(edge_embed)
        if self.dropout is not None:
            edge_update = self.dropout(edge_update)
        edge_embed = self.final_layer(edge_update + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        return edge_embed


class EdgeGate(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super().__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        trunk_layers.append(Linear(hidden_size, edge_embed_out, init="final"))
        self.trunk = nn.Sequential(*trunk_layers)

    def forward(self, node_embed, edge_embed, edge_index):
        node_embed = self.initial_embed(node_embed)

        node_dst = node_embed[edge_index[0]]
        node_src = node_embed[edge_index[1]]

        edge_bias = torch.cat([
            node_dst,
            node_src
        ], dim=-1)

        trunk_in = torch.cat([edge_embed, edge_bias], dim=-1)
        gating = F.sigmoid(self.trunk(trunk_in))

        return gating * edge_embed
