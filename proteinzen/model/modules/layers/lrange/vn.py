import numpy as np
import torch
from torch import nn

import torch_geometric.utils as pygu

from proteinzen.model.modules.openfold.layers import Linear


class VirtualNodeGatherAttn(nn.Module):
    def __init__(self,
                 c_s,
                 c_attn,
                 num_heads=4,
                 inf=1e8):
        super().__init__()
        self.c_s = c_s
        self.c_attn = c_attn
        self.num_heads = num_heads
        self.inf = inf

        self.lin_q = nn.Linear(c_s, c_attn * num_heads)
        self.lin_kv = nn.Linear(
            c_s,
            2 * c_attn * num_heads)

        self.lin_out = Linear(
            c_attn * num_heads,
            c_s,
            init='final'
        )

    def forward(self, node_features, vn_features, batch, node_mask):
        num_graphs, num_vn = vn_features.shape[:2]
        # B x N_vn x H x c_attn
        vn_q = self.lin_q(vn_features).view(num_graphs, num_vn, self.num_heads, self.c_attn)
        # N x H x c_attn
        node_kv = self.lin_kv(node_features).view(-1, self.num_heads, 2* self.c_attn)
        node_k, node_v = node_kv.split(self.c_attn, dim=-1)

        # N x N_vn x H x c_attn
        nodewise_vn_q = vn_q[batch]
        # N x N_vn x H
        attn = torch.einsum('...vhi, ...hi->...vh', nodewise_vn_q, node_k) / np.sqrt(self.c_attn)
        attn = attn * ((~node_mask[..., None, None]) * -self.inf)
        attn = pygu.softmax(
            attn,
            index=batch,
            dim=0
        )

        # B x N_vn x H x c_attn
        vn_update = pygu.scatter(
            node_v[:, None] * attn[..., None] * node_mask[..., None, None, None],
            index=batch,
            dim=0
        )
        # B x N_vn x c_s
        vn_update = self.lin_out(vn_update.view(num_graphs, num_vn, -1))
        return vn_update


class VirtualNodeScatterAttn(nn.Module):
    def __init__(self, c_s, c_attn, num_heads=4):
        super().__init__()
        self.c_s = c_s
        self.c_attn = c_attn
        self.num_heads = num_heads

        self.lin_q = nn.Linear(c_s, c_attn * num_heads)
        self.lin_kv = nn.Linear(
            c_s,
            2 * c_attn * num_heads)

        self.lin_out = Linear(
            c_attn * num_heads,
            c_s,
            init='final'
        )

    def forward(self, node_features, vn_features, batch, node_mask):
        num_graphs, num_vn = vn_features.shape[:2]
        # N x H x c_attn
        node_q = self.lin_q(node_features).view(-1, self.num_heads, self.c_attn)

        # N x N_vn x H x c_attn
        vn_kv = self.lin_kv(vn_features).view(num_graphs, num_vn, self.num_heads, 2*self.c_attn)
        nodewise_vn_kv = vn_kv[batch]
        nodewise_vn_k, nodewise_vn_v = nodewise_vn_kv.split(self.c_attn, dim=-1)
        # N x N_vn x H
        attn = torch.einsum("...hi, ...vhi->...vh", node_q, nodewise_vn_k) / np.sqrt(self.c_attn)
        attn = torch.softmax(attn, dim=-2)

        # N x H x c_attn
        node_update = torch.sum(nodewise_vn_v * attn[..., None], dim=1)
        # N x c_s
        node_update = self.lin_out(node_update.flatten(start_dim=-2))
        return node_update * node_mask[..., None]


class VirtualNodeGatherMPNN(nn.Module):
    def __init__(self,
                 c_s):
        super().__init__()
        self.c_s = c_s

        self.msg_fn = nn.Sequential(
            nn.Linear(c_s*2, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

    def forward(self, node_features, vn_features, batch, node_mask):
        num_vn = vn_features.shape[1]
        nodewise_vn = vn_features[batch]
        node_features_expand = node_features[..., None, :].expand(-1, num_vn, -1)
        msg = self.msg_fn(
            torch.cat([node_features_expand, nodewise_vn], dim=-1)
        )
        vn_update = pygu.scatter(
            msg[node_mask],
            index=batch[node_mask],
            dim=0,
            dim_size=vn_features.shape[0],
            reduce='mean'
        )
        return vn_update


class VirtualNodeScatterMPNN(nn.Module):
    def __init__(self,
                 c_s):
        super().__init__()
        self.c_s = c_s

        self.msg_fn = nn.Sequential(
            nn.Linear(c_s*2, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

    def forward(self, node_features, vn_features, batch, node_mask):
        num_vn = vn_features.shape[1]
        nodewise_vn = vn_features[batch]
        node_features_expand = node_features[..., None, :].expand(-1, num_vn, -1)
        msg = self.msg_fn(
            torch.cat([node_features_expand, nodewise_vn], dim=-1)
        )
        node_update = msg.mean(dim=-2)
        return node_update * node_mask[..., None]



class VirtualNodeAttnUpdate(nn.Module):
    def __init__(self, c_s, c_attn, num_heads=4):
        super().__init__()
        self.c_s = c_s

        self.gather = VirtualNodeGatherAttn(c_s, c_attn, num_heads)
        self.vn_ffn = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            Linear(c_s, c_s)
        )
        self.vn_ln1 = nn.LayerNorm(c_s)
        self.vn_ln2 = nn.LayerNorm(c_s)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=c_s,
            nhead=num_heads,
            dim_feedforward=c_s,
            batch_first=True,
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=2, enable_nested_tensor=False
        )

        self.scatter = VirtualNodeScatterAttn(c_s, c_attn, num_heads)
        self.node_ffn = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            Linear(c_s, c_s)
        )
        self.node_ln1 = nn.LayerNorm(c_s)
        self.node_ln2 = nn.LayerNorm(c_s)

    def forward(self, node_features, vn_features, batch, node_mask):
        # node -> vn
        vn_update = self.gather(node_features, vn_features, batch, node_mask)
        vn_features = self.vn_ln1(vn_features + vn_update)
        vn_update = self.vn_ffn(vn_features)
        vn_features = self.vn_ln2(vn_features + vn_update)

        # vn -> vn
        vn_features = self.transformer(vn_features)

        # vn -> node
        node_update = self.scatter(node_features, vn_features, batch, node_mask)
        node_features = self.node_ln1(node_features + node_update)
        node_update = self.node_ffn(node_features)
        node_features = self.node_ln2(node_features + node_update)

        return node_features, vn_features


class VirtualNodeMPNNUpdate(nn.Module):
    def __init__(self, c_s, num_heads=4):
        super().__init__()
        self.c_s = c_s

        self.gather = VirtualNodeGatherMPNN(c_s)
        self.vn_ffn = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            Linear(c_s, c_s, init='final')
        )
        self.vn_ln1 = nn.LayerNorm(c_s)
        self.vn_ln2 = nn.LayerNorm(c_s)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=c_s,
            nhead=num_heads,
            dim_feedforward=c_s,
            batch_first=True,
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=2, enable_nested_tensor=False
        )

        self.scatter = VirtualNodeScatterMPNN(c_s)
        self.node_ffn = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            Linear(c_s, c_s, init='final')
        )
        self.node_ln1 = nn.LayerNorm(c_s)
        self.node_ln2 = nn.LayerNorm(c_s)

    def forward(self, node_features, vn_features, batch, node_mask):
        # node -> vn
        vn_update = self.gather(node_features, vn_features, batch, node_mask)
        vn_features = self.vn_ln1(vn_features + vn_update)
        vn_update = self.vn_ffn(vn_features)
        vn_features = self.vn_ln2(vn_features + vn_update)

        # vn -> vn
        vn_features = self.transformer(vn_features)

        # vn -> node
        node_update = self.scatter(node_features, vn_features, batch, node_mask)
        node_features = self.node_ln1(node_features + node_update)
        node_update = self.node_ffn(node_features)
        node_features = self.node_ln2(node_features + node_update)

        return node_features, vn_features