import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy
import functools as fn
import numpy as np

from torch_geometric.nn import knn_graph
import torch_geometric.utils as pygu

from proteinzen.model.modules.openfold.layers_v2 import Linear, LayerNorm, swish
from proteinzen.utils.openfold import rigid_utils as ru


class GatherAdaLN(nn.Module):
    def __init__(self, c_s, c_cond):
        super().__init__()
        self.ln_s = nn.LayerNorm(c_s, elementwise_affine=False, bias=False)
        self.ln_cond = nn.LayerNorm(c_cond, bias=False)
        self.lin_cond = Linear(c_cond, c_s)
        self.lin_cond_nobias = Linear(c_cond, c_s, bias=False)

        self.c_s = c_s
        self.c_cond = c_cond

    def forward(self, s, cond, cond_to_s_idx, fastpass=False):
        s = self.ln_s(s)
        cond = self.ln_cond(cond)
        if fastpass:
            cond_gate = self.lin_cond(cond)
            cond_bias = self.lin_cond_nobias(cond)
            _s = s.view(*cond.shape[:-1], -1, s.shape[-1])
            _s = _s * torch.sigmoid(cond_gate)[..., None, :]
            _s = _s + cond_bias[..., None, :]
            s = _s.view(s.shape)
        else:
            cond_gate = torch.gather(
                self.lin_cond(cond),
                dim=1,
                index=cond_to_s_idx[..., None].expand(-1, -1, self.c_s),
            )
            cond_bias = torch.gather(
                self.lin_cond_nobias(cond),
                dim=1,
                index=cond_to_s_idx[..., None].expand(-1, -1, self.c_s)
            )
            s = s * torch.sigmoid(cond_gate)
            s = s + cond_bias

        return s


class GatherConditionedTransition(nn.Module):
    def __init__(self, c_s, c_cond, n=2):
        super().__init__()
        self.adaln = GatherAdaLN(c_s, c_cond)
        self.lin_1 = Linear(c_s, c_s*n, bias=False)
        self.lin_2 = Linear(c_s, c_s*n, bias=False)
        self.lin_cond = Linear(c_cond, c_s)
        with torch.no_grad():
            self.lin_cond.bias.fill_(-2.0)
        self.lin_b = Linear(c_s*n, c_s, bias=False, init='final')

        self.c_s = c_s

    def forward(self, s, cond, cond_to_s_idx, fastpass=False):
        s = self.adaln(s, cond, cond_to_s_idx)
        b = swish(self.lin_1(s)) * self.lin_2(s)
        if fastpass:
            cond_gate = self.lin_cond(cond)
            _s = torch.sigmoid(cond_gate)[..., None, :] * self.lin_b(b).view(*cond_gate.shape[:-1], -1, self.c_s)
            s = _s.view(s.shape)
        else:
            cond_gate = torch.gather(
                self.lin_cond(cond),
                dim=1,
                index=cond_to_s_idx[..., None].expand(-1, -1, self.c_s),
            )
            s = torch.sigmoid(cond_gate) * self.lin_b(b)
        return s


class BroadcastEdgeUpdate(nn.Module):
    def __init__(self, c_atompair, c_z):
        super().__init__()
        self.update = nn.Sequential(
            nn.LayerNorm(c_z),
            Linear(c_z, c_atompair, bias=False)
        )

    def forward(self, z, flat_atom_res_index, edge_index):
        n_batch, n_res, _, c_z = z.shape
        res_edge_index = flat_atom_res_index[edge_index]
        flatish_z = z.view(n_batch*n_res, n_res, c_z)
        flatish_atompair_update = self.update(flatish_z)
        atompair_update = flatish_atompair_update[res_edge_index[0], res_edge_index[1] % n_res]
        return atompair_update


# from boltz1
def get_indexing_matrix(K, W, H, device):
    assert W % 2 == 0
    assert H % (W // 2) == 0

    h = H // (W // 2)
    assert h % 2 == 0

    arange = torch.arange(2 * K, device=device)
    index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(
        min=0, max=h + 1
    )
    index = index.view(K, 2, 2 * K)[:, 0, :]
    onehot = F.one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
    return onehot.reshape(2 * K, h * K).float()

# from boltz1
def single_to_keys(single, indexing_matrix, W, H):
    B, N, D = single.shape
    K = N // W
    single = single.view(B, 2 * K, W // 2, D)
    return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
        B, K, H, D
    )

# adapted from boltz1
def pairs_to_atompairs(pairs, indexing_matrix, W, H):
    B, N, _, D = pairs.shape
    K = N // W
    pairs = pairs.view(B, K, W, 2 * K, W // 2, D)
    indexing_matrix = indexing_matrix.view(2 * K, K, -1)
    ret = torch.einsum("b x y j i d, j x h -> b x y h i d", pairs, indexing_matrix)
    ret = ret.reshape(
        B, K, W, H, D
    )
    return ret

class GatherUpdate(nn.Module):
    def __init__(self, c_atom, c_s):
        super().__init__()
        self.update = nn.Sequential(
            nn.LayerNorm(c_s),
            Linear(c_s, c_atom, bias=False)
        )
        self.c_atom = c_atom

    def forward(self, atom_embed, s, cond_to_s_idx, fastpass=False):
        if fastpass:
            new_atom_embed = atom_embed.view(*s.shape[:-1], -1, atom_embed.shape[-1]) + self.update(s)[..., None, :]
            return new_atom_embed.view(atom_embed.shape)
        else:
            atom_update = torch.gather(
                self.update(s),
                dim=1,
                index=cond_to_s_idx[..., None].expand(-1, -1, self.c_atom),
            )
            return atom_embed + atom_update


class SequenceAtomTransformerBlock(nn.Module):
    def __init__(self,
                 c_atom=128,
                 c_atompair=16,
                 c_s=256,
                 no_heads=4,
                 window_q=32,
                 window_k=128,
                 inf=1e8
                 ):
        super().__init__()

        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_s = c_s
        self.no_heads = no_heads
        self.c_head = c_atom // no_heads
        self.window_q = window_q
        self.window_k = window_k
        self.inf = inf

        self.adaln = GatherAdaLN(c_atom, c_s)

        self.lin_q = Linear(c_atom, c_atom)
        self.lin_kv = Linear(c_atom, 2*c_atom, bias=False)
        self.lin_b_ij = nn.Sequential(
            nn.LayerNorm(c_atompair),
            Linear(c_atompair, no_heads, bias=False)
        )
        self.lin_g = nn.Sequential(
            Linear(c_atom, c_atom),
            nn.Sigmoid()
        )

        self.lin_out = Linear(c_atom, c_atom)
        self.s_gate_lin = Linear(c_s, c_atom)
        with torch.no_grad():
            self.s_gate_lin.bias.fill_(-2.0)

        self.transition = GatherConditionedTransition(c_atom, c_s)

    def attn(
        self,
        atom_features,
        res_features,
        atompair_features,
        atom_res_idx,
        atompair_mask,
        to_queries,
        to_keys
    ):
        n_batch = atom_features.shape[0]
        n_atoms = atom_features.shape[1]
        atom_features = self.adaln(
            atom_features, res_features, atom_res_idx
        )

        atom_q = self.lin_q(atom_features)
        atom_kv = self.lin_kv(atom_features)
        atom_q = to_queries(atom_q).unflatten(-1, (self.no_heads, -1))
        atom_kv = to_keys(atom_kv).unflatten(-1, (2*self.no_heads, -1))
        atom_k, atom_v = atom_kv.split(self.no_heads, dim=-2)

        # n_batch x n_block x window_q x window_k x n_heads
        a = torch.einsum("bnqhc,bnkhc->bnqkh", atom_q, atom_k) / np.sqrt(self.c_head)
        # n_batch x n_block x window_q x window_k x n_heads
        b_ij = self.lin_b_ij(atompair_features)
        a = a + b_ij
        # print(a.shape, atompair_mask.shape)
        a = a - self.inf * (1 - atompair_mask.float()[..., None])
        a = a.permute(0, 1, 4, 2, 3)
        a = torch.softmax(a, dim=-1)

        atom_out = torch.einsum("bnhqk,bnkhc->bnqhc", a, atom_v)
        atom_out = atom_out.reshape(n_batch, n_atoms, -1)
        atom_gate = torch.sigmoid(self.s_gate_lin(res_features))
        atom_out = atom_out * torch.gather(
            atom_gate,
            dim=1,
            index=atom_res_idx[..., None].expand(-1, -1, atom_out.shape[-1])
        )
        return atom_out

    def forward(
        self,
        atom_features,
        res_features,
        atompair_features,
        atom_res_idx,
        atompos_mask,
        atompair_mask,
        to_queries,
        to_keys
    ):
        assert atom_features.shape[1] % self.window_q == 0
        # assert atompos.shape[1] % self.window_k == 0

        atom_update = self.attn(
            atom_features,
            res_features,
            atompair_features,
            atom_res_idx,
            atompair_mask,
            to_queries,
            to_keys
        )
        atom_features = atom_features + atom_update * atompos_mask[..., None]

        atom_features = atom_features + self.transition(atom_features, res_features, atom_res_idx) * atompos_mask[..., None]

        return atom_features


class SequenceAtomTransformer(nn.Module):
    def __init__(self,
                 c_atom=128,
                 c_atompair=16,
                 c_s=256,
                 no_heads=4,
                 num_blocks=2,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            SequenceAtomTransformerBlock(
                 c_atom=c_atom,
                 c_atompair=c_atompair,
                 c_s=c_s,
                 no_heads=no_heads,
            )
            for _ in range(num_blocks)
        ])

    def forward(
            self,
            atom_features,
            res_features,
            atompair_features,
            atom_res_idx,
            atompos_mask,
            atompair_mask,
            to_queries,
            to_keys
    ):
        for tfmr in self.blocks:
            atom_features = tfmr(
                atom_features,
                res_features,
                atompair_features,
                atom_res_idx,
                atompos_mask,
                atompair_mask,
                to_queries,
                to_keys
            )

        return atom_features


class AtomPairEmbedder(nn.Module):
    def __init__(self,
                 c_z,
                 c_atom,
                 c_atompair):
        super().__init__()

        self.pair_to_atompair = nn.Sequential(
            LayerNorm(c_z),
            Linear(c_z, c_atompair, bias=False)
        )
        self.atom_to_atompair_q = nn.Sequential(
            nn.ReLU(c_atom),
            Linear(c_atom, c_atompair, bias=False)
        )
        self.atom_to_atompair_k = nn.Sequential(
            nn.ReLU(c_atom),
            Linear(c_atom, c_atompair, bias=False)
        )
        self.dist_vec_proj = Linear(3, c_atompair, bias=False)
        self.dist_proj = Linear(1, c_atompair, bias=False)

        self.ffn = nn.Sequential(
            nn.ReLU(),
            Linear(c_atompair, c_atompair, bias=False),
            nn.ReLU(),
            Linear(c_atompair, c_atompair, bias=False),
            nn.ReLU(),
            Linear(c_atompair, c_atompair, bias=False),
        )

    def _broadcast_pairs(
        self,
        z,
        atom_to_res_idx,
        atom_mask,
        to_queries,
        to_keys
    ):
        # we flatten the 2d z dimension into one then gather across that
        # TODO: i'm not a huge fan of this method, i feel like there should be a better way...
        with torch.no_grad():
            n_batch = atom_to_res_idx.shape[0]
            n_res = z.shape[-2]
            q_res_idx = to_queries(atom_to_res_idx[..., None])
            k_res_idx = to_keys(atom_to_res_idx[..., None].float()).transpose(-1, -2).long()
            n_block = q_res_idx.shape[1]
            atompair_1d_idx = (q_res_idx * n_res + k_res_idx).view(n_batch, n_block, -1)

            q_mask = to_queries(atom_mask[..., None])
            k_mask = to_keys(atom_mask[..., None].float()).transpose(-1, -2).bool()
            atompair_mask = (q_mask * k_mask)

        z_flat = z.view(n_batch, 1, n_res ** 2, -1).expand(-1, n_block, -1, -1)
        atompair_embed = torch.gather(
            z_flat,
            -2,
            atompair_1d_idx[..., None].expand(-1, -1, -1, z.shape[-1])
        )
        atompair_embed = atompair_embed.view(*atompair_mask.shape, -1) * atompair_mask[..., None]

        return atompair_embed, atompair_mask

    def forward(self,
                atom_embed,
                atompos,
                z,
                atom_to_res_idx,
                atom_mask,
                to_queries,
                to_keys,
                eps=1e-8,
                init_atompair_embed=None
    ):
        # n_batch x n_block x window_q x 3
        atompos_q = to_queries(atompos)
        # n_batch x n_block x window_k x 3
        atompos_k = to_keys(atompos)
        # n_batch x n_block x window_q x window_k x 3
        dist_vec = atompos_q[..., None, :] - atompos_k[..., None, : ,:]
        # n_batch x n_block x window_q x window_k x 1
        dist = torch.linalg.vector_norm(dist_vec + eps, dim=-1, keepdims=True)

        # n_batch x n_block x window_q
        res_idx_q = to_queries(atom_to_res_idx[..., None])
        # n_batch x n_block x window_k
        res_idx_k = to_keys(atom_to_res_idx[..., None].float()).long().transpose(-1, -2)
        # n_batch x n_block x window_q x window_k
        same_res = (res_idx_q == res_idx_k)

        # n_batch x n_block x window_q x window_k x c_atompair
        atompair_features = self.dist_vec_proj(dist_vec) + self.dist_proj(1 / dist**2)
        atompair_features = atompair_features * same_res[..., None]

        broadcast_z, atompair_mask = self._broadcast_pairs(
            z,
            atom_to_res_idx,
            atom_mask,
            to_queries,
            to_keys
        )
        atompair_features = atompair_features + self.pair_to_atompair(broadcast_z)

        atom_embed_q = to_queries(atom_embed)
        atom_embed_k = to_keys(atom_embed)
        atompair_q = self.atom_to_atompair_q(atom_embed_q)
        atompair_k = self.atom_to_atompair_k(atom_embed_k)
        atompair_features = atompair_features + atompair_q[..., None, :] + atompair_k[..., None, :, :]

        if init_atompair_embed is not None:
            atompair_features = atompair_features + init_atompair_embed

        atompair_features = atompair_features + self.ffn(atompair_features)
        atompair_features = atompair_features * atompair_mask[..., None]

        return atompair_features, atompair_mask