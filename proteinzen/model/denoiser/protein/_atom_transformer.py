import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy
import functools as fn
import numpy as np

from torch_geometric.nn import knn_graph
import torch_geometric.utils as pygu
import dgl
import dgl.sparse
from dgl.ops import u_mul_e_sum, u_dot_v, u_add_v

from proteinzen.model.modules.openfold.layers_v2 import Linear, swish
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


class AtomTransformerBlock(nn.Module):
    def __init__(self,
                 c_atom=128,
                 c_atompair=16,
                 c_s=256,
                 c_z=128,
                 no_heads=4,
                 use_spatial_edges=False,
                 broadcast_edge_features=False,
                 add_nodes_to_edges=False,
                 use_edge_ffn=False,
                 ):
        super().__init__()

        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_s = c_s
        self.no_heads = no_heads
        self.c_head = c_atom // no_heads

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

        self.dist_vec_proj = Linear(3, c_atompair, bias=False)
        self.dist_proj = Linear(1, c_atompair, bias=False)

        self.use_spatial_edges = use_spatial_edges

        if broadcast_edge_features:
            self.broadcast_edges = BroadcastEdgeUpdate(c_atompair, c_z)
        else:
            self.broadcast_edges = None

        self.add_nodes_to_edges = add_nodes_to_edges
        if self.add_nodes_to_edges:
            self.u_proj = Linear(c_atom, c_atompair, bias=False)
            self.v_proj = Linear(c_atom, c_atompair, bias=False)

        self.use_edge_ffn = use_edge_ffn

        if self.use_edge_ffn:
            self.edge_ffn = nn.Sequential(
                nn.LayerNorm(c_atompair),
                Linear(c_atompair, c_atompair, bias=False),
                nn.ReLU(),
                Linear(c_atompair, c_atompair, bias=False),
                nn.ReLU(),
                Linear(c_atompair, c_atompair, bias=False),
            )
        else:
            self.edge_ffn = None


    def _gen_dgl_atompair_features(self,
                               flat_atompos,
                               rigids,
                               z,
                               atom_res_idx,
                               edge_index,
    ):
        atompair_vecs = flat_atompos[edge_index[1]] - flat_atompos[edge_index[0]]
        flat_rigids = rigids.view(-1)
        flat_res_idx = (
            atom_res_idx +
            torch.arange(atom_res_idx.shape[0], device=atom_res_idx.device)[:, None] * rigids.shape[1]
        ).view(-1)
        atompair_vecs_local = flat_rigids[flat_res_idx][edge_index[0]].invert_apply(atompair_vecs)
        dist = torch.linalg.vector_norm(atompair_vecs_local, dim=-1, keepdims=True)
        same_res = flat_res_idx[edge_index[1]] == flat_res_idx[edge_index[0]]

        atompair_features = self.dist_vec_proj(atompair_vecs_local) + self.dist_proj(dist)
        atompair_features = atompair_features * same_res[..., None]

        if self.broadcast_edges is not None:
            atompair_features = atompair_features + self.broadcast_edges(z, flat_res_idx, edge_index)


        return atompair_features

    def _dgl_attn(
        self,
        atom_features,
        atompos,
        res_features,
        rigids,
        z,
        atom_res_idx,
        edge_index=None,
        prev_atompair_features=None,
        fastpass=False
    ):
        ## keep track of batch with tensor
        batch = torch.tile(
            torch.arange(atompos.shape[0], device=atompos.device)[:, None],
            (1, atompos.shape[1])
        ).view(-1)
        ## flatten the batch dim
        flat_atom_features = atom_features.flatten(0, 1)
        flat_atompos = atompos.flatten(0, 1)

        if edge_index is None:
            with torch.no_grad():
                if self.use_spatial_edges:
                    # i want this to be 128 but apparently the cuda implementation maxes out at 100...
                    edge_index = knn_graph(flat_atompos, k=100, batch=batch, loop=True)
                else:
                    num_atoms = atompos.shape[1]
                    num_subset_centers = math.ceil(num_atoms / 32)
                    generic_block_q = torch.arange(32, device=atompos.device).long()
                    generic_block_k = torch.arange(128, device=atompos.device).long()
                    # 32 x 128 x 2
                    generic_block = torch.stack([
                        generic_block_k[None].tile(32, 1),
                        generic_block_q[:, None].tile(1, 128)
                    ], dim=-1)
                    block_offset = torch.arange(num_subset_centers, device=atompos.device).long() * 32
                    # n_block x 32 x 128 x 2
                    generic_block = generic_block[None] + block_offset[:, None, None, None]
                    # n_atoms x 128 x 2
                    generic_block = generic_block.view(-1, 128, 2)[:num_atoms]
                    # n_edge x 2
                    # we need to get rid of the edges to non-existant atoms
                    _select = generic_block < num_atoms
                    generic_block = generic_block[_select[..., 0] & _select[..., 1]]
                    batch_offset = torch.arange(atompos.shape[0], device=atompos.device).long() * num_atoms
                    # n_batch x n_edge x 2
                    edge_index = batch_offset[:, None, None] + generic_block[None]
                    edge_index = edge_index.view(-1, 2).permute(1, 0)
                    edge_index = pygu.sort_edge_index(edge_index)
        dgl_graph = dgl.graph(data=(edge_index[1], edge_index[0]))

        if prev_atompair_features is None:
            flat_atompair_features = self._gen_dgl_atompair_features(flat_atompos, rigids, z, atom_res_idx, edge_index)
        else:
            flat_atompair_features = prev_atompair_features

        if self.add_nodes_to_edges:
            relu_atom_embed = torch.relu(flat_atom_features)
            flat_atompair_features = flat_atompair_features + u_add_v(
                dgl_graph,
                self.u_proj(relu_atom_embed),
                self.v_proj(relu_atom_embed),
            )

        if self.use_edge_ffn:
            flat_atompair_features = flat_atompair_features + self.edge_ffn(flat_atompair_features)

        flat_atom_q = self.lin_q(flat_atom_features).view(-1, self.no_heads, self.c_head)
        flat_atom_kv = self.lin_kv(flat_atom_features).view(-1, 2*self.no_heads, self.c_head)
        flat_atom_k, flat_atom_v = flat_atom_kv.split(self.no_heads, dim=1)

        # print(flat_atom_k.shape, flat_atom_q.shape)
        flat_a = u_dot_v(dgl_graph, flat_atom_k, flat_atom_q) / np.sqrt(self.c_head)
        flat_a = flat_a.squeeze(-1)
        flat_b_ij = self.lin_b_ij(flat_atompair_features)
        # print(flat_a.shape, flat_b_ij.shape)
        flat_a = flat_a + flat_b_ij
        flat_a = pygu.softmax(
            flat_a,
            index=edge_index[0]
        )
        flat_atom_out = u_mul_e_sum(dgl_graph, flat_atom_v, flat_a[..., None])
        flat_atom_out = flat_atom_out.view(-1, self.c_atom) * self.lin_g(flat_atom_features)
        atom_out = flat_atom_out.view(atom_features.shape)
        if fastpass:
            atom_out = atom_out.view(*res_features.shape[:-1], -1, self.c_atom) + torch.sigmoid(self.s_gate_lin(res_features))[..., None, :]
            atom_out = atom_out.view(atom_features.shape)
        else:
            atom_out = atom_out + torch.gather(
                torch.sigmoid(self.s_gate_lin(res_features)),
                dim=1,
                index=atom_res_idx[..., None].expand(-1, -1, atom_out.shape[-1])
            )
        return atom_out, flat_atompair_features, edge_index

    def _torch_dgl_atompair_features(
        self,
        atompos,
        rigids,
        z,
        atom_res_idx,
        edge_index,
    ):
        atompos_k = torch.gather(
            atompos,
            dim=1,
            index=edge_index[..., None].expand(-1, -1, 3)
        )
        dist_vec = atompos_k - atompos[..., None, :]
        dist_vec_local = rigids[atom_res_idx].invert_apply(dist_vec)
        dist = torch.linalg.vector_norm(dist_vec_local, dim=-1, keepdims=True)
        res_idx_k = torch.gather(
            atom_res_idx,
            dim=1,
            index=edge_index
        )
        same_res = (res_idx_k == atom_res_idx)

        atompair_features = self.dist_vec_proj(dist_vec_local) + self.dist_proj(dist)
        atompair_features = atompair_features * same_res[..., None]

        # if self.broadcast_edges is not None:
        #     atompair_features = atompair_features + self.broadcast_edges(z, flat_res_idx, edge_index)


        return atompair_features

    def _torch_attn(
        self,
        atom_features,
        atompos,
        res_features,
        rigids,
        z,
        atom_res_idx,
        edge_index=None,
        prev_atompair_features=None,
        fastpass=False
    ):
        if self.use_spatial_edges:
            if edge_index is None:
                with torch.no_grad():
                    atom_dists = torch.cdist(atompos, atompos)
                    edge_index = torch.topk(atom_dists, k=128, largest=False)

            if prev_atompair_features is None:
                atompair_features = self._gen_torch_atompair_features(atompos, rigids, z, atom_res_idx, edge_index)
            else:
                atompair_features = prev_atompair_features


        if self.add_nodes_to_edges:
            relu_atom_embed = torch.relu(atom_features)
            atompair_features = atompair_features + self.u_proj(relu_atom_embed)[..., None, :]
            atompair_features = atompair_features + torch.gather(
                self.v_proj(relu_atom_embed),
                dim=1,
                index=edge_index[..., None].expand(-1, -1, atompair_features.shape[-1])
            )

        if self.use_edge_ffn:
            atompair_features = atompair_features + self.edge_ffn(atompair_features)

        atom_q = self.lin_q(atom_features).view(-1, self.no_heads, self.c_head)
        atom_kv = self.lin_kv(atom_features)
        atom_kv = torch.gather(
            atom_kv,
            dim=1,
            index=edge_index[..., None].expand(-1, -1, atom_kv.shape[-1])
        )
        atom_kv = atom_kv.view(atom_kv.shape[0], -1, 2*self.no_heads, self.c_head)
        atom_k, atom_v = atom_kv.split(self.no_heads, dim=-2)

        flat_a = torch.einsum("blhc,blmhc->blmh") / np.sqrt(self.c_head)
        flat_b_ij = self.lin_b_ij(atompair_features)
        # print(flat_a.shape, flat_b_ij.shape)
        flat_a = flat_a + flat_b_ij
        flat_a = pygu.softmax(
            flat_a,
            index=edge_index[0]
        )
        flat_atom_out = u_mul_e_sum(dgl_graph, flat_atom_v, flat_a[..., None])
        flat_atom_out = flat_atom_out.view(-1, self.c_atom) * self.lin_g(flat_atom_features)
        atom_out = flat_atom_out.view(atom_features.shape)
        if fastpass:
            atom_out = atom_out.view(*res_features.shape[:-1], -1, self.c_atom) + torch.sigmoid(self.s_gate_lin(res_features))[..., None, :]
            atom_out = atom_out.view(atom_features.shape)
        else:
            atom_out = atom_out + torch.gather(
                torch.sigmoid(self.s_gate_lin(res_features)),
                dim=1,
                index=atom_res_idx[..., None].expand(-1, -1, atom_out.shape[-1])
            )
        return atom_out, flat_atompair_features, edge_index


    def attn(self,
             atom_features,
             atompos,
             res_features,
             rigids,
             z,
             atom_res_idx,
             edge_index=None,
             prev_atompair_features=None,
             fastpass=False,
             use_dgl=True
    ):
        atom_features = self.adaln(
            atom_features, res_features, atom_res_idx
        )

        if use_dgl:
            return self._dgl_attn(
                atom_features,
                atompos,
                res_features,
                rigids,
                z,
                atom_res_idx,
                edge_index=edge_index,
                prev_atompair_features=prev_atompair_features,
                fastpass=fastpass,
            )
        else:
            return self._torch_attn(
                atom_features,
                atompos,
                res_features,
                rigids,
                z,
                atom_res_idx,
                edge_index=edge_index,
                prev_atompair_features=prev_atompair_features,
                fastpass=fastpass,
            )


    def forward(self,
                atom_features,
                atompos,
                res_features,
                rigids,
                z,
                atom_res_idx,
                edge_index=None,
                prev_atompair_features=None
    ):

        atom_update, flat_atompair_features, edge_index = self.attn(
            atom_features,
            atompos,
            res_features,
            rigids,
            z,
            atom_res_idx,
            edge_index=edge_index,
            prev_atompair_features=prev_atompair_features
        )
        atom_features = atom_features + atom_update

        atom_features = atom_features + self.transition(atom_features, res_features, atom_res_idx)

        return atom_features, flat_atompair_features, edge_index



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



class AtomTransformer(nn.Module):
    def __init__(self,
                 c_atom=128,
                 c_atompair=16,
                 c_s=256,
                 no_heads=4,
                 use_spatial_edges=False,
                 num_blocks=2,
                 broadcast_res_features=False,
                 add_nodes_to_edges=False,
                 broadcast_edge_features=False,
                 use_edge_ffn=False,
                 reuse_atompair_features=True,
    ):
        super().__init__()

        if broadcast_res_features:
            self.broadcast_res = GatherUpdate(c_atom, c_s)
        else:
            self.broadcast_res = None

        self.blocks = nn.ModuleList([
            AtomTransformerBlock(
                 c_atom=c_atom,
                 c_atompair=c_atompair,
                 c_s=c_s,
                 no_heads=no_heads,
                 use_spatial_edges=use_spatial_edges,
                 broadcast_edge_features=broadcast_edge_features,
                 add_nodes_to_edges=add_nodes_to_edges,
                 use_edge_ffn=use_edge_ffn
            )
            for _ in range(num_blocks)
        ])
        self.reuse_atompair_features = reuse_atompair_features

    def forward(self,
                atom_features,
                atompos,
                res_features,
                rigids,
                z,
                atom_res_idx,
    ):

        if self.broadcast_res is not None:
            atom_features = self.broadcast_res(atom_features, res_features, atom_res_idx)

        edge_index = None
        atompair_features = None
        for tfmr in self.blocks:
            atom_features, atompair_features, edge_index = tfmr(
                atom_features,
                atompos,
                res_features,
                rigids,
                z,
                atom_res_idx,
                edge_index=edge_index,
                prev_atompair_features=atompair_features
            )
            if not self.reuse_atompair_features:
                atompair_features = None
        return atom_features

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


class SequenceAtomTransformerBlock(nn.Module):
    def __init__(self,
                 c_atom=128,
                 c_atompair=16,
                 c_s=256,
                 c_z=128,
                 no_heads=4,
                 broadcast_edge_features=False,
                 add_nodes_to_edges=False,
                 use_edge_ffn=False,
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

        self.dist_vec_proj = Linear(3, c_atompair, bias=False)
        self.dist_proj = Linear(1, c_atompair, bias=False)

        if broadcast_edge_features:
            self.broadcast_edges = BroadcastEdgeUpdate(c_atompair, c_z)
        else:
            self.broadcast_edges = None

        self.add_nodes_to_edges = add_nodes_to_edges
        if self.add_nodes_to_edges:
            self.u_proj = Linear(c_atom, c_atompair, bias=False)
            self.v_proj = Linear(c_atom, c_atompair, bias=False)

        self.use_edge_ffn = use_edge_ffn

        if self.use_edge_ffn:
            self.edge_ffn = nn.Sequential(
                nn.LayerNorm(c_atompair),
                Linear(c_atompair, c_atompair, bias=False),
                nn.ReLU(),
                Linear(c_atompair, c_atompair, bias=False),
                nn.ReLU(),
                Linear(c_atompair, c_atompair, bias=False),
            )
        else:
            self.edge_ffn = None

    def _gen_atompair_features(
        self,
        atompos,
        rigids,
        z,
        atom_res_idx,
        edge_index,
    ):
        n_batch, n_atom = atompos.shape[:2]
        n_blocks = n_atom // self.window_q
        # n_batch x n_block x window_q x 3
        atompos_q = atompos.view(n_batch, n_blocks, self.window_q, -1)
        # n_batch x n_block x window_k x 3
        atompos_k = torch.gather(
            atompos,
            dim=1,
            index=torch.tile(edge_index.view(n_batch, -1, 1), (1, 1, atompos.shape[-1]))
        ).view(n_batch, n_blocks, self.window_k, atompos.shape[-1])
        # n_batch x n_block x window_q x window_k x 3
        dist_vec = atompos_q[..., None, :] - atompos_k[..., None, : ,:]
        block_rigids = torch.gather(
            rigids.to_tensor_7(),
            dim=1,
            index=atom_res_idx[..., None].tile((1, 1, 7))
        )
        block_rigids = ru.Rigid.from_tensor_7(block_rigids)
        block_rigids = block_rigids.view(n_batch, n_blocks, self.window_q, 1)
        dist_vec_local = block_rigids.invert_apply(dist_vec)
        # n_batch x n_block x window_q x window_k x 1
        dist = torch.linalg.vector_norm(dist_vec_local, dim=-1, keepdims=True)

        # n_batch x n_block x window_q
        res_idx_q = atom_res_idx.view(n_batch, n_blocks, self.window_q)
        # n_batch x n_block x window_k
        res_idx_k = torch.gather(
            atom_res_idx,
            dim=1,
            index=edge_index.view(n_batch, -1)
        ).view(n_batch, n_blocks, self.window_k)
        # n_batch x n_block x window_q x window_k
        same_res = (res_idx_q[..., None] == res_idx_k[..., None, :])

        # n_batch x n_block x window_q x window_k x c_atompair
        atompair_features = self.dist_vec_proj(dist_vec_local) + self.dist_proj(dist)
        atompair_features = atompair_features * same_res[..., None]

        return atompair_features

    def attn(
        self,
        atom_features,
        atompos,
        res_features,
        rigids,
        z,
        atom_res_idx,
        atompos_mask,
        prev_atompair_features=None,
        add_prev_atompair_features=False
    ):
        n_batch = atompos.shape[0]
        n_atoms = atompos.shape[1]
        n_blocks = n_atoms // self.window_q
        atom_features = self.adaln(
            atom_features, res_features, atom_res_idx
        )

        with torch.no_grad():
            num_subset_centers = math.ceil(n_atoms / 32)
            generic_block_k = torch.arange(128, device=atompos.device).long()
            block_offset = torch.arange(num_subset_centers, device=atompos.device).long() * 32
            # n_block x window_k
            generic_block = generic_block_k[None] + block_offset[..., None]
            # n_batch x n_block x window_k
            atompair_mask = generic_block[None] < atompos_mask.long().sum(dim=1)[..., None, None]
            # print(generic_block.shape, atompos_mask.shape)
            # print(atompos_mask.shape, atompair_mask.shape)
            atompair_mask = atompair_mask[..., None, :] * atompos_mask.view(n_batch, n_blocks, self.window_q, 1)
            # n_block x window_k
            generic_block = generic_block.clamp(min=0, max=n_atoms-1)
            # n_batch x n_block x window_k
            edge_index = torch.tile(generic_block[None], (n_batch, 1, 1))

        # n_batch x n_block x window_q x window_k x c_atompair
        if prev_atompair_features is None:
            atompair_features = self._gen_atompair_features(atompos, rigids, z, atom_res_idx, edge_index)
        elif add_prev_atompair_features:
            atompair_features = self._gen_atompair_features(atompos, rigids, z, atom_res_idx, edge_index)
            atompair_features = atompair_features + prev_atompair_features
        else:
            atompair_features = prev_atompair_features

        if self.add_nodes_to_edges:
            relu_atom_embed = torch.relu(atom_features)
            atompair_features = atompair_features + self.u_proj(relu_atom_embed).view(
                n_batch, n_blocks, self.window_q, 1, -1
            )
            atompair_features = atompair_features + torch.gather(
                self.v_proj(relu_atom_embed),
                dim=1,
                index=edge_index.view(n_batch, -1, 1).tile((1, 1, atompair_features.shape[-1]))
            ).view(n_batch, n_blocks, 1, self.window_k, atompair_features.shape[-1])

        if self.use_edge_ffn:
            atompair_features = atompair_features + self.edge_ffn(atompair_features)

        atom_q = self.lin_q(atom_features).view(n_batch, n_blocks, self.window_q, self.no_heads, self.c_head)
        atom_kv = self.lin_kv(atom_features)
        atom_kv = torch.gather(
            atom_kv,
            dim=1,
            index=edge_index.view(n_batch, -1, 1).tile((1, 1, atom_kv.shape[-1]))
        ).view(n_batch, n_blocks, self.window_k, 2*self.no_heads, self.c_head)
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
        return atom_out, atompair_features


    def forward(self,
                atom_features,
                atompos,
                res_features,
                rigids,
                z,
                atom_res_idx,
                atompos_mask,
                prev_atompair_features=None,
                add_prev_atompair_features=False
    ):
        assert atompos.shape[1] % self.window_q == 0
        assert atompos.shape[1] % self.window_k == 0

        atom_update, atompair_features = self.attn(
            atom_features,
            atompos,
            res_features,
            rigids,
            z,
            atom_res_idx,
            atompos_mask,
            prev_atompair_features=prev_atompair_features,
            add_prev_atompair_features=add_prev_atompair_features
        )
        atom_features = atom_features + atom_update * atompos_mask[..., None]

        atom_features = atom_features + self.transition(atom_features, res_features, atom_res_idx) * atompos_mask[..., None]

        return atom_features, atompair_features


class SequenceAtomTransformer(nn.Module):
    def __init__(self,
                 c_atom=128,
                 c_atompair=16,
                 c_s=256,
                 no_heads=4,
                 num_blocks=2,
                 broadcast_res_features=False,
                 add_nodes_to_edges=False,
                 broadcast_edge_features=False,
                 use_edge_ffn=False,
                 reuse_atompair_features=True,
    ):
        super().__init__()

        if broadcast_res_features:
            self.broadcast_res = GatherUpdate(c_atom, c_s)
        else:
            self.broadcast_res = None

        self.blocks = nn.ModuleList([
            SequenceAtomTransformerBlock(
                 c_atom=c_atom,
                 c_atompair=c_atompair,
                 c_s=c_s,
                 no_heads=no_heads,
                 broadcast_edge_features=broadcast_edge_features,
                 add_nodes_to_edges=add_nodes_to_edges,
                 use_edge_ffn=use_edge_ffn
            )
            for _ in range(num_blocks)
        ])
        self.reuse_atompair_features = reuse_atompair_features

    def forward(self,
                atom_features,
                atompos,
                res_features,
                rigids,
                z,
                atom_res_idx,
                atompos_mask,
                prev_atompair_features=None
    ):
        if self.broadcast_res is not None:
            atom_features = self.broadcast_res(atom_features, res_features, atom_res_idx)

        if self.reuse_atompair_features:
            atompair_features = prev_atompair_features
        else:
            atompair_features = None

        atompair_features = None
        for tfmr in self.blocks:
            atom_features, atompair_features = tfmr(
                atom_features,
                atompos,
                res_features,
                rigids,
                z,
                atom_res_idx,
                atompos_mask,
                prev_atompair_features=atompair_features,
                add_prev_atompair_features=(self.reuse_atompair_features and atompair_features is not None)
            )

            if not self.reuse_atompair_features:
                atompair_features = None

        return atom_features, atompair_features