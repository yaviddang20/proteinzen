import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy

import functools as fn

from proteinzen.model.modules.openfold.layers_v2 import Linear, ConditionedTransition, Transition, LayerNorm, AdaLN
from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.linear import EquiLinear
from gatr.interface.point import embed_point
from gatr.interface.plane import embed_oriented_plane, extract_oriented_plane
from gatr.interface.translation import embed_translation
from gatr.utils.tensors import construct_reference_multivector

from ._attn import ConditionedPairUpdateV2, PairEmbedderV2
from ._frame_transformer import (
    get_indexing_matrix, single_to_keys, pairs_to_framepairs,
)
from ._gatr_modules import GATrPairBiasBlock, SequenceBlockGATrPairBias, SequenceBlockConditionedGATrPairBias, EquiAdaLN
from ._geo_mlp import MLPConfig



def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def flatten_and_pad(tensor, dims, n_padding, value=0.):
    assert len(dims) == 2
    all_dims = list(range(tensor.dim()))
    dim_sizes = list(tensor.shape)
    abs_dims = sorted([all_dims[d] for d in dims])
    assert abs_dims[1] - abs_dims[0] == 1
    flat_dims = dim_sizes[:abs_dims[0]] + [-1] + dim_sizes[abs_dims[1]+1:]
    tail_dims = len(dim_sizes[abs_dims[1]+1:])
    flat_tensor = tensor.view(*flat_dims)
    pad_input = 2*[0 for _ in range(tail_dims)] + [0, n_padding]
    padded_flat_tensor = F.pad(flat_tensor, pad_input, value=value)
    return padded_flat_tensor


def unpad_and_unflatten(tensor, dim, unflat_shape, n_padding):
    narrow_len = tensor.shape[dim] - n_padding
    unpad_tensor = tensor.narrow(
        dim,
        start=0,
        length=narrow_len
    )
    return unpad_tensor.unflatten(dim, unflat_shape)


class SeqPredictorFromResidue(nn.Module):
    def __init__(self, c_s, n_aa=21):
        super().__init__()
        self.out = nn.Sequential(
            LayerNorm(c_s),
            Linear(c_s, n_aa)
        )

    def forward(self, node_s):
        return self.out(node_s)



class SeqPredictorFromAtoms(nn.Module):
    def __init__(self, c_atom_s, c_hidden, n_aa=21):
        super().__init__()
        self.proj = Linear(c_atom_s, c_hidden, bias=False)
        self.out = Linear(c_hidden, n_aa)

    def forward(self, atom_s):
        seq_embed = self.proj(atom_s)
        out = self.out(torch.relu(seq_embed).mean(dim=-2))
        return out


class EdgeDistPredictor(nn.Module):
    def __init__(self, c_z, n_bins=64):
        super().__init__()
        self.proj = Linear(c_z, c_z)
        self.out = nn.Sequential(
            LayerNorm(c_z),
            Linear(c_z, n_bins),
        )

    def forward(self, edge_features):
        z = self.proj(edge_features)
        z = z + z.transpose(-2, -3)
        return self.out(z)


class AtompairDistPredictor(nn.Module):
    def __init__(self, c_framepair, n_bins=22):
        super().__init__()
        self.n_bins = n_bins
        self.proj = Linear(c_framepair, n_bins, bias=False)

    def forward(self, framepair_embed):
        return self.proj(framepair_embed)


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
                eps=1e-8
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
        atompair_features = self.dist_proj(1 / dist**2)
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

        atompair_features = atompair_features + self.ffn(atompair_features)
        atompair_features = atompair_features * atompair_mask[..., None]

        return atompair_features, atompair_mask


class Embedder(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_mv,
                 c_atom,
                 c_atom_mv,
                 c_atompair,
                 attention_config,
                 mlp_config,
                 c_hidden=64,
                 num_heads=4,
                 block_q=32,
                 block_k=128,
                 n_tfmr_blocks=3,
                 n_pair_embed_blocks=2,
                 index_embed_size=32,
                 atoms_per_res=14
    ):
        super().__init__()
        self.c_s = c_s
        self.c_mv = c_mv
        self.c_z = c_z
        self.block_q = block_q
        self.block_k = block_k
        self.atoms_per_res = atoms_per_res

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=index_embed_size
        )
        self.node_init = Linear(index_embed_size*2, self.c_s, bias=False)
        self.atom_init = Linear(c_s, c_atom, bias=False)
        self.pos_embedder = nn.Embedding(atoms_per_res, c_atom)

        self.atompair_init = AtomPairEmbedder(
            c_z,
            c_atom,
            c_atompair
        )
        self.atom_gatr = SequenceBlockGATrPairBias(
            in_mv_channels=1,
            out_mv_channels=c_atom_mv,
            hidden_mv_channels=c_atom_mv,
            in_s_channels=c_atom,
            out_s_channels=c_atom,
            hidden_s_channels=c_atom,
            c_z=c_atompair,
            num_blocks=n_tfmr_blocks,
            attention=attention_config,
            mlp=mlp_config
        )
        self.proj_atom_to_node = EquiLinear(
            in_mv_channels=c_atom_mv,
            out_mv_channels=c_mv,
            in_s_channels=c_atom,
            out_s_channels=c_s
        )

        self.sc_atompair_init = AtomPairEmbedder(
            c_z,
            c_atom,
            c_atompair
        )
        self.sc_atom_gatr = SequenceBlockGATrPairBias(
            in_mv_channels=1,
            out_mv_channels=c_atom_mv,
            hidden_mv_channels=c_atom_mv,
            in_s_channels=c_atom,
            out_s_channels=c_atom,
            hidden_s_channels=c_atom,
            c_z=c_atompair,
            num_blocks=n_tfmr_blocks,
            attention=attention_config,
            mlp=mlp_config
        )
        self.sc_proj_atom_to_node = EquiLinear(
            in_mv_channels=c_atom_mv,
            out_mv_channels=c_mv,
            in_s_channels=c_atom,
            out_s_channels=c_s
        )

        self.node_adaln = EquiAdaLN(
            scalar_channels=c_s,
            mv_channels=c_mv,
            cond_scalar_channels=c_s,
            cond_mv_channels=c_mv
        )
        self.atom_adaln = EquiAdaLN(
            scalar_channels=c_atom,
            mv_channels=c_atom_mv,
            cond_scalar_channels=c_atom,
            cond_mv_channels=c_atom_mv
        )
        self.atompair_adaln = AdaLN(c_atompair, c_atompair)

        self.pair_embedder = PairEmbedderV2(
            c_z,
            c_hidden,
            no_blocks=n_pair_embed_blocks,
            no_heads=num_heads
        )

    def forward(
            self,
            *,
            seq_idx,
            t,
            fixed_mask,
            node_mask,
            noisy_atom14,
            sc_atom14=None,
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        n_batch = noisy_atom14.shape[0]
        n_atoms = noisy_atom14.shape[1] * noisy_atom14.shape[2]
        n_padding = (self.block_q - n_atoms % self.block_q) % self.block_q
        n_attn_blocks = (n_atoms + n_padding) // self.block_q

        indexing_matrix = get_indexing_matrix(
            n_attn_blocks,
            W=self.block_q,
            H=self.block_k,
            device=t.device
        )
        to_queries = lambda x: x.view(n_batch, n_attn_blocks, self.block_q, -1)
        to_keys = fn.partial(
            single_to_keys, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )
        to_flat = fn.partial(
            flatten_and_pad, n_padding=n_padding
        )
        to_res_batch = fn.partial(
            unpad_and_unflatten, n_padding=n_padding, unflat_shape=(-1, self.atoms_per_res)
        )

        edge_embed = self.pair_embedder(
            noisy_atom14[..., 1, :],
            node_mask,
            sc_X_ca=(
                sc_atom14[..., 1, :] if sc_atom14 is not None
                else None
            )
        )

        num_batch, num_res = seq_idx.shape
        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        seq_idx_embed = self.index_embedder(seq_idx)
        node_feats = [t_embed, seq_idx_embed]
        node_init = torch.cat(node_feats, dim=-1)
        node_init = self.node_init(node_init)

        atom_init = self.atom_init(
            torch.relu(node_init[..., None, :])
        ).tile((1, 1, self.atoms_per_res, 1))
        atom_init = atom_init + self.pos_embedder(
            torch.tensor([0] + [1 for _ in range(self.atoms_per_res-1)], device=atom_init.device)
        )[None, None]
        atom_to_res_idx = seq_idx[..., None].tile((1, 1, self.atoms_per_res))
        atom_mask = node_mask[..., None].tile((1, 1, self.atoms_per_res))

        def coord_to_mv(points):
            point_mv = embed_point(points)
            plane_mv = embed_oriented_plane(-points, points)
            trans_mv = embed_translation(points)
            # print("plane mv", plane_mv[0, 0, 0])
            return point_mv + plane_mv

        # atom_mv = embed_point(noisy_atom14)[..., None, :] + embed_translation(noisy_atom14)[..., None, :]
        atom_mv = coord_to_mv(noisy_atom14)[..., None, :]

        flat_atom_init = to_flat(atom_init, dims=(-2, -3))
        flat_atom_mv = to_flat(atom_mv, dims=(-3, -4))
        flat_atompos = to_flat(noisy_atom14, dims=(-2, -3))
        atom_to_res_idx = to_flat(atom_to_res_idx, dims=(-1, -2))
        atom_mask = to_flat(atom_mask, dims=(-1, -2))

        flat_atom_init = flat_atom_init.clone()
        atompair_embed, atompair_mask = self.atompair_init(
            flat_atom_init,
            flat_atompos,
            edge_embed,
            atom_to_res_idx,
            atom_mask,
            to_queries,
            to_keys,
        )

        flat_atom_init = flat_atom_init.clone()
        flat_atom_mv, flat_atom_s = self.atom_gatr(
            multivectors=flat_atom_mv,
            scalars=flat_atom_init,
            pair_scalars=atompair_embed,
            to_queries=to_queries,
            to_keys=to_keys,
            attention_mask=atompair_mask,
        )
        flat_atom_init = flat_atom_init.clone()

        atom_mv = to_res_batch(flat_atom_mv, dim=-3)
        atom_s = to_res_batch(flat_atom_s, dim=-2)

        node_mv, node_s = self.proj_atom_to_node(atom_mv, atom_s)
        node_mv = node_mv.mean(dim=-3)
        node_s = node_s.mean(dim=-2) + node_init

        if sc_atom14 is not None:
            # sc_atom_mv = embed_point(sc_atom14)[..., None, :] + embed_translation(sc_atom14)[..., None, :]
            sc_atom_mv = coord_to_mv(sc_atom14)[..., None, :]
            flat_sc_atom_mv = to_flat(sc_atom_mv, dims=(-3, -4))
            flat_sc_atompos = to_flat(sc_atom14, dims=(-2, -3))

            sc_atompair_embed, _ = self.sc_atompair_init(
                flat_atom_init,
                flat_sc_atompos,
                edge_embed,
                atom_to_res_idx,
                atom_mask,
                to_queries,
                to_keys,
            )
            flat_atom_init = flat_atom_init.clone()

            flat_sc_atom_mv, flat_sc_atom_s = self.sc_atom_gatr(
                multivectors=flat_sc_atom_mv,
                scalars=flat_atom_init,
                pair_scalars=sc_atompair_embed,
                to_queries=to_queries,
                to_keys=to_keys,
                attention_mask=atompair_mask,
            )
            flat_atom_init = flat_atom_init.clone()

            sc_atom_mv = to_res_batch(flat_sc_atom_mv, dim=-3)
            sc_atom_s = to_res_batch(flat_sc_atom_s, dim=-2)

            sc_node_mv, sc_node_s = self.proj_atom_to_node(sc_atom_mv, sc_atom_s)
            sc_node_mv = sc_node_mv.mean(dim=-3)
            sc_node_s = sc_node_s.mean(dim=-2) + node_init
        else:
            flat_sc_atom_mv = torch.zeros_like(flat_atom_mv)
            flat_sc_atom_s = torch.zeros_like(flat_atom_s)
            sc_atompair_embed = torch.zeros_like(atompair_embed)
            sc_node_mv = torch.zeros_like(node_mv)
            sc_node_s = torch.zeros_like(node_s)

        node_mv, node_s = self.node_adaln(
            node_mv, node_s,
            sc_node_mv, sc_node_s
        )
        flat_atom_mv, flat_atom_s = self.atom_adaln(
            flat_atom_mv, flat_atom_s,
            flat_sc_atom_mv, flat_sc_atom_s
        )
        atompair_embed = self.atompair_adaln(atompair_embed, sc_atompair_embed)

        return {
            "node_mv": node_mv,
            "node_s": node_s,
            "edge_embed": edge_embed,
            "atompos": flat_atompos,
            "atom_mv": flat_atom_mv,
            "atom_s": flat_atom_s,
            "atompair_embed": atompair_embed,
            "to_queries": to_queries,
            "to_keys": to_keys,
            "to_flat": to_flat,
            "to_res_batch": to_res_batch,
            "atom_to_res_idx": atom_to_res_idx,
            "atom_mask": atom_mask
        }

class AtomPositionUpdate(nn.Module):
    def __init__(self,
                 c_mv,
                 c_s
    ):
        super().__init__()
        self.lin = EquiLinear(
            in_mv_channels=c_mv,
            out_mv_channels=1,
            in_s_channels=c_s,
        )
    def forward(self, node_mv, node_s):
        out_mv, _ = self.lin(node_mv, node_s)
        pos_update = extract_oriented_plane(out_mv).squeeze(-2)
        return pos_update


class GATrDenoiser(nn.Module):
    def __init__(self,
                 node_attention_config,
                 node_mlp_config,
                 atom_attention_config,
                 atom_mlp_config,
                 c_mv=8,
                 c_s=256-8*16,
                 c_z=128,
                 c_atom_mv=4,
                 c_atom_s=128-4*16,
                 c_atompair=16,
                 num_blocks=4,
                 use_traj_predictions=False,
                 atom_gatr_num_blocks=3,
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.use_traj_predictions = use_traj_predictions
        self.num_blocks = num_blocks

        self.trunk = nn.ModuleDict()

        for b in range(num_blocks):
            self.trunk[f'gatr_block_{b}'] = GATrPairBiasBlock(
                mv_channels=c_mv,
                s_channels=c_s,
                z_channels=c_z,
                attention=node_attention_config,
                mlp=node_mlp_config
            )
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=c_s,
                nhead=4,
                dim_feedforward=c_s,
                batch_first=True,
                dropout=0.0,
                norm_first=True,
            )
            self.trunk[f'tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, 2)

            self.trunk[f'post_tfmr_{b}'] = Linear(
                c_s, c_s, init="final", bias=False)
            self.trunk[f'transition_{b}'] = Transition(
                c=c_s,
            )
            self.trunk[f'ca_update_{b}'] = AtomPositionUpdate(c_mv, c_s)

            if use_traj_predictions:
                self.trunk[f'seq_pred_{b}'] = SeqPredictorFromResidue(c_s)
                self.trunk[f'dist_pred_{b}'] = EdgeDistPredictor(c_z)

            if b < num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ConditionedPairUpdateV2(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_z//4,
                    no_heads=4,
                )

        self.node_to_atom_cond = EquiLinear(
            in_mv_channels=c_mv,
            in_s_channels=c_s,
            out_mv_channels=c_atom_mv,
            out_s_channels=c_atom_s,
            bias=False
        )
        self.node_to_atom_update = EquiLinear(
            in_mv_channels=c_mv,
            in_s_channels=c_s,
            out_mv_channels=c_atom_mv,
            out_s_channels=c_atom_s,
            bias=False
        )

        self.atompair_init = AtomPairEmbedder(
            c_z,
            c_atom_s,
            c_atompair
        )
        self.atompair_fuser = AdaLN(c_atompair, c_atompair)

        self.atom_gatr = SequenceBlockConditionedGATrPairBias(
            in_mv_channels=c_atom_mv,
            out_mv_channels=c_atom_mv,
            hidden_mv_channels=c_atom_mv,
            in_s_channels=c_atom_s,
            out_s_channels=c_atom_s,
            hidden_s_channels=c_atom_s,
            c_z=c_atompair,
            cond_mv_channels=c_atom_mv,
            cond_s_channels=c_atom_s,
            num_blocks=atom_gatr_num_blocks,
            attention=atom_attention_config,
            mlp=atom_mlp_config
        )
        self.atom_update = AtomPositionUpdate(c_atom_mv, c_atom_s)
        self.seq_pred = SeqPredictorFromAtoms(c_atom_s, c_hidden=c_s)


    def forward(self, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        node_mv = input_feats['node_mv']
        node_s = input_feats['node_s']
        edge_embed = input_feats['edge_embed']
        init_ca_atompos = input_feats['ca_atompos']
        ca_atompos = init_ca_atompos

        traj_data = {
            b: {}
            for b in range(self.num_blocks)
        }

        node_ref_mv = construct_reference_multivector("data", node_mv)

        for b in range(self.num_blocks):
            node_mv, node_s = self.trunk[f'gatr_block_{b}'](
                multivectors=node_mv,
                scalars=node_s,
                pair_scalars=edge_embed,
                node_mask=node_mask,
                reference_mv=node_ref_mv
            )

            seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                node_s, src_key_padding_mask=1 - node_mask)
            node_s = node_s + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_s = node_s * node_mask[..., None]

            node_s = node_s + self.trunk[f'transition_{b}'](node_s)
            node_s = node_s * node_mask[..., None]

            ca_atompos_update = self.trunk[f'ca_update_{b}'](
                node_mv, node_s
            )
            ca_atompos = ca_atompos + ca_atompos_update * diffuse_mask[..., None]

            if b < self.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_s,
                    edge_embed,
                    ca_atompos,
                    edge_mask
                )
                edge_embed = edge_embed * edge_mask[..., None]

            if self.use_traj_predictions:
                traj_data[b]['seq_logits'] = self.trunk[f"seq_pred_{b}"](node_s)
                traj_data[b]['dist_logits'] = self.trunk[f"dist_pred_{b}"](edge_embed)
                traj_data[b]['ca_pos'] = ca_atompos


        atom_mv = input_feats['atom_mv'].clone()
        atom_s = input_feats['atom_s'].clone()
        init_atompair_embed = input_feats['atompair_embed']
        to_queries = input_feats['to_queries']
        to_keys = input_feats['to_keys']
        # to_flat = input_feats['to_flat']
        to_res_batch = input_feats['to_res_batch']
        atom_to_res_idx = input_feats['atom_to_res_idx']
        atom_mask = input_feats['atom_mask']
        noisy_atom14 = input_feats['noisy_atom14']
        init_atom14_atompos = input_feats['atompos']
        # atom_ref_mv = construct_reference_multivector("data", atom_mv)

        atom_mv_update, atom_s_update = self.node_to_atom_update(node_mv, node_s)
        cond_atom_mv, cond_atom_s = self.node_to_atom_cond(node_mv, node_s)
        # broadcast
        atom_mv_update = torch.gather(
            atom_mv_update,
            -3,
            atom_to_res_idx[..., None, None].expand(-1, -1, *cond_atom_mv.shape[-2:])
        )
        atom_s_update = torch.gather(
            atom_s_update,
            -2,
            atom_to_res_idx[..., None].expand(-1, -1, cond_atom_s.shape[-1])
        )
        cond_atom_mv = torch.gather(
            cond_atom_mv,
            -3,
            atom_to_res_idx[..., None, None].expand(-1, -1, *cond_atom_mv.shape[-2:])
        )
        cond_atom_s = torch.gather(
            cond_atom_s,
            -2,
            atom_to_res_idx[..., None].expand(-1, -1, cond_atom_s.shape[-1])
        )

        atom_mv = atom_mv + atom_mv_update
        atom_s = atom_s + atom_s_update

        atompair_embed, atompair_mask = self.atompair_init(
            atom_s,
            init_atom14_atompos,
            edge_embed,
            atom_to_res_idx,
            atom_mask,
            to_queries,
            to_keys,
        )
        atompair_embed = self.atompair_fuser(atompair_embed, init_atompair_embed)

        atom_mv, atom_s = self.atom_gatr(
            multivectors=atom_mv,
            scalars=atom_s,
            pair_scalars=atompair_embed,
            to_queries=to_queries,
            to_keys=to_keys,
            cond_mv=cond_atom_mv,
            cond_s=cond_atom_s,
            attention_mask=atompair_mask,
            # reference_mv=atom_ref_mv
        )
        atompos_update = self.atom_update(atom_mv, atom_s) * atom_mask[..., None]
        print(atompos_update[0, 0], atom_mask[0, 0])
        atom14_update = to_res_batch(atompos_update, dim=-2)
        pred_atom14 = noisy_atom14 + atom14_update
        seq_logits = self.seq_pred(to_res_batch(atom_s, dim=-2))

        model_out = {
            'pred_atom14': pred_atom14,
            'traj_data': traj_data,
            'seq_logits': seq_logits
        }
        return model_out



class GATrAtom14Denoiser(nn.Module):

    def __init__(self,
                 c_mv=8,
                 c_s=256-8*16,
                 c_z=128,
                 c_atom_mv=4,
                 c_atom_s=128-4*16,
                 c_atompair=16,
                 use_traj_predictions=False,
                 num_blocks=8,
                 atom_gatr_num_blocks=3,
                 preconditioning=False,
                 block_q=32,
                 block_k=128,
                 ):
        super().__init__()
        # some compatibility code
        self.self_conditioning = True
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000

        self.num_blocks = num_blocks

        node_attention_config = SelfAttentionConfig(
            multi_query=False,
            in_mv_channels=c_mv,
            out_mv_channels=c_mv,
            in_s_channels=c_s,
            out_s_channels=c_s,
            num_heads=16,
            output_init="unit_scalar",
            checkpoint=False,
        )
        node_mlp_config = MLPConfig(
            mv_channels=c_mv,
            s_channels=c_s,
        )
        atom_attention_config = SelfAttentionConfig(
            multi_query=False,
            in_mv_channels=c_atom_mv,
            out_mv_channels=c_atom_mv,
            in_s_channels=c_atom_s,
            out_s_channels=c_atom_s,
            num_heads=4,
            output_init="unit_scalar",
            checkpoint=False,
        )
        atom_mlp_config = MLPConfig(
            mv_channels=c_atom_mv,
            s_channels=c_atom_s,
        )

        self.denoiser = GATrDenoiser(
            node_attention_config,
            node_mlp_config,
            atom_attention_config,
            atom_mlp_config,
            c_mv=c_mv,
            c_s=c_s,
            c_z=c_z,
            c_atom_mv=c_atom_mv,
            c_atom_s=c_atom_s,
            c_atompair=c_atompair,
            num_blocks=num_blocks,
            use_traj_predictions=use_traj_predictions,
            atom_gatr_num_blocks=atom_gatr_num_blocks,
        )

        self.embedder = Embedder(
            c_s,
            c_z,
            c_mv,
            c_atom=c_atom_s,
            c_atom_mv=c_atom_mv,
            c_atompair=c_atompair,
            attention_config=atom_attention_config,
            mlp_config=atom_mlp_config,
            c_hidden=64,
            num_heads=4,
            block_q=block_q,
            block_k=block_k,
            n_tfmr_blocks=atom_gatr_num_blocks,
        )
        self.preconditioning = preconditioning

    def forward(self, data, self_condition=None):
        res_data = data['residue']
        res_mask = (res_data['res_mask']).bool()

        data_list = data.to_data_list()
        for d in data_list:
            assert d.num_nodes == data_list[0].num_nodes

        seq_idx = [torch.arange(data_list[0].num_nodes, device=res_mask.device) for _ in data_list]
        seq_idx = torch.stack(seq_idx)
        t = data['t']
        batch_size = t.shape[0]
        fixed_mask = torch.zeros_like(res_mask).view(batch_size, -1)

        if self_condition is not None:
            sc_atom14 = self_condition['pred_atom14'] / 10
        else:
            sc_atom14 = None

        # center the training example at the mean of all atoms
        atom14_t = res_data['noised_atom14']
        atom14_t = atom14_t.view([t.shape[0], -1, 3])
        atom14_t_mask = res_data['atom14_mask'].view([t.shape[0], -1])
        center = (atom14_t * atom14_t_mask[..., None]).sum(dim=-2) / atom14_t_mask.sum(dim=-1)[..., None]
        atom14_t = atom14_t - center[..., None, :]
        atom14_t = atom14_t.view([t.shape[0], -1, 14, 3])

        if self.preconditioning:
            atom14_in = atom14_t * data['c_in'][..., None, None, None]
        else:
            atom14_in = atom14_t / 10

        input_feats = self.embedder(
            seq_idx=seq_idx,
            t=t,
            noisy_atom14=atom14_in,
            node_mask=res_mask.view(batch_size, -1),
            fixed_mask=fixed_mask,
            sc_atom14=sc_atom14,
        )

        input_feats.update({
            'fixed_mask': fixed_mask,
            'res_mask': res_mask.view(batch_size, -1),
            't': t,
            'noisy_atom14': atom14_in,
            'ca_atompos': atom14_in[..., 1, :]
        })

        score_dict = self.denoiser(input_feats)
        traj_data = score_dict['traj_data']
        atom14_out = score_dict['pred_atom14']
        if self.preconditioning:
            atom14_out = atom14_out * data['c_out'][..., None, None, None] + atom14_t * data['c_skip'][..., None, None, None]
        else:
            atom14_out = atom14_out * 10

        atom14_out = atom14_out + center[..., None, None, :]
        seq_logits = score_dict['seq_logits']

        ret = {}
        ret['pred_atom14'] = atom14_out
        ret['denoised_atom14'] = atom14_out.flatten(0, 1)
        ret['denoised_atom14_gt_seq'] = atom14_out.flatten(0, 1)
        ret['traj_data'] = traj_data
        ret['decoded_seq_logits'] = seq_logits.flatten(0, 1)

        return ret