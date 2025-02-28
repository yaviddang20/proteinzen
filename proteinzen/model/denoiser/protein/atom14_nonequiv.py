import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import copy

import functools as fn

from proteinzen.model.modules.layers.node.attention import FlashTransformerEncoder, ConditionedTransformerPairBias, TransformerPairBias
from proteinzen.model.modules.openfold.layers import Dropout
from proteinzen.model.modules.openfold.layers_v2 import Linear, ConditionedInvariantPointAttention, ConditionedTransition, BackboneUpdate, TorsionAngles, Transition, LayerNorm, AdaLN
import proteinzen.utils.openfold.rigid_utils as ru
from proteinzen.utils.openfold.rigid_utils import Rigid, batchwise_center

from ._attn import PairUpdate, ConditionedPairUpdateV2, MultiRigidPairEmbedder, PairEmbedderV2
from ._atom_transformer_nonequiv import (
    SequenceAtomTransformer, AtomPairEmbedder,
    get_indexing_matrix, single_to_keys
)

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

class TorsionAngles(nn.Module):
    def __init__(self, c, num_torsions, eps=1e-8):
        super(TorsionAngles, self).__init__()

        self.c = c
        self.eps = eps
        self.num_torsions = num_torsions

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        # TODO: Remove after published checkpoint is updated without these weights.
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.linear_final = Linear(
            self.c, self.num_torsions * 2, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)

        s = s + s_initial
        unnormalized_s = self.linear_final(s)
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        normalized_s = unnormalized_s / norm_denom

        return unnormalized_s, normalized_s


class SeqPredictor(nn.Module):
    def __init__(self, c_frame, n_aa=21):
        super().__init__()
        self.proj = nn.Sequential(
            LayerNorm(c_frame),
            Linear(c_frame, c_frame, bias=False)
        )
        self.out = Linear(c_frame, n_aa)
        self.c_frame = c_frame

    def forward(self, frame_embed):
        seq_embed = self.proj(frame_embed)
        out = self.out(seq_embed.mean(dim=-2))
        return out


class SeqPairPredictor(nn.Module):
    def __init__(self, c_z, n_aa=21):
        super().__init__()
        self.proj = nn.Sequential(
            Linear(c_z, n_aa * n_aa, bias=False)
        )

    def forward(self, edge_embed):
        two_body_logits = self.proj(edge_embed)
        return two_body_logits


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


class FramepairDistPredictor(nn.Module):
    def __init__(self, c_framepair, n_bins=22):
        super().__init__()
        self.n_bins = n_bins
        self.proj = Linear(c_framepair, n_bins, bias=False)

    def forward(self, framepair_embed):
        return self.proj(framepair_embed)


class GatherUpdate(nn.Module):
    def __init__(self, c_s, c_atom):
        super().__init__()
        self.lin = nn.Sequential(
            LayerNorm(c_s),
            Linear(c_s, c_atom, bias=False)
        )

    def forward(self,
                node_embed,
                atom_embed,
                atom_to_res_idx,
                atom_mask):
        broadcast_embed = self.lin(node_embed)
        broadcast_embed = torch.gather(
            node_embed,
            -2,
            atom_to_res_idx[..., None].expand([-1 for _ in range(atom_to_res_idx.dim())] + [atom_embed.shape[-1]])
        )
        atom_embed = atom_embed + broadcast_embed * atom_mask[..., None]
        return atom_embed


class ScatterUpdate(nn.Module):
    def __init__(self, c_s, c_atom):
        super().__init__()
        self.lin = Linear(c_atom, c_s, bias=False)

    def forward(self,
                atom_embed,
                node_embed,
                atom_to_res_idx,
                atom_mask):
        out = torch.zeros_like(node_embed)
        out.scatter_reduce_(
            -2,
            atom_to_res_idx[..., None].expand(-1, -1, node_embed.shape[-1]),
            F.relu(self.lin(atom_embed)) * atom_mask[..., None],
            reduce='mean'
        )
        out_denom = torch.zeros(node_embed.shape[:-1], device=out.device)
        denom = atom_mask.float()
        out_denom.scatter_add_(
            -1,
            atom_to_res_idx,
            denom,
        )
        out = out / out_denom[..., None]
        return out + node_embed


class Embedder(nn.Module):

    def __init__(self,
                 c_s,
                 c_z,
                 c_atom,
                 c_atompair,
                 c_hidden=64,
                 num_heads=4,
                 block_q=32,
                 block_k=128,
                 n_tfmr_blocks=3,
                 n_pair_embed_blocks=2,
                 index_embed_size=32,
                 break_symmetry=True,
                 atoms_per_residue=14,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_atom = c_atom
        self.block_q = block_q
        self.block_k = block_k
        self.break_symmetry = break_symmetry
        self.atoms_per_residue = atoms_per_residue

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=index_embed_size
        )
        self.mask_lin = Linear(1, self.c_s, bias=False)
        self.node_init = Linear(index_embed_size*2, self.c_s, bias=False)

        self.atompos_init = Linear(3, self.c_atom, bias=False)
        self.sc_atompos_init = Linear(3, self.c_atom, bias=False)

        self.atom_tfmr = SequenceAtomTransformer(
            c_s=c_s,
            c_atom=c_atom,
            c_atompair=c_atompair,
            no_heads=num_heads,
            num_blocks=n_tfmr_blocks,
        )
        self.sc_atom_tfmr = SequenceAtomTransformer(
            c_s=c_s,
            c_atom=c_atom,
            c_atompair=c_atompair,
            no_heads=num_heads,
            num_blocks=n_tfmr_blocks,
        )

        self.atom_gather_update = ScatterUpdate(c_s, c_atom)
        self.sc_atom_gather_update = ScatterUpdate(c_s, c_atom)


        self.node_adaln = AdaLN(c_s, c_s)
        self.atom_adaln = AdaLN(c_atom, c_atom)
        self.atompair_adaln = AdaLN(c_atompair, c_atompair)

        self.atompair_embedder = AtomPairEmbedder(
            c_z,
            c_atom,
            c_atompair
        )
        self.sc_atompair_embedder = AtomPairEmbedder(
            c_z,
            c_atom,
            c_atompair
        )
        self.pair_embedder = PairEmbedderV2(
            c_z,
            c_hidden,
            no_blocks=n_pair_embed_blocks
        )

        self.node_to_atom = Linear(c_s, c_atom, bias=False)
        self.pos_embedder = nn.Embedding(atoms_per_residue, c_atom)

    def forward(
            self,
            *,
            seq_idx,
            t,
            fixed_mask,
            node_mask,
            noised_atom14,
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
        n_batch = noised_atom14.shape[0]
        n_atoms = noised_atom14.shape[1] * noised_atom14.shape[2]
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
            unpad_and_unflatten, n_padding=n_padding, unflat_shape=(-1, self.atoms_per_residue)
        )
        num_batch, num_res = seq_idx.shape
        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        seq_idx_embed = self.index_embedder(seq_idx)
        node_feats = [t_embed, seq_idx_embed]
        node_embed = torch.cat(node_feats, dim=-1)
        node_embed = self.node_init(node_embed)
        node_embed = self.mask_lin(node_mask[..., None].float()) + node_embed
        node_init = node_embed

        atom_embed = self.node_to_atom(node_init[..., None, :]).tile((1, 1, self.atoms_per_residue, 1))
        atom_embed = atom_embed + self.pos_embedder(
            torch.arange(self.atoms_per_residue, device=atom_embed.device)
        )[None, None]
        atom_to_res_idx = seq_idx[..., None].tile((1, 1, self.atoms_per_residue))
        atom_mask = node_mask[..., None].tile((1, 1, self.atoms_per_residue))
        atom_init = atom_embed
        atom_init = atom_init + self.atompos_init(noised_atom14)

        sc_X_ca = sc_atom14[..., 1, :] if sc_atom14 is not None else None
        edge_embed = self.pair_embedder(noised_atom14[..., 1, :], node_mask, sc_X_ca=sc_X_ca)

        flat_atom_init = to_flat(atom_init, dims=(-2, -3))
        flat_atompos = to_flat(noised_atom14, dims=(-2, -3))
        atom_to_res_idx = to_flat(atom_to_res_idx, dims=(-1, -2))
        atom_mask = to_flat(atom_mask, dims=(-1, -2))

        atompair_embed, atompair_mask = self.atompair_embedder(
            flat_atom_init,
            flat_atompos,
            edge_embed,
            atom_to_res_idx,
            atom_mask,
            to_queries,
            to_keys
        )
        atom_init = self.atom_tfmr(
            flat_atom_init,
            node_embed,
            atompair_embed,
            atom_to_res_idx,
            atom_mask,
            atompair_mask,
            to_queries,
            to_keys,
        )

        node_init = self.atom_gather_update(
            atom_init,
            node_init,
            atom_to_res_idx,
            atom_mask
        )

        if sc_atom14 is not None:
            sc_atom_init = atom_embed + self.sc_atompos_init(sc_atom14)
            flat_sc_atom_init = to_flat(sc_atom_init, dims=(-2, -3))
            flat_sc_atompos = to_flat(sc_atom14, dims=(-2, -3))
            sc_atompair_embed, _ = self.sc_atompair_embedder(
                flat_sc_atom_init,
                flat_sc_atompos,
                edge_embed,
                atom_to_res_idx,
                atom_mask,
                to_queries,
                to_keys
            )
            sc_atom_init = self.sc_atom_tfmr(
                flat_sc_atom_init,
                node_embed,
                sc_atompair_embed,
                atom_to_res_idx,
                atom_mask,
                atompair_mask,
                to_queries,
                to_keys,
            )
            sc_node_init = self.sc_atom_gather_update(
                sc_atom_init,
                node_embed,
                atom_to_res_idx,
                atom_mask
            )
        else:
            sc_atom_init = torch.zeros_like(atom_init)
            sc_node_init = torch.zeros_like(node_init)
            sc_atompair_embed = torch.zeros_like(atompair_embed)

        atom_embed = self.atom_adaln(atom_init, sc_atom_init)
        node_embed = self.node_adaln(node_init, sc_node_init)
        atompair_embed = self.atompair_adaln(atompair_embed, sc_atompair_embed)

        return {
            "atom_embed": atom_embed,
            "node_embed": node_embed,
            "edge_embed": edge_embed,
            "noised_atompos": flat_atompos,
            "atompair_embed": atompair_embed,
            "atompair_mask": atompair_mask,
            "to_queries": to_queries,
            "to_keys": to_keys,
            "to_flat": to_flat,
            "to_res_batch": to_res_batch,
            "atom_to_res_idx": atom_to_res_idx,
            "atom_mask": atom_mask
        }


class AtomDenoiserCore(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_atom=64,
                 c_atompair=64,
                 num_heads=8,
                 num_blocks=4,
                 block_q=32,
                 block_k=128,
                 no_time_condition=False
                 ):
        super().__init__()

        self.trunk = nn.ModuleDict()

        self.num_blocks = num_blocks
        self.block_q = block_q
        self.block_k = block_k
        self.no_time_condition = no_time_condition

        self.time_embed = GaussianRandomFourierBasis(n_basis=128, c_out=c_s)

        for b in range(num_blocks):
            if self.no_time_condition:
                self.trunk[f'tfmr_{b}'] = TransformerPairBias(
                    c_s=c_s,
                    c_z=c_z,
                    no_heads=num_heads,
                    n_layers=2,
                )
            else:
                self.trunk[f'tfmr_{b}'] = ConditionedTransformerPairBias(
                    c_s=c_s,
                    c_cond=c_s,
                    c_z=c_z,
                    no_heads=num_heads,
                    n_layers=2,
                )

            self.trunk[f'broadcast_to_atoms_{b}'] = GatherUpdate(
                c_s,
                c_atom
            )

            self.trunk[f'atompair_update_{b}'] = AtomPairEmbedder(
                c_z,
                c_atom,
                c_atompair
            )

            self.trunk[f'atom_tfmr_{b}'] = SequenceAtomTransformer(
                c_atom,
                c_atompair,
                c_s,
                no_heads=4,
                num_blocks=1#2
            )

            self.trunk[f'scatter_to_nodes_{b}'] = ScatterUpdate(
                c_s,
                c_atom
            )
            self.trunk[f'atom_update_{b}'] = nn.Sequential(
                LayerNorm(c_atom),
                Linear(c_atom, 3, bias=False, init='final')
            )

            if b < num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ConditionedPairUpdateV2(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_z//4,
                    no_heads=4,
                )

        self.seq_pred = SeqPredictor(c_atom)

    def forward(self, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        time_embed = self.time_embed(input_feats['t'])
        time_embed = time_embed[:, None].expand(-1, node_mask.shape[1], -1)
        # Main trunk
        node_embed = input_feats['node_embed']
        edge_embed = input_feats['edge_embed']
        node_embed = node_embed * node_mask[..., None]
        atompos = input_feats['noised_atompos']
        atom_embed = input_feats['atom_embed']
        atompair_embed = input_feats['atompair_embed']
        atompair_mask = input_feats['atompair_mask']
        to_queries = input_feats['to_queries']
        to_keys = input_feats['to_keys']
        to_res_batch = input_feats['to_res_batch']
        atom_to_res_idx = input_feats['atom_to_res_idx']
        atom_mask = input_feats['atom_mask']

        for b in range(self.num_blocks):
            if self.no_time_condition:
                node_embed = self.trunk[f'tfmr_{b}'](
                    node_embed, edge_embed, node_mask)
            else:
                node_embed = self.trunk[f'tfmr_{b}'](
                    node_embed, time_embed, edge_embed, node_mask)

            atom_embed = self.trunk[f'broadcast_to_atoms_{b}'](
                node_embed,
                atom_embed,
                atom_to_res_idx,
                atom_mask
            )
            atompair_embed, _ = self.trunk[f'atompair_update_{b}'](
                atom_embed,
                atompos,
                edge_embed,
                atom_to_res_idx,
                atom_mask,
                to_queries,
                to_keys,
                init_atompair_embed=atompair_embed
            )
            atom_embed = self.trunk[f'atom_tfmr_{b}'](
                atom_embed,
                node_embed,
                atompair_embed,
                atom_to_res_idx,
                atom_mask,
                atompair_mask,
                to_queries,
                to_keys,
            )
            atompos = atompos + self.trunk[f'atom_update_{b}'](atom_embed) * atom_mask[..., None]

            if b < self.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, to_res_batch(atompos, -2)[..., 1, :], edge_mask)
                edge_embed *= edge_mask[..., None]

        seq_logits = self.seq_pred(to_res_batch(atom_embed, -2))
        model_out = {
            'denoised_atompos': atompos,
            'seq_logits': seq_logits
        }
        return model_out


class GaussianRandomFourierBasis(nn.Module):
    """ Damped random Fourier Feature encoding layer """
    def __init__(self, n_basis=64, c_out=256):
        super().__init__()
        kappa = torch.randn((1, n_basis,))
        self.register_buffer('kappa', kappa)
        self.proj = nn.Sequential(
            LayerNorm(n_basis * 2),
            Linear(n_basis*2, c_out, bias=False)
        )

    def forward(self, ts):
        tp = 2 * np.pi * ts[..., None] * self.kappa
        n = torch.cat([torch.cos(tp), torch.sin(tp)], dim=-1)
        return self.proj(n)


class AtomDenoiser(nn.Module):

    def __init__(self,
                 # diffuser,
                 c_s=256,
                 c_z=128,
                 c_atom=128,
                 c_atompair=16,
                 num_heads=16,
                 num_blocks=8,
                 trans_preconditioning=False,
                 block_q=32,
                 block_k=128,
                 atoms_per_residue=14,
                 no_time_condition=False
                 ):
        super().__init__()
        # some compatibility code
        self.self_conditioning = True
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000

        self.denoiser = AtomDenoiserCore(
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_atompair=c_atompair,
            num_heads=num_heads,
            num_blocks=num_blocks,
            block_q=block_q,
            block_k=block_k,
            no_time_condition=no_time_condition
        )

        self.embedder = Embedder(
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_atompair=c_atompair,
            block_q=block_q,
            block_k=block_k,
        )
        self.c_s = c_s
        self.trans_preconditioning = trans_preconditioning
        self.atoms_per_residue = atoms_per_residue
        self.no_time_condition = no_time_condition

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
        print(fixed_mask.shape)

        if self_condition is not None:
            sc_atom14 = self_condition['denoised_atom14'].view(batch_size, -1, self.atoms_per_residue, 3)
        else:
            sc_atom14 = None

        atom14_t = res_data['noised_atom14'].view(batch_size, -1, self.atoms_per_residue, 3)
        if self.trans_preconditioning:
            atom14_in = atom14_t * data['c_in'][..., None, None, None]
        else:
            atom14_in = atom14_t / 10

        denoiser_data = self.embedder(
            seq_idx=seq_idx,
            t=t,
            noised_atom14=atom14_in,
            node_mask=res_mask.view(batch_size, -1),
            fixed_mask=fixed_mask,
            sc_atom14=sc_atom14,
        )
        denoiser_data['res_mask'] = res_data['res_mask'].view(batch_size, -1)
        denoiser_data['t'] = t

        score_dict = self.denoiser(denoiser_data)
        atompos_out = score_dict['denoised_atompos']
        atom14_out = denoiser_data['to_res_batch'](atompos_out, -2)
        if self.trans_preconditioning:
            atom14_out = (atom14_out - atom14_in) * data['c_out'][..., None, None, None] + atom14_t * data['c_skip'][..., None, None, None]
        else:
            atom14_out = atom14_out * 10

        seq_logits = score_dict['seq_logits']
        ret = {}
        ret['denoised_atom14'] = atom14_out.flatten(0, 1)
        ret['denoised_atom14_gt_seq'] = atom14_out.flatten(0, 1)
        ret['decoded_seq_logits'] = seq_logits.flatten(0, 1)

        return ret