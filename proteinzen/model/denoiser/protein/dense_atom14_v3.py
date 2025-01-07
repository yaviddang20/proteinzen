import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy

import functools as fn

from proteinzen.model.modules.layers.node.attention import FlashTransformerEncoder
from proteinzen.model.modules.openfold.layers import InvariantPointAttention
from proteinzen.model.modules.openfold.layers_v2 import Linear, ConditionedInvariantPointAttention, ConditionedTransition, BackboneUpdate, TorsionAngles, Transition
import proteinzen.utils.openfold.rigid_utils as ru
from proteinzen.utils.openfold.rigid_utils import Rigid, batchwise_center
from proteinzen.utils.framediff.all_atom import compute_backbone
from proteinzen.data.openfold.residue_constants import restype_atom37_rigid_group_positions

from ._attn import ConditionedPairUpdate, Atom14PairEmbedder
from ._atom_transformer_v2 import get_indexing_matrix, single_to_keys, GatherUpdate, AtomPairEmbedder, SequenceAtomTransformer

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
    def __init__(self, c_atom, n_atoms, n_aa=21):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(c_atom),
            Linear(c_atom, c_atom, bias=False)
        )
        self.out = Linear(c_atom, n_aa)
        self.c_atom = c_atom
        self.n_atoms = n_atoms

    def forward(self, atom_embed, atom_res_idx, fastpass=True):
        atom_update = self.proj(atom_embed)
        if fastpass:
            return self.out(atom_update.view(*atom_update.shape[:-2], self.n_atoms, self.c_atom).mean(dim=-2))
        else:
            seq_out = torch.zeros(
                (atom_res_idx.shape[0], atom_res_idx.max().item()+1, self.c_atom),
                device=atom_update.device,
            )
            seq_out.scatter_reduce_(
                dim=1,
                index=atom_res_idx[..., None].expand(-1, -1, self.c_atom),
                src=atom_update,
                reduce='mean',
                include_self=True
            )
            return self.out(seq_out)


class EdgeDistPredictor(nn.Module):
    def __init__(self, c_z, n_bins=64):
        super().__init__()
        self.proj = Linear(c_z, c_z)
        self.out = nn.Sequential(
            nn.LayerNorm(c_z),
            Linear(c_z, n_bins),
        )

    def forward(self, edge_features):
        z = self.proj(edge_features)
        z = z + z.transpose(-2, -3)
        return self.out(z)


class AtomUpdateFromRes(nn.Module):
    def __init__(self, c_s, n_atoms=10):
        super().__init__()
        self.out = nn.Sequential(
            nn.LayerNorm(c_s),
            Linear(c_s, 3*n_atoms, bias=False, init='final')
        )
        self.n_atoms = n_atoms

    def forward(self, s):
        n_batch, n_res, _ = s.shape
        out = self.out(s).view(n_batch, n_res, self.n_atoms, 3)
        return out


class AtomUpdate(nn.Module):
    def __init__(self, c_s, c_out=3):
        super().__init__()
        self.out = nn.Sequential(
            nn.LayerNorm(c_s),
            Linear(c_s, c_out, bias=False, init='final')
        )

    def forward(self, s):
        return self.out(s)


class ScatterUpdate(nn.Module):
    def __init__(self, c_atom, c_s):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(c_atom),
            Linear(c_atom, c_s, bias=False, init='final')
        )
        self.c_s = c_s

    def forward(self, atom_features, s, atom_res_idx, atom_mask):
        atom_update = F.relu(self.layer(atom_features)) * atom_mask[..., None]
        s = s.clone()
        s.scatter_reduce_(
            dim=1,
            index=atom_res_idx[..., None].expand(-1, -1, self.c_s),
            src=atom_update,
            reduce='mean',
            # include_self=True
        )
        return s


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
                 atoms_per_res=10
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.block_q = block_q
        self.block_k = block_k
        self.break_symmetry = break_symmetry
        self.atoms_per_res = atoms_per_res

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=index_embed_size
        )

        node_in = (
            index_embed_size * 2
            + 4 * 3
            + atoms_per_res * 3
        )

        self.node_init = Linear(node_in, self.c_s, bias=False)

        self.atompair_init = AtomPairEmbedder(
            c_z,
            c_atom,
            c_atompair
        )
        self.sc_atompair_init = AtomPairEmbedder(
            c_z,
            c_atom,
            c_atompair
        )

        self.atompair_fuser = nn.Sequential(
            nn.LayerNorm(c_atompair*2),
            Linear(c_atompair*2, c_atompair, bias=False)
        )

        self.atom_tfmr = SequenceAtomTransformer(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=c_s,
            num_blocks=n_tfmr_blocks,
            no_heads=num_heads,
        )

        self.pair_embedder = Atom14PairEmbedder(
            c_s,
            c_z,
            c_hidden,
            no_blocks=n_pair_embed_blocks,
            no_heads=num_heads,
        )

        self.node_to_atom = Linear(c_s, c_atom, bias=False)
        self.pos_embedder = nn.Embedding(atoms_per_res, c_atom)
        self.atom_to_node = ScatterUpdate(c_atom, c_s)


    def forward(
            self,
            *,
            noised_atom10_local,
            seq_idx,
            t,
            fixed_mask,
            node_mask,
            rigids,
            sc_rigids=None,
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
        n_batch = rigids.shape[0]
        n_atoms = rigids.shape[1] * self.atoms_per_res
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

        # init node embeddings
        num_batch, num_res = seq_idx.shape
        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        seq_idx_embed = self.index_embedder(seq_idx)
        bb_pos = torch.as_tensor(restype_atom37_rigid_group_positions[0, :4], device=t_embed.device).view(-1)
        node_feats = [
            t_embed,
            seq_idx_embed,
            bb_pos[None, None].expand(*t_embed.shape[:2], -1),
            noised_atom10_local.view(*noised_atom10_local.shape[:-2], -1),
        ]
        node_embed = torch.cat(node_feats, dim=-1)
        node_embed = self.node_init(node_embed)

        # init edge embeddings
        edge_embed = self.pair_embedder(node_embed, rigids, node_mask, sc_rigids=sc_rigids)

        # make some helper tensors
        atom_to_res_idx = torch.arange(num_res, device=edge_embed.device)[None, :, None].expand(
            num_batch, -1, self.atoms_per_res
        )
        atom_to_res_idx = atom_to_res_idx.reshape(num_batch, -1)

        # init atom and atompair embeddings
        atom_embed = self.node_to_atom(node_embed[..., None, :]).tile((1, 1, self.atoms_per_res, 1))
        atom_embed = atom_embed + self.pos_embedder(
            torch.arange(self.atoms_per_res, device=atom_embed.device)
        )[None, None]

        atom_embed = atom_embed.view(num_batch, -1, atom_embed.shape[-1])
        atompos = rigids[..., None].apply(noised_atom10_local).view(num_batch, -1, 3)
        atom_mask = node_mask[..., None].expand(-1, -1, self.atoms_per_res).reshape(num_batch, -1)

        # make some padding so we can form blocks
        atompos = F.pad(atompos, (0, 0, 0, n_padding), value=0)
        atom_embed = F.pad(atom_embed, (0, 0, 0, n_padding), value=0)
        atom_to_res_idx = F.pad(atom_to_res_idx, (0, n_padding), value=0)
        atom_mask = F.pad(atom_mask, (0, n_padding), value=0)

        atompair_embed, atompair_mask = self.atompair_init(
            atom_embed,
            rigids,
            atompos,
            edge_embed,
            atom_to_res_idx,
            atom_mask,
            to_queries,
            to_keys,
        )
        atom_embed = atom_embed.clone()

        if sc_atom14 is not None and sc_rigids is not None:
            sc_atompos = sc_atom14[..., 4:, :].reshape(num_batch, -1, 3)
            sc_atompos = F.pad(sc_atompos, (0, 0, 0, n_padding), value=0)
            sc_atompair_embed, atompair_mask = self.atompair_init(
                atom_embed,
                sc_rigids,
                sc_atompos,
                edge_embed,
                atom_to_res_idx,
                atom_mask,
                to_queries,
                to_keys,
            )
        else:
            sc_atompair_embed = torch.zeros_like(atompair_embed)
        atompair_embed = self.atompair_fuser(
            torch.cat([atompair_embed, sc_atompair_embed], dim=-1)
        )

        atom_embed = self.atom_tfmr(
            atom_embed,
            atompos,
            node_embed,
            atompair_embed,
            atom_to_res_idx,
            atom_mask,
            atompair_mask,
            to_queries,
            to_keys
        )
        atom_embed = atom_embed * atom_mask[..., None]

        # update node with new atom embed
        node_embed = self.atom_to_node(
            atom_embed,
            node_embed,
            atom_to_res_idx,
            atom_mask
        )

        return node_embed, edge_embed, atompos, atom_embed, atompair_embed, atompair_mask, to_queries, to_keys, n_atoms, atom_to_res_idx, atom_mask


class IpaScore(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_atom=64,
                 c_atompair=16,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 use_traj_predictions=False,
                 force_flash_transformer=False,
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.use_traj_predictions = use_traj_predictions
        self.force_flash_transformer = force_flash_transformer

        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        self.num_blocks = num_blocks


        for b in range(num_blocks):
            self.trunk[f'ipa_{b}'] = InvariantPointAttention(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                pre_ln=True,
                lin_bias=False
            )
            if force_flash_transformer:
                self.trunk[f'tfmr_{b}'] = FlashTransformerEncoder(
                    h_dim=c_s,
                    n_layers=2,
                    no_heads=4,
                    h_ff=c_s,
                    dropout=0.0,
                    ln_first=True,
                    dtype=None
                )
            else:
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
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(c_s)

            if use_traj_predictions:
                self.trunk[f'dist_pred_{b}'] = EdgeDistPredictor(c_z)

            if b < num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ConditionedPairUpdate(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_z//4,
                    no_heads=4
                )

        self.atompair_init = AtomPairEmbedder(
            c_z,
            c_atom,
            c_atompair
        )

        self.atompair_fuser = nn.Sequential(
            nn.LayerNorm(c_atompair*2),
            Linear(c_atompair*2, c_atompair, bias=False)
        )
        self.seq_pred = SeqPredictor(c_atom, n_atoms=10)
        self.atom_tfmr = SequenceAtomTransformer(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=c_s,
            num_blocks=3,
            no_heads=4
        )
        self.atom_update = nn.Sequential(
            nn.LayerNorm(c_atom),
            Linear(c_atom, 3, bias=False, init='final')
        )

        self.torsion_pred = TorsionAngles(c_s, 1)


    def forward(self,
                node_embed,
                edge_embed,
                input_feats):
        init_frames = input_feats['rigids_t'].type(torch.float32)
        noised_atompos = input_feats['atompos']
        atom_mask = input_feats['atom_mask']
        atom_embed = input_feats['atom_embed']
        atom_to_res_idx = input_feats['atom_to_res_idx']
        atompair_embed_skip = input_feats['atompair_embed']
        atompair_mask = input_feats['atompair_mask']
        n_atoms = input_feats['n_atoms']
        to_queries = input_feats['to_queries']
        to_keys = input_feats['to_keys']
        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))

        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)
        noised_atompos = self.scale_pos(noised_atompos)
        node_embed = node_embed * node_mask[..., None]

        traj_data = {
            b: {}
            for b in range(self.num_blocks)
        }

        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                s=node_embed,
                z=edge_embed,
                r=curr_rigids,
                mask=node_mask)
            node_embed = (node_embed + ipa_embed) * node_mask[..., None]
            if self.force_flash_transformer:
                seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                    node_embed, node_mask)
            else:
                seq_tfmr_out = self.trunk[f'tfmr_{b}'](
                    node_embed, src_key_padding_mask=1 - node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = node_embed * node_mask[..., None]

            node_embed = node_embed + self.trunk[f'transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]

            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * diffuse_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update * diffuse_mask[..., None])

            if b < self.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, curr_rigids, edge_mask)
                edge_embed *= edge_mask[..., None]

            if self.use_traj_predictions:
                traj_data[b]['dist_logits'] = self.trunk[f"dist_pred_{b}"](edge_embed)
                traj_data[b]['rigids'] = self.unscale_rigids(curr_rigids)

            # print(b, node_embed.isnan().any(), edge_embed.isnan().any())

        atom_embed = atom_embed.clone()
        atompair_embed, _ = self.atompair_init(
            atom_embed,
            curr_rigids,
            noised_atompos,
            edge_embed,
            atom_to_res_idx,
            atom_mask,
            to_queries,
            to_keys,
        )
        atompair_embed = self.atompair_fuser(
            torch.cat([atompair_embed, atompair_embed_skip], dim=-1)
        )
        atom_embed = self.atom_tfmr(
            atom_embed,
            noised_atompos,
            node_embed,
            atompair_embed,
            atom_to_res_idx,
            atom_mask,
            atompair_mask,
            to_queries,
            to_keys
        )

        n_batch, n_res = node_embed.shape[:2]
        noised_atompos_local = curr_rigids[..., None].invert_apply(
            noised_atompos[..., :n_atoms, :].view(n_batch, n_res, -1, 3)
        )
        atom_embed = atom_embed[..., :n_atoms, :].view(
            n_batch, n_res, -1, atom_embed.shape[-1])

        # print(atom_embed.isnan().any(), atompair_embed.isnan().any())

        atompos_local_update = self.atom_update(atom_embed)
        atompos_local = noised_atompos_local + atompos_local_update
        seq_logits = self.seq_pred(atom_embed, atom_to_res_idx)

        for b in range(self.num_blocks):
            traj_data[b]['seq_logits'] = seq_logits
            traj_data[b]['atom10_local'] = self.unscale_pos(atompos_local)

        curr_rigids = self.unscale_rigids(curr_rigids)
        _, psi_pred = self.torsion_pred(node_embed)
        atompos_local = self.unscale_pos(atompos_local)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'final_atom10_local': atompos_local,
            'node_embed': node_embed,
            'traj_data': traj_data
        }
        return model_out


class IpaAtom10DenoiserV3(nn.Module):
    def __init__(self,
                 # diffuser,
                 c_s=256,
                 c_z=128,
                 c_atom=128,
                 c_atompair=16,
                 c_hidden=16,
                 num_heads=16,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=8,
                 use_traj_predictions=False,
                 force_flash_transformer=False,
                 use_v2=False,
                 use_v3=False,
                 preconditioning=False,
                 nonlocal_preconditioning=False,
                 ):
        super().__init__()
        # some compatibility code
        self.self_conditioning = True
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000

        assert not (use_v2 and use_v3), "can only use v2 or v3"

        self.ipa_score = IpaScore(
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            num_blocks=num_blocks,
            use_traj_predictions=use_traj_predictions,
            force_flash_transformer=force_flash_transformer,
            coordinate_scaling=1 if preconditioning else 0.1,
        )
        self.embedder = Embedder(
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_atompair=c_atompair
        )
        self.c_s = c_s
        self.preconditioning = preconditioning
        self.nonlocal_preconditioning = nonlocal_preconditioning

    def forward(self, data, self_condition=None):
        res_data = data['residue']
        res_mask = (res_data['res_mask']).bool()

        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'])
        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, res_data.batch, res_data['res_mask'].bool())
        rigids_t = rigids_t.translate(-center)

        data_list = data.to_data_list()
        for d in data_list:
            assert d.num_nodes == data_list[0].num_nodes

        seq_idx = [torch.arange(data_list[0].num_nodes, device=center.device) for _ in data_list]
        seq_idx = torch.stack(seq_idx)
        t = data['t']
        batch_size = t.shape[0]
        fixed_mask = torch.zeros_like(res_mask).view(batch_size, -1)

        if self_condition is not None:
            sc_rigids = self_condition['final_rigids'].view(batch_size, -1)
            sc_atom14 = self_condition['denoised_atom14'].view(batch_size, -1, 14, 3)
        else:
            sc_rigids = None
            sc_atom14 = None

        # center the training example at the mean of the x_cas
        rigids_t = Rigid.from_tensor_7(res_data['rigids_t'])
        center = batchwise_center(rigids_t, res_data.batch, res_mask)
        rigids_t = rigids_t.translate(-center)
        rigids_t = rigids_t.view([t.shape[0], -1])
        rigids_t_original_scale = rigids_t
        noised_atom10_local = res_data['noised_atom10_local'].view([batch_size, -1, 10, 3])

        noised_atom10_local = noised_atom10_local - data['atom10_prior_offset'][None, None, None]
        if self.preconditioning:
            rigids_t = rigids_t.apply_trans_fn(lambda x: x * data['trans_c_in'][..., None, None])
            noised_atom10_local_in = noised_atom10_local * data['atom10_c_in'][..., None, None, None]
        else:
            noised_atom10_local_in = noised_atom10_local

        node_embed, edge_embed, atompos, atom_embed, atompair_embed, atompair_mask, to_queries, to_keys, n_atoms, atom_to_res_idx, atom_mask = self.embedder(
            noised_atom10_local=noised_atom10_local_in,
            seq_idx=seq_idx,
            t=t,
            node_mask=res_mask.view(batch_size, -1),
            fixed_mask=fixed_mask,
            rigids=rigids_t,
            sc_rigids=sc_rigids,
            sc_atom14=sc_atom14
        )

        input_feats = {
            'fixed_mask': fixed_mask,
            'res_mask': res_mask.view(batch_size, -1),
            'rigids_t': rigids_t.to_tensor_7(),
            't': t,
            "noised_atom10_local": noised_atom10_local_in,
            "atompos": atompos,
            "atom_embed": atom_embed,
            "atom_mask": atom_mask,
            "atompair_embed": atompair_embed,
            "atompair_mask": atompair_mask,
            "to_queries": to_queries,
            "to_keys": to_keys,
            "n_atoms": n_atoms,
            "atom_to_res_idx": atom_to_res_idx,
        }

        score_dict = self.ipa_score(node_embed, edge_embed, input_feats)
        rigids = score_dict['final_rigids']
        final_atom10_local = score_dict['final_atom10_local']
        if self.preconditioning:
            rigids = rigids.apply_trans_fn(lambda x: x * data['trans_c_out'][..., None, None] + rigids_t_original_scale.get_trans() * data['trans_c_skip'][..., None, None])
            final_atom10_local = final_atom10_local * data['atom10_c_out'][..., None, None, None] + noised_atom10_local * data['atom10_c_skip'][..., None, None, None]
        rigids = rigids.view(-1).translate(center)
        final_atom10_local = final_atom10_local + data['atom10_prior_offset'][None, None, None]
        final_atom10_local = final_atom10_local.view(-1, 10, 3)

        psi = score_dict['psi'].view(-1, 2)

        ret = {}
        ret['denoised_frames'] = rigids
        ret['final_rigids'] = rigids
        ret['denoised_atom10_local'] = final_atom10_local
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_atom14 = denoised_bb_items[-1].squeeze(0)
        denoised_atom14[..., 4:, :] = rigids[..., None].apply(final_atom10_local)
        denoised_bb = denoised_atom14[:, :5]

        ret['denoised_bb'] = denoised_bb
        ret['denoised_atom14'] = denoised_atom14
        ret['psi'] = psi
        traj_data = score_dict['traj_data']
        ret['traj_data'] = traj_data
        ret['decoded_seq_logits'] = traj_data[max(traj_data.keys())]['seq_logits'].flatten(0, 1)

        return ret