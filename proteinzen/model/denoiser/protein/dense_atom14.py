import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy

import functools as fn

from proteinzen.model.modules.layers.node.attention import FlashTransformerEncoder
from proteinzen.model.modules.openfold.layers_v2 import Linear, ConditionedInvariantPointAttention, ConditionedTransition, BackboneUpdate, TorsionAngles
import proteinzen.utils.openfold.rigid_utils as ru
from proteinzen.utils.openfold.rigid_utils import Rigid, batchwise_center
from proteinzen.utils.framediff.all_atom import compute_backbone

from ._attn import ConditionedPairUpdate

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
    def __init__(self, c_s, n_aa=21, atom_input=False):
        super().__init__()
        self.atom_input = atom_input
        if atom_input:
            self.proj = nn.Sequential(
                Linear(c_s + 3*10+10, c_s),
                nn.LayerNorm(c_s),
                Linear(c_s, n_aa)
            )
        else:
            self.proj = nn.Sequential(
                Linear(c_s, c_s),
                nn.LayerNorm(c_s),
                Linear(c_s, n_aa)
            )

    def forward(self, node_features, flat_atom10):
        if self.atom_input:
            return self.proj(
                torch.cat([
                    node_features,
                    flat_atom10,
                    torch.linalg.vector_norm(flat_atom10.view(*flat_atom10.shape[:-1], 10, 3), dim=-1)
                ], dim=-1)
            )
        else:
            return self.proj(node_features)


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


class AtomUpdate(nn.Module):
    def __init__(self, c_s, c_out=3*10):
        super().__init__()
        self.out = nn.Sequential(
            nn.LayerNorm(c_s),
            Linear(c_s, c_out, bias=False, init='final')
        )

    def forward(self, s):
        return self.out(s)


class Embedder(nn.Module):

    def __init__(self,
                 c_s,
                 c_z,
                 index_embed_size=32,
                 use_init_distogram=False,
    ):
        super(Embedder, self).__init__()

        # Time step embedding
        t_embed_size = index_embed_size
        node_embed_dims = (
            t_embed_size    # time
            + 1             # noised state
            + 3 * 10        # atom pos in local frame
            + 10            # atom dist
            + index_embed_size  # res index
        )
        # self conditioning
        node_embed_dims += (
            3 * 10        # atom pos in local frame
            + 10             # atom dist
        )

        node_embed_size = c_s
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        edge_in = (
            t_embed_size    # time
            + 1             # noised state
            + 3 * 10        # atom pos in local frame
            + 10             # atom dist
        ) * 2
        edge_in += index_embed_size
        edge_in += 22

        self.use_init_distogram = use_init_distogram
        if use_init_distogram:
            edge_in += 22

        edge_embed_size = c_z
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=index_embed_size
        )

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

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
            sc_atom14=None
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
        num_batch, num_res = seq_idx.shape
        node_feats = []

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))

        atom10_dist_to_ca = torch.linalg.vector_norm(noised_atom10_local, dim=-1)
        prot_t_embed = torch.cat([
            prot_t_embed,
            fixed_mask,
            noised_atom10_local.flatten(-2, -1),
            1/(1 + atom10_dist_to_ca**2)
        ], dim=-1)
        node_feats = [prot_t_embed]
        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])

        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]
        pair_feats.append(self.index_embedder(rel_seq_offset))

        if sc_rigids is not None:
            sc_dgram = calc_distogram(
                sc_rigids.get_trans(),
                1e-5,
                20,
                22
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))

            if sc_atom14 is not None:
                sidechain_atoms = sc_atom14[..., 4:, :]
                sidechain_atoms_local_frame = sc_rigids[..., None].invert_apply(sidechain_atoms)
                sidechain_atoms_dist_to_ca = torch.linalg.vector_norm(sidechain_atoms_local_frame, dim=-1)
                node_feats += [
                    sidechain_atoms_local_frame.flatten(-2, -1),
                    1/(1 + sidechain_atoms_dist_to_ca)
                ]
            else:
                raise ValueError("we expect both sc_atom14 and sc_rigids at the same time")
        else:
            pair_feats.append(torch.zeros([num_batch, num_res**2, 22], device=prot_t_embed.device))
            node_feats.append(torch.zeros([num_batch, num_res, 3*10+10], device=prot_t_embed.device))

        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        return node_embed, edge_embed


class IpaScore(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 use_traj_predictions=False,
                 force_flash_transformer=False,
                 cond_ffn=False,
                 seq_pred_atom_input=False,
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
            c_cond = (
                3 * 10  # atom pos
                + 10     # atom dist
            )
            if cond_ffn:
                self.trunk[f'cond_ffn_1_{b}'] = nn.Sequential(
                    Linear(c_cond, c_s, bias=False),
                    nn.ReLU(),
                    Linear(c_s, c_s, bias=False),
                    nn.ReLU(),
                    Linear(c_s, c_s, bias=False)
                )
                self.trunk[f'cond_ffn_2_{b}'] = nn.Sequential(
                    Linear(c_cond, c_s, bias=False),
                    nn.ReLU(),
                    Linear(c_s, c_s, bias=False),
                    nn.ReLU(),
                    Linear(c_s, c_s, bias=False)
                )
                c_cond = c_s
            else:
                self.trunk[f'cond_ffn_1_{b}'] = None
                self.trunk[f'cond_ffn_2_{b}'] = None

            self.trunk[f'ipa_{b}'] = ConditionedInvariantPointAttention(
                 c_s=c_s,
                 c_cond=c_cond,
                 c_z=c_z,
                 c_hidden=c_hidden,
                 num_heads=num_heads,
                 num_qk_points=num_qk_points,
                 num_v_points=num_v_points,
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
                    norm_first=True
                )
                self.trunk[f'tfmr_{b}'] = torch.nn.TransformerEncoder(
                    tfmr_layer, 2)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                c_s, c_s, init="final", bias=False)
            self.trunk[f'transition_{b}'] = ConditionedTransition(
                c_s=c_s,
                c_cond=c_cond
            )
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(c_s)
            self.trunk[f'atom_update_{b}'] = AtomUpdate(c_s)

            if use_traj_predictions:
                self.trunk[f'seq_pred_{b}'] = SeqPredictor(c_s, atom_input=seq_pred_atom_input)
                self.trunk[f'dist_pred_{b}'] = EdgeDistPredictor(c_z)

            if b < num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ConditionedPairUpdate(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_z//4,
                    no_heads=4
                )

        self.torsion_pred = TorsionAngles(c_s, 1)


    def forward(self, init_node_embed, edge_embed, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = input_feats['rigids_t'].type(torch.float32)
        noised_atom10_local = input_feats['noised_atom10_local']

        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)
        curr_atoms = noised_atom10_local.flatten(-2, -1)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]

        traj_data = {
            b: {}
            for b in range(self.num_blocks)
        }

        for b in range(self.num_blocks):
            atoms_cond = torch.cat([
                curr_atoms,
                torch.linalg.vector_norm(curr_atoms.view(*curr_rigids.shape, 10, 3), dim=-1)
            ], dim=-1)
            if self.trunk[f'cond_ffn_1_{b}'] is not None:
                atoms_cond_1 = self.trunk[f'cond_ffn_1_{b}'](atoms_cond)
                atoms_cond_2 = self.trunk[f'cond_ffn_2_{b}'](atoms_cond)
            else:
                atoms_cond_1 = atoms_cond
                atoms_cond_2 = atoms_cond

            ipa_embed = self.trunk[f'ipa_{b}'](
                s=node_embed,
                cond=atoms_cond_1,
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
            node_embed = self.trunk[f'transition_{b}'](node_embed, atoms_cond_2)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * diffuse_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update * diffuse_mask[..., None])
            curr_atoms = curr_atoms + (
                self.trunk[f'atom_update_{b}'](node_embed) * diffuse_mask[..., None]
            )

            if b < self.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, curr_rigids, edge_mask)
                edge_embed *= edge_mask[..., None]

            if self.use_traj_predictions:
                traj_data[b]['seq_logits'] = self.trunk[f"seq_pred_{b}"](node_embed, curr_atoms)
                traj_data[b]['dist_logits'] = self.trunk[f"dist_pred_{b}"](edge_embed)
                traj_data[b]['rigids'] = self.unscale_rigids(curr_rigids)
                traj_data[b]['atom10_local'] = curr_atoms

        curr_rigids = self.unscale_rigids(curr_rigids)
        _, psi_pred = self.torsion_pred(node_embed)
        curr_atom10 = curr_atoms.view(*curr_rigids.shape, 10, 3)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'final_atom10_local': curr_atom10,
            'node_embed': node_embed,
            'traj_data': traj_data
        }
        return model_out


class IpaAtom10Denoiser(nn.Module):
    def __init__(self,
                 # diffuser,
                 c_s=256,
                 c_z=128,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=8,
                 use_init_dgram=False,
                 use_traj_predictions=False,
                 force_flash_transformer=False,
                 use_v2=False,
                 use_v3=False,
                 cond_ffn=False,
                 seq_pred_atom_input=False,
                 atom10_preconditioning=False,
                 trans_preconditioning=False,
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
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            num_blocks=num_blocks,
            use_traj_predictions=use_traj_predictions,
            force_flash_transformer=force_flash_transformer,
            cond_ffn=cond_ffn,
            seq_pred_atom_input=seq_pred_atom_input,
            coordinate_scaling=1 if trans_preconditioning else 0.1
        )
        self.embedder = Embedder(
            c_s=c_s,
            c_z=c_z,
            use_init_distogram=use_init_dgram,
        )
        self.c_s = c_s
        self.atom10_preconditioning = atom10_preconditioning
        self.trans_preconditioning = trans_preconditioning

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
        if self.atom10_preconditioning:
            # TODO: we shouldn't hardcode this
            cb_local = torch.tensor([-5.537e-01, -7.773e-01, -1.201e+00], device=noised_atom10_local.device)
            noised_atom10_local = noised_atom10_local - cb_local[None, None, None]
            noised_atom10_local = noised_atom10_local * data['atom10_c_in'][..., None, None, None]

        if self.trans_preconditioning:
            rigids_t = rigids_t.apply_trans_fn(lambda x: x * data['trans_c_in'][..., None, None])

        node_embed, edge_embed = self.embedder(
            noised_atom10_local=noised_atom10_local,
            seq_idx=seq_idx,
            t=t,
            node_mask=res_mask,
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
            "noised_atom10_local": noised_atom10_local,
        }

        score_dict = self.ipa_score(node_embed, edge_embed, input_feats)
        rigids = score_dict['final_rigids']
        if self.trans_preconditioning:
            rigids = rigids.apply_trans_fn(lambda x: x * data['trans_c_out'][..., None, None] + rigids_t_original_scale.get_trans() * data['trans_c_skip'][..., None, None])
        rigids = rigids.view(-1).translate(center)
        if self.atom10_preconditioning:
            final_atom10_local = score_dict['final_atom10_local']
            # TODO: we shouldn't hardcode this
            cb_local = torch.tensor([-5.537e-01, -7.773e-01, -1.201e+00], device=final_atom10_local.device)
            final_atom10_local = final_atom10_local * data['atom10_c_out'][..., None, None, None] + noised_atom10_local * data['atom10_c_skip'][..., None, None, None]
            final_atom10_local = final_atom10_local + cb_local[None, None, None]
            final_atom10_local = final_atom10_local.view(-1, 10, 3)
        else:
            final_atom10_local = score_dict['final_atom10_local'].view(-1, 10, 3)

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