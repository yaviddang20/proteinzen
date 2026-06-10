import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import copy
import functools as fn

from scipy.optimize import linear_sum_assignment

from proteinzen.boltz.data import const

from proteinzen.model.modules.attention import TransformerPairBias
from proteinzen.openfold.layers.layers import InvariantPointAttention, Dropout, TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming, permute_final_dims
from proteinzen.openfold.layers.layers_v2 import (
    Linear, ConditionedInvariantPointAttention, BackboneUpdate, TorsionAngles, LayerNorm, AdaLN, ConditionedTransition, lecun_normal_init_
)

import proteinzen.openfold.utils.rigid_utils as ru
from proteinzen.openfold.utils.rigid_utils import Rigid
from proteinzen.stoch_interp.so3_utils import rotquat_to_rotvec

from proteinzen.model.modules.pair_modules import MultiRigidPairEmbedder

from proteinzen.model.modules.frame_transformer import ScatterUpdate, GatherUpdate
from proteinzen.model.modules.pair_factorize import TransitionFactorizer, AttentionFactorizer
from proteinzen.model.modules.fafpb import FactorizedPairBiasTransformer
from proteinzen.model.modules.flash_ipa import FlashInvariantPointAttention, EdgeTransition


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
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size)))# .to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size)))# .to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding

def get_timestep_embedding_flexshape(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[..., None] * emb.view(*[1 for _ in timesteps.shape], -1)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
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


def gather_helper(tensor, token_gather_idx):
    new_dims = tensor.dim() - token_gather_idx.dim()
    idx_expand = token_gather_idx.view(
        *token_gather_idx.shape, *[1 for _ in range(new_dims)]
    ).expand(
        *[-1 for _ in token_gather_idx.shape],
        *tensor.shape[-new_dims:]
    ).long()
    return torch.gather(
        tensor,
        1,
        idx_expand
    )



class SeqPredictor(nn.Module):
    def __init__(self, c_s, c_frame, n_aa=21):
        super().__init__()
        self.ln = LayerNorm(c_frame)
        self.scatter_update = ScatterUpdate(c_s, c_frame)
        self.out = Linear(c_s, n_aa)
        self.c_s = c_s

    def forward(self, rigids_embed_flat, rigids_to_res_idx, rigids_mask, out):
        rigids_embed_flat = self.ln(rigids_embed_flat)
        seq_embed = self.scatter_update(
            rigids_embed_flat,
            out,
            rigids_to_res_idx,
            rigids_mask
        )
        return self.out(seq_embed)


class RigidTransformer(nn.Module):
    def __init__(
        self,
        c_s,
        c_z_factor=128,
        z_factor_rank=4,
        no_blocks=4,
        num_ipa_heads=16,
        num_attn_pair_bias_heads=16,
        num_qk_points=8,
        num_v_points=12,
    ):
        super().__init__()

        self.trunk = nn.ModuleDict()
        self.no_blocks = no_blocks

        for b in range(no_blocks):
            self.trunk[f'ipa_{b}'] = FlashInvariantPointAttention(
                c_s=c_s,
                c_z=c_z_factor,
                z_factor_rank=z_factor_rank,
                c_hidden=c_s // num_ipa_heads,
                no_heads=num_ipa_heads,
                no_qk_points=num_qk_points,
                no_v_points=num_v_points,
            )

            self.trunk[f'tfmr_{b}'] = FactorizedPairBiasTransformer(
                num_blocks=2,
                c_s=c_s,
                c_z=c_z_factor,
                z_factor_rank=z_factor_rank,
                c_hidden=c_s // num_attn_pair_bias_heads,
                no_heads=num_attn_pair_bias_heads,
            )
            self.trunk[f'edge_transition_{b}'] = EdgeTransition(
                mode='1d',
                node_embed_size=c_s,
                edge_embed_in=c_z_factor,
                edge_embed_out=c_z_factor,
                z_factor_rank=z_factor_rank,
            )

    def forward(
        self,
        rigids_embed,
        rigids,
        rigids_z1,
        rigids_z2,
        rigids_mask
    ):
        for b in range(self.no_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                s=rigids_embed,
                z=None,
                z_factor_1=rigids_z1,
                z_factor_2=rigids_z2,
                r=rigids,
                mask=rigids_mask)
            rigids_embed = (rigids_embed + ipa_embed) * rigids_mask[..., None]

            rigids_embed = self.trunk[f'tfmr_{b}'](
                rigids_embed,
                rigids_z1,
                rigids_z2,
                mask=rigids_mask
            )

            rigids_z1, rigids_z2 = self.trunk[f'edge_transition_{b}'](
                node_embed=rigids_embed,
                edge_embed=None,
                z_factor_1=rigids_z1,
                z_factor_2=rigids_z2
            )
            rigids_z1 = rigids_z1 * rigids_mask[..., None, None]
            rigids_z2 = rigids_z2 * rigids_mask[..., None, None]

        return rigids_embed, rigids_z1, rigids_z2


class Embedder(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_frame,
                 c_hidden=64,
                 c_z_factor=128,
                 z_factor_rank=4,
                 num_pair_embed_blocks=4,
                 num_node_embed_blocks=24,
                 num_rigid_embed_blocks=4,
                 index_embed_size=256,
                 max_rigids_idx=3+14,
                 use_qk_norm=True,
                 restype_dict=const.token_ids,
                 num_elements=const.num_elements,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_sidechain = c_frame

        self.timestep_embedder = fn.partial(
            get_timestep_embedding_flexshape,
            embedding_dim=index_embed_size
        )
        self.time_condition_embed = Linear(index_embed_size, c_s, bias=False)
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=index_embed_size
        )

        self.restype_dict = restype_dict
        self.num_aa = len(restype_dict)
        self.mask_token = restype_dict['UNK']
        self.node_init = Linear(index_embed_size, self.c_s, bias=False)
        self.node_seq_embed = Linear(
            self.num_aa, self.c_s, bias=False
        )
        self.node_is_unindexed_embed = Linear(1, c_s, bias=False)
        self.node_is_copy_embed = Linear(1, c_s, bias=False)
        self.node_hotspot_type_embed = nn.Embedding(3, c_s)

        self.rigid_init = Linear(self.c_s, c_frame, bias=False)
        self.rigid_time_embed = Linear(index_embed_size, c_frame, bias=False)
        self.rigid_idx_embed = nn.Embedding(max_rigids_idx, c_frame)
        self.rigid_is_atomized_embed = Linear(1, c_frame, bias=False)
        self.rigid_element_embed = nn.Embedding(num_elements, c_frame)
        self.rigid_charge_embed = Linear(1, c_frame, bias=False)

        self.pair_embedder = MultiRigidPairEmbedder(
            c_z,
            c_hidden,
            no_blocks=num_pair_embed_blocks,
            use_qk_norm=use_qk_norm,
            use_self_folding=False,
            add_same_chain_feature=True
        )
        self.pair_factorizer = TransitionFactorizer(
            c_factor=c_z_factor,
            c_z=c_z,
            z_factor_rank=z_factor_rank
        )
        # self.pair_factorizer = AttentionFactorizer(
        #     c_s=c_s,
        #     c_factor=c_z_factor,
        #     c_z=c_z,
        #     z_factor_rank=z_factor_rank
        # )

        self.rigid_tfmr = RigidTransformer(
            c_s=c_frame,
            c_z_factor=c_z_factor,
            z_factor_rank=z_factor_rank,
            no_blocks=num_rigid_embed_blocks
        )

        self.rigid_to_node = ScatterUpdate(c_s, c_frame)
        self.node_to_rigid = GatherUpdate(c_s, c_frame)

        self.node_tfmr = TransformerPairBias(
            c_s=c_s,
            c_z=c_z,
            no_heads=16,
            n_layers=num_node_embed_blocks,
            dropout=0.0,
            use_qk_norm=True
        )


    def _gen_node_features(
        self,
        seq,
        seq_idx,
        seq_noising_mask,
        seq_mask,
        rigids,
        token_gather_idx,
        is_unindexed_mask,
        is_copy_mask,
        token_hotspot_type,
    ):
        rigids_to_nodes = fn.partial(gather_helper, token_gather_idx=token_gather_idx)
        node_rigids = rigids_to_nodes(rigids)

        visible_seq = seq * seq_mask + self.mask_token * (~seq_mask)
        visible_seq = visible_seq * (~seq_noising_mask) + self.mask_token * seq_noising_mask
        seq_embed = F.one_hot(visible_seq, num_classes=self.num_aa).float()
        node_init = (
            self.node_seq_embed(seq_embed)
            + self.node_is_unindexed_embed(is_unindexed_mask[..., None].float())
            + self.node_is_copy_embed(is_copy_mask[..., None].float())
            + self.node_hotspot_type_embed(token_hotspot_type.long())
        )
        # node_seq_idx_embed = self.index_embedder(seq_idx)
        # node_init = node_init + self.node_init(node_seq_idx_embed)

        return {
            "node_init": node_init,
            "node_rigids": ru.Rigid.from_tensor_7(node_rigids),
        }


    def _gen_rigid_features(
        self,
        node_init,
        rigids_element,
        rigids_charge,
        t,
        rigids_token_uid,
        rigids_idx,
        rigids_is_atomized_mask,
    ):
        nodes_to_rigids = fn.partial(gather_helper, token_gather_idx=rigids_token_uid)

        rigids_init = self.rigid_init(nodes_to_rigids(node_init))
        time_embed = self.rigid_time_embed(self.timestep_embedder(t))
        rigids_idx_embed = self.rigid_idx_embed(rigids_idx)
        is_atomized_embed = self.rigid_is_atomized_embed(rigids_is_atomized_mask[..., None].float())
        element_mask = (rigids_element != -1)
        element_embed = self.rigid_element_embed(rigids_element * element_mask) * element_mask[..., None]
        charge_embed = self.rigid_charge_embed(rigids_charge.unsqueeze(-1))

        rigids_init = (
            rigids_init
            + time_embed
            + rigids_idx_embed
            + is_atomized_embed
            + element_embed
            + charge_embed
        )
        return rigids_init

    def forward(
            self,
            *,
            token_mask,
            token_seq,
            token_seq_idx,
            token_seq_noising_mask,
            token_seq_mask,
            token_chain_idx,
            # token_is_atomized_mask,
            token_is_unindexed_mask,
            token_is_copy_mask,
            token_hotspot_type,
            token_gather_idx,
            t,
            rigids,
            rigids_element,
            rigids_charge,
            rigids_token_uid,
            rigids_idx,
            rigids_mask,
            rigids_noising_mask,
            rigids_is_atomized_mask,
            token_bonds,
        ):
        # compute node embeddings
        node_data = self._gen_node_features(
            token_seq,
            token_seq_idx,
            token_seq_noising_mask,
            token_seq_mask,
            rigids,
            token_gather_idx,
            token_is_unindexed_mask,
            token_is_copy_mask,
            token_hotspot_type,
        )
        node_init = node_data['node_init']

        # compute edge embedding and factorize it
        edge_embed = self.pair_embedder(
            node_data['node_rigids'],
            token_mask,
            token_seq_idx,
            token_seq_idx, # TODO: this is incorrect, just putting it here to test some other code
            token_chain_idx,
            token_is_unindexed_mask,
            token_bonds,
        )
        raise Exception("fix ur temp todo")
        z1, z2 = self.pair_factorizer(edge_embed)
        # z1, z2 = self.pair_factorizer(node_init, edge_embed, token_mask)

        nodes_to_rigids = fn.partial(gather_helper, token_gather_idx=rigids_token_uid)
        rigids_edge_z1 = nodes_to_rigids(z1)
        rigids_edge_z2 = nodes_to_rigids(z2)

        # compute rigid embeddings from node embeddings
        rigids_init = self._gen_rigid_features(
            node_init,
            rigids_element,
            rigids_charge,
            t,
            rigids_token_uid,
            rigids_idx,
            rigids_is_atomized_mask,
        )

        rigids_init, rigids_edge_z1, rigids_edge_z2 = self.rigid_tfmr(
            rigids_init,
            ru.Rigid.from_tensor_7(rigids).apply_trans_fn(lambda x: x * 0.1),
            rigids_edge_z1,
            rigids_edge_z2,
            rigids_mask
        )

        node_embed = self.rigid_to_node(rigids_init, node_init, rigids_token_uid, rigids_mask)
        node_embed = self.node_tfmr(
            node_embed,
            edge_embed,
            token_mask
        )
        rigids_init = self.node_to_rigid(node_embed, rigids_init, rigids_token_uid, rigids_mask)

        rigids_data = {
            "rigids_t": ru.Rigid.from_tensor_7(rigids),
            "rigids_init": rigids_init,
            "rigids_mask": rigids_mask,
            "rigids_token_uid": rigids_token_uid,
            "rigids_is_atomized_mask": rigids_is_atomized_mask,
            "rigids_noising_mask": rigids_noising_mask
        }

        ret = {
            "node_init": node_embed,
            "rigids_edge_z1": rigids_edge_z1,
            "rigids_edge_z2": rigids_edge_z2,
            "time_condition_embed": self.time_condition_embed(self.timestep_embedder(t)),
        }
        ret.update(rigids_data)
        return ret


class IpaDenoiser(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z_factor=128,
                 z_factor_rank=2,
                 num_ipa_heads=16,
                 num_attn_pair_bias_heads=16,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 num_aa=21,
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        self.num_blocks = num_blocks
        self.num_aa = num_aa
        self.c_z_factor = c_z_factor
        self.z_factor_rank = z_factor_rank

        for b in range(num_blocks):
            self.trunk[f'ipa_{b}'] = FlashInvariantPointAttention(
                c_s=c_s,
                c_z=c_z_factor,
                z_factor_rank=z_factor_rank,
                c_hidden=c_s // num_ipa_heads,
                no_heads=num_ipa_heads,
                no_qk_points=num_qk_points,
                no_v_points=num_v_points,
            )

            self.trunk[f'tfmr_{b}'] = FactorizedPairBiasTransformer(
                num_blocks=2,
                c_s=c_s,
                c_z=c_z_factor,
                z_factor_rank=z_factor_rank,
                c_hidden=c_s // num_attn_pair_bias_heads,
                no_heads=num_attn_pair_bias_heads,
            )

            self.trunk[f'rigids_update_{b}'] = BackboneUpdate(c_s)

            if b < num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = EdgeTransition(
                    mode='1d',
                    node_embed_size=c_s,
                    edge_embed_in=c_z_factor,
                    edge_embed_out=c_z_factor,
                    z_factor_rank=z_factor_rank,
                )

        self.torsion_pred = TorsionAngles(c_s, 1)
        self.seq_pred = SeqPredictor(c_s, c_s, n_aa=num_aa)


    def forward(self, input_feats):
        node_init = input_feats['node_init']

        init_rigids = input_feats['rigids_t']
        rigids_embed = input_feats['rigids_init']
        rigids_token_uid = input_feats['rigids_token_uid']
        rigids_mask = input_feats['rigids_mask']
        rigids_z1 = input_feats['rigids_edge_z1']
        rigids_z2 = input_feats['rigids_edge_z2']
        rigids_noising_mask_flat = input_feats['rigids_noising_mask']

        curr_rigids = self.scale_rigids(init_rigids)

        # Main trunk
        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                s=rigids_embed,
                z=None,
                z_factor_1=rigids_z1,
                z_factor_2=rigids_z2,
                r=curr_rigids,
                mask=rigids_mask)
            rigids_embed = (rigids_embed + ipa_embed) * rigids_mask[..., None]

            rigids_embed = self.trunk[f'tfmr_{b}'](
                rigids_embed,
                rigids_z1,
                rigids_z2,
                mask=rigids_mask
            )

            rigid_update = self.trunk[f'rigids_update_{b}'](
                rigids_embed * rigids_noising_mask_flat[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update *  rigids_noising_mask_flat[..., None])

            if b < self.num_blocks-1:
                rigids_z1, rigids_z2 = self.trunk[f'edge_transition_{b}'](
                    node_embed=rigids_embed,
                    edge_embed=None,
                    z_factor_1=rigids_z1,
                    z_factor_2=rigids_z2
                )
                rigids_z1 = rigids_z1 * rigids_mask[..., None, None]
                rigids_z2 = rigids_z2 * rigids_mask[..., None, None]

        seq_logits = self.seq_pred(
            rigids_embed,
            rigids_token_uid,
            rigids_mask,
            out=torch.zeros_like(node_init)
        )

        curr_rigids = self.unscale_rigids(curr_rigids)
        _, psi_pred = self.torsion_pred(node_init)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'seq_logits': seq_logits
        }
        return model_out


class FlashIpaMultiRigidDenoiser(nn.Module):
    def __init__(self,
                 c_s=768,
                 c_frame=512,
                 c_z=256,
                 c_z_factor=128,
                 z_factor_rank=4,
                 num_ipa_heads=16,
                 num_attn_pair_bias_heads=4,
                 num_qk_points=8,
                 num_v_points=12,
                 num_pair_embed_blocks=4,
                 num_node_embed_blocks=4,
                 num_encoder_blocks=4,
                 num_decoder_blocks=12,
                 trans_preconditioning=False,
                 rot_preconditioning=True,
                 cg_version=1,
                 use_amp=True,
                 rot_vf_scaling=1,
                 ):
        super().__init__()

        self.use_amp = use_amp
        self.self_conditioning = False

        self.ipa_denoiser = IpaDenoiser(
            c_s=c_frame,
            c_z_factor=c_z_factor,
            z_factor_rank=z_factor_rank,
            num_ipa_heads=num_ipa_heads,
            num_attn_pair_bias_heads=num_attn_pair_bias_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            num_blocks=num_decoder_blocks,
            coordinate_scaling=1 if trans_preconditioning else 0.1,
            num_aa=len(const.tokens)
        )

        self.embedder = Embedder(
            c_s=c_s,
            c_frame=c_frame,
            c_z=c_z,
            c_z_factor=c_z_factor,
            z_factor_rank=z_factor_rank,
            num_node_embed_blocks=num_node_embed_blocks,
            num_pair_embed_blocks=num_pair_embed_blocks,
            num_rigid_embed_blocks=num_encoder_blocks
        )

        self.c_s = c_s
        self.trans_preconditioning = trans_preconditioning
        self.rot_preconditioning = rot_preconditioning
        self.cg_version = cg_version
        self.rot_vf_scaling = rot_vf_scaling


    def forward(self, data, self_condition=None, self_folding=None, sanitize_motif_idx=False):
        token_data = data['token']
        rigids_data = data['rigids']

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            input_feats = self.embedder(
                token_mask=token_data['token_mask'],
                token_seq_idx=token_data['residue_idx'],
                token_seq=token_data['res_type'],
                token_seq_mask=token_data['token_mask'],
                token_seq_noising_mask=token_data['seq_noising_mask'],
                token_chain_idx=token_data['asym_id'],
                token_is_unindexed_mask=token_data['token_is_unindexed_mask'],
                token_is_copy_mask=token_data['token_is_copy_mask'],
                token_hotspot_type=token_data['hotspot_type'],
                token_gather_idx=token_data['token_to_rep_rigid'],
                t=data['t'],
                rigids=rigids_data['rigids_t'],
                rigids_element=rigids_data['rigids_ref_element'],
                rigids_charge=rigids_data['rigids_ref_charge'],
                rigids_token_uid=rigids_data['rigids_to_token'],
                rigids_idx=rigids_data['rigids_sidechain_idx'],
                rigids_mask=rigids_data['rigids_mask'],
                rigids_is_atomized_mask=rigids_data['rigids_is_atom_mask'],
                rigids_noising_mask=rigids_data['rigids_noising_mask'],
                token_bonds=token_data['token_bonds'],
            )

            input_feats['condition_embed'] = input_feats['time_condition_embed']
            input_feats['token_mask'] = token_data['token_mask']
            input_feats['token_gather_idx'] = token_data['token_to_rep_rigid']
            input_feats['t'] = data['t']

            score_dict = self.ipa_denoiser(input_feats)

        rigids_out = score_dict['final_rigids']

        if self.rot_preconditioning:
            t = data['t']
            def scale_rot(rot_in, rot_out):
                rel_rot = rot_out.compose_q(rot_in.invert())
                rel_rotquat = rel_rot.get_quats()
                rel_rotvec = rotquat_to_rotvec(rel_rotquat.view(-1, 4)).view(*rel_rotquat.shape[:-1], -1)
                angle = torch.linalg.vector_norm(rel_rotvec + 1e-8, dim=-1)
                scaled_angle = angle * (1 - t)
                axis = F.normalize(rel_rotvec, dim=-1)
                scaled_rotquat = torch.cat([
                    torch.cos(scaled_angle/2)[..., None], torch.sin(scaled_angle/2)[..., None] * axis
                ], dim=-1)
                scaled_rot = ru.Rotation(quats=scaled_rotquat)
                new_rot = scaled_rot.compose_q(rot_in)
                return new_rot

            rigids_in = ru.Rigid.from_tensor_7(data['rigids']['rigids_t'])
            rots_in = rigids_in.get_rots()
            rots_out = rigids_out.get_rots()
            rigids_out = Rigid(
                rots=scale_rot(rots_in, rots_out),
                trans=rigids_out.get_trans()
            )

        seq_logits = score_dict['seq_logits']

        pred_seq = seq_logits[..., :-1].argmax(dim=-1)
        seq_noising_mask = token_data['seq_noising_mask']
        pred_seq = pred_seq * seq_noising_mask + token_data['seq'] * (~seq_noising_mask)

        if rigids_out.to_tensor_7().isnan().any() or pred_seq.isnan().any():
            print("caught a nan in forward")
            exit()

        ret = {}
        ret['denoised_rigids'] = rigids_out
        ret['decoded_seq_logits'] = seq_logits
        ret['pred_seq'] = pred_seq

        with torch.no_grad():
            token_rigids = gather_helper(rigids_out.to_tensor_7(), token_data['token_to_rep_rigid'])
            token_rigids = ru.Rigid.from_tensor_7(token_rigids)

            motif_rigid_mask = token_data['token_is_copy_mask']
            protein_rigid_mask = ~token_data['token_is_copy_mask']
            dist_mask = motif_rigid_mask[..., None] & protein_rigid_mask[..., None, :]
            res_CA_pos = token_rigids.get_trans()
            trans_dist = torch.cdist(res_CA_pos, res_CA_pos)
            trans_dist[~dist_mask] = 1e6
            closest_neighbors = torch.argsort(trans_dist)
            motif_idx = closest_neighbors[..., 0]
            ret['motif_idx'] = motif_idx
            if sanitize_motif_idx and not self.training:
                # there are rare cases where
                # two motif residues will be assigned to the same residue
                # this can sometimes happen because of a bad generated structure
                # if prompted, we patch this out using linear sum assignment
                # with the distance matrix as the cost matrix
                for i, sample_motif_rigid_mask in enumerate(motif_rigid_mask):
                    motif_assignments = motif_idx[i, sample_motif_rigid_mask]
                    if motif_assignments.unique().numel() == motif_assignments.numel():
                        continue
                    # if we have duplicate motif assignments, resolve them via linear sum assignment
                    cost_mat = trans_dist[i, sample_motif_rigid_mask].numpy(force=True)
                    row_assign, col_assign = linear_sum_assignment(cost_mat)
                    motif_idx[i, sample_motif_rigid_mask] = torch.as_tensor(col_assign, device=motif_idx.device)
                    # print(motif_assignments, col_assign)

        return ret


class MonotonicIncreasingFn(nn.Module):
    def __init__(
        self,
        c_hidden=1024,
        n_res_ident=21,
        n_t_per_res=2
    ):
        super().__init__()
        self.c_hidden = c_hidden
        self.n_res_ident = n_res_ident
        self.n_t_per_res = n_t_per_res
        self.l1 = nn.Linear(1, n_res_ident * n_t_per_res, bias=False)
        self.l2_weight = nn.Parameter(
            torch.randn(n_res_ident * n_t_per_res, c_hidden)
        )
        self.l2_bias = nn.Parameter(
            torch.randn(n_res_ident * n_t_per_res, c_hidden)
        )
        self.l3_weight = nn.Parameter(
            torch.randn(n_res_ident * n_t_per_res, c_hidden)
        )
        with torch.no_grad():
            lecun_normal_init_(self.l1.weight)
            lecun_normal_init_(self.l2_weight)
            lecun_normal_init_(self.l2_bias)
            lecun_normal_init_(self.l3_weight)

    def gamma(self, t):
        t_per_res_ident = F.linear(t, torch.abs(self.l1.weight))
        t_per_res_ident = t_per_res_ident.view(*t_per_res_ident.shape[:-1], self.n_res_ident * self.n_t_per_res, 1)
        l2_weight_view = self.l2_weight.view(
            [1 for _ in t_per_res_ident.shape[:-2]]
            + list(self.l2_weight.shape)
        )
        l2_bias_view = self.l2_bias.view(
            [1 for _ in t_per_res_ident.shape[:-2]]
            + list(self.l2_bias.shape)
        )
        out = t_per_res_ident * torch.abs(l2_weight_view) + l2_bias_view
        out = torch.sigmoid(out)
        l3_weight_view = self.l3_weight.view(
            [1 for _ in out.shape[:-2]]
            + list(self.l3_weight.shape)
        )
        out = torch.sum(out * torch.abs(l3_weight_view), dim=-1)

        return out

    def forward(self, t):
        gamma_t = self.gamma(t)
        gamma_0 = self.gamma(torch.zeros_like(t))
        gamma_1 = self.gamma(torch.ones_like(t))
        gamma = (gamma_t - gamma_0) / (gamma_1 - gamma_0 + 1e-6)
        gamma = gamma.view(*gamma.shape[:-1], self.n_res_ident, self.n_t_per_res)
        return gamma