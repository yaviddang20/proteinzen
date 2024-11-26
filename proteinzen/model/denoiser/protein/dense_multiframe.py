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
from proteinzen.utils.framediff.all_atom import compute_backbone, compute_atom14_from_frame5

from ._attn import ConditionedPairUpdate
from ._frame_transformer import SequenceFrameTransformerUpdate, get_indexing_matrix, single_to_keys, KnnFrameTransformerUpdate, KnnInvariantPointAttention

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
    def __init__(self, c_frame, n_aa=21):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(c_frame),
            Linear(c_frame, c_frame, bias=False)
        )
        self.out = Linear(c_frame, n_aa)
        self.c_frame = c_frame

    def forward(self, frame_embed):
        seq_embed = self.proj(frame_embed)
        out = self.out(seq_embed.mean(dim=-2))
        return out


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


class Embedder(nn.Module):

    def __init__(self,
                 c_s,
                 c_z,
                 c_frame,
                 index_embed_size=32,
                 use_init_distogram=False,
    ):
        super(Embedder, self).__init__()
        self.c_sidechain = c_frame

        # Time step embedding
        t_embed_size = index_embed_size
        node_embed_dims = (
            t_embed_size    # time
            + 1             # noised state
            + index_embed_size  # res index
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

        self.node_to_sidechain_frame = Linear(c_s, c_frame, bias=False)
        self.pos_embedder = nn.Embedding(5, c_frame)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

    def forward(
            self,
            *,
            seq_idx,
            t,
            fixed_mask,
            sc_rigids=None,
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

        prot_t_embed = torch.cat([
            prot_t_embed,
            fixed_mask,
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
                sc_rigids[..., 0].get_trans(),
                1e-5,
                20,
                22
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))

        else:
            pair_feats.append(torch.zeros([num_batch, num_res**2, 22], device=prot_t_embed.device))

        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())

        sidechain_frame_embed = self.node_to_sidechain_frame(node_embed[..., None, :]).tile((1, 1, 5, 1))
        sidechain_frame_embed = sidechain_frame_embed + self.pos_embedder(
            torch.arange(5, device=sidechain_frame_embed.device)
        )[None, None]

        return node_embed, edge_embed, sidechain_frame_embed


class IpaScore(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=64,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 use_traj_predictions=False,
                 force_flash_transformer=False,
                 block_q=32,
                 block_k=128,
                 propagate_framepair_embed=False,
                 use_knn=False,
                 k=30,
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.use_traj_predictions = use_traj_predictions
        self.force_flash_transformer = force_flash_transformer
        self.propagate_framepair_embed = propagate_framepair_embed
        self.use_knn = use_knn
        self.k = k

        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        self.num_blocks = num_blocks
        self.block_q = block_q
        self.block_k = block_k

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
            if self.use_knn:
                self.trunk[f'knn_ipa_{b}'] = KnnInvariantPointAttention(
                    c_s=c_s,
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
                    norm_first=True,
                )
                self.trunk[f'tfmr_{b}'] = torch.nn.TransformerEncoder(
                    tfmr_layer, 2)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                c_s, c_s, init="final", bias=False)
            self.trunk[f'transition_{b}'] = Transition(
                c=c_s,
            )

            self.trunk[f'rigids_tfmr_{b}'] = SequenceFrameTransformerUpdate(
                c_s=c_s,
                c_z=c_z,
                c_frame=c_frame,
                c_framepair=c_framepair,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                block_q=block_q,
                block_k=block_k,
                n_blocks=1
            )
            self.trunk[f'rigids_update_{b}'] = BackboneUpdate(c_frame)

            if use_traj_predictions:
                self.trunk[f'seq_pred_{b}'] = SeqPredictor(c_frame)
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

    @torch.no_grad()
    def _construct_knn_graph(self, rigids):
        trans = rigids.get_trans()
        bb_trans = trans[..., 0, :]
        dist_mat = torch.cdist(bb_trans, bb_trans)
        _, res_edge_index = torch.topk(dist_mat, k=self.k, dim=-1, largest=False)
        return res_edge_index

    def forward(self, init_node_embed, edge_embed, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = input_feats['rigids_t'].type(torch.float32)
        rigids_embed = input_feats['rigids_embed']
        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))

        n_batch = curr_rigids.shape[0]
        n_rigids = curr_rigids.shape[1] * curr_rigids.shape[2]
        n_padding = (self.block_q - n_rigids % self.block_q) % self.block_q
        n_attn_blocks = (n_rigids + n_padding) // self.block_q

        indexing_matrix = get_indexing_matrix(
            n_attn_blocks,
            W=self.block_q,
            H=self.block_k,
            device=rigids_embed.device
        )
        to_queries = lambda x: x.view(n_batch, n_attn_blocks, self.block_q, -1)
        to_keys = fn.partial(
            single_to_keys, indexing_matrix=indexing_matrix, W=self.block_q, H=self.block_k
        )

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]

        traj_data = {
            b: {}
            for b in range(self.num_blocks)
        }

        framepair_embed = None
        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                s=node_embed,
                z=edge_embed,
                r=curr_rigids[..., 0],
                mask=node_mask)
            node_embed = (node_embed + ipa_embed) * node_mask[..., None]

            if self.use_knn:
                edge_index = self._construct_knn_graph(curr_rigids)
                knn_edge_embed = torch.gather(
                    edge_embed,
                    2,
                    edge_index[..., None].tile(1, 1, 1, edge_embed.shape[-1])
                )
                knn_ipa_embed = self.trunk[f'knn_ipa_{b}'](
                    s=node_embed,
                    edge_index=edge_index,
                    z=knn_edge_embed,
                    r=curr_rigids[..., 0],
                    s_mask=node_mask)
                node_embed = (node_embed + knn_ipa_embed) * node_mask[..., None]

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

            rigids_embed, node_embed, framepair_embed = self.trunk[f'rigids_tfmr_{b}'](
                node_embed,
                curr_rigids,
                rigids_embed,
                to_queries,
                to_keys,
                framepair_embed=framepair_embed
            )
            if not self.propagate_framepair_embed:
                framepair_embed = None

            rigid_update = self.trunk[f'rigids_update_{b}'](
                rigids_embed * diffuse_mask[..., None, None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update * diffuse_mask[..., None, None])

            if b < self.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, curr_rigids[..., 0], edge_mask)
                edge_embed *= edge_mask[..., None]

            if self.use_traj_predictions:
                traj_data[b]['seq_logits'] = self.trunk[f"seq_pred_{b}"](rigids_embed)
                traj_data[b]['dist_logits'] = self.trunk[f"dist_pred_{b}"](edge_embed)
                traj_data[b]['rigids'] = self.unscale_rigids(curr_rigids)

        curr_rigids = self.unscale_rigids(curr_rigids)
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'node_embed': node_embed,
            'traj_data': traj_data
        }
        return model_out


class IpaMultiRigidDenoiser(nn.Module):

    def __init__(self,
                 # diffuser,
                 c_s=256,
                 c_z=128,
                 c_frame=64,
                 c_framepair=16,
                 c_hidden=16,
                 num_heads=16,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=8,
                 use_init_dgram=False,
                 use_traj_predictions=False,
                 force_flash_transformer=False,
                 trans_preconditioning=False,
                 block_q=16,
                 block_k=64,
                 propagate_framepair_embed=False,
                 use_knn=False,
                 ):
        super().__init__()
        # some compatibility code
        self.self_conditioning = True
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000

        self.ipa_score = IpaScore(
            c_s=c_s,
            c_z=c_z,
            c_frame=c_frame,
            c_framepair=c_framepair,
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            num_blocks=num_blocks,
            use_traj_predictions=use_traj_predictions,
            force_flash_transformer=force_flash_transformer,
            coordinate_scaling=1 if trans_preconditioning else 0.1,
            block_q=block_q,
            block_k=block_k,
            propagate_framepair_embed=propagate_framepair_embed,
            use_knn=use_knn,
        )
        self.embedder = Embedder(
            c_s=c_s,
            c_z=c_z,
            c_frame=c_frame,
            use_init_distogram=use_init_dgram,
        )
        self.c_s = c_s
        self.trans_preconditioning = trans_preconditioning

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
            sc_rigids = self_condition['final_rigids'].view(batch_size, -1, 5)
        else:
            sc_rigids = None

        # center the training example at the mean of the x_cas
        rigids_t = Rigid.from_tensor_7(res_data['rigids_t'])
        rigids_t = rigids_t.view([t.shape[0], -1, rigids_t.shape[-1]])
        center = rigids_t.get_trans().mean(dim=(-2, -3))
        rigids_t = rigids_t.translate(-center[..., None, None, :])

        if self.trans_preconditioning:
            rigids_in = rigids_t.apply_trans_fn(lambda x: x * data['trans_c_in'][..., None, None, None])
        else:
            rigids_in = rigids_t

        node_embed, edge_embed, rigids_embed = self.embedder(
            seq_idx=seq_idx,
            t=t,
            fixed_mask=fixed_mask,
            sc_rigids=sc_rigids,
        )


        input_feats = {
            'fixed_mask': fixed_mask,
            'res_mask': res_mask.view(batch_size, -1),
            'rigids_t': rigids_in.to_tensor_7(),
            't': t,
            "rigids_embed": rigids_embed,
        }

        score_dict = self.ipa_score(node_embed, edge_embed, input_feats)
        traj_data = score_dict['traj_data']
        rigids_out = score_dict['final_rigids']
        if self.trans_preconditioning:
            rigids_out = rigids_out.apply_trans_fn(
                lambda x: x * data['trans_c_out'][..., None, None, None] + rigids_t.get_trans() * data['trans_c_skip'][..., None, None, None]
            )

        rigids_out = rigids_out.translate(center[..., None, None, :])
        seq_logits = traj_data[max(traj_data.keys())]['seq_logits']
        denoised_atom14_gt_seq = compute_atom14_from_frame5(rigids_out, res_mask, res_data['seq'].view(batch_size, -1))
        denoised_atom14_pred_seq = compute_atom14_from_frame5(rigids_out, res_mask, seq_logits.argmax(dim=-1))
        psi = score_dict['psi'].view(-1, 2)

        ret = {}
        ret['denoised_frames'] = rigids_out.view(-1, rigids_out.shape[-1])
        ret['final_rigids'] = rigids_out.view(-1, rigids_out.shape[-1])
        ret['denoised_atom14'] = denoised_atom14_pred_seq.flatten(0, 1)
        ret['denoised_atom14_gt_seq'] = denoised_atom14_gt_seq.flatten(0, 1)
        ret['denoised_bb'] = compute_backbone(rigids_out[..., 0], score_dict['psi'], impute_O=True)[-1][..., :5, :].flatten(0, 1)
        ret['psi'] = psi
        ret['traj_data'] = traj_data
        ret['decoded_seq_logits'] = seq_logits.flatten(0, 1)

        return ret