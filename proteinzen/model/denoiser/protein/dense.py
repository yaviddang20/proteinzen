import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy

import functools as fn

from proteinzen.model.modules.layers.node.conv import SequenceUpscaler, SequenceDownscaler
from proteinzen.model.modules.layers.node.attention import FlashTransformerEncoder
from proteinzen.model.modules.layers.edge.conv import EdgeUpscaler, EdgeDownscaler
from proteinzen.data.datasets.featurize.common import _rbf
from proteinzen.model.modules.openfold.layers import Linear, InvariantPointAttention, StructureModuleTransition, BackboneUpdate
from proteinzen.model.modules.openfold.layers import LocalTriangleAttentionNew
import proteinzen.utils.openfold.rigid_utils as ru
from proteinzen.utils.openfold.rigid_utils import Rigid, batchwise_center
from proteinzen.utils.framediff.all_atom import compute_backbone

from proteinzen.model.encoder.chimera import ProteinAtomicChimeraEmbedder

from ._attn import PairUpdate, PairEmbedder

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


class EdgeTransition(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

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

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        edge_embed = torch.cat(
            [edge_embed, edge_bias], axis=-1).reshape(
                batch_size * num_res**2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(
            batch_size, num_res, num_res, -1
        )
        return edge_embed


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
    def __init__(self, c_s, n_aa=21):
        super().__init__()
        self.proj = nn.Sequential(
            Linear(c_s, c_s),
            nn.LayerNorm(c_s),
            Linear(c_s, n_aa)
        )

    def forward(self, node_features):
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



class Embedder(nn.Module):

    def __init__(self,
                 c_s,
                 c_latent,
                 c_z,
                 index_embed_size=32,
                 use_init_distogram=False,
                 conv_downsample_factor=0,
                 use_pair_embedder=False,
                 use_seq_mask_features=False,
    ):
        super(Embedder, self).__init__()

        c_latent_init = c_latent
        c_latent = c_latent * (2 ** conv_downsample_factor)
        self.conv_downsample_factor = conv_downsample_factor
        self.use_pair_embedder = use_pair_embedder
        self.use_seq_mask_features = use_seq_mask_features
        if self.conv_downsample_factor > 0:
            self.upscale = SequenceUpscaler(
                conv_downsample_factor,
                c_in=c_latent_init,
                c_out=c_latent
            )

        # Time step embedding
        t_embed_size = index_embed_size
        node_embed_dims = t_embed_size + 1 + c_latent

        # Sequence index embedding
        node_embed_dims += index_embed_size
        if self.use_seq_mask_features:
            node_embed_dims += 1

        node_embed_size = c_s
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        if self.use_pair_embedder:
            self.pair_embedder = PairEmbedder(
                c_z=c_z,
                c_latent=c_latent,
                c_hidden=c_z//2,
                latent_pairs=False
            )

        else:
            edge_in = (t_embed_size + c_latent + 1) * 2
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
            latent_features,
            seq_idx,
            t,
            fixed_mask,
            node_mask,
            seq_mask,
            self_conditioning_ca,
            init_ca=None,
            rigids=None,
            sc_rigids=None
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

        if self.conv_downsample_factor > 0:
            latent_features = self.upscale(latent_features, seq_len=seq_idx.shape[1])

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        prot_t_embed = torch.cat([
            prot_t_embed,
            fixed_mask,
            latent_features,
        ], dim=-1)
        node_feats = [prot_t_embed]
        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        if self.use_seq_mask_features:
            node_feats.append(seq_mask[..., None].float())
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])

        if self.use_pair_embedder:
            assert rigids is not None
            edge_embed = self.pair_embedder(
                latent_features=latent_features,
                rigids=rigids,
                node_mask=node_mask,
                sc_rigids=sc_rigids
            )
        else:
            pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]
            pair_feats.append(self.index_embedder(rel_seq_offset))

            sc_dgram = calc_distogram(
                self_conditioning_ca,
                1e-5,
                20,
                22
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))
            if self.use_init_distogram:
                init_dgram = calc_distogram(
                    init_ca,
                    1e-5,
                    20,
                    22
                )
                pair_feats.append(init_dgram.reshape([num_batch, num_res**2, -1]))
            edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
            edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        return node_embed, edge_embed


class IpaScore(nn.Module):

    def __init__(self,
                 #diffuser,
                 c_s=256,
                 c_latent=128,
                 c_z=128,
                 c_skip=64,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 update_edges_with_dist=False,
                 use_proteus_edge_transition=False,
                 use_pair_update=False,
                 conv_downsample_factor=0,
                 use_traj_predictions=False,
                 force_flash_transformer=False
                 ):
        super(IpaScore, self).__init__()
        # self.diffuser = diffuser
        self.update_edges_with_dist = update_edges_with_dist
        self.use_proteus_edge_transition = use_proteus_edge_transition
        self.use_pair_update = use_pair_update
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
            )
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(c_s)
            self.trunk[f'skip_embed_{b}'] = Linear(
                c_s,
                c_skip,
                init="final"
            )
            tfmr_in = c_s + c_skip
            if force_flash_transformer:
                self.trunk[f'seq_tfmr_{b}'] = FlashTransformerEncoder(
                    h_dim=tfmr_in,
                    n_layers=2,
                    no_heads=4,
                    h_ff=tfmr_in,
                    dropout=0.0,
                    ln_first=False,
                    dtype=None
                )
            else:
                tfmr_layer = torch.nn.TransformerEncoderLayer(
                    d_model=tfmr_in,
                    nhead=4,
                    dim_feedforward=tfmr_in,
                    batch_first=True,
                    dropout=0.0,
                    norm_first=False
                )
                self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                    tfmr_layer, 2)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                tfmr_in, c_s, init="final")
            self.trunk[f'node_transition_{b}'] = StructureModuleTransition(
                c=c_s)
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(c_s)

            if use_traj_predictions:
                self.trunk[f'seq_pred_{b}'] = SeqPredictor(c_s)
                self.trunk[f'dist_pred_{b}'] = EdgeDistPredictor(c_z)

            if b < num_blocks-1:
                # No edge update on the last block.
                if update_edges_with_dist:
                    edge_in = c_z + 22
                else:
                    edge_in = c_z
                if self.use_proteus_edge_transition:
                    self.trunk[f'edge_transition_{b}'] = LocalTriangleAttentionNew(
                        c_s=c_s,
                        c_z=c_z,
                        c_rbf=64,
                        c_gate_s=16,
                        c_hidden=128,
                        c_hidden_mul=128,
                        no_heads=4,
                        transition_n=2,
                        k_neighbour=32,
                        k_linear=0,
                        inf=1e9,
                        pair_dropout=0.25
                    )
                elif self.use_pair_update:
                    self.trunk[f'edge_transition_{b}'] = PairUpdate(
                        c_z=c_z,
                        c_hidden=c_z//4,
                        no_heads=4
                    )

                else:
                    self.trunk[f'edge_transition_{b}'] = EdgeTransition(
                        node_embed_size=c_s,
                        edge_embed_in=edge_in,
                        edge_embed_out=c_z,
                    )

        self.torsion_pred = TorsionAngles(c_s, 1)

        self.conv_downsample_factor = conv_downsample_factor
        if self.conv_downsample_factor > 0:
            c_in = c_latent * (2 ** conv_downsample_factor)
            self.project = Linear(c_s, c_in, bias=False)
            self.downsample = SequenceDownscaler(
                conv_downsample_factor,
                c_in,
                c_latent
            )
            self.final = Linear(c_latent, c_latent)#, init='final')
            self.final_gate = Linear(c_latent, c_latent, init='gating')
        else:
            self.latent_delta = nn.Sequential(
                Linear(c_s+c_latent, c_s, init='relu'),
                nn.ReLU(),
                Linear(c_s, c_s, init='relu'),
                nn.ReLU(),
                Linear(c_s, c_s, init='relu'),
                nn.LayerNorm(c_s),
                Linear(c_s, c_latent, init='final')
            )

    def forward(self, init_node_embed, edge_embed, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = input_feats['rigids_t'].type(torch.float32)
        latent_features = input_feats['noised_latent_features']

        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]

        traj_data = {
            b: {}
            for b in range(self.num_blocks)
        }

        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_in = torch.cat([
                node_embed, self.trunk[f'skip_embed_{b}'](init_node_embed)
            ], dim=-1)
            if self.force_flash_transformer:
                seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                    seq_tfmr_in, node_mask)
            else:
                seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                    seq_tfmr_in, src_key_padding_mask=1 - node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * diffuse_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update * diffuse_mask[..., None])

            if b < self.num_blocks-1:
                if self.update_edges_with_dist:
                    pos = curr_rigids.get_trans()
                    dists_2d = torch.linalg.norm(
                        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)
                    curr_dgram = _rbf(dists_2d, 1e-5, 20, D_count=22, device=dists_2d.device)
                    edge_embed_in = torch.cat(
                        [edge_embed, curr_dgram],
                        dim=-1
                    )
                else:
                    edge_embed_in = edge_embed

                if self.use_proteus_edge_transition:
                    edge_embed = self.trunk[f'edge_transition_{b}'](
                        node_embed, edge_embed, curr_rigids, edge_mask)
                elif self.use_pair_update:
                    edge_embed = self.trunk[f'edge_transition_{b}'](
                        edge_embed, curr_rigids, edge_mask)
                else:
                    edge_embed = self.trunk[f'edge_transition_{b}'](
                        node_embed, edge_embed_in)
                edge_embed *= edge_mask[..., None]

            if self.use_traj_predictions:
                traj_data[b]['seq_logits'] = self.trunk[f"seq_pred_{b}"](node_embed)
                traj_data[b]['dist_logits'] = self.trunk[f"dist_pred_{b}"](edge_embed)
                traj_data[b]['rigids'] = self.unscale_rigids(curr_rigids)

        if self.conv_downsample_factor > 0:
            latent_update = self.downsample(self.project(node_embed))
            gate = self.final_gate(latent_update)
            latent_features = latent_features + self.final(latent_update) * torch.sigmoid(gate)
        else:
            latent_features = latent_features + self.latent_delta(
                torch.cat([node_embed, latent_features], dim=-1)
            ) * node_mask[..., None]

        curr_rigids = self.unscale_rigids(curr_rigids)
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'final_latent': latent_features,
            'node_embed': node_embed,
            'traj_data': traj_data
        }
        return model_out


class IpaDenoiser(nn.Module):
    def __init__(self,
                 # diffuser,
                 c_s=256,
                 c_latent=128,
                 c_z=128,
                 c_skip=64,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 use_init_dgram=False,
                 update_edges_with_dgram=False,
                 use_proteus_transition=False,
                 use_pair_embedder=False,
                 use_pair_update=False,
                 use_seq_mask_features=False,
                 conv_downsample_factor=0,
                 use_traj_predictions=False,
                 force_flash_transformer=False,
                 sc_atomic_embedder=False
                 ):
        super().__init__()
        # some compatibility code
        self.self_conditioning = True
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000
        self.c_latent = c_latent
        self.use_pair_embedder = use_pair_embedder

        self.ipa_score = IpaScore(
            # diffuser,
            c_s=c_s,
            c_latent=c_latent,
            c_z=c_z,
            c_skip=c_skip,
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            num_blocks=num_blocks,
            update_edges_with_dist=update_edges_with_dgram,
            use_proteus_edge_transition=use_proteus_transition,
            use_pair_update=use_pair_update,
            conv_downsample_factor=conv_downsample_factor,
            use_traj_predictions=use_traj_predictions,
            force_flash_transformer=force_flash_transformer
        )
        self.embedder = Embedder(
            c_s=c_s,
            c_latent=c_latent,
            c_z=c_z,
            use_init_distogram=use_init_dgram,
            conv_downsample_factor=conv_downsample_factor,
            use_pair_embedder=use_pair_embedder,
            use_seq_mask_features=use_seq_mask_features
        )
        self.c_s = c_s
        if sc_atomic_embedder:
            self.sc_atomic_embedder = ProteinAtomicChimeraEmbedder(
                h_frame=c_s,
                use_masking_features=False,
                n_layers=4,
                use_ffn=True,
                compat_mode=False,
                atom_max_neighbors=128,
                atomic_r=8
            )
        else:
            self.sc_atomic_embedder = None

    def forward(self, data, intermediates, self_condition=None):
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
            self_conditioning_ca = self_condition['final_rigids'].get_trans().view(batch_size, -1, 3)
        else:
            self_conditioning_ca = torch.zeros_like(rigids_t.get_trans().view(batch_size, -1, 3))

        # center the training example at the mean of the x_cas
        rigids_t = Rigid.from_tensor_7(res_data['rigids_t'])
        center = batchwise_center(rigids_t, res_data.batch, res_mask)
        rigids_t = rigids_t.translate(-center)
        rigids_t = rigids_t.view([t.shape[0], -1])
        latent_sidechain_t = intermediates['noised_latent_sidechain'].view(batch_size, -1, self.c_latent)

        if self.use_pair_embedder:
            node_embed, edge_embed = self.embedder(
                latent_features=latent_sidechain_t,
                seq_idx=seq_idx,
                t=t,
                fixed_mask=fixed_mask,
                node_mask=res_mask.view(batch_size, -1),
                seq_mask=res_data['seq_noising_mask'].view(batch_size, -1),
                self_conditioning_ca=self_conditioning_ca,
                init_ca=rigids_t.get_trans(),
                rigids=rigids_t,
                sc_rigids=(
                    self_condition['final_rigids'].view(batch_size, -1) if self_condition is not None
                    else None
                )
            )
        else:
            node_embed, edge_embed = self.embedder(
                latent_features=latent_sidechain_t,
                seq_idx=seq_idx,
                t=t,
                node_mask=res_mask,
                seq_mask=res_data['seq_noising_mask'].view(batch_size, -1),
                fixed_mask=fixed_mask,
                self_conditioning_ca=self_conditioning_ca,
                init_ca=rigids_t.get_trans()
            )

        if self.sc_atomic_embedder is not None:
            if self_condition is not None:
                sc_data = copy.copy(data)
                sc_res_data = sc_data['res_data']
                sc_res_data['atom14_gt_positions'] = self_condition['decoded_atom14']
                sc_res_data['atom14_gt_exists'] = self_condition['decoded_atom14_mask']
                sc_res_data['atom14_noising_mask'] = self_condition['decoded_atom14_mask']
                sc_res_data['seq'] = self_condition['decoded_seq_logits'].argmax(dim=-1)
                sc_res_data['rigids_1'] = self_condition['final_rigids'].to_tensor_7()
                sc_res_data['x'] = self_condition['final_rigids'].get_trans()
                sc_res_data['bb'] = self_condition['denoised_bb'][:, :4]
                sc_node_embed = self.sc_atomic_embedder(sc_data)['latent_mu']
                node_embed = node_embed + sc_node_embed.view(batch_size, -1, self.c_s)


        input_feats = {
            'fixed_mask': fixed_mask,
            'res_mask': res_mask.view(batch_size, -1),
            'rigids_t': rigids_t.to_tensor_7(),
            't': t,
            'noised_latent_features': latent_sidechain_t
        }

        score_dict = self.ipa_score(node_embed, edge_embed, input_feats)
        rigids = score_dict['final_rigids'].view(-1)
        rigids = rigids.translate(center)

        psi = score_dict['psi'].view(-1, 2)

        ret = {}
        ret['denoised_frames'] = rigids
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)[:, :5]
        ret['denoised_bb'] = denoised_bb
        # ret['node_features'] = score_dict['node_embed']
        ret['psi'] = psi
        ret['pred_latent_sidechain'] = score_dict['final_latent'].view(-1, self.c_latent)
        ret['traj_data'] = score_dict['traj_data']

        return ret


class DenseEmbedder(nn.Module):

    def __init__(self,
                 c_s,
                 c_z,
                 c_s_latent,
                 c_z_latent,
                 index_embed_size=32,
                 use_init_distogram=False,
                 use_seq_mask_features=False,
                 use_pair_embedder=False,
                 conv_downsample_factor=0.
        ):
        super().__init__()
        self.use_seq_mask_features = use_seq_mask_features

        c_s_interim = c_s_latent * (2 ** conv_downsample_factor)
        c_z_interim = c_z_latent * (2 ** conv_downsample_factor)
        self.conv_downsample_factor = conv_downsample_factor
        self.use_pair_embedder = use_pair_embedder
        if self.conv_downsample_factor > 0:
            self.node_upscale = SequenceUpscaler(
                conv_downsample_factor,
                c_in=c_s_latent,
                c_out=c_s_interim
            )
            self.edge_upscale = EdgeUpscaler(
                conv_downsample_factor,
                c_in=c_z_latent,
                c_out=c_z_interim
            )

        # Time step embedding
        t_embed_size = index_embed_size
        node_embed_dims = t_embed_size + 1 + c_s_latent
        if use_seq_mask_features:
            node_embed_dims += 1

        # Sequence index embedding
        node_embed_dims += index_embed_size

        node_embed_size = c_s
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        if self.use_pair_embedder:
            self.pair_embedder = PairEmbedder(
                c_z=c_z,
                c_latent=c_z_latent,
                c_hidden=c_z//2,
                latent_pairs=True
            )

        else:
            edge_in = (t_embed_size + c_s_latent + 1) * 2 + c_z_latent
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
            latent_features,
            latent_edge_features,
            seq_idx,
            t,
            fixed_mask,
            self_conditioning_ca,
            seq_mask,
            init_ca=None,
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

        if self.conv_downsample_factor > 0:
            latent_features = self.node_upscale(latent_features, seq_len=seq_idx.shape[1])
            latent_edge_features = self.edge_upscale(latent_edge_features, seq_len=seq_idx.shape[1])

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        prot_t_embed = torch.cat([
            prot_t_embed,
            fixed_mask,
            latent_features,
        ], dim=-1)
        node_feats = [prot_t_embed]
        if self.use_seq_mask_features:
            node_feats.append(seq_mask[..., None])

        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])
        pair_feats.append(self.index_embedder(rel_seq_offset))

        sc_dgram = calc_distogram(
            self_conditioning_ca,
            1e-5,
            20,
            22
        )
        pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))
        pair_feats.append(latent_edge_features.view([num_batch, num_res**2, -1]))
        if self.use_init_distogram:
            init_dgram = calc_distogram(
                init_ca,
                1e-5,
                20,
                22
            )
            pair_feats.append(init_dgram.reshape([num_batch, num_res**2, -1]))


        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        return node_embed, edge_embed

class DenseIpaScore(nn.Module):

    def __init__(self,
                 #diffuser,
                 c_s=256,
                 c_s_latent=4,
                 c_z_latent=4,
                 c_z=128,
                 c_skip=64,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 update_edges_with_dist=False,
                 use_proteus_edge_transition=False,
                 use_pair_update=False,
                 conv_downsample_factor=0.
                 ):
        super().__init__()
        # self.diffuser = diffuser
        self.update_edges_with_dist = update_edges_with_dist
        self.use_proteus_edge_transition = use_proteus_edge_transition
        self.use_pair_update = use_pair_update

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
            )
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(c_s)
            self.trunk[f'skip_embed_{b}'] = Linear(
                c_s,
                c_skip,
                init="final"
            )
            tfmr_in = c_s + c_skip
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=4,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, 2)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                tfmr_in, c_s, init="final")
            self.trunk[f'node_transition_{b}'] = StructureModuleTransition(
                c=c_s)
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(c_s)

            if b < num_blocks-1:
                # No edge update on the last block.
                if update_edges_with_dist:
                    edge_in = c_z + 22
                else:
                    edge_in = c_z
                if self.use_proteus_edge_transition:
                    self.trunk[f'edge_transition_{b}'] = LocalTriangleAttentionNew(
                        c_s=c_s,
                        c_z=c_z,
                        c_rbf=64,
                        c_gate_s=16,
                        c_hidden=128,
                        c_hidden_mul=128,
                        no_heads=4,
                        transition_n=2,
                        k_neighbour=32,
                        k_linear=0,
                        inf=1e9,
                        pair_dropout=0.25
                    )
                elif self.use_pair_update:
                    self.trunk[f'edge_transition_{b}'] = PairUpdate(
                        c_z=c_z,
                        c_hidden=c_z,
                    )

                else:
                    self.trunk[f'edge_transition_{b}'] = EdgeTransition(
                        node_embed_size=c_s,
                        edge_embed_in=edge_in,
                        edge_embed_out=c_z,
                    )

        self.torsion_pred = TorsionAngles(c_s, 1)

        self.conv_downsample_factor = conv_downsample_factor
        if self.conv_downsample_factor > 0:
            c_s_in = int(c_s_latent * (2 ** conv_downsample_factor))
            self.node_project = nn.Sequential(
                nn.LayerNorm(c_s),
                Linear(c_s, c_s_in, bias=False)
            )
            self.node_downsample = SequenceDownscaler(
                conv_downsample_factor,
                c_s_in,
                c_s_latent
            )
            self.final_s = Linear(c_s_latent, c_s_latent)#, init='final')
            self.final_s_gate = Linear(c_s_latent, c_s_latent, init='gating')

            c_z_in = int(c_z_latent * (2 ** conv_downsample_factor))
            self.edge_project = nn.Sequential(
                nn.LayerNorm(c_z),
                Linear(c_z, c_z_in, bias=False)
            )
            self.edge_downsample = EdgeDownscaler(
                conv_downsample_factor,
                c_z_in,
                c_z_latent
            )
            self.final_z = Linear(c_z_latent, c_z_latent)#, init='final')
            self.final_z_gate = Linear(c_z_latent, c_z_latent, init='gating')
        else:
            self.latent_delta = nn.Sequential(
                Linear(c_s+c_s_latent, c_s, init='relu'),
                nn.ReLU(),
                Linear(c_s, c_s, init='relu'),
                nn.ReLU(),
                Linear(c_s, c_s, init='relu'),
                nn.LayerNorm(c_s),
                Linear(c_s, c_s_latent, init='final')
            )
            self.latent_edge_delta = nn.Sequential(
                Linear(c_z+c_z_latent, c_z, init='relu'),
                nn.ReLU(),
                Linear(c_z, c_z, init='relu'),
                nn.ReLU(),
                Linear(c_z, c_z, init='relu'),
                nn.LayerNorm(c_z),
                Linear(c_z, c_z_latent, init='final')
            )


    def forward(self, init_node_embed, edge_embed, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = input_feats['rigids_t'].type(torch.float32)
        latent_features = input_feats['noised_latent_features']
        latent_edge_features = input_feats['noised_latent_edges']

        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_in = torch.cat([
                node_embed, self.trunk[f'skip_embed_{b}'](init_node_embed)
            ], dim=-1)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                seq_tfmr_in, src_key_padding_mask=1 - node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * diffuse_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update * diffuse_mask[..., None])

            if b < self.num_blocks-1:
                if self.update_edges_with_dist:
                    pos = curr_rigids.get_trans()
                    dists_2d = torch.linalg.norm(
                        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)
                    curr_dgram = _rbf(dists_2d, 1e-5, 20, D_count=22, device=dists_2d.device)
                    edge_embed_in = torch.cat(
                        [edge_embed, curr_dgram],
                        dim=-1
                    )
                else:
                    edge_embed_in = edge_embed

                if self.use_proteus_edge_transition:
                    edge_embed = self.trunk[f'edge_transition_{b}'](
                        node_embed, edge_embed, curr_rigids, edge_mask)
                else:
                    edge_embed = self.trunk[f'edge_transition_{b}'](
                        node_embed, edge_embed_in)
                edge_embed *= edge_mask[..., None]

        if self.conv_downsample_factor > 0:
            latent_update = self.node_downsample(self.node_project(node_embed))
            s_gate = self.final_s_gate(latent_update)
            latent_features = latent_features + self.final_s(latent_update) * torch.sigmoid(s_gate)

            latent_edge_update = self.edge_downsample(self.edge_project(edge_embed))
            z_gate = self.final_z_gate(latent_edge_update)
            latent_edge_features = latent_edge_features + self.final_z(latent_edge_update) * torch.sigmoid(z_gate)
        else:
            latent_features = latent_features + self.latent_delta(
                torch.cat([latent_features, node_embed], dim=-1)
            )
            latent_edge_features = latent_edge_features + self.latent_edge_delta(
                torch.cat([latent_edge_features, edge_embed], dim=-1)
            )

        curr_rigids = self.unscale_rigids(curr_rigids)
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'final_latent': latent_features,
            'final_latent_edge': latent_edge_features,
            'node_embed': node_embed
        }
        return model_out

class DenseIpaDenoiser(nn.Module):
    def __init__(self,
                 # diffuser,
                 c_s=256,
                 c_s_latent=4,
                 c_z_latent=4,
                 c_z=128,
                 c_skip=64,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 use_init_dgram=False,
                 update_edges_with_dgram=False,
                 use_proteus_transition=False,
                 use_seq_mask_features=False,
                 conv_downsample_factor=0,
                 ):
        super().__init__()
        # some compatibility code
        self.self_conditioning = True
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000
        self.c_s_latent = c_s_latent
        self.c_z_latent = c_z_latent

        self.ipa_score = DenseIpaScore(
            # diffuser,
            c_s=c_s,
            c_s_latent=c_s_latent,
            c_z_latent=c_z_latent,
            c_z=c_z,
            c_skip=c_skip,
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            num_blocks=num_blocks,
            update_edges_with_dist=update_edges_with_dgram,
            use_proteus_edge_transition=use_proteus_transition,
            conv_downsample_factor=conv_downsample_factor
        )
        self.embedder = DenseEmbedder(
            c_s=c_s,
            c_s_latent=c_s_latent,
            c_z_latent=c_z_latent,
            c_z=c_z,
            use_init_distogram=use_init_dgram,
            use_seq_mask_features=use_seq_mask_features,
            conv_downsample_factor=conv_downsample_factor
        )

    def forward(self, data, intermediates, self_condition=None):
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
        seq_mask = res_data['seq_noising_mask'].float().view(batch_size, -1)

        if self_condition is not None:
            self_conditioning_ca = self_condition['final_rigids'].get_trans().view(batch_size, -1, 3)
        else:
            self_conditioning_ca = torch.zeros_like(rigids_t.get_trans().view(batch_size, -1, 3))

        # center the training example at the mean of the x_cas
        rigids_t = Rigid.from_tensor_7(res_data['rigids_t'])
        center = batchwise_center(rigids_t, res_data.batch, res_mask)
        rigids_t = rigids_t.translate(-center)
        rigids_t = rigids_t.view([t.shape[0], -1])
        latent_sidechain_t = intermediates['noised_latent_sidechain'].view(batch_size, -1, self.c_latent)
        latent_edge_t = intermediates['noised_latent_edge']

        node_embed, edge_embed = self.embedder(
            latent_features=latent_sidechain_t,
            latent_edge_features=latent_edge_t,
            seq_idx=seq_idx,
            t=t,
            fixed_mask=fixed_mask,
            self_conditioning_ca=self_conditioning_ca,
            init_ca=rigids_t.get_trans(),
            seq_mask=seq_mask
        )

        input_feats = {
            'fixed_mask': fixed_mask,
            'res_mask': res_mask.view(batch_size, -1),
            'rigids_t': rigids_t.to_tensor_7(),
            't': t,
            'noised_latent_features': latent_sidechain_t,
            'noised_latent_edges': latent_edge_t
        }

        score_dict = self.ipa_score(node_embed, edge_embed, input_feats)
        rigids = score_dict['final_rigids'].view(-1)
        rigids = rigids.translate(center)

        psi = score_dict['psi'].view(-1, 2)

        ret = {}
        ret['denoised_frames'] = rigids
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)[:, :5]
        ret['denoised_bb'] = denoised_bb
        # ret['node_features'] = score_dict['node_embed']
        ret['psi'] = psi
        ret['pred_latent_sidechain'] = score_dict['final_latent']
        ret['pred_latent_edge'] = score_dict['final_latent_edge']

        return ret