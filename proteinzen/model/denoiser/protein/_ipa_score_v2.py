import torch
import torch.nn as nn
import torch.nn.functional as F

from proteinzen.model.modules.layers.node.attention import FlashTransformerEncoder
from proteinzen.data.datasets.featurize.common import _rbf
from proteinzen.model.modules.openfold.layers_v2 import Linear, ConditionedInvariantPointAttention, ConditionedTransition, BackboneUpdate, TorsionAngles
from proteinzen.utils.openfold.rigid_utils import Rigid

from ._attn import ConditionedPairUpdate


class SeqPredictor(nn.Module):
    def __init__(self, c_s, n_aa=21):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(c_s),
            Linear(c_s, n_aa)
        )

    def forward(self, node_features):
        return self.proj(node_features)


class EdgeDistPredictor(nn.Module):
    def __init__(self, c_z, n_bins=64):
        super().__init__()
        self.out = nn.Sequential(
            nn.LayerNorm(c_z),
            Linear(c_z, n_bins),
        )

    def forward(self, z):
        z = z + z.transpose(-2, -3)
        return self.out(z)

class LatentUpdate(nn.Module):
    def __init__(self, c_s, c_latent):
        super().__init__()
        self.out = nn.Sequential(
            nn.LayerNorm(c_s),
            Linear(c_s, c_latent, bias=False, init='final')
        )

    def forward(self, s):
        return self.out(s)


class IpaScoreV2(nn.Module):

    def __init__(self,
                 #diffuser,
                 c_s=256,
                 c_latent=128,
                 c_z=128,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_points=8,
                 num_v_points=12,
                 num_blocks=4,
                 coordinate_scaling=0.1,
                 use_traj_predictions=False,
                 force_flash_transformer=False
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
            self.trunk[f'ipa_{b}'] = ConditionedInvariantPointAttention(
                 c_s=c_s,
                 c_cond=c_latent,
                 c_z=c_z,
                 c_hidden=c_hidden,
                 num_heads=num_heads,
                 num_qk_points=num_qk_points,
                 num_v_points=num_v_points,
            )
            if force_flash_transformer:
                self.trunk[f'seq_tfmr_{b}'] = FlashTransformerEncoder(
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
                self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                    tfmr_layer, 2)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                c_s, c_s, init="final", bias=False)
            self.trunk[f'node_transition_{b}'] = ConditionedTransition(
                c_s=c_s,
                c_cond=c_latent
            )
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(c_s)
            self.trunk[f'latent_update_{b}'] = LatentUpdate(c_s, c_latent)

            if use_traj_predictions:
                self.trunk[f'seq_pred_{b}'] = SeqPredictor(c_s)
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
        latent_features = input_feats['noised_latent_features']

        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)
        curr_latent = latent_features
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]

        traj_data = {
            b: {}
            for b in range(self.num_blocks)
        }

        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                s=node_embed,
                cond=curr_latent,
                z=edge_embed,
                r=curr_rigids,
                mask=node_mask)
            node_embed = (node_embed + ipa_embed) * node_mask[..., None]
            if self.force_flash_transformer:
                seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                    node_embed, node_mask)
            else:
                seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                    node_embed, src_key_padding_mask=1 - node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = node_embed * node_mask[..., None]
            node_embed = self.trunk[f'node_transition_{b}'](node_embed, latent_features)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * diffuse_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update * diffuse_mask[..., None])
            curr_latent = curr_latent + self.trunk[f'latent_update_{b}'](node_embed) * diffuse_mask[..., None]

            if b < self.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed, curr_rigids, edge_mask)
                edge_embed *= edge_mask[..., None]

            if self.use_traj_predictions:
                traj_data[b]['seq_logits'] = self.trunk[f"seq_pred_{b}"](node_embed)
                traj_data[b]['dist_logits'] = self.trunk[f"dist_pred_{b}"](edge_embed)
                traj_data[b]['rigids'] = self.unscale_rigids(curr_rigids)
                traj_data[b]['latent'] = curr_latent

        curr_rigids = self.unscale_rigids(curr_rigids)
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'final_rigids': curr_rigids,
            'final_latent': curr_latent,
            'node_embed': node_embed,
            'traj_data': traj_data
        }
        return model_out