import math

import torch
from torch import nn

from proteinzen.model.design.ipmp import IPMPDecoder as DesignIPMPDecoder
from proteinzen.model.design.ipa import IPADecoder, NodeIPADecoder
from proteinzen.model.design.mlp import MLPDecoder
from proteinzen.model.encoder.ipmp import IPMPEncoder as MLMIPMPEncoder
from proteinzen.model.encoder.tfn import ProteinAtomicEmbedder
from proteinzen.model.encoder.chimera import ProteinAtomicChimeraEmbedder
from proteinzen.model.encoder.chimera_dense import ProteinAtomicChimeraDenseEmbedder
from proteinzen.model.denoiser.protein.frames import DynamicGraphIpaFrameDenoiser
from proteinzen.model.denoiser.protein.dense import IpaDenoiser, DenseIpaDenoiser
from proteinzen.model.discriminator.chimera import ProteinAtomicChimeraDiscriminator
from proteinzen.model.discriminator.ipmp import IPMPDiscriminator

from ._supervisor import LatentNodeSupervisor


class IPMPLatentWrapper(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_latent=128,
                 c_z=128,
                 c_hidden=256,
                 c_s_in=6,
                 num_rbf=16,
                 num_layers=4,
                 num_heads=8,
                 num_qk_pts=8,
                 num_v_pts=12,
                 h_time=64,
                 sidechain_k=30,
                 knn_k=20,
                 lrange_k=40,
                 self_conditioning=True,
                 mlp_decoder=False
                 ):
        super().__init__()
        self.encoder = MLMIPMPEncoder(
            c_s=c_s//2,
            c_z=c_z,
            c_hidden=c_s//2,
            c_s_out=c_latent,
            c_s_in=c_s_in,
            num_rbf=num_rbf,
            num_layers=num_layers * (2 if mlp_decoder else 1),
            k=sidechain_k,
        )
        if mlp_decoder:
            self.decoder = MLPDecoder(
                c_s=c_s//2,
                c_latent=c_latent,
            )
        else:
            self.decoder = DesignIPMPDecoder(
                c_s=c_s//2,
                c_z=c_z,
                c_latent=c_latent,
                c_hidden=c_s//2,
                c_s_in=c_s_in,
                num_rbf=num_rbf,
                num_layers=num_layers,
                k=sidechain_k,
            )
        self.denoiser = DynamicGraphIpaFrameDenoiser(
            c_s=c_s,
            c_latent=c_latent,
            c_z=c_z,
            c_hidden=16,
            num_heads=16,
            num_qk_pts=num_qk_pts,
            num_v_pts=num_v_pts,
            n_layers=8,
            h_time=h_time,
            knn_k=knn_k,
            lrange_k=lrange_k,
            self_conditioning=self_conditioning,
        )
        self.c_latent = c_latent
        self.self_conditioning = self_conditioning

    def sample_prior(self, num_nodes, device):
        return {
            "noised_latent_sidechain": torch.randn((num_nodes, self.c_latent), device=device)
        }


class TFNLatentWrapper(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_latent=128,
                 c_z=128,
                 c_hidden=256,
                 c_s_in=6,
                 num_rbf=16,
                 num_layers=4,
                 denoiser_num_layers=8,
                 num_heads=8,
                 num_qk_pts=8,
                 num_v_pts=12,
                 h_time=64,
                 sidechain_k=30,
                 knn_k=20,
                 lrange_k=40,
                 res_h_mult_factor=2,
                 res_edge_mult_factor=1,
                 encoder_res_edge_updates=False,
                 self_conditioning=True,
                 broadcast_to_atoms=False,
                 lrange_embedding=False,
                 smooth_embedding=False,
                 mlp_decoder=False,
                 use_masking_features=False,
                 denoiser_masked_latent_feature=False,
                 compatibility_mode=False
                 ):
        super().__init__()
        self.encoder = ProteinAtomicEmbedder(
            h_frame=c_latent,
            res_num_rbf=num_rbf,
            knn_k=sidechain_k,
            broadcast_to_atoms=broadcast_to_atoms,
            lrange_graph=lrange_embedding,
            smooth_nodewise=smooth_embedding,
            use_masking_features=use_masking_features,
            res_h_mult_factor=res_h_mult_factor,
            res_edge_mult_factor=res_edge_mult_factor,
            res_edge_update=encoder_res_edge_updates,
            compatibility_mode=compatibility_mode
        )
        if mlp_decoder:
            self.decoder = MLPDecoder(
                c_s=c_s//2,
                c_latent=c_latent,
            )
        else:
            self.decoder = DesignIPMPDecoder(
                c_s=c_s//2,
                c_z=c_z,
                c_latent=c_latent,
                c_hidden=c_s//2,
                c_s_in=c_s_in,
                num_rbf=num_rbf,
                num_layers=num_layers,
                k=sidechain_k,
            )
        self.denoiser = DynamicGraphIpaFrameDenoiser(
            c_s=c_s,
            c_latent=c_latent,
            c_z=c_z,
            c_hidden=16,
            num_heads=16,
            num_qk_pts=num_qk_pts,
            num_v_pts=num_v_pts,
            n_layers=denoiser_num_layers,
            h_time=h_time,
            knn_k=knn_k,
            lrange_k=lrange_k,
            self_conditioning=self_conditioning,
            masked_latent_feature=denoiser_masked_latent_feature
        )
        self.c_latent = c_latent
        self.self_conditioning = self_conditioning

    def sample_prior(self, num_nodes, device):
        return {
            "noised_latent_sidechain": torch.randn((num_nodes, self.c_latent), device=device)
        }


class ChimeraLatentWrapper(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_latent=128,
                 c_z=128,
                 c_hidden=256,
                 c_s_in=6,
                 num_rbf=16,
                 num_layers=4,
                 num_heads=8,
                 num_qk_pts=8,
                 num_v_pts=12,
                 h_time=64,
                 sidechain_k=30,
                 knn_k=20,
                 lrange_k=40,
                 res_h_mult_factor=2,
                 res_edge_mult_factor=1,
                 encoder_res_edge_updates=False,
                 self_conditioning=True,
                 broadcast_to_atoms=False,
                 lrange_embedding=False,
                 smooth_embedding=False,
                 mlp_decoder=False,
                 use_masking_features=True,
                 compatibility_mode=True
                 ):
        super().__init__()
        self.encoder = ProteinAtomicChimeraEmbedder(
            c_s=128,
            c_z=128,
            h_frame=c_latent,
            res_num_rbf=num_rbf,
            knn_k=sidechain_k,
            broadcast_to_atoms=broadcast_to_atoms,
            lrange_graph=lrange_embedding,
            use_masking_features=use_masking_features,
            res_h_mult_factor=res_h_mult_factor,
        )
        if mlp_decoder:
            self.decoder = MLPDecoder(
                c_s=c_s//2,
                c_latent=c_latent,
            )
        else:
            self.decoder = DesignIPMPDecoder(
                c_s=c_s//2,
                c_z=c_z,
                c_latent=c_latent,
                c_hidden=c_s//2,
                c_s_in=c_s_in,
                num_rbf=num_rbf,
                num_layers=num_layers,
                k=sidechain_k,
            )
        self.denoiser = DynamicGraphIpaFrameDenoiser(
            c_s=c_s,
            c_latent=c_latent,
            c_z=c_z,
            c_hidden=16,
            num_heads=16,
            num_qk_pts=num_qk_pts,
            num_v_pts=num_v_pts,
            n_layers=8,
            h_time=h_time,
            knn_k=knn_k,
            lrange_k=lrange_k,
            self_conditioning=self_conditioning,
        )
        self.c_latent = c_latent
        self.self_conditioning = self_conditioning

    def sample_prior(self, num_nodes, device):
        return {
            "noised_latent_sidechain": torch.randn((num_nodes, self.c_latent), device=device)
        }


class IPMPDenseLatentWrapper(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_latent=128,
                 c_z=128,
                 c_hidden=256,
                 c_s_in=6,
                 num_rbf=16,
                 num_layers=4,
                 num_heads=8,
                 num_qk_pts=8,
                 num_v_pts=12,
                 h_time=64,
                 sidechain_k=30,
                 self_conditioning=True,
                 use_init_dgram=False,
                 update_edges_with_dgram=False,
                 use_proteus_transition=False,
                 ):
        super().__init__()
        self.encoder = MLMIPMPEncoder(
            c_s=c_s//2,
            c_z=c_z,
            c_hidden=c_s//2,
            c_s_out=c_latent,
            c_s_in=c_s_in,
            num_rbf=num_rbf,
            num_layers=num_layers,
            k=sidechain_k,
        )
        self.decoder = DesignIPMPDecoder(
            c_s=c_s//2,
            c_z=c_z,
            c_latent=c_latent,
            c_hidden=c_s//2,
            c_s_in=c_s_in,
            num_rbf=num_rbf,
            num_layers=num_layers,
            k=sidechain_k,
        )
        self.denoiser = IpaDenoiser(
            c_s=c_s,
            c_latent=c_latent,
            c_z=c_z,
            c_hidden=16,
            num_heads=16,
            num_qk_points=num_qk_pts,
            num_v_points=num_v_pts,
            num_blocks=8,
            use_init_dgram=use_init_dgram,
            update_edges_with_dgram=update_edges_with_dgram,
            use_proteus_transition=use_proteus_transition,
        )
        self.c_latent = c_latent
        self.self_conditioning = self_conditioning

        # some compatibility code
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000

    def sample_prior(self, num_nodes, device):
        return {
            "noised_latent_sidechain": torch.randn((num_nodes, self.c_latent), device=device)
        }


class TFNDenseLatentWrapper(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_latent=128,
                 c_z=128,
                 c_hidden=256,
                 c_s_in=6,
                 num_rbf=16,
                 num_layers=4,
                 num_heads=8,
                 num_qk_pts=8,
                 num_v_pts=12,
                 h_time=64,
                 sidechain_k=30,
                 self_conditioning=True,
                 use_init_dgram=False,
                 update_edges_with_dgram=False,
                 use_proteus_transition=False,
                 use_pair_update=False,
                 use_pair_embedder=False,
                 res_h_mult_factor=2,
                 res_edge_mult_factor=1,
                 encoder_res_edge_updates=False,
                 encoder_use_ffn=False,
                 encoder_expanded_bb_rbfs=False,
                 broadcast_to_atoms=False,
                 lrange_embedding=False,
                 smooth_embedding=False,
                 use_masking_features=False,
                 denoiser_masked_latent_feature=False,
                 compatibility_mode=False,
                 latent_conv_downsample_factor=0,
                 use_ipa_decoder=False,
                 use_mlp_decoder=False,
                 decoder_bb_struct_node_dropout=1.0,
                 decoder_bb_struct_edge_dropout=1.0,
                 decoder_bb_struct_dropout_mode="itemwise",
                 use_traj_predictions=False,
                 ):
        super().__init__()

        self.latent_conv_downsample_factor = latent_conv_downsample_factor

        self.encoder = ProteinAtomicEmbedder(
            h_frame=c_latent,
            res_num_rbf=num_rbf,
            knn_k=sidechain_k,
            broadcast_to_atoms=broadcast_to_atoms,
            lrange_graph=lrange_embedding,
            smooth_nodewise=smooth_embedding,
            use_masking_features=use_masking_features,
            res_h_mult_factor=res_h_mult_factor,
            res_edge_mult_factor=res_edge_mult_factor,
            res_edge_update=encoder_res_edge_updates,
            compatibility_mode=compatibility_mode,
            n_layers=4,
            conv_downsample_factor=latent_conv_downsample_factor,
            use_ffn=encoder_use_ffn,
            expanded_bb_rbfs=encoder_expanded_bb_rbfs
        )
        assert not (use_ipa_decoder and use_mlp_decoder)
        if use_ipa_decoder:
            self.decoder = NodeIPADecoder(
                c_s=c_s//2,
                c_z=c_z,
                c_latent=c_latent,
                c_hidden=c_s//2,
                num_rbf=num_rbf,
                num_layers=num_layers,
                bb_struct_node_dropout=decoder_bb_struct_node_dropout,
                bb_struct_edge_dropout=decoder_bb_struct_edge_dropout,
                bb_struct_dropout_mode=decoder_bb_struct_dropout_mode,
                conv_downsample_factor=latent_conv_downsample_factor,
            )
        elif use_mlp_decoder:
            self.decoder = MLPDecoder(
                c_s=c_s//2,
                c_latent=c_latent,
            )
        else:
            self.decoder = DesignIPMPDecoder(
                c_s=c_s//2,
                c_latent=c_latent,
                c_z=c_z,
                c_hidden=c_latent,
                c_s_in=c_s_in,
                num_rbf=num_rbf,
                num_layers=num_layers,
                k=sidechain_k,
                conv_downsample_factor=latent_conv_downsample_factor,
                bb_struct_node_dropout=decoder_bb_struct_node_dropout,
                bb_struct_edge_dropout=decoder_bb_struct_edge_dropout,
                bb_struct_dropout_mode=decoder_bb_struct_dropout_mode,
            )
        self.denoiser = IpaDenoiser(
            c_s=c_s,
            c_latent=c_latent,
            c_z=c_z,
            c_hidden=16,
            num_heads=16,
            num_qk_points=num_qk_pts,
            num_v_points=num_v_pts,
            num_blocks=8,
            use_init_dgram=use_init_dgram,
            update_edges_with_dgram=update_edges_with_dgram,
            use_proteus_transition=use_proteus_transition,
            use_pair_update=use_pair_update,
            use_pair_embedder=use_pair_embedder,
            conv_downsample_factor=latent_conv_downsample_factor,
            use_traj_predictions=use_traj_predictions
        )
        self.c_latent = c_latent
        self.self_conditioning = self_conditioning

        # some compatibility code
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000

    def sample_prior(self, num_nodes, device, num_batch=None):
        if self.latent_conv_downsample_factor > 0:
            assert num_batch is not None
            num_latent_nodes = num_nodes
            for _ in range(self.latent_conv_downsample_factor):
                num_latent_nodes = math.ceil(num_latent_nodes / 2)
            return {
                "noised_latent_sidechain": torch.randn((num_batch * num_latent_nodes, self.c_latent), device=device)
            }


        else:
            return {
                "noised_latent_sidechain": torch.randn((num_nodes, self.c_latent), device=device)
            }


class ChimeraDenseLatentWrapper(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_latent=128,
                 c_z=128,
                 c_hidden=256,
                 c_s_in=6,
                 num_rbf=16,
                 num_layers=4,
                 num_heads=8,
                 num_qk_pts=8,
                 num_v_pts=12,
                 h_time=64,
                 sidechain_k=30,
                 self_conditioning=True,
                 use_init_dgram=False,
                 update_edges_with_dgram=False,
                 use_proteus_transition=False,
                 use_pair_update=False,
                 use_pair_embedder=False,
                 res_h_mult_factor=2,
                 res_edge_mult_factor=1,
                 encoder_res_edge_updates=False,
                 broadcast_to_atoms=False,
                 lrange_embedding=False,
                 smooth_embedding=False,
                 use_masking_features=False,
                 denoiser_seq_mask_features=False,
                 denoiser_use_v2=False,
                 denoiser_use_v3=False,
                 decoder_seq_mask_features=False,
                 denoiser_masked_latent_feature=False,
                 compatibility_mode=False,
                 latent_conv_downsample_factor=0,
                 use_ipa_decoder=False,
                 use_mlp_decoder=False,
                 decoder_bb_struct_node_dropout=1.0,
                 decoder_bb_struct_edge_dropout=1.0,
                 decoder_bb_struct_dropout_mode="itemwise",
                 decoder_gumbel_softmax=False,
                 use_traj_predictions=False,
                 encoder_use_ffn=False,
                 encoder_tanh_out=False,
                 atom_max_neighbors=32,
                 atomic_r=5,
                 force_flash_transformer=False,
                 compat_mode=True,
                 latent_supervisor=False,
                 sc_atomic_embedder=False,
                 discriminator=False,
                 ipmp_discriminator=False,
                 ):
        super().__init__()

        self.latent_conv_downsample_factor = latent_conv_downsample_factor

        self.encoder = ProteinAtomicChimeraEmbedder(
            h_frame=c_latent,
            res_num_rbf=num_rbf,
            knn_k=sidechain_k,
            broadcast_to_atoms=broadcast_to_atoms,
            lrange_graph=lrange_embedding,
            use_masking_features=use_masking_features,
            res_h_mult_factor=res_h_mult_factor,
            n_layers=4,
            conv_downsample_factor=latent_conv_downsample_factor,
            use_ffn=encoder_use_ffn,
            compat_mode=compat_mode,
            atom_max_neighbors=atom_max_neighbors,
            atomic_r=atomic_r,
            tanh_out=encoder_tanh_out
        )
        assert not (use_ipa_decoder and use_mlp_decoder)
        if use_ipa_decoder:
            self.decoder = NodeIPADecoder(
                c_s=c_s//2,
                c_z=c_z,
                c_latent=c_latent,
                c_hidden=c_s//2,
                num_rbf=num_rbf,
                num_layers=num_layers,
                bb_struct_node_dropout=decoder_bb_struct_node_dropout,
                bb_struct_edge_dropout=decoder_bb_struct_edge_dropout,
                bb_struct_dropout_mode=decoder_bb_struct_dropout_mode,
                conv_downsample_factor=latent_conv_downsample_factor,
            )
        elif use_mlp_decoder:
            self.decoder = MLPDecoder(
                c_s=c_s//8,
                c_latent=c_latent,
            )
        else:
            self.decoder = DesignIPMPDecoder(
                c_s=c_s//2,
                c_latent=c_latent,
                c_z=c_z,
                c_hidden=c_latent,
                c_s_in=c_s_in,
                num_rbf=num_rbf,
                num_layers=num_layers,
                k=sidechain_k,
                conv_downsample_factor=latent_conv_downsample_factor,
                bb_struct_node_dropout=decoder_bb_struct_node_dropout,
                bb_struct_edge_dropout=decoder_bb_struct_edge_dropout,
                bb_struct_dropout_mode=decoder_bb_struct_dropout_mode,
                use_seq_mask_features=decoder_seq_mask_features,
                gumbel_softmax_pred_seq=decoder_gumbel_softmax
            )
        self.denoiser = IpaDenoiser(
            c_s=c_s,
            c_latent=c_latent,
            c_z=c_z,
            c_hidden=16,
            num_heads=16,
            num_qk_points=num_qk_pts,
            num_v_points=num_v_pts,
            num_blocks=8,
            use_init_dgram=use_init_dgram,
            update_edges_with_dgram=update_edges_with_dgram,
            use_proteus_transition=use_proteus_transition,
            use_pair_update=use_pair_update,
            use_pair_embedder=use_pair_embedder,
            conv_downsample_factor=latent_conv_downsample_factor,
            use_traj_predictions=use_traj_predictions,
            force_flash_transformer=force_flash_transformer,
            use_seq_mask_features=denoiser_seq_mask_features,
            sc_atomic_embedder=sc_atomic_embedder,
            use_v2=denoiser_use_v2,
            use_v3=denoiser_use_v3
        )
        self.c_latent = c_latent
        self.self_conditioning = self_conditioning

        if latent_supervisor:
            self.latent_supervisor = LatentNodeSupervisor(c_latent=c_latent, c_hidden=c_s//2)
        if discriminator:
            if ipmp_discriminator:
                self.discriminator = IPMPDiscriminator(
                    c_s=c_s//2,
                    c_z=c_z,
                )
            else:
                self.discriminator = ProteinAtomicChimeraDiscriminator(
                    c_s=c_s//4,
                    c_z=c_z//4,
                    res_num_rbf=num_rbf,
                    knn_k=sidechain_k,
                    broadcast_to_atoms=broadcast_to_atoms,
                    lrange_graph=lrange_embedding,
                    use_masking_features=False,
                    res_h_mult_factor=res_h_mult_factor,
                    n_layers=4,
                    use_ffn=encoder_use_ffn,
                    compat_mode=compat_mode,
                    atom_max_neighbors=atom_max_neighbors,
                    atomic_r=atomic_r,
                    use_gumbel_softmax_logits=decoder_gumbel_softmax
                )

        # some compatibility code
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000

    def sample_prior(self, num_nodes, device, num_batch=None):
        if self.latent_conv_downsample_factor > 0:
            assert num_batch is not None
            num_latent_nodes = num_nodes
            for _ in range(self.latent_conv_downsample_factor):
                num_latent_nodes = math.ceil(num_latent_nodes / 2)
            return {
                "noised_latent_sidechain": torch.randn((num_batch * num_latent_nodes, self.c_latent), device=device)
            }


        else:
            return {
                "noised_latent_sidechain": torch.randn((num_nodes, self.c_latent), device=device)
            }


class DenseChimeraLatentWrapper(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_s_latent=4,
                 c_z_latent=4,
                 c_hidden=256,
                 c_s_in=6,
                 num_rbf=16,
                 num_layers=4,
                 num_heads=8,
                 num_qk_pts=8,
                 num_v_pts=12,
                 h_time=64,
                 sidechain_k=30,
                 res_h_mult_factor=2,
                 self_conditioning=True,
                 broadcast_to_atoms=False,
                 lrange_embedding=False,
                 mlp_decoder=False,
                 use_masking_features=True,
                 use_init_dgram=False,
                 update_edges_with_dgram=False,
                 use_proteus_transition=False,
                 use_pair_update=False,
                 quantize_encoder_codebook_size=None,
                 denoiser_use_seq_mask_features=False,
                 decoder_bb_struct_node_dropout=1.0,
                 decoder_bb_struct_edge_dropout=1.0,
                 decoder_bb_struct_dropout_mode="itemwise",
                 latent_conv_downsample_factor=0,
                 ):
        super().__init__()

        self.encoder = ProteinAtomicChimeraDenseEmbedder(
            c_s=c_s//2,
            c_z=c_z,
            c_s_out=c_s_latent,
            c_z_out=c_z_latent,
            res_num_rbf=num_rbf,
            knn_k=sidechain_k,
            broadcast_to_atoms=broadcast_to_atoms,
            lrange_graph=lrange_embedding,
            use_masking_features=use_masking_features,
            res_h_mult_factor=res_h_mult_factor,
            n_layers=4,
            quantize_codebook_size=quantize_encoder_codebook_size,
            conv_downsample_factor=latent_conv_downsample_factor
        )
        self.decoder = IPADecoder(
            c_s=c_s//2,
            c_z=c_z,
            c_s_in=c_s_latent,
            c_z_in=c_z_latent,
            c_hidden=c_s//2,
            num_layers=num_layers,
            bb_struct_node_dropout=decoder_bb_struct_node_dropout,
            bb_struct_edge_dropout=decoder_bb_struct_edge_dropout,
            bb_struct_dropout_mode=decoder_bb_struct_dropout_mode,
            conv_downsample_factor=latent_conv_downsample_factor
        )
        self.denoiser = DenseIpaDenoiser(
            c_s=c_s,
            c_z=c_z,
            c_s_latent=c_s_latent,
            c_z_latent=c_z_latent,
            c_hidden=16,
            num_heads=16,
            num_qk_points=num_qk_pts,
            num_v_points=num_v_pts,
            num_blocks=8,
            use_init_dgram=use_init_dgram,
            update_edges_with_dgram=update_edges_with_dgram,
            use_proteus_transition=use_proteus_transition,
            use_seq_mask_features=denoiser_use_seq_mask_features,
            conv_downsample_factor=latent_conv_downsample_factor
        )
        self.c_s_latent = c_s_latent
        self.c_z_latent = c_z_latent
        self.self_conditioning = self_conditioning

        # some compatibility code
        self.lrange_k = 10000
        self.knn_k = 10000
        self.lrange_logn_scale = 10000
        self.lrange_logn_offset = 10000

    def sample_prior(self, num_batch, num_nodes, device):
        if self.latent_conv_downsample_factor > 0:
            assert num_batch is not None
            num_latent_nodes = num_nodes
            for _ in range(self.latent_conv_downsample_factor):
                num_latent_nodes = math.ceil(num_latent_nodes / 2)
            return {
                "noised_latent_sidechain": torch.randn((num_batch * num_latent_nodes, self.c_s_latent), device=device),
                "noised_latent_edge": torch.randn((num_batch * num_latent_nodes * num_latent_nodes, self.c_z_latent), device=device)
            }
        else:
            return {
                "noised_latent_sidechain": torch.randn((num_nodes, self.c_s_latent), device=device),
                "noised_latent_edge": torch.randn((num_nodes, self.c_z_latent), device=device)
            }