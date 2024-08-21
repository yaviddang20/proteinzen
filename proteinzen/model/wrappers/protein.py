import torch
from torch import nn

from proteinzen.model.design.ipmp import IPMPDecoder as DesignIPMPDecoder
from proteinzen.model.encoder.ipmp import IPMPEncoder as MLMIPMPEncoder
from proteinzen.model.encoder.tfn import ProteinAtomicEmbedder
from proteinzen.model.denoiser.protein.frames import DynamicGraphIpaFrameDenoiser
from proteinzen.model.denoiser.protein.dense import IpaDenoiser


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
                 num_heads=8,
                 num_qk_pts=8,
                 num_v_pts=12,
                 h_time=64,
                 sidechain_k=30,
                 knn_k=20,
                 lrange_k=40,
                 self_conditioning=True,
                 ):
        super().__init__()
        self.encoder = ProteinAtomicEmbedder(
            h_frame=c_latent,
            res_num_rbf=num_rbf,
            knn_k=sidechain_k,
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
                 ):
        super().__init__()
        self.encoder = ProteinAtomicEmbedder(
            h_frame=c_latent,
            res_num_rbf=num_rbf,
            knn_k=sidechain_k,
        )
        self.decoder = DesignIPMPDecoder(
            c_s=c_s//2,
            c_latent=c_latent,
            c_z=c_z,
            c_hidden=c_latent,
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