import torch
from torch import nn

from ligbinddiff.model.design.ipmp import IPMPDecoder as DesignIPMPDecoder
from ligbinddiff.model.mlm.ipmp import IPMPEncoder as MLMIPMPEncoder
from ligbinddiff.model.denoiser.protein.frames import DynamicGraphIpaFrameDenoiser


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
                 num_vn=4,
                 self_conditioning=True,
                 ):
        super().__init__()
        self.encoder = MLMIPMPEncoder(
            c_s=c_latent,
            c_z=c_z,
            c_hidden=c_latent,
            c_s_in=c_s_in,
            num_rbf=num_rbf,
            num_layers=num_layers,
            k=sidechain_k,
        )
        self.decoder = DesignIPMPDecoder(
            c_s=c_latent,
            c_z=c_z,
            c_hidden=c_latent,
            c_s_in=c_s_in,
            num_rbf=num_rbf,
            num_layers=num_layers,
            k=sidechain_k,
        )
        self.denoiser = DynamicGraphIpaFrameDenoiser(
            c_s=c_s,
            c_latent=c_latent,
            c_z=c_z,
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_pts=num_qk_pts,
            num_v_pts=num_v_pts,
            h_time=h_time,
            knn_k=knn_k,
            lrange_k=lrange_k,
            num_vn=num_vn,
            self_conditioning=self_conditioning,
        )
        self.c_latent = c_latent
        self.self_conditioning = self_conditioning

    def sample_prior(self, num_nodes, device):
        return {
            "noised_latent_sidechain": torch.randn((num_nodes, self.c_latent), device=device)
        }

