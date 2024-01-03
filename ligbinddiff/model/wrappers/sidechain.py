import torch
from torch import nn

from ligbinddiff.model.autoencoder.density import DensityEncoder
from ligbinddiff.model.autoencoder.ipa import IPAEncoder, IPADecoder
from ligbinddiff.model.design.ipmp import IPMPEncoder as DesignIPMPEncoder, IPMPDecoder as DesignIPMPDecoder
from ligbinddiff.model.autoencoder.ipmp import IPMPEncoder as AutoEncIPMPEncoder, IPMPDecoder as AutoEncIPMPDecoder
from ligbinddiff.model.denoiser.sidechain.ipmp_latent import IPMPDenoiser

class LatentSidechainWrapper(nn.Module):
    def __init__(self, encoder, decoder, denoiser):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.denoiser = denoiser

class IPMPLatentSidechainWrapper(nn.Module):
    def __init__(self,
                 c_s=128,
                 c_z=128,
                 c_hidden=128,
                 c_s_in=6,
                 c_time=64,
                 num_rbf=16,
                 num_layers=4,
                 k=30):
        super().__init__()
        self.encoder = DesignIPMPEncoder(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            c_s_in=c_s_in,
            num_rbf=num_rbf,
            num_layers=num_layers,
            k=k
        )
        self.decoder = DesignIPMPDecoder(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            c_s_in=c_s_in,
            num_rbf=num_rbf,
            num_layers=num_layers,
            k=k
        )
        self.denoiser = IPMPDenoiser(
            c_s=c_s,
            c_latent=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            c_s_in=c_s_in,
            c_time=c_time,
            num_rbf=num_rbf,
            num_layers=num_layers,
            k=k
        )
        self.c_s = c_s

    def sample_prior(self, num_nodes, device):
        return {
            "noised_latent_sidechain": torch.randn((num_nodes, self.c_s), device=device)
        }


class IPALatentSidechainWrapper(nn.Module):
    def __init__(self,
                 c_s=128,
                 c_z=128,
                 c_hidden=128,
                 c_s_in=6,
                 c_time=64,
                 num_rbf=16,
                 num_layers=4,
                 k=30,
                 attn='qkv'):
        super().__init__()
        self.encoder = IPAEncoder(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            c_s_in=c_s_in,
            num_rbf=num_rbf,
            num_layers=num_layers,
            k=k,
            attn=attn
        )
        self.decoder = IPADecoder(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            c_s_in=c_s_in,
            num_rbf=num_rbf,
            num_layers=num_layers,
            k=k,
            attn=attn
        )
        self.denoiser = IPMPDenoiser(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            c_s_in=c_s_in,
            c_time=c_time,
            num_rbf=num_rbf,
            num_layers=num_layers,
            k=k
        )
        self.c_s = c_s

    def sample_prior(self, num_nodes, device):
        return {
            "noised_latent_sidechain": torch.randn((num_nodes, self.c_s), device=device)
        }


class DensityLatentSidechainWrapper(nn.Module):
    def __init__(self,
                 c_s=128,
                 c_latent=16,
                 c_z=128,
                 c_hidden=128,
                 c_s_in=6,
                 c_time=64,
                 num_rbf=16,
                 num_layers=4,
                 k=30):
        super().__init__()
        self.encoder = DensityEncoder(
            c_s=[c_s//4, c_s//8, c_s//16, c_s//32],
            c_z=c_z,
            c_output=c_latent,
            num_rbf=num_rbf,
            num_layers=num_layers,
            k=k
        )
        self.decoder = AutoEncIPMPDecoder(
            c_s=c_s,
            c_latent=c_latent,
            c_z=c_z,
            c_hidden=c_hidden,
            c_s_in=c_s_in,
            num_rbf=num_rbf,
            num_layers=1,#num_layers,
            k=k
        )
        self.denoiser = IPMPDenoiser(
            c_s=c_s,
            c_latent=c_latent,
            c_z=c_z,
            c_hidden=c_hidden,
            c_s_in=c_s_in,
            c_time=c_time,
            num_rbf=num_rbf,
            num_layers=num_layers,
            k=k
        )
        self.c_latent = c_latent

    def sample_prior(self, num_nodes, device):
        return {
            "noised_latent_sidechain": torch.randn((num_nodes, self.c_latent), device=device)
        }