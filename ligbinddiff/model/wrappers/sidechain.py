import torch
from torch import nn

from ligbinddiff.model.autoencoder.ipmp import IPMPEncoder, IPMPDecoder
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
        self.encoder = IPMPEncoder(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            c_s_in=c_s_in,
            num_rbf=num_rbf,
            num_layers=num_layers,
            k=k
        )
        self.decoder = IPMPDecoder(
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
        return torch.randn((num_nodes, self.c_s), device=device)
