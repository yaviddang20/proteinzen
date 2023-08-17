""" Diffusion modules """
import abc
import operator

import torch
from torch import nn
import tqdm
import numpy as np

from ligbinddiff.utils.type_l import type_l_randn_like, type_l_mult, type_l_add, type_l_sub
from ligbinddiff.utils.fiber import rand_compact_fiber_density
from ligbinddiff.utils.so3_embedding import so3_add, so3_sub, so3_mult, so3_randn_like
from ligbinddiff.model.modules.equiformer_v2.so3 import SO3_Embedding

class Diffuser(nn.Module, abc.ABC):
    def __init__(self,
                 denoiser,
                 scheduler,
                 _add,
                 _sub,
                 _mult,
                 _randn_like,
                 x_0_key,
                 x_t_key):
        super().__init__()

        self.denoiser = denoiser
        self.scheduler = scheduler

        self._add = _add
        self._sub = _sub
        self._mult = _mult
        self._randn_like = _randn_like

        self.x_0_key = x_0_key
        self.x_t_key = x_t_key

    def forward_noising(self, data):
        t = self.scheduler.sample_t()
        data_coeff, noise_coeff = self.scheduler.noising_coeffs(t)
        noised_data = self.noise(data, data_coeff, noise_coeff)
        noised_data['t'] = t
        noised_data['loss_weight'] = self.scheduler.weight(t)
        return noised_data

    def noise(self, data, data_coeff, noise_coeff):
        x_0 = data[self.x_0_key]
        noise = self._randn_like(x_0)
        scaled_noise = self._mult(noise_coeff, noise)
        scaled_x_0 = self._mult(data_coeff, x_0)
        x_t = self._add(scaled_x_0, scaled_noise)
        data[self.x_t_key] = x_t
        return data

    def reverse_noising(self, data):
        return self.denoiser(data)

    def forward(self, data):
        noised_batch = self.forward_noising(data)
        outputs = self.reverse_noising(noised_batch)
        return noised_batch, outputs

    def score_fn(self, data):
        x_t = data[self.x_t_key]
        t = data['t']
        denoiser_output = self.denoiser(data)
        x_0_pred = denoiser_output[self.x_0_key]
        alphabar_t = self.scheduler.alphabar(t)

        score = self._sub(
            self._mult(
                torch.sqrt(alphabar_t),
                x_0_pred
            ),
            x_t
        )
        score = self._mult(1 / (1 - alphabar_t), score)
        return score, denoiser_output

    def reverse_step(self, data, delta_t):
        assert delta_t < 0
        t = data['t']
        score, denoiser_output = self.score_fn(data)
        x_t = data[self.x_t_key]
        beta_t = self.scheduler.beta(t)

        f = -0.5 * beta_t
        g = torch.sqrt(beta_t)

        noise = self._randn_like(x_t)
        drift = self._mult(
            self._sub(self._mult(f, x_t), self._mult(g**2, score)),
            delta_t
        )
        diffusion = self._mult(g * np.sqrt(np.abs(delta_t)), noise)
        delta_x = self._add(drift, diffusion)

        x_tm1 = self._add(x_t, delta_x)
        tm1 = t + delta_t
        return x_tm1, tm1, denoiser_output

    @abc.abstractmethod
    def sample_prior(self, num_nodes, device):
        raise NotImplemented

    def sample(self,
               data,
               steps=None,
               show_progress=False,
               device=None):
        if device is None:
            device = data['x'].device
        num_nodes = data.num_nodes
        x_T = self.sample_prior(num_nodes, device)

        if steps is not None:
            if self.scheduler.discrete:
                assert self.scheduler.T % steps == 0
        else:
            steps = self.scheduler.T

        data['t'] = steps - 1
        data[self.x_t_key] = x_T
        delta_t = - self.scheduler.T // steps
        denoiser_output = None

        with torch.no_grad():
            if show_progress:
                pbar = tqdm.tqdm(total=steps)

            while data['t'] > 0:
                x_tm1, tm1, denoiser_output = self.reverse_step(data, delta_t)
                data['t'] = tm1
                data[self.x_t_key] = x_tm1

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        return data, denoiser_output


class DensityDiffuser(Diffuser):
    def __init__(self, denoiser, scheduler, n_channels=4, density_n_max=5):
        super().__init__(denoiser,
                         scheduler,
                         _add=type_l_add,
                         _sub=type_l_sub,
                         _mult=type_l_mult,
                         _randn_like=type_l_randn_like,
                         x_0_key='density',
                         x_t_key='noised_density')

        self.n_channels = n_channels
        self.density_n_max = density_n_max

    def sample_prior(self, num_nodes, device):
        density = rand_compact_fiber_density(
            num_nodes,
            self.density_n_max,
            num_channels=self.n_channels,
            device=device)
        return density


class SuperpositionDiffuser(Diffuser):
    def __init__(self, denoiser, scheduler, num_atoms=91):
        super().__init__(denoiser,
                         scheduler,
                         _add=so3_add,
                         _sub=so3_sub,
                         _mult=so3_mult,
                         _randn_like=so3_randn_like,
                         x_0_key='atom91_centered',
                         x_t_key='noised_atom91')

        self.num_atoms = num_atoms

    def sample_prior(self, num_nodes, device):
        superposition = SO3_Embedding(
            num_nodes,
            lmax_list=[1],
            num_channels=self.num_atoms,
            device=device,
            dtype=torch.float
        )
        superposition.embedding = torch.randn_like(superposition.embedding)
        return superposition


class LatentDiffuser(Diffuser):
    def __init__(self,
                 denoiser,
                 scheduler,
                 encoder,
                 decoder,
                 latent_lmax_list,
                 latent_n_channels):
        super().__init__(denoiser,
                         scheduler,
                         _add=so3_add,
                         _sub=so3_sub,
                         _mult=so3_mult,
                         _randn_like=so3_randn_like,
                         x_0_key='latent',
                         x_t_key='noised_latent')
        self.encoder = encoder
        self.decoder = decoder
        self.latent_lmax_list = latent_lmax_list
        self.latent_n_channels = latent_n_channels

    def sample_prior(self, num_nodes, device):
        latent = SO3_Embedding(
            num_nodes,
            lmax_list=self.latent_lmax_list,
            num_channels=self.latent_n_channels,
            device=device,
            dtype=torch.float
        )
        latent.embedding = torch.randn_like(latent.embedding)
        return latent

    def forward(self, data):
        latent_data = self.encoder(data)
        noised_latent = self.forward_noising(latent_data)
        latent_outputs = self.reverse_noising(noised_latent)
        decoded_outputs = self.decoder(latent_outputs)
        return noised_latent, latent_outputs, decoded_outputs
