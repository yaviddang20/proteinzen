""" Diffusion modules """
import abc
import operator

import torch
from torch import nn
import tqdm
import numpy as np

from ligbinddiff.utils.type_l import type_l_randn_like, type_l_mult, type_l_add, type_l_sub
# from ligbinddiff.utils.fiber import rand_compact_fiber_density
from ligbinddiff.utils.so3_embedding import so3_add, so3_sub, so3_mult, so3_randn_like, so3_ones_like, gen_so3_unop
from ligbinddiff.model.modules.equiformer_v2.so3 import SO3_Embedding

from ligbinddiff.model.seq_des.latent.equiformer_v2.denoiser import LatentDenoiser
from ligbinddiff.model.seq_des.latent.equiformer_v2.discriminator import Atom91Discriminator
from ligbinddiff.model.seq_des.latent.equiformer_v2.autoencoder import LatentEncoder, LatentDecoder
from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2


# TODO: brought this in to avoid importing dgl in the utils, do this in a less hacky way
def gen_compact_nmax_fiber(n_max):
    fiber_dict = {l: 0 for l in range(n_max+1)}
    n_levels = {l: [] for l in range(n_max+1)}
    for l in fiber_dict.keys():
        for n in range(n_max+1):
            if n < l: continue
            if (n-l) % 2 == 0:
                fiber_dict[l] = fiber_dict[l] + 1
                n_levels[l].append(n)
    return fiber_dict

def rand_compact_fiber_density(num_nodes, n_max, num_channels=1, device='cpu'):
    fiber = gen_compact_nmax_fiber(n_max)
    density = {}
    for l, num_vecs in fiber.items():
        m_tot = 2*l+1
        density[l] = torch.randn((num_nodes, num_vecs * num_channels, m_tot), device=device)
    return density


class Diffuser(nn.Module, abc.ABC):
    def __init__(self,
                 denoiser,
                 scheduler,
                 _add,
                 _sub,
                 _mult,
                 _randn_like,
                 x_0_key,
                 x_0_pred_key,
                 x_t_key):
        super().__init__()

        self.denoiser = denoiser
        self.scheduler = scheduler

        self._add = _add
        self._sub = _sub
        self._mult = _mult
        self._randn_like = _randn_like

        self.x_0_key = x_0_key
        self.x_0_pred_key = x_0_pred_key
        self.x_t_key = x_t_key

    def forward_noising(self, data):
        t = self.scheduler.sample_t([data.num_graphs])
        t = t.to(data['x'].device)
        data_coeff, noise_coeff = self.scheduler.noising_coeffs(t)
        noised_data = self.noise(data, data_coeff, noise_coeff)
        noised_data['t'] = t
        noised_data['loss_weight'] = self.scheduler.weight(t)
        return noised_data

    def noise(self, data, data_coeff, noise_coeff):
        data_splits = data._slice_dict['x']
        data_lens = data_splits[1:] - data_splits[:-1]

        data_coeff = torch.cat([
            data_coeff[i].expand(l) for i, l in enumerate(data_lens)
        ]).view(-1, 1, 1)
        noise_coeff = torch.cat([
            noise_coeff[i].expand(l) for i, l in enumerate(data_lens)
        ]).view(-1, 1, 1)

        x_0 = data[self.x_0_key]
        noise = self._randn_like(x_0)
        scaled_noise = self._mult(noise_coeff, noise)
        scaled_x_0 = self._mult(data_coeff, x_0)
        x_t = self._add(scaled_x_0, scaled_noise)
        data[self.x_t_key] = x_t
        return data

    def reverse_noising(self, data):
        return self.denoiser(data)

    def forward(self, data, warmup=None):
        noised_batch = self.forward_noising(data)
        outputs = self.reverse_noising(noised_batch)
        return noised_batch, outputs

    def score_fn(self, data):
        x_t = data[self.x_t_key]
        t = data['t']
        denoiser_output = self.denoiser(data)
        x_0_pred = denoiser_output[self.x_0_pred_key]
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

        data['t'] = torch.ones([num_nodes], device=device).view(
            -1, 1, 1).float() * (steps - 1)
        data[self.x_t_key] = x_T
        delta_t = - self.scheduler.T // steps
        denoiser_output = None

        with torch.no_grad():
            if show_progress:
                pbar = tqdm.tqdm(total=steps)

            while (data['t'] > 0).all():
                x_tm1, tm1, denoiser_output = self.reverse_step(data, delta_t)
                data['t'] = tm1
                data[self.x_t_key] = x_tm1

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        return data, denoiser_output


class LegacyDensityDiffuser(Diffuser):
    def __init__(self, denoiser, scheduler, n_channels=4, density_n_max=5):
        super().__init__(denoiser,
                         scheduler,
                         _add=type_l_add,
                         _sub=type_l_sub,
                         _mult=type_l_mult,
                         _randn_like=type_l_randn_like,
                         x_0_key='density',
                         x_0_pred_key='denoised_density',
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

    def sample(self,
               data,
               steps=None,
               show_progress=True,
               device=None):
        if device is None:
            device = data['x'].device
        num_nodes = data.num_nodes
        n_max = self.density_n_max
        num_channels = self.n_channels
        density = rand_compact_fiber_density(num_nodes, n_max, num_channels=num_channels, device=device)
        if steps:
            assert self.scheduler.T % steps == 0
        else:
            steps = self.scheduler.T

        step = steps - 1
        delta_t = -1/steps
        data['t'] = step
        data[self.x_t_key] = density
        denoiser_output = None

        with torch.no_grad():
            if show_progress:
                pbar = tqdm.tqdm(total=steps)
            while step > 0:
                # from https://github.com/huggingface/diffusers/blob/v0.18.2/src/diffusers/schedulers/scheduling_ddpm.py#L91
                alpha_prod_t = self.scheduler.alphas_cumprod[step]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[step-1]
                beta_prod_t = 1 - alpha_prod_t
                beta_prod_t_prev = 1 - alpha_prod_t_prev
                current_alpha_t = alpha_prod_t / alpha_prod_t_prev
                current_beta_t = 1 - current_alpha_t

                denoiser_output = self.reverse_noising(data)
                one_shot_density = denoiser_output['density']
                # one_shot_seq = denoiser_output['seq_logits']
                # one_shot_atom91 = denoiser_output['atom91']

                pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
                current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
                density = type_l_add(
                    type_l_mult(pred_original_sample_coeff, one_shot_density),
                    type_l_mult(current_sample_coeff, density)
                )
                # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
                # and sample from it to get previous sample
                # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
                variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

                # we always take the log of variance, so clamp it to ensure it's not 0
                variance = torch.clamp(variance, min=1e-20)
                noise = type_l_mult(
                    variance,
                    rand_compact_fiber_density(num_nodes, n_max, num_channels=num_channels, device=device)
                )
                density = type_l_add(
                    density,
                    noise
                )

                step -= 1

                data['t'] = step
                data[self.x_t_key] = density

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
                         x_0_pred_key='denoised_density',
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
                         _add=operator.add, #so3_add,
                         _sub=operator.sub, #so3_sub,
                         _mult=operator.mul, #so3_mult,
                         _randn_like=torch.randn_like, #so3_randn_like,
                         x_0_key='atom91_centered',
                         x_0_pred_key='denoised_atom91',
                         x_t_key='noised_atom91')

        self.num_atoms = num_atoms

    def sample_prior(self, num_nodes, device):
        # superposition = SO3_Embedding(
        #     num_nodes,
        #     lmax_list=[1],
        #     num_channels=self.num_atoms,
        #     device=device,
        #     dtype=torch.float
        # )
        # superposition.embedding = torch.randn_like(superposition.embedding)
        superposition = torch.randn((num_nodes, self.num_atoms, 3), device=device)
        return superposition



## TODO: is there a way to not break the original structure?
class LatentDiffuser(Diffuser):
    def __init__(self,
                 scheduler,
                 node_lmax_list,
                 edge_channels_list,
                 h_time=64,
                 scalar_h_dim=128,
                 bb_lmax_list=[1],
                 bb_channels=6,
                 atom_lmax_list=[1],
                 atom_channels=91,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4,
                 ):
        # build these expensive coeff stores
        atom_super_lmax_list = [max(l1, l2) for l1, l2 in zip(atom_lmax_list, node_lmax_list)]
        bb_super_lmax_list = [max(l1, l2) for l1, l2 in zip(bb_lmax_list, node_lmax_list)]
        atom_super_SO3_rotation_list = nn.ModuleList()
        bb_super_SO3_rotation_list = nn.ModuleList()
        node_SO3_rotation_list = nn.ModuleList()
        for lmax in atom_super_lmax_list:
            atom_super_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in bb_super_lmax_list:
            bb_super_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in node_lmax_list:
            node_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )

        atom_super_SO3_grid_list = nn.ModuleList()
        bb_super_SO3_grid_list = nn.ModuleList()
        node_SO3_grid_list = nn.ModuleList()
        for l in range(max(atom_super_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            atom_super_SO3_grid_list.append(SO3_m_grid)
        for l in range(max(bb_super_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            bb_super_SO3_grid_list.append(SO3_m_grid)
        bb_super_SO3_grid_list = nn.ModuleList()
        for l in range(max(node_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            node_SO3_grid_list.append(SO3_m_grid)

        mappingReduced_super_atoms = CoefficientMappingModule(atom_super_lmax_list, atom_super_lmax_list)
        mappingReduced_super_bb = CoefficientMappingModule(bb_super_lmax_list, bb_super_lmax_list)
        mappingReduced_nodes = CoefficientMappingModule(node_lmax_list, node_lmax_list)

        denoiser = LatentDenoiser(
            node_lmax_list=node_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_nodes=mappingReduced_nodes,
            node_SO3_rotation=node_SO3_rotation_list,
            node_SO3_grid=node_SO3_grid_list,
            num_heads=num_heads,
            h_channels=h_channels,
            h_time=h_time,
            scalar_h_dim=scalar_h_dim,
            n_layers=num_layers,
        )
        # init late so we can use common structure
        super().__init__(denoiser,
                         scheduler,
                         _add=so3_add,
                         _sub=so3_sub,
                         _mult=so3_mult,
                         _randn_like=so3_randn_like,
                         x_0_key='latent',
                         x_0_pred_key='denoised_latent',
                         x_t_key='noised_latent')
        self.encoder = LatentEncoder(
            node_lmax_list=node_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_nodes=mappingReduced_nodes,
            mappingReduced_super_bb=mappingReduced_super_bb,
            mappingReduced_super_atoms=mappingReduced_super_atoms,
            node_SO3_rotation=node_SO3_rotation_list,
            node_SO3_grid=node_SO3_grid_list,
            bb_super_SO3_rotation=bb_super_SO3_rotation_list,
            bb_super_SO3_grid=bb_super_SO3_grid_list,
            atom_super_SO3_rotation=atom_super_SO3_rotation_list,
            atom_super_SO3_grid=atom_super_SO3_grid_list,
            bb_lmax_list=bb_lmax_list,
            bb_channels=bb_channels,
            atom_lmax_list=atom_lmax_list,
            atom_channels=atom_channels,
            num_heads=num_heads,
            h_channels=h_channels,
            num_layers=num_layers
        )
        self.decoder = LatentDecoder(
            node_lmax_list=node_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_nodes=mappingReduced_nodes,
            mappingReduced_super_bb=mappingReduced_super_bb,
            mappingReduced_super_atoms=mappingReduced_super_atoms,
            node_SO3_rotation=node_SO3_rotation_list,
            node_SO3_grid=node_SO3_grid_list,
            bb_super_SO3_rotation=bb_super_SO3_rotation_list,
            bb_super_SO3_grid=bb_super_SO3_grid_list,
            atom_super_SO3_rotation=atom_super_SO3_rotation_list,
            atom_super_SO3_grid=atom_super_SO3_grid_list,
            bb_lmax_list=bb_lmax_list,
            bb_channels=bb_channels,
            atom_lmax_list=atom_lmax_list,
            atom_channels=atom_channels,
            num_heads=num_heads,
            h_channels=h_channels,
            num_layers=num_layers
        )

        self.latent_lmax_list = node_lmax_list
        self.latent_n_channels = h_channels
        self.mappingReduced_super_atoms = mappingReduced_super_atoms
        self.mappingReduced_super_bb = mappingReduced_super_bb
        self.mappingReduced_nodes = mappingReduced_nodes
        self._apply_so3 = gen_so3_unop

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

    def forward(self, data, warmup=False, deterministic=False):
        if not warmup:
            latent_data = self.encoder(data)

            if deterministic:
                latent_data['latent'] = latent_data['latent_mu']
            else:
                latent_sigma = self._apply_so3(torch.exp)(
                    self._mult(
                        latent_data['latent_logvar'],
                        0.5
                    ),
                )
                latent_data['latent'] = self._add(
                    latent_data['latent_mu'],
                    self._mult(
                        latent_sigma,
                        self._randn_like(latent_sigma)
                    )
                )

            decoded_outputs = self.decoder(latent_data)
            # latent_data['latent'].embedding = latent_data['latent'].embedding.detach()

            noised_latent = self.forward_noising(latent_data)
            latent_outputs = self.reverse_noising(noised_latent)
            decoded_outputs.update(latent_outputs)

        else:
            latent_data = self.encoder(data)

            if deterministic:
                latent_data['latent'] = latent_data['latent_mu']
            else:
                latent_sigma = self._apply_so3(torch.exp)(
                    self._mult(
                        latent_data['latent_logvar'],
                        0.5
                    ),
                )
                latent_data['latent'] = self._add(
                    latent_data['latent_mu'],
                    self._mult(
                        latent_sigma,
                        self._randn_like(latent_sigma)
                    )
                )
            decoded_outputs = self.decoder(latent_data)
            noised_latent = latent_data

        return noised_latent, decoded_outputs

    def sample(self,
               data,
               steps=None,
               show_progress=False,
               device=None):
        latent_outputs, denoiser_output = super().sample(data,
                                               steps,
                                               show_progress,
                                               device)
        latent_outputs['latent'] = latent_outputs['noised_latent']
        decoded_outputs = self.decoder(latent_outputs)
        # we do this to recover the "ground truth" encoding
        decoded_outputs = self.encoder(decoded_outputs)
        decoded_outputs['latent'] = decoded_outputs['latent_mu']
        # decoded_outputs['latent_mu'] = latent_outputs['latent']
        # decoded_outputs['latent_logvar'] = self._apply_so3(torch.zeros_like)(latent_outputs['latent'])
        decoded_outputs['seq_logits'] = decoded_outputs['decoded_seq_logits']

        return decoded_outputs, denoiser_output
