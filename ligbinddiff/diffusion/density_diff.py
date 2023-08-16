""" Diffusion modules """
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import tqdm
import torch.distributions as dist

from ligbinddiff.utils.type_l import type_l_randn_like, type_l_mult, type_l_add, type_l_sub
from ligbinddiff.utils.fiber import rand_compact_fiber_density


class LinearDiscreteVPDensityDiffuser(nn.Module):
    """ Diffusion in density space

    Adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py """
    def __init__(self,
                denoiser,
                beta_min=0.1,
                beta_max=20,
                N=1000,
                n_channels=1):
        super().__init__()
        """Construct a Variance Preserving SDE.

        Args:
        beta_min: value of beta(0)
        beta_max: value of beta(1)
        N: number of discretization steps
        """

        self.denoiser = denoiser
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.t_dist = dist.Categorical(torch.ones(N) / N)

        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.n_channels = n_channels

    def forward_diffusion(self, density):
        """
        Parameters
        ----------
        coords: torch.Tensor
            one single protein coord set
        """
        t_idx = self.t_dist.sample()
        # t_idx = torch.zeros_like(t_idx)
        ts = t_idx / self.N

        # t_idx = torch.zeros_like(t_idx)

        noise = type_l_randn_like(density)
        scaled_noise = type_l_mult(self.sqrt_1m_alphas_cumprod[t_idx], noise)
        scaled_density = type_l_mult(self.sqrt_alphas_cumprod[t_idx], density)
        noised_density = type_l_add(scaled_density, scaled_noise)

        return noised_density, ts

    def snr_derivative(self, ts):
        # hacky way to accomedate the snr interface for reweighting
        t_idx = round(ts.item() * self.N)
        return None, -0.5 / self.discrete_betas[t_idx]  # only neg bc of how the snr interface works

    def denoising_step(self, node_features, edge_features, noised_density, ts, graph):
        return self.denoiser(node_features, edge_features, noised_density, ts, graph)

    def score_fn(self, node_features, edge_features, graph):
        def score(noised_density, ts):
            t_idx = round(ts.item() * self.N)
            denoised_density = self.denoiser(node_features, edge_features, noised_density, ts, graph)
            alpha_t = self.alphas[t_idx]
            return

    def sample(self,
               node_features,
               edge_features,
               graph,
               steps=None,
               return_trajectory=False,
               show_progress=False,
               eps=1e-6):
        device = graph.device
        num_nodes = graph.number_of_nodes()
        n_max = self.denoiser.density_n_max
        num_channels = self.n_channels
        density = rand_compact_fiber_density(num_nodes, n_max, num_channels=num_channels, device=device)
        if steps:
            assert self.N % steps == 0
        else:
            steps = self.N

        step = steps - 1
        delta_t = -1/steps
        trajectory = []
        with torch.no_grad():
            trajectory.append((density, None, None))
            if show_progress:
                pbar = tqdm.tqdm(total=steps)
            while step > 0:
                # from https://github.com/huggingface/diffusers/blob/v0.18.2/src/diffusers/schedulers/scheduling_ddpm.py#L91
                alpha_prod_t = self.alphas_cumprod[step]
                alpha_prod_t_prev = self.alphas_cumprod[step-1]
                beta_prod_t = 1 - alpha_prod_t
                beta_prod_t_prev = 1 - alpha_prod_t_prev
                current_alpha_t = alpha_prod_t / alpha_prod_t_prev
                current_beta_t = 1 - current_alpha_t

                t = torch.as_tensor(step / steps, device=device)

                one_shot_density, one_shot_seq, one_shot_atom91 = self.denoising_step(node_features, edge_features, density, t, graph)

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

                # # drift term
                # alphabar_t = self.alphas_cumprod[step]
                # beta_t = self.discrete_betas[step]
                # t = torch.as_tensor(step / steps, device=device)

                # one_shot_density, one_shot_seq, one_shot_atom91 = self.denoising_step(node_features, edge_features, density, t, graph)
                # density_term = type_l_mult(0.5 * (alphabar_t + 1) / (1 - alphabar_t), density)
                # one_shot_term = type_l_mult(torch.sqrt(alphabar_t)/(1 - alphabar_t), one_shot_density)
                # diff = type_l_sub(density_term, one_shot_term)
                # a = type_l_mult(beta_t * delta_t, diff)

                # # noise term
                # b = type_l_mult(
                #     torch.sqrt(beta_t) * np.sqrt(np.abs(delta_t)),
                #     rand_compact_fiber_density(num_nodes, n_max, num_channels=num_channels, device=device)
                # )

                # # update
                # density = type_l_add(
                #     density,
                #     type_l_add(a, b)
                # )

                step -= 1
                trajectory.append((density, one_shot_seq, one_shot_atom91))
                if show_progress:
                    pbar.update(1)
            if show_progress:
                pbar.close()


        if return_trajectory:
            return trajectory
        else:
            return trajectory[-1]




class PositiveLinear(nn.Module):
    def forward(self, weight):
        return torch.abs(weight)


class LearnableVPDensityDiffuser(nn.Module):
    """ Diffusion in density space """
    def __init__(self,
                denoiser,
                gamma_0=-7,
                gamma_1=13.5,
                ):
        super().__init__()

        self.denoiser = denoiser
        self.t_dist = dist.Uniform(0, 1)

        ## beta schedule parameters
        self.l1 = nn.Linear(1, 1)
        self.l2 = nn.Linear(1, 1024)
        self.l3 = nn.Linear(1024, 1)
        nn.utils.parametrize.register_parametrization(self.l1, "weight", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l1, "bias", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l2, "weight", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l2, "bias", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l3, "weight", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l3, "bias", PositiveLinear())

        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1

    def _gamma_tilde(self, t):
        ret = self.l1(t) + self.l3(F.sigmoid(self.l2(self.l1(t))))  # B x 1
        return ret

    def _gamma(self, t):
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * (
                self._gamma_tilde(t) - self._gamma_tilde(torch.zeros_like(t))
            ) / (
                self._gamma_tilde(torch.ones_like(t)) - self._gamma_tilde(torch.zeros_like(t))
            )
        return gamma  # B x 1

    def snr(self, t):
        return torch.expm1(-self._gamma(t)) + 1 # B x 1

    def snr_derivative(self, t):
        snr_t, tau_t = torch.func.jvp(self.snr, (t,), (torch.ones_like(t),)) # B x B
        return snr_t, tau_t  # B x 1, B x 1

    def alpha(self, t):
        return F.sigmoid(-self._gamma(t))  # B x 1

    def forward_diffusion(self, density, ts=None):
        """
        Parameters
        ----------
        coords: torch.Tensor
            one single protein coord set
        """
        # device = density[0].device # self.l1.weight.device
        # print(device)
        device = self.l1.weight.device
        # density = {k:v.to(device) for k,v in density.items()}

        # sample timepoints
        if ts is None:
            ts = self.t_dist.sample((1,1)).to(device)  # (1,1)
        alpha_ts = self.alpha(ts)  # (1,1)
        # print(alpha_ts.device)

        noise = type_l_randn_like(density)
        scaled_noise = type_l_mult(torch.sqrt(1 - alpha_ts), noise)
        scaled_density = type_l_mult(torch.sqrt(alpha_ts), density)
        noised_density = type_l_add(scaled_density, scaled_noise)

        return noised_density, ts

    def denoising_step(self, node_features, edge_features, noised_density, ts, graph):
        return self.denoiser(node_features, edge_features, noised_density, ts, graph)

    def sample(self,
               node_features,
               edge_features,
               graph,
               steps=100,
               return_trajectory=False,
               eps=1e-6):
        device = graph.device
        num_nodes = graph.number_of_nodes()
        n_max = self.denoiser.density_n_max
        density = rand_compact_fiber_density(num_nodes, n_max, device=device)

        trajectory = []
        with torch.no_grad():
            delta_t = 1/steps
            t = torch.tensor([[1.]]).to(device) - delta_t
            trajectory.append((density, None, None))
            pbar = tqdm.tqdm(total=steps)
            while t > delta_t:
                s = t - delta_t
                alpha_t = self.alpha(t)
                alpha_s = self.alpha(s)
                c = -torch.expm1(self._gamma(s) - self._gamma(t))
                one_shot_density, one_shot_seq, one_shot_atom91 = self.denoising_step(node_features, edge_features, density, t, graph)
                density_term = type_l_mult(1 - c, density)
                one_shot_term = type_l_mult(torch.sqrt(alpha_t) * c, one_shot_density)
                drift = type_l_mult(
                    torch.sqrt(alpha_s / alpha_t),
                    type_l_add(density_term, one_shot_term)
                )
                diffusion = type_l_mult(
                    torch.sqrt((1 - alpha_s) * c),
                    rand_compact_fiber_density(num_nodes, n_max, device=device)
                )
                density = type_l_add(
                    drift,
                    diffusion
                )

                # snr_t, tau_t = self.snr_derivative(t)
                # beta_t = -tau_t * (1 - alpha_t) ** 2 # * -1?
                # print(alpha_t, beta_t, snr_t, tau_t)

                # # drift term
                # one_shot_density, one_shot_seq, one_shot_atom91 = self.denoising_step(node_features, edge_features, density, t, graph)
                # density_term = type_l_mult(0.5 * (alpha_t + 1) / (1 - alpha_t), density)
                # one_shot_term = type_l_mult(torch.sqrt(alpha_t)/(1 - alpha_t), one_shot_density)
                # diff = type_l_sub(density_term, one_shot_term)
                # a = type_l_mult(beta_t * (-delta_t), diff)

                # # noise term
                # b = type_l_mult(
                #     torch.sqrt(beta_t) * np.sqrt(delta_t),
                #     rand_compact_fiber_density(num_nodes, n_max, device=device)
                # )

                # # update
                # density = type_l_add(
                #     density,
                #     type_l_add(a, b)
                # )
                t -= delta_t
                trajectory.append((density, one_shot_seq, one_shot_atom91))
                pbar.update(1)
            pbar.close()


        if return_trajectory:
            return trajectory
        else:
            return trajectory[-1]
