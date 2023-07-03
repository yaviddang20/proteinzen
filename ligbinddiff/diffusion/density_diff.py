""" Diffusion modules """
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributions as dist

from ligbinddiff.utils.type_l import type_l_randn_like, type_l_mult, type_l_add, type_l_sub
from ligbinddiff.utils.fiber import rand_fiber_density


class PositiveLinear(nn.Module):
    def forward(self, weight):
        return torch.abs(weight)


class DensityDiffuser(nn.Module):
    """ Diffusion in density space """
    def __init__(self,
                denoiser,
                gamma_0=-7,
                gamma_1=13.5,
                omega=1,
                delta=10,
                r_gamma=1.54  # average C-C bond length?
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
        g_t, g_t_grad = torch.func.jvp(self.snr, (t,), (torch.ones_like(t),)) # B x B
        return g_t, g_t_grad  # B x 1, B x 1

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
               return_trajectory=False):
        device = graph.device
        num_nodes = graph.number_of_nodes
        n_max = self.denoiser.density_n_max
        density = rand_fiber_density(num_nodes, n_max, device=device)

        trajectory = []
        with torch.no_grad():
            delta_t = 1/steps
            t = torch.tensor([[1.]]).to(device)
            trajectory.append((density, None, None))
            while t > 0:
                alpha_t = self.alpha(t)
                snr_t, tau_t = self.snr_derivative(t)
                beta_t = - tau_t * (1 - alpha_t) ** 2 # * -1?

                # drift term
                one_shot_density, one_shot_seq, one_shot_atom91 = self.denoising_step(node_features, edge_features, density, t, graph)
                scaled_one_shot = type_l_mult(torch.sqrt(alpha_t), one_shot_density)
                diff = type_l_sub(scaled_one_shot, density)
                a = type_l_mult(0.5 * beta_t / (1-alpha_t) * delta_t, diff)

                # noise term
                b = type_l_mult(
                    torch.sqrt(beta_t) * np.sqrt(delta_t),
                    rand_fiber_density(num_nodes, n_max, device=device)
                )

                # update
                density = type_l_add(
                    type_l_add(a, b),
                    density
                )
                t -= delta_t
                trajectory.append((density, one_shot_seq, one_shot_atom91))


        if return_trajectory:
            return trajectory
        else:
            return trajectory[-1]
