""" Noise scheduler for latent reps (SO3_Embedding s)

Adapted from FrameDiff, https://github.com/jasonkyuyim/se3_diffusion/blob/master/data/r3_diffuser.py """
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from ligbinddiff.diffusion.schedulers import LearnedScheduler
from ligbinddiff.utils.so3_embedding import so3_mult, so3_randn_like, so3_ones_like, gen_so3_unop
from ligbinddiff.model.modules.equiformer_v2.so3 import SO3_Embedding


class FixedBetaSchedule:
    def __init__(self,
                 min_beta,
                 max_beta):
        self.min_beta = min_beta
        self.max_beta = max_beta

    def b_t(self, t):
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return self.min_beta + t*(self.max_beta - self.min_beta)

    def marginal_b_t(self, t):
        return t*self.min_beta + (1/2)*(t**2)*(self.max_beta-self.min_beta)


class LearnedBetaSchedule(nn.Module):
    def __init__(self,
                 min_beta,
                 max_beta):
        self._schedule = LearnedScheduler()
        self.min_beta = min_beta
        self.max_beta = max_beta

    def b_t(self, t):
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return self._schedule.beta(t)

    def marginal_b_t(self, t):
        return t*self.min_beta + (1/2)*(t**2)*(self.max_beta-self.min_beta)


class SidechainDiffuser(nn.Module):
    """VP-SDE diffuser class for latent spaces."""

    def __init__(self,
                 lmax_list=None,
                 h_dim=128,
                 min_beta=0.1,
                 max_beta=20,
                 schedule='fixed'):
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        super().__init__()
        if schedule == 'fixed':
            self.scheduler = FixedBetaSchedule(min_beta, max_beta)
        elif schedule == 'learned':
            raise NotImplementedError("I haven't implemented this yet")
        else:
            raise ValueError(f"unknown value of schedule {schedule}")

        self.lmax_list = lmax_list
        self.h_dim = h_dim
        if lmax_list is None:
            self.mode = "scalars"
        else:
            self.mode = "so3"
            self.num_coeffs = sum([(lmax+1) ** 2 for lmax in lmax_list])

    def b_t(self, t):
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return self.scheduler.b_t(t)

    def marginal_b_t(self, t):
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return self.scheduler.marginal_b_t(t)

    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return torch.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t) * x

    def sample_ref(self,
                   n_samples: int=1,
                   impute=None,
                   noising_mask=None,
                   device='cpu',
                   dtype=torch.float):
        if self.mode == 'scalars':
            sample = torch.randn(
                size=(n_samples, self.h_dim),
                device=device)
            if impute is not None and noising_mask is not None:
                sample[noising_mask] = impute[noising_mask]

            return {
                "noised_latent_sidechain": sample
            }

        elif self.mode == 'so3':
            sample_embedding = torch.randn(
                size=(n_samples, self.num_coeffs, self.h_dim),
                device=device)
            if impute is not None and noising_mask is not None:
                sample_embedding[noising_mask] = impute.embedding[noising_mask]

            sample = SO3_Embedding(
                n_samples,
                self.lmax_list,
                self.num_channels,
                device=device,
                dtype=dtype
            )
            sample.set_embedding(sample_embedding)
            return {
                "noised_latent_sidechain": sample
            }

    def calc_x_0(self, score_t: SO3_Embedding, x_t: SO3_Embedding, t) -> SO3_Embedding:
        """
        Args
        ----
        score_t : torch.Tensor, shape=[..., 3]

        x_t : torch.Tensor, shape=[..., 3]

        t : torch.Tensor, shape=[...]

        """
        t = t[:, None, None]
        beta_t = self.marginal_b_t(t)
        cond_var = self.conditional_var(t)
        x_0_embedding = (score_t * cond_var + x_t) / torch.exp(-1/2*beta_t)

        x_0 = x_t.clone()
        x_0.set_embedding(x_0_embedding)
        return x_0

    def forward_marginal(self, x_0, t: torch.Tensor, noising_mask: torch.Tensor):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., 3] initial positions in Angstroms.
            t: [...] continuous time in [0, 1].

        Returns:
            x_t: [..., 3] positions at time t in Angstroms.
            score_t: [..., 3] score at time t in Angstroms.
        """
        if self.mode == 'scalars':
            t_ = t[:, None]
        elif self.mode == 'so3':
            t_ = t[:, None, None]
        else:
            raise ValueError()

        mu_t = torch.exp(-1/2*self.marginal_b_t(t_)) * x_0
        eps_t = torch.randn(mu_t.shape, device=t.device)
        sigma_t = torch.sqrt(1 - torch.exp(-self.marginal_b_t(t_)))

        x_t = mu_t + eps_t * sigma_t
        score_t = self.score(x_t, x_0, t)
        score_scaling = self.score_scaling(t)

        x_t[~noising_mask] = x_0[~noising_mask]
        score_t[~noising_mask] = 0
        return {
            "noised_latent_sidechain": x_t,
            "latent_sidechain_score": score_t,
            "latent_sidechain_score_scaling": score_scaling
        }

    def score_scaling(self, t: torch.Tensor) -> torch.Tensor:
        return 1 / torch.sqrt(self.conditional_var(t))

    def reverse(
            self,
            x_t,
            score_t,
            t: torch.Tensor,
            dt: float,
            mask: Optional[torch.Tensor]=None,
            noise_scale: float=1.0,
        ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        if self.mode == 'scalars':
            t = t[:, None]
        elif self.mode == 'so3':
            t = t[:, None, None]
        else:
            raise ValueError()

        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * torch.randn(size=score_t.shape, device=score_t.device)
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * np.sqrt(np.abs(dt)) * z

        if mask is not None:
            if self.mode == 'scalars':
                perturb *= mask[..., None]
            else:
                perturb *= mask[..., None, None]
        else:
            mask = torch.ones(x_t.shape[:-2])
        x_tm1 = x_t - perturb
        return x_tm1

    def conditional_var(self, t: torch.Tensor) -> torch.Tensor:
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I
        """
        return 1 - torch.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t: torch.Tensor):
        if self.mode == 'scalars':
            t = t[:, None]
        elif self.mode == 'so3':
            t = t[:, None, None]
        else:
            raise ValueError()
        score_t = -(x_t - torch.exp(-1/2*self.marginal_b_t(t)) * x_0) / self.conditional_var(t)
        return score_t
