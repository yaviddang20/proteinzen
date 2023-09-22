""" Noise scheduler for SO3

Adapted from FrameDiff, https://github.com/jasonkyuyim/se3_diffusion/blob/master/data/r3_diffuser.py """
from typing import Optional

import numpy as np
import torch
from torch import nn
from ligbinddiff.diffusion.schedulers import LearnedScheduler


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


class R3Diffuser(nn.Module):
    """VP-SDE diffuser class for translations."""

    def __init__(self,
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

    def sample_ref(self, n_samples: int=1):
        return torch.randn(size=(n_samples, 3))

    def calc_trans_0(self, score_t, x_t, t):
        """
        Args
        ----
        score_t : torch.Tensor, shape=[..., 3]

        x_t : torch.Tensor, shape=[..., 3]

        t : torch.Tensor, shape=[...]

        """
        t = t[:, None]
        beta_t = self.marginal_b_t(t)
        cond_var = self.conditional_var(t)
        return (score_t * cond_var + x_t) / torch.exp(-1/2*beta_t)

    def forward_marginal(self, x_0: torch.Tensor, t: torch.Tensor):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., 3] initial positions in Angstroms.
            t: [...] continuous time in [0, 1].

        Returns:
            x_t: [..., 3] positions at time t in Angstroms.
            score_t: [..., 3] score at time t in Angstroms.
        """
        t_ = t[:, None]
        mu_t = torch.exp(-1/2*self.marginal_b_t(t_)) * x_0
        eps_t = torch.randn(mu_t.shape, device=t.device)
        sigma_t = torch.sqrt(1 - torch.exp(-self.marginal_b_t(t_)))
        x_t = mu_t + eps_t * sigma_t

        score_t = self.score(x_t, x_0, t)
        return x_t, score_t

    def score_scaling(self, t: torch.Tensor):
        return 1 / torch.sqrt(self.conditional_var(t))

    def reverse(
            self,
            *,
            x_t: torch.Tensor,
            score_t: torch.Tensor,
            t: torch.Tensor,
            dt: float,
            mask: Optional[torch.Tensor]=None,
            center: bool=True,
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
        t = t[:, None]
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * torch.randn(size=score_t.shape, device=score_t.device)
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * np.sqrt(np.abs(dt)) * z

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = torch.ones(x_t.shape[:-1], device=x_t.device)
        x_t_1 = x_t - perturb
        if center:
            com = torch.sum(x_t_1, dim=-2) / torch.sum(mask, dim=-1)[..., None]
            x_t_1 -= com[..., None, :]
        return x_t_1

    def conditional_var(self, t):
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I

        """
        return 1 - torch.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t):
        t = t[:, None]
        return -(x_t - torch.exp(-1/2*self.marginal_b_t(t)) * x_0) / self.conditional_var(t)
