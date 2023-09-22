"""
Diffusion in SO(3)

Based on FrameDiff,
https://github.com/jasonkyuyim/se3_diffusion/blob/master/data/so3_diffuser.py,
https://github.com/jasonkyuyim/se3_diffusion/blob/unsupported_refactor/data/so3_diffuser.py """

"""SO(3) diffusion methods."""
from typing import Optional

import numpy as np
import torch
from torch import nn

from . import so3_utils, igso3


class SO3Diffuser(nn.Module):
    def __init__(self,
                 min_sigma=0.1,
                 max_sigma=0.25,
                 num_sigma=1000,
                 num_omega=1000,
                 schedule='logarithmic'):
        super().__init__()
        self.schedule = schedule

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        self.num_sigma = num_sigma

        # Discretize omegas for calculating CDFs. Skip omega=0.
        self.igso3 = igso3.IGSO3(min_t=self.min_sigma**2, max_t=self.max_sigma**2,
                L=2000, num_ts=self.num_sigma, num_omegas=num_omega)

        _score_scaling = torch.sqrt(torch.abs(
            torch.sum(
                self.igso3._d_logf_d_omega**2 * self.igso3._pdf_angle, dim=-1) / torch.sum(
                    self.igso3._pdf_angle, dim=-1)
        )) / np.sqrt(3)
        self._score_scaling = _score_scaling#torch.as_tensor(_score_scaling)

    @property
    def discrete_sigma(self):
        return self.sigma(torch.linspace(0.0, 1.0, self.num_sigma))

    def sigma_idx(self, sigma: torch.Tensor):
        """Calculates the index for discretized sigma during IGSO(3) initialization."""
        return torch.bucketize(sigma, self.discrete_sigma) - 1

    def sigma(self, t: torch.Tensor):
        r"""Extract \sigma(t) corresponding to chosen sigma schedule."""
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        if self.schedule == 'logarithmic':
            max_sigma = torch.tensor(self.max_sigma)
            min_sigma = torch.tensor(self.max_sigma)
            return torch.log(t * torch.exp(max_sigma) + (1 - t) * torch.exp(min_sigma))
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')

    def diffusion_coef(self, t):
        """Compute diffusion coefficient (g_t)."""
        if self.schedule == 'logarithmic':
            max_sigma = torch.tensor(self.max_sigma)
            min_sigma = torch.tensor(self.max_sigma)
            g_t = torch.sqrt(
                2 * (torch.exp(max_sigma) - torch.exp(min_sigma)) * self.sigma(t) / torch.exp(self.sigma(t))
            )
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')
        return g_t

    def sample_ref(self, n_samples: float=1):
        return so3_utils.sample_uniform(n_samples)

    def score(self, R_t: torch.Tensor, R_0: torch.Tensor, t: torch.Tensor,
              eps: float=1e-6):
        """Computes the score of IGSO(3) density

        grad_R log IGSO3(R ; R_0, t)

        Args:
            R: [..., 3, 3] array of rotation matrices at which to compute the score.
            R_0: [..., 3, 3] initial rotations used for IGSO3 location
                parameter.
            t: continuous time in [0, 1].

        Returns:
            [..., 3, 3] score vector in the direction of the sampled vector with
            magnitude given by _score_norms.
        """
        R_0t = torch.einsum('...ji,...jk->...ik', R_0, R_t) # compute R_0^T R
        score = torch.einsum(
                '...ij,...jk->...ik',
                R_0, self.igso3.score(R_0t, t, eps))
        return score

    def score_scaling(self, t: torch.Tensor):
        """Calculates scaling used for scores during trianing."""
        self_device = self._score_scaling.device
        return self._score_scaling[self.igso3.t_idx(t).to(self_device)].to(t.device)

    def forward_marginal(self, R_0: torch.Tensor, t: torch.Tensor):
        """Samples from the forward diffusion process at time index t.

        Args:
            R_0: [..., 3, 3] initial rotations.
            t: continuous time in [0, 1].

        Returns:
            R_t: [..., 3, 3] noised rotation vectors.
            rot_score: [..., 3, 3] score of rot_t as a rotation vector.
        """
        n_samples = np.cumprod(R_0.shape[:-2])[-1]
        assert t.numel() == n_samples
        sampled_rots = self.igso3.sample(t)
        R_t = torch.einsum('...ij,...jk->...ik', sampled_rots, R_0)
        rot_score = self.score(R_t, R_0, t).reshape(R_0.shape)
        return R_t, rot_score

    def reverse(
            self,
            R_t: torch.Tensor,
            score_t: torch.Tensor,
            t: torch.Tensor,
            dt: float,
            mask: Optional[torch.Tensor]=None,
            noise_scale: float=1.0,
            ):
        """Simulates the reverse SDE for 1 step using the Geodesic random walk.

        Args:
            rot_t: [..., 3] current rotations at time t.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            add_noise: set False to set diffusion coefficent to 0.
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] rotation vector at next step.
        """
        # Convert to pytorch tensors
        g_t = self.diffusion_coef(t[:, None, None])
        perturb = ( g_t ** 2 ) * dt * score_t + noise_scale * g_t * np.sqrt(np.abs(dt)) * so3_utils.tangent_gaussian(R_t)
        if mask is not None: perturb *= mask[..., None, None]
        R_t_1 = so3_utils.expmap(R_t, perturb)
        return R_t_1
