"""SE(3) diffusion methods.

Adapted from https://github.com/jasonkyuyim/se3_diffusion/blob/unsupported_refactor/data/se3_diffuser.py """

from typing import Optional, Tuple

import numpy as np
import torch
from . import so3
from . import r3


class SE3Diffuser:
    def __init__(self):
        self._so3_diffuser = so3.SO3Diffuser()
        self._r3_diffuser = r3.R3Diffuser()

    def forward_marginal(
            self,
            frames_0: Tuple[torch.Tensor, torch.Tensor],
            t: torch.Tensor,
            diffuse_mask: Optional[np.ndarray] = None,
        ):
        """
        Args:
            rigids_0: [..., N] openfold Rigid objects
            t: [...] continuous time in [0, 1].

        Returns:
            rigids_t: [..., N] noised rigid. [..., N, 7] if as_tensor_7 is true.
            trans_score: [..., N, 3] translation score
            rot_score: [..., N, 3, 3] rotation score
            trans_score_norm: [...] translation score norm
            rot_score_norm: [...] rotation score norm
        """
        R_0, trans_0 = frames_0

        R_t, rot_score = self._so3_diffuser.forward_marginal(
            R_0.numpy(), t)
        rot_score_scaling = self._so3_diffuser.score_scaling(t)
        trans_t, trans_score = self._r3_diffuser.forward_marginal(
            trans_0.numpy(), t)
        trans_score_scaling = self._r3_diffuser.score_scaling(t)

        if diffuse_mask is not None:
            # diffuse_mask = torch.tensor(diffuse_mask).to(rot_t.device)
            R_t = self._apply_mask(
                R_t, R_0, diffuse_mask[..., None, None])
            trans_t = self._apply_mask(
                trans_t, trans_0, diffuse_mask[..., None])

            trans_score = self._apply_mask(
                trans_score,
                torch.zeros_like(trans_score),
                diffuse_mask[..., None])
            rot_score = self._apply_mask(
                rot_score,
                torch.zeros_like(rot_score),
                diffuse_mask[..., None, None])
        R_t = torch.tensor(R_t, dtype=torch.float64)
        trans_t = torch.tensor(trans_t, dtype=torch.float64)
        frames_t = (R_t, trans_t)

        return {
            'R_t': R_t,
            'trans_t': trans_t,
            'frames_t': frames_t,
            'trans_score': trans_score,
            'rot_score': rot_score,
            'trans_score_scaling': trans_score_scaling,
            'rot_score_scaling': rot_score_scaling,
        }

    def calc_trans_0(self, trans_score, trans_t, t):
        return self._r3_diffuser.calc_trans_0(trans_score, trans_t, t)

    def calc_trans_score(self, trans_t, trans_0, t):
        return self._r3_diffuser.score(
            trans_t, trans_0, t)

    def calc_rot_score(self, R_t, R_0, t):
        """Returns conditional score as object in tangent space at R_t"""
        return self._so3_diffuser.score(
            R_t.cpu(),
            R_0.cpu(),
            t.cpu()
        )

    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def score(self,
              rigid_0: Tuple[torch.Tensor, torch.Tensor],
              rigid_t: Tuple[torch.Tensor, torch.Tensor],
              t: torch.Tensor):
        R_0, trans_0 = rigid_0
        R_t, trans_t = rigid_t

        rot_score = self._so3_diffuser.score(R_t, R_0, t)
        trans_score = self._r3_diffuser.score(trans_t, trans_0, t)

        return trans_score, rot_score

    def score_scaling(self, t):
        rot_score_scaling = self._so3_diffuser.score_scaling(t)
        trans_score_scaling = self._r3_diffuser.score_scaling(t)
        return rot_score_scaling, trans_score_scaling

    def reverse(
            self,
            rigid_t: Tuple[torch.Tensor, torch.Tensor],
            rot_score: torch.Tensor,
            trans_score: torch.Tensor,
            t: torch.Tensor,
            dt: float,
            diffuse_mask: Optional[torch.Tensor] = None,
            center: bool=True,
            noise_scale: float=1.0,
        ):
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigid_t: [..., N] protein rigid objects at time t.
            rot_score: [..., N, 3, 3] rotation score.
            trans_score: [..., N, 3] translation score.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            center: true to set center of mass to zero after step

        Returns:
            rigid_t_1: [..., N] protein rigid objects at time t-1.
        """
        R_t, trans_t = rigid_t
        R_t_1 = self._so3_diffuser.reverse(
            R_t=R_t,
            score_t=rot_score,
            t=t,
            dt=dt,
            noise_scale=noise_scale,
            )
        trans_t_1 = self._r3_diffuser.reverse(
            x_t=trans_t,
            score_t=trans_score,
            t=t,
            dt=dt,
            center=center,
            noise_scale=noise_scale
            )

        if diffuse_mask is not None:
            trans_t_1 = self._apply_mask(
                trans_t_1, trans_t, diffuse_mask[..., None])
            R_t_1 = self._apply_mask(
                R_t_1, R_t, diffuse_mask[..., None, None])

        return (R_t_1, trans_t_1)

    def sample_ref(
            self,
            n_samples: int,
            impute: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
            diffuse_mask: Optional[np.ndarray]=None
        ):
        """Samples rigids from reference distribution.

        Args:
            n_samples: Number of samples.
            impute: Rigid objects to use as imputation values if either
                translations or rotations are not diffused.
        """
        if impute is not None:
            rot_impute, trans_impute = impute
            assert rot_impute.shape[0] == n_samples
            rot_impute = rot_impute.reshape((n_samples, 3, 3))
            trans_impute = trans_impute.reshape((n_samples, 3))

        if diffuse_mask is not None and impute is None:
            raise ValueError('Must provide imputation values if using a diffusion mask.')

        rot_ref = self._so3_diffuser.sample_ref(n_samples=n_samples)
        trans_ref = self._r3_diffuser.sample_ref(n_samples=n_samples)

        if diffuse_mask is not None:
            rot_ref = self._apply_mask(
                rot_ref, rot_impute, diffuse_mask[..., None, None])
            trans_ref = self._apply_mask(
                trans_ref, trans_impute, diffuse_mask[..., None])
        return {
            'frames_t': (rot_ref, trans_ref)
        }
