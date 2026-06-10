import abc
from typing import Optional, Sequence
import functools as fn

from hydra_zen import load_from_yaml
import torch
import torch.nn.functional as F
import numpy as np

from proteinzen.stoch_interp.model_wrapper import ModelForwardWrapper
from proteinzen.stoch_interp import so3_utils
from proteinzen.stoch_interp.multiframe import _trans_diffuse_mask, _rots_diffuse_mask

class DifferentialEquation(abc.ABC):
    @abc.abstractmethod
    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        raise NotImplementedError

    def regularize_rotvec(self, rotvec):
        """ Adapted from Geomstats:

        Regularize a point to be in accordance with convention.

        In 3D, regularize the norm of the rotation vector,
        to be between 0 and pi, following the axis-angle
        representation's convention.

        If the angle is between pi and 2pi,
        the function computes its complementary in 2pi and
        inverts the direction of the rotation axis.

        Parameters
        ----------
        point : array-like, shape=[...,3]
            Point.

        Returns
        -------
        regularized_point : array-like, shape=[..., 3]
            Regularized point.
        """
        theta = torch.linalg.norm(rotvec, axis=-1)
        k = torch.floor(theta / 2.0 / torch.pi)

        # angle in [0;2pi)
        angle = theta - 2 * k * torch.pi

        # this avoids dividing by 0
        theta_eps = torch.where(
            torch.isclose(theta, torch.zeros_like(theta)),
            torch.ones_like(theta),
            theta
        )

        # angle in [0, pi]
        normalized_angle = torch.where(angle <= torch.pi, angle, 2 * torch.pi - angle)
        norm_ratio = torch.where(
            torch.isclose(theta, torch.zeros_like(theta)),
            torch.ones_like(theta),
            normalized_angle / theta_eps)

        # reverse sign if angle was greater than pi
        norm_ratio = torch.where(angle > torch.pi, -norm_ratio, norm_ratio)
        return torch.einsum("...,...i->...i", norm_ratio, rotvec)


def get_g_t(t, g_t_fn):
    if g_t_fn == 'fn1':
        return (1 - t) / (t + 0.1) ** 2
    elif g_t_fn == 'fn2':
        return (1 - t) / (t + 0.01)
    elif g_t_fn == 'fn3':
        return 1 / (t + 0.01)
    elif g_t_fn == 'fn4':
        pi_div_2 = torch.pi / 2
        return pi_div_2 * torch.tan((0.99-t) * pi_div_2)
    elif g_t_fn == 'fn5':
        pi_div_2 = torch.pi / 2
        scale = pi_div_2 * torch.tan((np.sqrt(0.99)-t)**2 * pi_div_2)
        ret = torch.zeros_like(scale)
        scale[t > 0.99] = 0
        ret[torch.isfinite(scale)] = scale
        return ret
    elif g_t_fn == 'fn6':
        pi_div_2 = torch.pi / 2
        scale = pi_div_2 * torch.tan(torch.sqrt(0.98-t) * pi_div_2)
    elif g_t_fn == 'fn7':
        return (1 - t) / (t + 0.2) ** 2
    elif g_t_fn == 'fn8':
        return (1 - t) / (t + 0.05)
    elif g_t_fn == 'fn9':
        return 1 / (1 + torch.e ** (10 * (t - 0.5)))
    elif g_t_fn == 'fn10':
        return 10 * 1.5 * (1 - t)
    elif g_t_fn == 'fn11':
        return (1 - t)
    elif g_t_fn == 'fn12':
        return t
    elif g_t_fn.startswith("poly"):
        exponent = float(g_t_fn[4:])
        return (1 - t) / (
            t + (0.01 ** (1/exponent))
        ) ** exponent
    elif g_t_fn.startswith("tan"):
        exponent = float(g_t_fn[3:])
        pi_div_2 = torch.pi / 2
        scale = pi_div_2 * torch.tan(
            (0.99 ** (1/exponent) - t) ** exponent
            * pi_div_2
        )
        ret = torch.zeros_like(scale)
        scale[t > 0.99] = 0
        ret[torch.isfinite(scale)] = scale
        return ret
    else:
        raise ValueError(f"we don't recognize function specifier {g_t_fn}")



class BaseEulerODEStep(DifferentialEquation):
    def __init__(
        self,
        trans_step_scale: float = 1.0,
        rot_step_scale: float = 1.0,
    ):
        self.trans_step_scale = trans_step_scale
        self.rot_step_scale = rot_step_scale

    def _lambda_t(self, t, lambda_0):
        return (t**2 + (1-t) ** 2) / (t**2 / lambda_0 + (1-t) ** 2)

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        trans_change = self.trans_step_scale * scores_and_vfs['base_trans_vf'] * trans_d_t
        trans_t_2 = trans_t_1 + trans_change
        # in theory this is unnecessary
        # but we're doing it for consistency
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        # step_scale = self._lambda_t(rot_time, lambda_0=self.trans_step_scale)
        rot_change = self.rot_step_scale * scores_and_vfs['base_rot_vf'] * rot_d_t
        mat_t = so3_utils.rotvec_to_rotmat(rot_change)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)
        # in theory this is unnecessary
        # but we're doing it for consistency
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new


class BaseEulerSDEStep(DifferentialEquation):
    def __init__(
        self,
        trans_step_scale: float = 1.0,
        rot_step_scale: float = 1.0,
        trans_noise_scale: float = 0.16,
        rot_noise_scale: float = 0.16,
        trans_g_t_fn: str = 'fn1',
        rot_g_t_fn: str = 'fn1',
        trans_noise_std: float = 16.,
        threshold_rots: bool=False,
        reflecting_diff_rots: bool=False
    ):
        self.trans_step_scale = trans_step_scale
        self.rot_step_scale = rot_step_scale
        self.trans_noise_scale = trans_noise_scale
        self.rot_noise_scale = rot_noise_scale
        self.trans_g_t_fn = trans_g_t_fn
        self.rot_g_t_fn = rot_g_t_fn

        self.trans_noise_std = trans_noise_std
        self.threshold_rots = threshold_rots
        self.reflecting_diff_rots = reflecting_diff_rots

    def g_t(self, t, g_t_fn):
        if g_t_fn == 'fn1':
            return (1 - t) / (t + 0.1) ** 2
        elif g_t_fn == 'fn2':
            return (1 - t) / (t + 0.01)
        elif g_t_fn == 'fn3':
            return 1 / (t + 0.01)
        elif g_t_fn == 'fn4':
            pi_div_2 = torch.pi / 2
            return pi_div_2 * torch.tan((0.99-t) * pi_div_2)
        elif g_t_fn == 'fn5':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan((np.sqrt(0.99)-t)**2 * pi_div_2)
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret
        elif g_t_fn == 'fn6':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(torch.sqrt(0.98-t) * pi_div_2)
        elif g_t_fn == 'fn7':
            return (1 - t) / (t + 0.2) ** 2
        elif g_t_fn == 'fn8':
            return (1 - t) / (t + 0.05)
        elif g_t_fn == 'fn9':
            return 1 / (1 + torch.e ** (10 * (t - 0.5)))
        elif g_t_fn == 'fn10':
            return 10 * 1.5 * (1 - t)
        elif g_t_fn == 'fn11':
            return (1 - t) * (t < 0.5)
        elif g_t_fn == 'fn12':
            return (1 - t) / (t + 0.0625)
        elif g_t_fn.startswith("poly"):
            exponent = float(g_t_fn[4:])
            return (1 - t) / (
                t + (0.01 ** (1/exponent))
            ) ** exponent
        elif g_t_fn.startswith("tan"):
            exponent = float(g_t_fn[3:])
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(
                (0.99 ** (1/exponent) - t) ** exponent
                * pi_div_2
            )
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret

        raise ValueError(f"we don't recogize the g_t fn specifier {g_t_fn}")

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = self.g_t(trans_time, g_t_fn=self.trans_g_t_fn)
        vf_step = self.trans_step_scale * scores_and_vfs['base_trans_vf'] * trans_d_t
        score_step = self.trans_step_scale * g_t[..., None] * scores_and_vfs['base_trans_score'] * trans_d_t
        dW_t = self.trans_noise_std * torch.randn_like(trans_t_1) * torch.sqrt(trans_d_t)
        noise_step = torch.sqrt(2 * g_t[..., None] * self.trans_noise_scale) * dW_t
        total_step = vf_step + (score_step + noise_step) * (trans_time < 0.99)[..., None]
        trans_t_2 = trans_t_1 + total_step

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = self.g_t(rot_time, g_t_fn=self.rot_g_t_fn)
        vf_step = self.rot_step_scale * scores_and_vfs['base_rot_vf'] * rot_d_t
        score_step = self.rot_step_scale * g_t[..., None] * scores_and_vfs['base_rot_score'] * rot_d_t

        dB_rot = torch.randn_like(vf_step) * torch.sqrt(rot_d_t) * 1.5
        noise_step = torch.sqrt(2 * g_t[..., None] * self.rot_noise_scale) * dB_rot

        total_step = vf_step
        total_step = vf_step + (score_step + noise_step) * (rot_time < 0.99)[..., None]
        if self.threshold_rots:
            total_angle = torch.linalg.vector_norm(total_step + 1e-8, dim=-1)
            total_angle = torch.where(total_angle < (1 - rot_time) * torch.pi, total_angle, (1 - rot_time) * torch.pi)
            total_step = F.normalize(total_step, dim=-1) * total_angle[..., None]
        elif self.reflecting_diff_rots:
            vf_angle = torch.linalg.vector_norm(scores_and_vfs['base_rot_vf'] + 1e-8, dim=-1)
            total_angle = torch.linalg.vector_norm(total_step + 1e-8, dim=-1)

            reflect = total_angle > vf_angle + (1 - rot_time) * torch.pi
            reflected_angle = 2 * ((1 - rot_time) * torch.pi + vf_angle) - total_angle
            # double_reflected_angle =

            # 3 * (1 - rot_time) * torch.pi + vf_angle
            # mod_angle = total_angle - (4 * (1 - rot_time) * torch.pi - vf_angle)
            total_angle = torch.where(reflect, reflected_angle, total_angle)
            print(total_angle)
            total_step = F.normalize(total_step, dim=-1) * total_angle[..., None]

        mat_t = so3_utils.rotvec_to_rotmat(total_step)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new


class LowTemperatureSDEStep(DifferentialEquation):
    def __init__(
        self,
        inv_temp: float,
        isothermal: bool = False,
        equilibrate_at_inv_temp: bool=False,
        trans_equil_factor: float = 1.0,
        rot_equil_factor: float = 1.0,
        trans_noise_scale: float = 1.0,
        rot_noise_scale: float = 1.0,
        t_cutoff: float = 0.9
    ):
        self.lambda_0 = inv_temp
        self.isothermal = isothermal
        self.equilibrate_at_inv_temp = equilibrate_at_inv_temp
        self.trans_psi = trans_equil_factor
        self.rot_psi = rot_equil_factor
        self.trans_noise_scale = trans_noise_scale
        self.rot_noise_scale = rot_noise_scale
        self.t_cutoff = t_cutoff

    def _lambda_t_trans(self, t, lambda_0):
        return (t**2 + (1-t) ** 2) / (t**2 / lambda_0 + (1-t) ** 2)

    def _lambda_t_rot(self, t, lambda_0):
        prior_var = 1.5 ** 2
        return (1 + prior_var * (1-t) ** 2) / (1 / lambda_0 + prior_var * (1-t) ** 2)

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        if self.isothermal:
            lambda_t = self.lambda_0
        else:
            lambda_t = self._lambda_t_trans(trans_time, self.lambda_0)[..., None]
        # g_t = (1-trans_time) / (trans_time + 0.0625)
        g_t = (1-trans_time) / (trans_time + 0.01)

        # lower temperature scaling with drift corrected
        trans_vf = lambda_t * scores_and_vfs['base_trans_vf'] - (lambda_t - 1) * trans_t_1 / (trans_time + 0.0625)[..., None]
        vf_step = trans_vf * trans_d_t
        if self.equilibrate_at_inv_temp:
            score_step = self.lambda_0 * g_t[..., None] * scores_and_vfs['base_trans_score'] * trans_d_t
        else:
            score_step = lambda_t * g_t[..., None] * scores_and_vfs['base_trans_score'] * trans_d_t

        # score_step = (lambda_t * 0.5 * g_t**2)[..., None] * scores_and_vfs['base_trans_score'] * trans_d_t
        dW_t = torch.randn_like(trans_t_1) * torch.sqrt(self.trans_noise_scale * trans_d_t)
        noise_step = torch.sqrt(2 * self.trans_psi * g_t[..., None]) * dW_t
        total_step = vf_step + (self.trans_psi * score_step + noise_step) * (trans_time < self.t_cutoff)[..., None]
        trans_t_2 = trans_t_1 + total_step

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def regularize_rotvec(self, rotvec):
        """Regularize a point to be in accordance with convention.
        Adapted from geomstats.

        In 3D, regularize the norm of the rotation vector,
        to be between 0 and pi, following the axis-angle
        representation's convention.

        If the angle is between pi and 2pi,
        the function computes its complementary in 2pi and
        inverts the direction of the rotation axis.

        Parameters
        ----------
        point : array-like, shape=[...,3]
            Point.

        Returns
        -------
        regularized_point : array-like, shape=[..., 3]
            Regularized point.
        """
        theta = torch.linalg.norm(rotvec, axis=-1)
        k = torch.floor(theta / 2.0 / torch.pi)

        # angle in [0;2pi)
        angle = theta - 2 * k * torch.pi

        # this avoids dividing by 0
        theta_eps = torch.where(
            torch.isclose(theta, torch.zeros_like(theta)),
            torch.ones_like(theta),
            theta
        )

        # angle in [0, pi]
        normalized_angle = torch.where(angle <= torch.pi, angle, 2 * torch.pi - angle)
        norm_ratio = torch.where(
            torch.isclose(theta, torch.zeros_like(theta)),
            torch.ones_like(theta),
            normalized_angle / theta_eps)

        # reverse sign if angle was greater than pi
        norm_ratio = torch.where(angle > torch.pi, -norm_ratio, norm_ratio)
        return torch.einsum("...,...i->...i", norm_ratio, rotvec)

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        if self.isothermal:
            lambda_t = self.lambda_0
        else:
            # lambda_t = self._lambda_t_rot(rot_time, self.lambda_0)
            lambda_t = self._lambda_t_trans(rot_time, self.lambda_0)[..., None]

        g_t = torch.sqrt(
            torch.clip(4.5-4.52 * rot_time, min=0)
        )

        vf_step = lambda_t * scores_and_vfs['base_rot_vf'] * rot_d_t
        if self.equilibrate_at_inv_temp:
            score_step = self.lambda_0 * 0.5 * g_t[..., None]**2 * scores_and_vfs['base_rot_score'] * rot_d_t
        else:
            score_step = lambda_t * 0.5 * g_t[..., None]**2 * scores_and_vfs['base_rot_score'] * rot_d_t

        dB_rot = torch.randn_like(vf_step) * torch.sqrt(self.rot_noise_scale * rot_d_t)
        noise_step = g_t[..., None] * np.sqrt(self.rot_psi) * dB_rot
        total_step = vf_step + (self.rot_psi * score_step + noise_step) * (rot_time < self.t_cutoff)[..., None]
        total_step = self.regularize_rotvec(total_step)
        mat_t = so3_utils.rotvec_to_rotmat(total_step)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new


def tuple_sched_to_fn(tuple_sched):
    def _fn(t):
        for (start, end), value in tuple_sched:
            select = (t >= start) & (t < end)
            if select.all():
                return torch.full_like(t, value)
        raise ValueError()
    return _fn


class PiecewiseSDEStep(DifferentialEquation):
    def __init__(
        self,
        yaml_path: str,
        t_cutoff: float = 0.9
    ):
        cfg = load_from_yaml(yaml_path)
        self.lambda_t_fn = tuple_sched_to_fn(cfg.lambda_t)
        self.trans_g_t_fn = tuple_sched_to_fn(cfg.trans_g_t_fn)
        self.rot_g_t_fn = tuple_sched_to_fn(cfg.rot_g_t_fn)
        self.t_cutoff = t_cutoff

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        lambda_t = self.lambda_t_fn(trans_time)[..., None]
        g_t = self.trans_g_t_fn(trans_time)

        # lower temperature scaling with drift corrected
        trans_vf = lambda_t * scores_and_vfs['base_trans_vf'] - (lambda_t - 1) * trans_t_1 / (trans_time + 0.0625)[..., None]
        vf_step = trans_vf * trans_d_t
        score_step = lambda_t * g_t[..., None] * scores_and_vfs['base_trans_score'] * trans_d_t

        # score_step = (lambda_t * 0.5 * g_t**2)[..., None] * scores_and_vfs['base_trans_score'] * trans_d_t
        dW_t = torch.randn_like(trans_t_1) * torch.sqrt(trans_d_t)
        noise_step = torch.sqrt(2 * g_t[..., None]) * dW_t
        total_step = vf_step + (score_step + noise_step) * (trans_time < self.t_cutoff)[..., None]
        trans_t_2 = trans_t_1 + total_step

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def regularize_rotvec(self, rotvec):
        """Regularize a point to be in accordance with convention.
        Adapted from geomstats.

        In 3D, regularize the norm of the rotation vector,
        to be between 0 and pi, following the axis-angle
        representation's convention.

        If the angle is between pi and 2pi,
        the function computes its complementary in 2pi and
        inverts the direction of the rotation axis.

        Parameters
        ----------
        point : array-like, shape=[...,3]
            Point.

        Returns
        -------
        regularized_point : array-like, shape=[..., 3]
            Regularized point.
        """
        theta = torch.linalg.norm(rotvec, axis=-1)
        k = torch.floor(theta / 2.0 / torch.pi)

        # angle in [0;2pi)
        angle = theta - 2 * k * torch.pi

        # this avoids dividing by 0
        theta_eps = torch.where(
            torch.isclose(theta, torch.zeros_like(theta)),
            torch.ones_like(theta),
            theta
        )

        # angle in [0, pi]
        normalized_angle = torch.where(angle <= torch.pi, angle, 2 * torch.pi - angle)
        norm_ratio = torch.where(
            torch.isclose(theta, torch.zeros_like(theta)),
            torch.ones_like(theta),
            normalized_angle / theta_eps)

        # reverse sign if angle was greater than pi
        norm_ratio = torch.where(angle > torch.pi, -norm_ratio, norm_ratio)
        return torch.einsum("...,...i->...i", norm_ratio, rotvec)

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        lambda_t = self.lambda_t_fn(rot_time)[..., None]

        g_t = self.rot_g_t_fn(rot_time)

        vf_step = lambda_t * scores_and_vfs['base_rot_vf'] * rot_d_t
        score_step = lambda_t * 0.5 * g_t[..., None]**2 * scores_and_vfs['base_rot_score'] * rot_d_t

        dB_rot = torch.randn_like(vf_step) * torch.sqrt(rot_d_t)
        noise_step = g_t[..., None] * dB_rot
        total_step = vf_step + (score_step + noise_step) * (rot_time < self.t_cutoff)[..., None]
        total_step = self.regularize_rotvec(total_step)
        mat_t = so3_utils.rotvec_to_rotmat(total_step)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new


def sweep_trans_g_t(t, mode, fn_max):
    if mode == "1-t/t":
        eps = 1 / fn_max
        num = 1-t
        denom = t + eps
        return num / denom
    elif mode == "1-t/t2":
        eps = 1 / np.sqrt(fn_max)
        num = 1 - t
        denom = (t + eps) ** 2
        return num / denom
    elif mode == "1/t":
        eps = 1 / fn_max
        num = 1
        denom = t + eps
        return num / denom
    elif mode == "1/t2":
        eps = 1 / np.sqrt(fn_max)
        num = 1
        denom = (t + eps) ** 2
        return num / denom
    elif mode == "tan":
        eps = 1 / fn_max
        num = torch.sin((1.0 - t) * torch.pi / 2.0)
        denom = torch.cos((1.0 - t) * torch.pi / 2.0)
        return (torch.pi / 2.0) * num / (denom + eps)
    else:
        raise ValueError(f"invalid trans g_t mode {mode}")

def sweep_rot_g_t(t, mode, fn_max):
    if mode == "sqrt":
        g_t = torch.sqrt(1-t.clip(0.999))
    elif mode == "1-t":
        g_t = (1-t)
    elif mode == "1-t2":
        g_t = (1 - t**2)
    elif mode == "(1-t)2":
        g_t  = (1 - t) ** 2
    elif mode == "4t(1-t)":
        g_t = 4 * t * (1-t)
    else:
        raise ValueError(f"invalid rot g_t mode {mode}")

    return g_t * fn_max


def sweep_lambda_t(t, mode, lambda_0):
    if mode == "trans":
        return (t**2 + (1-t) ** 2) / (t**2 / lambda_0 + (1-t) ** 2)
    elif mode == "chroma":
        return 1 / (t**2 / lambda_0 + 1-t ** 2)
    elif mode == "fast_increase":
        return ((1-t)**2 + lambda_0 * (1 - (1-t) ** 2))
    elif mode == "slow_decrease":
        return (t**2 + lambda_0 * (1-t) ** 2)
    elif mode == "linear_decrease":
        return t + lambda_0 * (1-t)
    elif mode == "isothermal":
        return torch.full_like(t, lambda_0)
    else:
        raise ValueError(f"invalid lambda_t mode {mode}")


class SweepLowTemperatureSDEStep(DifferentialEquation):
    def __init__(
        self,
        inv_temp: float,
        trans_g_t_fn: str,
        trans_g_t_fn_max: int,
        rot_g_t_fn: str,
        rot_g_t_fn_max: int,
        lambda_t_fn: str,
        trans_noise_scale: float = 1.0,
        rot_noise_scale: float = 1.0,
        equilibrate_at_inv_temp: bool = False,
        t_cutoff: float = 0.9
    ):
        self.lambda_0 = inv_temp
        self.trans_g_t = fn.partial(sweep_trans_g_t, mode=trans_g_t_fn, fn_max=trans_g_t_fn_max)
        self.rot_g_t = fn.partial(sweep_rot_g_t, mode=rot_g_t_fn, fn_max=rot_g_t_fn_max)
        self.lambda_t = fn.partial(sweep_lambda_t, mode=lambda_t_fn, lambda_0=inv_temp)
        self.equilibrate_at_inv_temp = equilibrate_at_inv_temp
        self.trans_noise_scale = trans_noise_scale
        self.rot_noise_scale = rot_noise_scale
        self.t_cutoff = t_cutoff

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        lambda_t = self.lambda_t(trans_time)[..., None]
        g_t = self.trans_g_t(trans_time)

        # lower temperature scaling with drift corrected
        trans_vf = lambda_t * scores_and_vfs['base_trans_vf'] - (lambda_t - 1) * trans_t_1 / (trans_time + 0.0625)[..., None]
        vf_step = trans_vf * trans_d_t
        if self.equilibrate_at_inv_temp:
            score_step = self.lambda_0 * g_t[..., None] * scores_and_vfs['base_trans_score'] * trans_d_t
        else:
            score_step = lambda_t * g_t[..., None] * scores_and_vfs['base_trans_score'] * trans_d_t

        # score_step = (lambda_t * 0.5 * g_t**2)[..., None] * scores_and_vfs['base_trans_score'] * trans_d_t
        dW_t = torch.randn_like(trans_t_1) * torch.sqrt(self.trans_noise_scale * trans_d_t)
        noise_step = torch.sqrt(2 * g_t[..., None]) * dW_t
        total_step = vf_step + (score_step + noise_step) * (trans_time < self.t_cutoff)[..., None]
        trans_t_2 = trans_t_1 + total_step

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def regularize_rotvec(self, rotvec):
        """Regularize a point to be in accordance with convention.
        Adapted from geomstats.

        In 3D, regularize the norm of the rotation vector,
        to be between 0 and pi, following the axis-angle
        representation's convention.

        If the angle is between pi and 2pi,
        the function computes its complementary in 2pi and
        inverts the direction of the rotation axis.

        Parameters
        ----------
        point : array-like, shape=[...,3]
            Point.

        Returns
        -------
        regularized_point : array-like, shape=[..., 3]
            Regularized point.
        """
        theta = torch.linalg.norm(rotvec, axis=-1)
        k = torch.floor(theta / 2.0 / torch.pi)

        # angle in [0;2pi)
        angle = theta - 2 * k * torch.pi

        # this avoids dividing by 0
        theta_eps = torch.where(
            torch.isclose(theta, torch.zeros_like(theta)),
            torch.ones_like(theta),
            theta
        )

        # angle in [0, pi]
        normalized_angle = torch.where(angle <= torch.pi, angle, 2 * torch.pi - angle)
        norm_ratio = torch.where(
            torch.isclose(theta, torch.zeros_like(theta)),
            torch.ones_like(theta),
            normalized_angle / theta_eps)

        # reverse sign if angle was greater than pi
        norm_ratio = torch.where(angle > torch.pi, -norm_ratio, norm_ratio)
        return torch.einsum("...,...i->...i", norm_ratio, rotvec)

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        lambda_t = self.lambda_t(rot_time)[..., None]
        g_t = self.rot_g_t(rot_time)

        vf_step = lambda_t * scores_and_vfs['base_rot_vf'] * rot_d_t
        if self.equilibrate_at_inv_temp:
            score_step = self.lambda_0 * 0.5 * g_t[..., None]**2 * scores_and_vfs['base_rot_score'] * rot_d_t
        else:
            score_step = lambda_t * 0.5 * g_t[..., None]**2 * scores_and_vfs['base_rot_score'] * rot_d_t

        dB_rot = torch.randn_like(vf_step) * torch.sqrt(self.rot_noise_scale * rot_d_t)
        noise_step = g_t[..., None] * dB_rot
        total_step = vf_step + (score_step + noise_step) * (rot_time < self.t_cutoff)[..., None]
        total_step = self.regularize_rotvec(total_step)
        mat_t = so3_utils.rotvec_to_rotmat(total_step)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new


class ChromaEulerSDEStep(DifferentialEquation):
    def __init__(
        self,
        temp_annealing: float,
        langevin_rate: float,
        use_isothermal_annealing: bool,
        trans_noise_scale: float = 1.0,
        rot_noise_scale: float = 1.0,
        trans_g_t_fn: str = 'fn1',
        rot_g_t_fn: str = 'fn1',
        trans_noise_std: float = 16.
    ):
        self.temp_annealing = temp_annealing
        self.langevin_rate = langevin_rate
        self.use_isothermal_annealing = use_isothermal_annealing

        self.trans_noise_scale = trans_noise_scale
        self.rot_noise_scale = rot_noise_scale
        self.trans_g_t_fn = trans_g_t_fn
        self.rot_g_t_fn = rot_g_t_fn

        self.trans_noise_std = trans_noise_std

    def _lambda_t_trans(self, t, lambda_0):
        return (t**2 + (1-t) ** 2) / (t**2 / lambda_0 + (1-t) ** 2)

    def _lambda_t_rot(self, t, lambda_0):
        uniform_variance = np.pi ** 2 / 3 + 2
        return (1 + uniform_variance * (1-t) ** 2) / (1 / lambda_0 + uniform_variance * (1-t) ** 2)

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = get_g_t(trans_time, g_t_fn=self.trans_g_t_fn)
        lambda_t = self._lambda_t_trans(trans_time, self.temp_annealing)

        if self.use_isothermal_annealing:
            lambda_L = self.temp_annealing
        else:
            lambda_L = lambda_t

        if self.temp_annealing < 1:
            # no annealing if the temp scaling factor is less than 1
            score_scale = (self.langevin_rate * lambda_L / 2)
        else:
            score_scale = (lambda_t + self.langevin_rate * lambda_L / 2 - 1/2)

        f_t = 1 / trans_time.clip(min=0.01)
        # lower temperature scaling with drift corrected
        vf = ((1 - lambda_t) * f_t)[..., None] * trans_t_1 + lambda_t[..., None] * scores_and_vfs['base_trans_vf']
        vf_step = vf * trans_d_t

        score_step = (score_scale * g_t)[..., None] * scores_and_vfs['base_trans_score'] * trans_d_t
        dW_t = self.trans_noise_std * torch.randn_like(trans_t_1) * torch.sqrt(trans_d_t)
        noise_step = torch.sqrt(
            (1 * (self.temp_annealing >= 1) + self.langevin_rate)
            * g_t[..., None]
            * self.trans_noise_scale
        ) * dW_t
        total_step = vf_step + (score_step + noise_step) * (trans_time < 0.99)[..., None]
        trans_t_2 = trans_t_1 + total_step

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = get_g_t(rot_time, g_t_fn=self.trans_g_t_fn)
        lambda_t = self._lambda_t_rot(rot_time, self.temp_annealing)
        if self.use_isothermal_annealing:
            lambda_L = self.temp_annealing
        else:
            lambda_L = lambda_t
        if self.temp_annealing < 1:
            # no annealing if the temp scaling factor is less than 1
            score_scale = (self.langevin_rate * lambda_L / 2)
        else:
            score_scale = (lambda_t + self.langevin_rate * lambda_L / 2 - 1/2)

        vf_step = lambda_t[..., None] * scores_and_vfs['base_rot_vf'] * rot_d_t
        score_step = (score_scale * g_t)[..., None] * scores_and_vfs['base_rot_score'] * rot_d_t
        dB_rot = 1.5 * torch.randn_like(vf_step) * torch.sqrt(rot_d_t)
        noise_step = torch.sqrt(
            (1 * (self.temp_annealing >= 1) + self.langevin_rate)
            * g_t[..., None]
            * self.rot_noise_scale
        ) * dB_rot

        total_step = vf_step + (score_step + noise_step) * (rot_time < 0.99)[..., None]
        mat_t = so3_utils.rotvec_to_rotmat(total_step)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new


class EDMEulerSDEStep(DifferentialEquation):
    def __init__(
        self,
        trans_step_scale: float = 1.5,
        rot_step_scale: float = 1.5,
        trans_noise_scale: float = 0.16,
        rot_noise_scale: float = 0.16,
        trans_g_t_fn: str = 'fn1',
        rot_g_t_fn: str = 'fn1',
        trans_noise_std: float = 16.,
        churn: float = 0.4,
        v2=False
    ):
        self.trans_step_scale = trans_step_scale
        self.rot_step_scale = rot_step_scale
        self.trans_noise_scale = trans_noise_scale
        self.rot_noise_scale = rot_noise_scale
        self.trans_g_t_fn = trans_g_t_fn
        self.rot_g_t_fn = rot_g_t_fn

        self.trans_noise_std = trans_noise_std
        self.churn = churn
        self.v2 = v2

    def g_t(self, t, g_t_fn):
        if g_t_fn == 'fn1':
            return (1 - t) / (t + 0.1) ** 2
        elif g_t_fn == 'fn2':
            return (1 - t) / (t + 0.01)
        elif g_t_fn == 'fn3':
            return 1 / (t + 0.01)
        elif g_t_fn == 'fn4':
            pi_div_2 = torch.pi / 2
            return pi_div_2 * torch.tan((0.99-t) * pi_div_2)
        elif g_t_fn == 'fn5':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan((np.sqrt(0.99)-t)**2 * pi_div_2)
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret
        elif g_t_fn == 'fn6':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(torch.sqrt(0.98-t) * pi_div_2)
        elif g_t_fn == 'fn7':
            return (1 - t) / (t + 0.2) ** 2
        elif g_t_fn == 'fn8':
            return (1 - t) / (t + 0.05)
        elif g_t_fn == 'fn9':
            return 1 / (1 + torch.e ** (10 * (t - 0.5)))
        elif g_t_fn == 'fn10':
            return 1.5 * (1 - t)
        elif g_t_fn == 'fn11':
            return 1.5 * (1 - t)
        elif g_t_fn == 'fn12':
            return (1 - t) / (t + 0.0625)
        elif g_t_fn.startswith("poly"):
            exponent = float(g_t_fn[4:])
            return (1 - t) / (
                t + (0.01 ** (1/exponent))
            ) ** exponent
        elif g_t_fn.startswith("tan"):
            exponent = float(g_t_fn[3:])
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(
                (0.99 ** (1/exponent) - t) ** exponent
                * pi_div_2
            )
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret
        elif g_t_fn == "zero":
            return torch.zeros_like(t)

        raise ValueError(f"we don't recogize the g_t fn specifier {g_t_fn}")

    def trans_churn(
        self,
        trans_t,
        trans_time,
        trans_d_t,
        noising_mask,
        seed=None,
        noise_scale_override=None
    ):
        t = trans_time
        t_hat = torch.clamp(trans_time - self.churn * trans_d_t, min=0)
        d_t_hat = trans_d_t * (1 + self.churn)

        if noise_scale_override is not None:
            noise_scale = noise_scale_override
        else:
            noise_scale = self.trans_noise_scale

        if self.v2:
            if t_hat > 0:
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                trans_0 = torch.randn_like(trans_t) * self.trans_noise_std
                reverse_trans_vf = (trans_0 - trans_t) / t_hat
                trans_t_hat = trans_t + self.churn * trans_d_t * reverse_trans_vf
            else:
                trans_t_hat = trans_t
        noise_scale = torch.sqrt(
            2 * self.g_t(t_hat, self.trans_g_t_fn) * noise_scale - 2 * self.g_t(t, self.trans_g_t_fn) * noise_scale
        )
        if seed is not None:
            torch.manual_seed(seed)
        noise = torch.randn_like(trans_t) * noise_scale * 10
        trans_t_hat = trans_t + noise * (trans_time < 0.99)[..., None]

        trans_t_hat = _trans_diffuse_mask(trans_t_hat, trans_t, noising_mask)
        return t_hat, d_t_hat, trans_t_hat

    def rot_churn(
        self,
        rotmats_t,
        rot_time,
        rot_d_t,
        noising_mask,
        seed=None,
        noise_scale_override=None
    ):
        t = rot_time
        t_hat = torch.clamp(rot_time - self.churn * rot_d_t, min=0)
        d_t_hat = rot_d_t * (1 + self.churn)

        if noise_scale_override is not None:
            noise_scale = noise_scale_override
        else:
            noise_scale = self.rot_noise_scale

        if self.v2:
            from proteinzen.stoch_interp.multiframe import _uniform_so3
            if t_hat > 0:
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                rotmats_0 = _uniform_so3(rotmats_t.shape[0], rotmats_t.shape[1], rotmats_t.device)
                rotmats_t_hat = so3_utils.geodesic_t(rot_d_t * self.churn / t_hat, rotmats_0, rotmats_t)
            else:
                rotmats_t_hat = rotmats_t

        noise_scale = torch.sqrt(
            2 * self.g_t(t_hat, self.rot_g_t_fn) * noise_scale - 2 * self.g_t(t, self.rot_g_t_fn) * noise_scale
        )
        if seed is not None:
            torch.manual_seed(seed)
        dB_rot = torch.randn(rotmats_t.shape[:-1], device=rotmats_t.device) * noise_scale
        rotmats_t_hat = so3_utils.rot_mult(
            rotmats_t,
            so3_utils.rotvec_to_rotmat(dB_rot)
        )
        noising_mask = noising_mask & (rot_time < 0.99)[..., None]
        rotmats_t_hat = _rots_diffuse_mask(rotmats_t_hat, rotmats_t, noising_mask)
        return t_hat, d_t_hat, rotmats_t_hat

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = self.g_t(trans_time, g_t_fn=self.trans_g_t_fn)
        vf_step = self.trans_step_scale * scores_and_vfs['base_trans_vf'] * trans_d_t
        score_step = self.trans_step_scale * g_t[..., None] * scores_and_vfs['base_trans_score'] * trans_d_t
        total_step = vf_step + score_step * (trans_time < 0.99)[..., None]
        trans_t_2 = trans_t_1 + total_step

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = self.g_t(rot_time, g_t_fn=self.rot_g_t_fn)
        vf_step = scores_and_vfs['base_rot_vf'].float() * rot_d_t
        score_step = g_t[..., None] * scores_and_vfs['base_rot_score'] * rot_d_t
        total_step = vf_step + score_step * (rot_time < 0.99)[..., None]
        mat_t = so3_utils.rotvec_to_rotmat(total_step * self.rot_step_scale)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new


class EDMChromaSDEStep(DifferentialEquation):
    def __init__(
        self,
        temp_annealing: float,
        langevin_rate: float,
        use_isothermal_annealing: bool,
        trans_noise_scale: float = 0.16,
        rot_noise_scale: float = 0.16,
        trans_g_t_fn: str = 'fn1',
        rot_g_t_fn: str = 'fn1',
        trans_noise_std: float = 16.,
        churn: float = 0.4,
        v2=False
    ):
        self.temp_annealing = temp_annealing
        self.langevin_rate = langevin_rate
        self.use_isothermal_annealing = use_isothermal_annealing
        self.trans_noise_scale = trans_noise_scale
        self.rot_noise_scale = rot_noise_scale
        self.trans_g_t_fn = trans_g_t_fn
        self.rot_g_t_fn = rot_g_t_fn

        self.trans_noise_std = trans_noise_std
        self.churn = churn
        self.v2 = v2

    def _lambda_t_trans(self, t, lambda_0):
        return (t**2 + (1-t) ** 2) / (t**2 / lambda_0 + (1-t) ** 2)

    def _lambda_t_rot(self, t, lambda_0):
        uniform_variance = np.pi ** 2 / 3 + 2
        return (1 + uniform_variance * (1-t) ** 2) / (1 / lambda_0 + uniform_variance * (1-t) ** 2)

    def g_t(self, t, g_t_fn):
        if g_t_fn == 'fn1':
            return (1 - t) / (t + 0.1) ** 2
        elif g_t_fn == 'fn2':
            return (1 - t) / (t + 0.01)
        elif g_t_fn == 'fn3':
            return 1 / (t + 0.01)
        elif g_t_fn == 'fn4':
            pi_div_2 = torch.pi / 2
            return pi_div_2 * torch.tan((0.99-t) * pi_div_2)
        elif g_t_fn == 'fn5':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan((np.sqrt(0.99)-t)**2 * pi_div_2)
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret
        elif g_t_fn == 'fn6':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(torch.sqrt(0.98-t) * pi_div_2)
        elif g_t_fn == 'fn7':
            return (1 - t) / (t + 0.2) ** 2
        elif g_t_fn == 'fn8':
            return (1 - t) / (t + 0.05)
        elif g_t_fn == 'fn9':
            return 1 / (1 + torch.e ** (10 * (t - 0.5)))
        elif g_t_fn == 'fn10':
            return 1.5 * (1 - t)
        elif g_t_fn == 'fn11':
            return 1.5 * (1 - t)
        elif g_t_fn == 'fn12':
            return (1 - t) / (t + 0.0625)
        elif g_t_fn.startswith("poly"):
            exponent = float(g_t_fn[4:])
            return (1 - t) / (
                t + (0.01 ** (1/exponent))
            ) ** exponent
        elif g_t_fn.startswith("tan"):
            exponent = float(g_t_fn[3:])
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(
                (0.99 ** (1/exponent) - t) ** exponent
                * pi_div_2
            )
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret

        raise ValueError(f"we don't recogize the g_t fn specifier {g_t_fn}")

    def trans_churn(
        self,
        trans_t,
        trans_time,
        trans_d_t,
        noising_mask,
        seed=None,
        noise_scale_override=None
    ):
        t = trans_time
        t_hat = torch.clamp(trans_time - self.churn * trans_d_t, min=0)
        d_t_hat = trans_d_t * (1 + self.churn)

        if noise_scale_override is not None:
            noise_scale = noise_scale_override
        else:
            noise_scale = self.trans_noise_scale

        if self.v2:
            if t_hat > 0:
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                trans_0 = torch.randn_like(trans_t) * self.trans_noise_std
                reverse_trans_vf = (trans_0 - trans_t) / t_hat
                trans_t_hat = trans_t + self.churn * trans_d_t * reverse_trans_vf
            else:
                trans_t_hat = trans_t

        noise_scale = torch.sqrt(
            2 * self.g_t(t_hat, self.trans_g_t_fn) * noise_scale - 2 * self.g_t(t, self.trans_g_t_fn) * noise_scale
        )
        noise_scale = noise_scale * np.sqrt(1 * (self.temp_annealing >= 1) + self.langevin_rate)
        if seed is not None:
            torch.manual_seed(seed)
        noise = torch.randn_like(trans_t) * noise_scale * 10
        trans_t_hat = trans_t + noise * (trans_time < 0.99)[..., None]

        trans_t_hat = _trans_diffuse_mask(trans_t_hat, trans_t, noising_mask)
        return t_hat, d_t_hat, trans_t_hat

    def rot_churn(
        self,
        rotmats_t,
        rot_time,
        rot_d_t,
        noising_mask,
        seed=None,
        noise_scale_override=None
    ):
        t = rot_time
        t_hat = torch.clamp(rot_time - self.churn * rot_d_t, min=0)
        d_t_hat = rot_d_t * (1 + self.churn)

        if noise_scale_override is not None:
            noise_scale = noise_scale_override
        else:
            noise_scale = self.rot_noise_scale

        if self.v2:
            from proteinzen.stoch_interp.multiframe import _uniform_so3
            if t_hat > 0:
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                rotmats_0 = _uniform_so3(rotmats_t.shape[0], rotmats_t.shape[1], rotmats_t.device)
                rotmats_t_hat = so3_utils.geodesic_t(rot_d_t * self.churn / t_hat, rotmats_0, rotmats_t)
            else:
                rotmats_t_hat = rotmats_t

        noise_scale = torch.sqrt(
            2 * self.g_t(t_hat, self.rot_g_t_fn) * noise_scale - 2 * self.g_t(t, self.rot_g_t_fn) * noise_scale
        )
        noise_scale = noise_scale * np.sqrt(1 * (self.temp_annealing >= 1) + self.langevin_rate)
        if seed is not None:
            torch.manual_seed(seed)
        dB_rot = torch.randn(rotmats_t.shape[:-1], device=rotmats_t.device) * noise_scale
        rotmats_t_hat = so3_utils.rot_mult(
            rotmats_t,
            so3_utils.rotvec_to_rotmat(dB_rot)
        )
        noising_mask = noising_mask & (rot_time < 0.99)[..., None]
        rotmats_t_hat = _rots_diffuse_mask(rotmats_t_hat, rotmats_t, noising_mask)
        return t_hat, d_t_hat, rotmats_t_hat

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = get_g_t(trans_time, g_t_fn=self.trans_g_t_fn)
        lambda_t = self._lambda_t_trans(trans_time, self.temp_annealing)
        if self.use_isothermal_annealing:
            lambda_L = self.temp_annealing
        else:
            lambda_L = lambda_t

        if self.temp_annealing < 1:
            # no annealing if the temp scaling factor is less than 1
            score_scale = (self.langevin_rate * lambda_L / 2)
        else:
            score_scale = (lambda_t + self.langevin_rate * lambda_L / 2 - 1/2)

        f_t = 1 / trans_time.clip(min=0.01)
        # lower temperature scaling with drift corrected
        vf = ((1 - lambda_t) * f_t)[..., None] * trans_t_1 + lambda_t[..., None] * scores_and_vfs['base_trans_vf']
        vf_step = vf * trans_d_t

        score_step = (score_scale * g_t)[..., None] * scores_and_vfs['base_trans_score'] * trans_d_t
        total_step = vf_step + score_step * (trans_time < 0.99)[..., None]
        trans_t_2 = trans_t_1 + total_step

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = get_g_t(rot_time, g_t_fn=self.rot_g_t_fn)
        lambda_t = self._lambda_t_rot(rot_time, self.temp_annealing)
        if self.use_isothermal_annealing:
            lambda_L = self.temp_annealing
        else:
            lambda_L = lambda_t
        if self.temp_annealing < 1:
            # no annealing if the temp scaling factor is less than 1
            score_scale = (self.langevin_rate * lambda_L / 2)
        else:
            score_scale = (lambda_t + self.langevin_rate * lambda_L / 2 - 1/2)
        vf_step =  scores_and_vfs['base_rot_vf'] * rot_d_t
        score_step = (score_scale * g_t)[..., None] * scores_and_vfs['base_rot_score'] * rot_d_t
        total_step = vf_step + score_step * (rot_time < 0.99)[..., None]
        mat_t = so3_utils.rotvec_to_rotmat(total_step)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new

class GradientGuidedEulerODEStep(DifferentialEquation):
    def __init__(
        self,
        trans_step_scale: float,
        trans_guidance_scale: float,
        rot_step_scale: float,
        rot_guidance_scale: float,
    ):
        self.trans_step_scale = trans_step_scale
        self.trans_guidance_scale = trans_guidance_scale
        self.rot_step_scale = rot_step_scale
        self.rot_guidance_scale = rot_guidance_scale

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        base_vf_step = self.trans_step_scale * scores_and_vfs['base_trans_vf'] * trans_d_t
        gradient_vf_step = self.trans_step_scale * scores_and_vfs['gradient_trans_vf'] * trans_d_t
        # gradient_vf_step = gradient_vf_step  * (trans_time > 0.01)[..., None]
        print(
            trans_time, torch.norm(scores_and_vfs['gradient_trans_vf'], dim=-1).mean()
        )
        trans_t_2 = trans_t_1 + base_vf_step + self.trans_guidance_scale * gradient_vf_step
        # in theory this is unnecessary
        # but we're doing it for consistency
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        base_vf_step = self.rot_step_scale * scores_and_vfs['base_rot_vf'] * rot_d_t
        gradient_vf_step = self.rot_step_scale * scores_and_vfs['gradient_rot_vf'] * rot_d_t
        # gradient_vf_step = gradient_vf_step  * (rot_time > 0.01)[..., None]

        grad_vf_norm = torch.norm(scores_and_vfs['gradient_rot_vf'], dim=-1)
        num_nan = torch.isnan(grad_vf_norm).sum()

        print(
            rot_time, torch.norm(scores_and_vfs['gradient_rot_vf'], dim=-1).mean(), num_nan, grad_vf_norm.numel()
        )
        mat_t = so3_utils.rotvec_to_rotmat(base_vf_step + self.rot_guidance_scale * gradient_vf_step)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)
        # in theory this is unnecessary
        # but we're doing it for consistency
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new


class EulerODEStepWithGradientLangevinChurn(DifferentialEquation):
    def __init__(
        self,
        trans_step_scale: float = 1.,
        rot_step_scale: float = 1.,
        trans_noise_scale: float = 0.16,
        rot_noise_scale: float = 0.16,
        trans_g_t_fn: str = 'fn1',
        rot_g_t_fn: str = 'fn1',
        trans_noise_std: float = 16.
    ):
        self.trans_step_scale = trans_step_scale
        self.rot_step_scale = rot_step_scale
        self.trans_noise_scale = trans_noise_scale
        self.rot_noise_scale = rot_noise_scale
        self.trans_g_t_fn = trans_g_t_fn
        self.rot_g_t_fn = rot_g_t_fn

        self.trans_noise_std = trans_noise_std

    def g_t(self, t, g_t_fn):
        if g_t_fn == 'fn1':
            return (1 - t) / (t + 0.1) ** 2
        elif g_t_fn == 'fn2':
            return (1 - t) / (t + 0.01)
        elif g_t_fn == 'fn3':
            return 1 / (t + 0.01)
        elif g_t_fn == 'fn4':
            pi_div_2 = torch.pi / 2
            return pi_div_2 * torch.tan((0.99-t) * pi_div_2)
        elif g_t_fn == 'fn5':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan((np.sqrt(0.99)-t)**2 * pi_div_2)
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret
        elif g_t_fn == 'fn6':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(torch.sqrt(0.98-t) * pi_div_2)
        elif g_t_fn == 'fn7':
            return (1 - t) / (t + 0.2) ** 2
        elif g_t_fn == 'fn8':
            return (1 - t) / (t + 0.05)
        elif g_t_fn == 'fn9':
            return 1 / (1 + torch.e ** (10 * (t - 0.5)))
        elif g_t_fn == 'fn10':
            return 10 * 1.5 * (1 - t)
        elif g_t_fn.startswith("poly"):
            exponent = float(g_t_fn[4:])
            return (1 - t) / (
                t + (0.01 ** (1/exponent))
            ) ** exponent
        elif g_t_fn.startswith("tan"):
            exponent = float(g_t_fn[3:])
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(
                (0.99 ** (1/exponent) - t) ** exponent
                * pi_div_2
            )
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret

        raise ValueError(f"we don't recogize the g_t fn specifier {g_t_fn}")

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = self.g_t(trans_time, g_t_fn=self.trans_g_t_fn)
        vf_step = self.trans_step_scale * scores_and_vfs['base_trans_vf'] * trans_d_t
        score_step = self.trans_step_scale * g_t[..., None] * scores_and_vfs['gradient_trans_score'] * trans_d_t
        dW_t = self.trans_noise_std * torch.randn_like(trans_t_1) * torch.sqrt(trans_d_t)
        noise_step = torch.sqrt(2 * g_t[..., None] * self.trans_noise_scale) * dW_t
        total_step = vf_step + (score_step + noise_step) * (trans_time < 0.99)[..., None]
        trans_t_2 = trans_t_1 + total_step

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = self.g_t(rot_time, g_t_fn=self.rot_g_t_fn)
        vf_step = self.rot_step_scale * scores_and_vfs['base_rot_vf'] * rot_d_t
        score_step = self.rot_step_scale * g_t[..., None] * scores_and_vfs['gradient_rot_score'] * rot_d_t
        dB_rot = torch.randn_like(vf_step) * torch.sqrt(rot_d_t)
        noise_step = torch.sqrt(2 * g_t[..., None] * self.rot_noise_scale) * dB_rot

        total_step = vf_step + (score_step + noise_step) * (rot_time < 0.99)[..., None]
        mat_t = so3_utils.rotvec_to_rotmat(total_step)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new


class EulerSDEStepWithGradientLangevinChurn(DifferentialEquation):
    def __init__(
        self,
        trans_step_scale: float = 1.,
        rot_step_scale: float = 1.,
        trans_noise_scale: float = 0.16,
        rot_noise_scale: float = 0.16,
        trans_g_t_fn: str = 'fn1',
        rot_g_t_fn: str = 'fn1',
        trans_guidance_scale: float = 1.0,
        rot_guidance_scale: float = 1.0,
        trans_noise_std: float = 16.
    ):
        self.trans_step_scale = trans_step_scale
        self.rot_step_scale = rot_step_scale
        self.trans_noise_scale = trans_noise_scale
        self.rot_noise_scale = rot_noise_scale
        self.trans_g_t_fn = trans_g_t_fn
        self.rot_g_t_fn = rot_g_t_fn
        self.trans_guidance_scale = trans_guidance_scale
        self.rot_guidance_scale = rot_guidance_scale

        self.trans_noise_std = trans_noise_std

    def g_t(self, t, g_t_fn):
        if g_t_fn == 'fn1':
            return (1 - t) / (t + 0.1) ** 2
        elif g_t_fn == 'fn2':
            return (1 - t) / (t + 0.01)
        elif g_t_fn == 'fn3':
            return 1 / (t + 0.01)
        elif g_t_fn == 'fn4':
            pi_div_2 = torch.pi / 2
            return pi_div_2 * torch.tan((0.99-t) * pi_div_2)
        elif g_t_fn == 'fn5':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan((np.sqrt(0.99)-t)**2 * pi_div_2)
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret
        elif g_t_fn == 'fn6':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(torch.sqrt(0.98-t) * pi_div_2)
        elif g_t_fn == 'fn7':
            return (1 - t) / (t + 0.2) ** 2
        elif g_t_fn == 'fn8':
            return (1 - t) / (t + 0.05)
        elif g_t_fn == 'fn9':
            return 1 / (1 + torch.e ** (10 * (t - 0.5)))
        elif g_t_fn == 'fn10':
            return 10 * 1.5 * (1 - t)
        elif g_t_fn.startswith("poly"):
            exponent = float(g_t_fn[4:])
            return (1 - t) / (
                t + (0.01 ** (1/exponent))
            ) ** exponent
        elif g_t_fn.startswith("tan"):
            exponent = float(g_t_fn[3:])
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(
                (0.99 ** (1/exponent) - t) ** exponent
                * pi_div_2
            )
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret

        raise ValueError(f"we don't recogize the g_t fn specifier {g_t_fn}")

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = self.g_t(trans_time, g_t_fn=self.trans_g_t_fn)
        vf_step = self.trans_step_scale * scores_and_vfs['base_trans_vf'] * trans_d_t
        score = self.trans_guidance_scale * scores_and_vfs['gradient_trans_score'] + scores_and_vfs['base_trans_score']
        score_step = self.trans_step_scale * g_t[..., None] * score * trans_d_t
        dW_t = self.trans_noise_std * torch.randn_like(trans_t_1) * torch.sqrt(trans_d_t)
        noise_step = torch.sqrt(2 * g_t[..., None] * self.trans_noise_scale) * dW_t
        total_step = vf_step + (score_step + noise_step) * (trans_time < 0.99)[..., None]
        trans_t_2 = trans_t_1 + total_step

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = self.g_t(rot_time, g_t_fn=self.rot_g_t_fn)
        vf_step = self.rot_step_scale * scores_and_vfs['base_rot_vf'] * rot_d_t
        score = self.rot_guidance_scale * scores_and_vfs['gradient_rot_score'] + scores_and_vfs['base_rot_score']
        score_step = self.rot_step_scale * g_t[..., None] * score * rot_d_t
        dB_rot = torch.randn_like(vf_step) * torch.sqrt(rot_d_t)
        noise_step = torch.sqrt(2 * g_t[..., None] * self.rot_noise_scale) * dB_rot

        total_step = vf_step + (score_step + noise_step) * (rot_time < 0.99)[..., None]
        mat_t = so3_utils.rotvec_to_rotmat(total_step)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new

class GradientGuidedEulerSDEStepWithGradientLangevinChurn(DifferentialEquation):
    def __init__(
        self,
        trans_step_scale: float = 1.,
        rot_step_scale: float = 1.,
        trans_noise_scale: float = 0.16,
        rot_noise_scale: float = 0.16,
        trans_g_t_fn: str = 'fn1',
        rot_g_t_fn: str = 'fn1',
        trans_guidance_scale: float = 1.0,
        rot_guidance_scale: float = 1.0,
        trans_noise_std: float = 16.
    ):
        self.trans_step_scale = trans_step_scale
        self.rot_step_scale = rot_step_scale
        self.trans_noise_scale = trans_noise_scale
        self.rot_noise_scale = rot_noise_scale
        self.trans_g_t_fn = trans_g_t_fn
        self.rot_g_t_fn = rot_g_t_fn
        self.trans_guidance_scale = trans_guidance_scale
        self.rot_guidance_scale = rot_guidance_scale

        self.trans_noise_std = trans_noise_std

    def g_t(self, t, g_t_fn):
        if g_t_fn == 'fn1':
            return (1 - t) / (t + 0.1) ** 2
        elif g_t_fn == 'fn2':
            return (1 - t) / (t + 0.01)
        elif g_t_fn == 'fn3':
            return 1 / (t + 0.01)
        elif g_t_fn == 'fn4':
            pi_div_2 = torch.pi / 2
            return pi_div_2 * torch.tan((0.99-t) * pi_div_2)
        elif g_t_fn == 'fn5':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan((np.sqrt(0.99)-t)**2 * pi_div_2)
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret
        elif g_t_fn == 'fn6':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(torch.sqrt(0.98-t) * pi_div_2)
        elif g_t_fn == 'fn7':
            return (1 - t) / (t + 0.2) ** 2
        elif g_t_fn == 'fn8':
            return (1 - t) / (t + 0.05)
        elif g_t_fn == 'fn9':
            return 1 / (1 + torch.e ** (10 * (t - 0.5)))
        elif g_t_fn == 'fn10':
            return 10 * 1.5 * (1 - t)
        elif g_t_fn.startswith("poly"):
            exponent = float(g_t_fn[4:])
            return (1 - t) / (
                t + (0.01 ** (1/exponent))
            ) ** exponent
        elif g_t_fn.startswith("tan"):
            exponent = float(g_t_fn[3:])
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(
                (0.99 ** (1/exponent) - t) ** exponent
                * pi_div_2
            )
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret

        raise ValueError(f"we don't recogize the g_t fn specifier {g_t_fn}")

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = self.g_t(trans_time, g_t_fn=self.trans_g_t_fn)
        vf = self.trans_guidance_scale * scores_and_vfs['gradient_trans_vf'] + scores_and_vfs['base_trans_vf']
        vf_step = self.trans_step_scale * vf * trans_d_t
        score = self.trans_guidance_scale * scores_and_vfs['gradient_trans_score'] + scores_and_vfs['base_trans_score']
        score_step = self.trans_step_scale * g_t[..., None] * score * trans_d_t
        dW_t = self.trans_noise_std * torch.randn_like(trans_t_1) * torch.sqrt(trans_d_t)
        noise_step = torch.sqrt(2 * g_t[..., None] * self.trans_noise_scale) * dW_t
        total_step = vf_step + (score_step + noise_step) * (trans_time < 0.99)[..., None]
        trans_t_2 = trans_t_1 + total_step

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = self.g_t(rot_time, g_t_fn=self.rot_g_t_fn)
        vf = self.rot_guidance_scale * scores_and_vfs['gradient_rot_vf'] + scores_and_vfs['base_rot_vf']
        vf_step = self.rot_step_scale * scores_and_vfs['base_rot_vf'] * rot_d_t
        score = self.rot_guidance_scale * scores_and_vfs['gradient_rot_score'] + scores_and_vfs['base_rot_score']
        score_step = self.rot_step_scale * g_t[..., None] * score * rot_d_t
        dB_rot = torch.randn_like(vf_step) * torch.sqrt(rot_d_t)
        noise_step = torch.sqrt(2 * g_t[..., None] * self.rot_noise_scale) * dB_rot

        total_step = vf_step + (score_step + noise_step) * (rot_time < 0.99)[..., None]
        mat_t = so3_utils.rotvec_to_rotmat(total_step)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new

class LangevinSDEStep(DifferentialEquation):
    def __init__(
        self,
        trans_step_scale: float = 1.5,
        rot_step_scale: float = 1.5,
        trans_noise_scale: float = 1.0,
        rot_noise_scale: float = 1.0,
        trans_g_t_fn: str = 'fn2',
        rot_g_t_fn: str = 'fn10',
        trans_noise_std: float = 16.
    ):
        self.trans_step_scale = trans_step_scale
        self.rot_step_scale = rot_step_scale
        self.trans_noise_scale = trans_noise_scale
        self.rot_noise_scale = rot_noise_scale
        self.trans_g_t_fn = trans_g_t_fn
        self.rot_g_t_fn = rot_g_t_fn

        self.trans_noise_std = trans_noise_std

    def g_t(self, t, g_t_fn):
        if g_t_fn == 'fn1':
            return (1 - t) / (t + 0.1) ** 2
        elif g_t_fn == 'fn2':
            return (1 - t) / (t + 0.01)
        elif g_t_fn == 'fn3':
            return 1 / (t + 0.01)
        elif g_t_fn == 'fn4':
            pi_div_2 = torch.pi / 2
            return pi_div_2 * torch.tan((0.99-t) * pi_div_2)
        elif g_t_fn == 'fn5':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan((np.sqrt(0.99)-t)**2 * pi_div_2)
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret
        elif g_t_fn == 'fn6':
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(torch.sqrt(0.98-t) * pi_div_2)
        elif g_t_fn == 'fn7':
            return (1 - t) / (t + 0.2) ** 2
        elif g_t_fn == 'fn8':
            return (1 - t) / (t + 0.05)
        elif g_t_fn == 'fn9':
            return 1 / (1 + torch.e ** (10 * (t - 0.5)))
        elif g_t_fn == 'fn10':
            return 10 * 1.5 * (1 - t)
        elif g_t_fn.startswith("poly"):
            exponent = float(g_t_fn[4:])
            return (1 - t) / (
                t + (0.01 ** (1/exponent))
            ) ** exponent
        elif g_t_fn.startswith("tan"):
            exponent = float(g_t_fn[3:])
            pi_div_2 = torch.pi / 2
            scale = pi_div_2 * torch.tan(
                (0.99 ** (1/exponent) - t) ** exponent
                * pi_div_2
            )
            ret = torch.zeros_like(scale)
            scale[t > 0.99] = 0
            ret[torch.isfinite(scale)] = scale
            return ret

        raise ValueError(f"we don't recogize the g_t fn specifier {g_t_fn}")

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = self.g_t(trans_time, g_t_fn=self.trans_g_t_fn)
        vf_step = self.trans_step_scale * scores_and_vfs['base_trans_vf'] * trans_d_t
        score_step = self.trans_step_scale * g_t[..., None] * scores_and_vfs['base_trans_score'] * trans_d_t
        dW_t = self.trans_noise_std * torch.randn_like(trans_t_1) * torch.sqrt(trans_d_t)
        noise_step = torch.sqrt(2 * g_t[..., None] * self.trans_noise_scale) * dW_t
        total_step = vf_step * (trans_time >= 0.99)[..., None] + (score_step + noise_step) * (trans_time < 0.99)[..., None]
        trans_t_2 = trans_t_1 + total_step

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = self.g_t(rot_time, g_t_fn=self.rot_g_t_fn)
        vf_step = self.rot_step_scale * scores_and_vfs['base_rot_vf'] * rot_d_t
        score_step = self.rot_step_scale * g_t[..., None] * scores_and_vfs['base_rot_score'] * rot_d_t
        dB_rot = torch.randn_like(vf_step) * torch.sqrt(rot_d_t)
        noise_step = torch.sqrt(2 * g_t[..., None] * self.rot_noise_scale) * dB_rot

        total_step = vf_step * (rot_time >= 0.99)[..., None] + (score_step + noise_step) * (rot_time < 0.99)[..., None]
        mat_t = so3_utils.rotvec_to_rotmat(total_step)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new


class LangevinEquilibrationStep(DifferentialEquation):
    def __init__(
        self,
        trans_step_scale: float = 1.0,
        rot_step_scale: float = 1.0,
        trans_noise_scale: float = 1.0,
        rot_noise_scale: float = 1.0,
        trans_psi: float = 0.0,
        rot_psi: float = 0.0,
        trans_noise_std: float = 16.
    ):
        self.trans_step_scale = trans_step_scale
        self.rot_step_scale = rot_step_scale
        self.trans_noise_scale = trans_noise_scale
        self.rot_noise_scale = rot_noise_scale
        self.trans_psi = trans_psi
        self.rot_psi = rot_psi

        self.trans_noise_std = trans_noise_std

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        score_step = self.trans_step_scale * self.trans_psi / 2 * scores_and_vfs['base_trans_score'] * trans_d_t
        dW_t = self.trans_noise_std * torch.randn_like(trans_t_1) * torch.sqrt(trans_d_t)
        noise_step = np.sqrt(self.trans_psi * self.trans_noise_scale) * dW_t
        total_step = score_step + noise_step
        trans_t_2 = trans_t_1 + total_step

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        score_step = self.rot_step_scale * self.rot_psi / 2 * scores_and_vfs['base_rot_score'] * rot_d_t
        dB_rot = torch.randn_like(score_step) * torch.sqrt(rot_d_t)
        noise_step = np.sqrt(self.rot_psi * self.rot_noise_scale) * dB_rot

        total_step = score_step + noise_step
        mat_t = so3_utils.rotvec_to_rotmat(total_step)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new


class CustomLangevinEquilibrationStep(DifferentialEquation):
    def __init__(
        self,
        trans_step_scale: float = 1.0,
        rot_step_scale: float = 1.0,
        trans_noise_scale: float = 1.0,
        rot_noise_scale: float = 1.0,
        trans_psi: float = 0.0,
        rot_psi: float = 0.0,
        trans_noise_std: float = 16.
    ):
        self.trans_step_scale = trans_step_scale
        self.rot_step_scale = rot_step_scale
        self.trans_noise_scale = trans_noise_scale
        self.rot_noise_scale = rot_noise_scale
        self.trans_psi = trans_psi
        self.rot_psi = rot_psi

        self.trans_noise_std = trans_noise_std

    def g_t(self, t, g_t_fn):
        if g_t_fn == 'trans_natural':
            return (1 - t) / (t + 0.0625)
        elif g_t_fn == 'rots_natural':
            return 1.5 * (1 - t)

        raise ValueError(f"we don't recogize the g_t fn specifier {g_t_fn}")

    def trans_step(
        self,
        trans_t_1,
        trans_time,
        trans_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = self.g_t(trans_time, g_t_fn="trans_natural") * self.trans_psi
        score_step = self.trans_step_scale * g_t[..., None] * scores_and_vfs['base_trans_score'] * trans_d_t
        dW_t = self.trans_noise_std * torch.randn_like(trans_t_1) * torch.sqrt(trans_d_t)
        noise_step = torch.sqrt(2 * g_t[..., None] * self.trans_noise_scale) * dW_t
        total_step = score_step + noise_step
        trans_t_2 = trans_t_1 + total_step

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        trans_new = trans_t_1.clone()
        trans_new[rigids_noising_mask] = trans_t_2[rigids_noising_mask]
        return trans_new

    def rot_step(
        self,
        rotmats_t_1,
        rot_time,
        rot_d_t,
        scores_and_vfs,
        rigids_noising_mask,
    ):
        g_t = self.g_t(rot_time, g_t_fn="rots_natural") * self.rot_psi
        score_step = self.rot_step_scale * g_t[..., None] * scores_and_vfs['base_rot_score'] * rot_d_t
        dB_rot = torch.randn_like(score_step) * torch.sqrt(rot_d_t)
        noise_step = torch.sqrt(2 * g_t[..., None] * self.rot_noise_scale) * dB_rot

        total_step = score_step + noise_step
        mat_t = so3_utils.rotvec_to_rotmat(total_step)
        rotmats_t_2 = torch.einsum("...ij,...jk->...ik", rotmats_t_1, mat_t)

        # we do imputation because sometimes the score calculation
        # is unstable for fixed rigids
        rotmats_new = rotmats_t_1.clone()
        rotmats_new[rigids_noising_mask] = rotmats_t_2[rigids_noising_mask]
        return rotmats_new