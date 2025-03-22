import tqdm
import torch
from . import so3_utils
from . import utils as du
from scipy.spatial.transform import Rotation
import copy
from functools import partial
from scipy.optimize import linear_sum_assignment, differential_evolution, Bounds, NonlinearConstraint

from proteinzen.utils.framediff import all_atom
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.model.utils.graph import batchwise_to_nodewise

from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import scatter
import dataclasses
import numpy as np

# from eigenfold
class HarmonicPrior:
    def __init__(self, N = 256, a=3/(3.8**2)):
        J = torch.zeros(N, N)
        for i, j in zip(np.arange(N-1), np.arange(1, N)):
            J[i,i] += a
            J[j,j] += a
            J[i,j] = J[j,i] = -a
        D, P = torch.linalg.eigh(J)
        D_inv = 1/D
        D_inv[0] = 0
        self.P, self.D_inv = P, D_inv
        self.N = N

    def to(self, device):
        self.P = self.P.to(device)
        self.D_inv = self.D_inv.to(device)

    def sample(self, batch_dims=()):
        return self.P @ (torch.sqrt(self.D_inv)[:,None] * torch.randn(*batch_dims, self.N, 3, device=self.P.device))


def _centered_gaussian(batch, rigids_per_res, device):
    noise = torch.randn(batch.shape[0], rigids_per_res, 3, device=device)
    center = scatter(
        noise,
        index=batch,
        dim=0,
        reduce='mean'
    ).mean(dim=-2)
    return noise - center[batch][..., None, :]


def _uniform_so3(num_res, rigids_per_res, device):
    return torch.tensor(
        Rotation.random(num_res * rigids_per_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_res, rigids_per_res, 3, 3)


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (~diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return rotmats_t * diffuse_mask[..., None, None] + rotmats_1 * (
        ~diffuse_mask[..., None, None]
    )

@dataclasses.dataclass
class RotConfig:
    train_schedule="linear"
    sample_schedule="exp"
    exp_rate=10

@dataclasses.dataclass
class TransConfig:
    train_schedule="linear"
    sample_schedule="linear"

@dataclasses.dataclass
class SamplingConfig:
    num_timesteps=100

class SE3InterpolantConfig:
    min_t = 1e-2
    rots=RotConfig()
    trans=TransConfig()
    sampling=SamplingConfig()
    self_condition=True


class MultiSE3Interpolant:
    def __init__(self,
                 cfg,
                 separate_ot=True,
                 prealign_noise=True,
                 trans_preconditioning=False,
                 trans_preconditioning_std=16,
                 rigids_per_res=3,#5,
                 lognorm_t_sched=False,
                 lognorm_mu=0.0,
                 lognorm_sig=1.0,
                 mixed_beta_t_sched=False,
                 beta_p1=1.9,
                 beta_p2=1.0,
                 shift_time_scale=False,
                 trans_gamma=0.16,
                 trans_step_size=1.5,
                 rot_gamma=0.16,
                 rot_step_size=1.5,
                 sampling_churn=0.4,
                 use_diffusion_forcing=False,
    ):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None
        self.separate_ot = separate_ot
        self.prealign_noise = prealign_noise
        self.trans_preconditioning = trans_preconditioning
        self.trans_preconditioning_std = trans_preconditioning_std
        self.lognorm_t_sched = lognorm_t_sched
        self.lognorm_mu = lognorm_mu
        self.lognorm_sig = lognorm_sig
        self.shift_time_scale = shift_time_scale
        self.mixed_beta_t_sched = mixed_beta_t_sched
        self.beta_p1 = beta_p1
        self.beta_p2 = beta_p2

        self.trans_gamma = trans_gamma
        self.trans_step_size = trans_step_size
        self.rot_gamma = rot_gamma
        self.rot_step_size = rot_step_size
        self.churn = sampling_churn

        print(self.igso3)

        self.rigids_per_res = rigids_per_res
        self.use_diffusion_forcing = use_diffusion_forcing

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        if self.lognorm_t_sched:
            ln_sig = self.lognorm_mu + torch.randn(num_batch, device=self._device).float() * self.lognorm_sig
            t = torch.sigmoid(ln_sig)
            return t
        elif self.mixed_beta_t_sched:
            u = torch.rand(1)
            if u < 0.02:
                t = torch.rand(num_batch, device=self._device).float()
                return t * (1 - self._cfg.min_t)
            else:
                dist = torch.distributions.beta.Beta(self.beta_p1, self.beta_p2)
                t = dist.sample((num_batch,)).to(self._device)
                return t * (1 - self._cfg.min_t)
        else:
            t = torch.rand(num_batch, device=self._device).float()
            return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t

    def time_shift(self, t, n_res):
        shift_scale = np.sqrt(n_res / 100)
        return t / (t * (1 - shift_scale) + shift_scale)

    def _sample_trans_0(self, batch, device):
        if self.trans_preconditioning:
            trans_0 = _centered_gaussian(batch, self.rigids_per_res, device)
            trans_0 = trans_0 * self.trans_preconditioning_std
        else:
            trans_nm_0 = _centered_gaussian(batch, self.rigids_per_res, device)
            # TODO: this is a typo but i'm keeping it so i can compare against previous training runs
            trans_0 = trans_nm_0 * 16 # du.NM_TO_ANG_SCALE

        return trans_0.to(device)

    def _corrupt_trans(self, trans_1, trans_0, t, res_mask, diffuse_mask, batch):
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None, None]

    def _sample_rotmats_0(self, rotmats_1):
        num_res = rotmats_1.shape[0]
        noisy_rotmats = self.igso3.sample(torch.tensor([1.5]), num_res * self.rigids_per_res).to(rotmats_1.device)
        noisy_rotmats = noisy_rotmats.squeeze(0).view(num_res, self.rigids_per_res, 3, 3).float()
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        return rotmats_0

    def _corrupt_rotmats(self, rotmats_1, rotmats_0, t, res_mask, diffuse_mask, batch):
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = rotmats_t * res_mask[..., None, None, None] + identity[None, None] * (
            ~res_mask[..., None, None, None]
        )

        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)


    @torch.no_grad()
    def corrupt_batch(self, batch: HeteroData):
        res_data = batch["residue"]

        rigids_1 = ru.Rigid.from_tensor_7(res_data["rigids_1"])
        # [N, 5, 3]
        trans_1 = rigids_1.get_trans()
        # [N, 5, 3, 3]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()

        # [N]
        res_mask = res_data["res_mask"]
        rigids_noising_mask = res_data["rigids_noising_mask"]

        # [B]
        rigidwise_t = batch['rigidwise_t']
        if self.shift_time_scale:
            rigidwise_t = self.time_shift(rigidwise_t, (res_data.batch == 0).float().sum().item())

        rotmats_0 = self._sample_rotmats_0(rotmats_1)
        trans_0 = self._sample_trans_0(res_data.batch, trans_1.device)

        if self.prealign_noise:
            # rotate each structure to align as best as possible with noise
            aligned_trans_0, _, _ = du.align_structures(
                trans_0.flatten(0, 1),
                torch.repeat_interleave(res_data.batch, self.rigids_per_res),
                trans_1.flatten(0, 1)
            )
            trans_0 = aligned_trans_0.view(trans_0.shape)

        trans_t = self._corrupt_trans(
            trans_1,
            trans_0,
            rigidwise_t,
            res_mask,
            rigids_noising_mask,
            res_data.batch
        )
        rotmats_t = self._corrupt_rotmats(
            rotmats_1,
            rotmats_0,
            rigidwise_t,
            res_mask,
            rigids_noising_mask,
            res_data.batch
        )
        rigids_t = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t), trans=trans_t
        )
        res_data["rigids_t"] = rigids_t.to_tensor_7()
        res_data["rotmats_t"] = rotmats_t
        res_data["trans_t"] = trans_t

        return batch

    def g_t(self, t):
        return (1 - t) / (t + 0.1) ** 2

    def _trans_churn(self, d_t, t, trans_t):
        t_hat = torch.clamp(t - self.churn * d_t, min=0)
        d_t_hat = d_t * (1 + self.churn)
        noise_scale = torch.sqrt(
            2 * self.g_t(t_hat) * self.trans_gamma - 2 * self.g_t(t) * self.trans_gamma
        )
        trans_t_hat = trans_t + torch.randn_like(trans_t) * noise_scale * 10
        return t_hat, d_t_hat, trans_t_hat

    def _trans_score(self, t, trans_1, trans_t):
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return (t * trans_vf - trans_t) / (1-t)

    def _trans_euler_step(
            self,
            d_t,
            t,
            trans_1,
            trans_t,
            add_noise=True,
    ):
        trans_vf = (trans_1 - trans_t) / (1 - t)

        step_size = self.trans_step_size
        g_t = self.g_t(t)
        gamma = self.trans_gamma
        dW_t = torch.randn_like(trans_t) * torch.sqrt(d_t) * 10
        trans_score = self._trans_score(t, trans_1, trans_t)

        if t > 0.99:
            return trans_t + (trans_vf * d_t) * step_size
        else:
            if add_noise:
                noise_t = torch.sqrt(2 * g_t * gamma) * dW_t
                return trans_t + (trans_vf * d_t + trans_score * g_t * d_t) * step_size + noise_t
            else:
                return trans_t + (trans_vf * d_t + trans_score * g_t * d_t) * step_size


    def _rot_churn(self, d_t, t, rotmats_t):
        gamma = self.rot_gamma
        churn = self.churn
        t_hat = torch.clamp(t - churn * d_t, min=0)
        d_t_hat = d_t * (1 + churn)
        noise_scale = torch.sqrt(
            2 * self.g_t(t_hat) * gamma - 2 * self.g_t(t) * gamma
        )
        dB_rot = torch.randn(rotmats_t.shape[:-1], device=rotmats_t.device) * noise_scale
        rotmats_t_hat = so3_utils.rot_mult(
            rotmats_t,
            so3_utils.rotvec_to_rotmat(dB_rot)
        )
        return t_hat, d_t_hat, rotmats_t_hat

    def _rots_score(self, t, rotmats_1, rotmats_t):
        ls = torch.arange(1000, device=rotmats_t.device)
        rel_rotmat = torch.einsum("...ij,...jk->...ik", rotmats_t.transpose(-1, -2), rotmats_1).view(-1, 3, 3)
        omega, _, _ = so3_utils.angle_from_rotmat(rel_rotmat)
        omega = omega.view(-1)
        sigma = (
            ((1-t) * 1.5).square()
            + (t * 0.1).square()
        ).sqrt()
        sigma = sigma[None].expand(omega.shape).to(omega.device)
        prefactor = so3_utils.dlog_igso3_expansion(omega, sigma, ls)
        prefactor = prefactor.view(rotmats_t.shape[:-2])
        omega = omega.view(rotmats_t.shape[:-2])
        rot_score = (prefactor / omega)[..., None] * so3_utils.calc_rot_vf(rotmats_1, rotmats_t)
        return rot_score

    def _rots_euler_step(
            self,
            d_t,
            t,
            rotmats_1,
            rotmats_t,
            add_noise=True,
    ):
        rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1) / (1 - t)
        rot_score = self._rots_score(t, rotmats_1, rotmats_t)
        g_t = self.g_t(t)
        step_size = self.rot_step_size
        gamma = self.rot_gamma
        dB_rot = torch.randn_like(rot_vf) * torch.sqrt(d_t)
        if t > 0.99:
            mat_t = so3_utils.rotvec_to_rotmat(d_t * rot_vf * step_size)
        else:
            if add_noise:
                rotvec_t = d_t * rot_vf + d_t * g_t * rot_score
                noise_t = torch.sqrt(2 * g_t * gamma) * dB_rot
                mat_t = so3_utils.rotvec_to_rotmat(rotvec_t * step_size + noise_t)
            else:
                rotvec_t = d_t * rot_vf + d_t * g_t * rot_score
                mat_t = so3_utils.rotvec_to_rotmat(step_size * rotvec_t)

        return torch.einsum("...ij,...jk->...ik", rotmats_t, mat_t)
