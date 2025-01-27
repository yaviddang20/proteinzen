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
    return trans_t * diffuse_mask[..., None, None] + trans_1 * (~diffuse_mask[..., None, None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return rotmats_t * diffuse_mask[..., None, None, None] + rotmats_1 * (
        ~diffuse_mask[..., None, None, None]
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
                 use_trans_sfm=False,
                 trans_sfm_g=0.1,
                 use_rot_sfm=False,
                 rot_sfm_g=0.1
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
        self.use_trans_sfm = use_trans_sfm
        self.trans_sfm_g = trans_sfm_g
        self.use_rot_sfm = use_rot_sfm
        self.rot_sfm_g = rot_sfm_g

        print(self.igso3)

        self.rigids_per_res = rigids_per_res

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
        else:
            t = torch.rand(num_batch, device=self._device).float()
            return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t


    def _sample_trans_0(self, batch, device):
        if self.trans_preconditioning:
            trans_0 = _centered_gaussian(batch, self.rigids_per_res, device)
            trans_0 = trans_0 * self.trans_preconditioning_std
        else:
            trans_nm_0 = _centered_gaussian(batch, self.rigids_per_res, device)
            # TODO: this is a typo but i'm keeping it so i can compare against previous training runs
            trans_0 = trans_nm_0 * 16 # du.NM_TO_ANG_SCALE

        return trans_0.to(device)

    def _corrupt_trans(self, trans_1, trans_0, t, res_mask, batch):
        trans_t = (1 - t[..., None, None]) * trans_0 + t[..., None, None] * trans_1

        if self.use_trans_sfm:
            g = self.trans_sfm_g
            std = g * (t * (1-t)).sqrt()
            # TODO: this is a typo but i'm keeping it so i can compare against previous training runs
            scaling = self.trans_preconditioning_std if self.trans_preconditioning else 16 #du.NM_TO_ANG_SCALE
            dW_trans = torch.randn_like(trans_0) * std[batch, None, None] * scaling
            trans_t = trans_t + dW_trans

        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None, None]


    def _sample_rotmats_0(self, rotmats_1):
        num_res = rotmats_1.shape[0]
        noisy_rotmats = self.igso3.sample(torch.tensor([1.5]), num_res * self.rigids_per_res).to(rotmats_1.device)
        noisy_rotmats = noisy_rotmats.squeeze(0).view(num_res, self.rigids_per_res, 3, 3).float()
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        return rotmats_0

    def _corrupt_rotmats(self, rotmats_1, rotmats_0, t, res_mask, batch):
        rotmats_t = so3_utils.geodesic_t(t[..., None, None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)

        if self.use_rot_sfm:
            g = self.rot_sfm_g
            std = g * (t * (1-t)).sqrt()
            dB_rot = self.igso3.sample(std, self.rigids_per_res).to(rotmats_1.device)
            dB_rot = dB_rot[batch]
            rotmats_t = torch.einsum("...ij,...jk->...ik", rotmats_t, dB_rot)

        rotmats_t = rotmats_t * res_mask[..., None, None, None] + identity[None, None] * (
            ~res_mask[..., None, None, None]
        )

        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)


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
        noising_mask = res_data["res_noising_mask"]
        mask = res_mask & noising_mask

        # [B]
        if "t" not in batch:
            t = self.sample_t(batch.num_graphs)
            batch["t"] = t
        else:
            t = batch["t"]

        nodewise_t = batchwise_to_nodewise(t, res_data.batch)

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

        trans_t = self._corrupt_trans(trans_1, trans_0, nodewise_t, mask, res_data.batch)
        rotmats_t = self._corrupt_rotmats(rotmats_1, rotmats_0, nodewise_t, mask, res_data.batch)
        rigids_t = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t), trans=trans_t
        )
        res_data["rigids_t"] = rigids_t.to_tensor_7()
        res_data["rotmats_t"] = rotmats_t
        res_data["trans_t"] = trans_t

        var_scaling_dict = self.var_scaling_factors(t)
        # print(var_scaling_dict)
        batch['trans_c_skip'] = var_scaling_dict['c_skip']
        batch['trans_c_in'] = var_scaling_dict['c_in']
        batch['trans_c_out'] = var_scaling_dict['c_out']
        batch['trans_loss_weighting'] = var_scaling_dict['loss_weighting']

        return batch

    def var_scaling_factors(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)

        sig_1 = self.trans_preconditioning_std
        sig_0 = self.trans_preconditioning_std
        sig_signal = sig_1 * t
        sig_noise = sig_0 * (1-t)
        var_t = sig_signal ** 2 + sig_noise ** 2
        # TODO: i did some emperical calculations to adjust c_out
        # to have the vector field target be Var 1 rather than
        # the denoiser target but you should check this math
        # old: c_out = (1-t) * sig_1 * sig_0 / torch.sqrt(var_t)
        c_skip = t * sig_1**2 / (var_t)
        # c_out = sig_1 * sig_0 / torch.sqrt(var_t)
        c_out = (1-t) * sig_1 * sig_0 / torch.sqrt(var_t)
        c_in = 1 / torch.sqrt(var_t)
        loss_weighting = 1 / (c_out ** 2)
        return {
            "c_skip": c_skip,
            "c_out": c_out,
            "c_in": c_in,
            "loss_weighting": loss_weighting
        }

    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == "exp":
            return 1 - torch.exp(-t * self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == "linear":
            return t
        else:
            raise ValueError(f"Invalid schedule: {self._rots_cfg.sample_schedule}")

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        trans_vf = (trans_1 - trans_t) / (1 - t)
        if self.use_trans_sfm:
            g = self.trans_sfm_g
            std = g * (t * (1-t)).sqrt()
            # TODO: this is a typo but i'm keeping it so i can compare against previous training runs
            scaling = self.trans_preconditioning_std if self.trans_preconditioning else 16 #du.NM_TO_ANG_SCALE
            dW_trans = torch.randn_like(trans_t) * std * scaling * torch.sqrt(d_t)
            trans_t = trans_t + dW_trans

        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        # scaling = 1 / (1 - t)
        # rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
        # rot_angle = torch.linalg.vector_norm(rot_vf, dim=-1)
        # rot_angle = torch.min(torch.pi * (1 - t), rot_angle)
        # rot_vf = torch.nn.functional.normalize(rot_vf, dim=-1) * rot_angle[..., None]
        # mat_t = so3_utils.rotvec_to_rotmat(d_t * rot_vf * scaling)
        # return torch.einsum("...ij,...jk->...ik", rotmats_t, mat_t)

        if self._rots_cfg.sample_schedule == "linear":
            scaling = 1 / (1 - t.clip(max=0.9))
        elif self._rots_cfg.sample_schedule == "exp":
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f"Unknown sample schedule {self._rots_cfg.sample_schedule}"
            )
        if self.use_rot_sfm:
            rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1) * scaling

            g = self.rot_sfm_g
            std = g * (t * (1-t)).sqrt()
            dB_rot = torch.randn_like(rot_vf) * std * torch.sqrt(d_t)
            mat_t = so3_utils.rotvec_to_rotmat(d_t * rot_vf + dB_rot)
            return torch.einsum("...ij,...jk->...ik", rotmats_t, mat_t)
        else:
            return so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)