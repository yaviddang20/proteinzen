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


# from genie2
def compute_frenet_frames(coords, chains, mask, eps=1e-10):
    """
    Construct Frenet-Serret frames based on a sequence of coordinates.

    Since the Frenet-Serret frame is constructed based on three consecutive
    residues, for each chain, the rotational component of its first residue
    is assigned with the rotational component of its second residue; and the
    rotational component of its last residue is assigned with the rotational
    component of its second last residue.

    Args:
        coords:
            [B, N, 3] Per-residue atom positions.
        chains:
            [B, N] Per-residue chain indices.
        mask:
            [B, N] Residue mask.
        eps:
            Epsilon for computational stability. Default to 1e-10.

    Returns:
        rots:
            [B, N, 3, 3] Rotational components for the constructed frames.
    """

    # [B, N-1, 3]
    t = coords[:, 1:] - coords[:, :-1]
    t_norm = torch.sqrt(eps + torch.sum(t ** 2, dim=-1))
    t = t / t_norm.unsqueeze(-1)

    # [B, N-2, 3]
    b = torch.cross(t[:, :-1], t[:, 1:], dim=-1)
    b_norm = torch.sqrt(eps + torch.sum(b ** 2, dim=-1))
    b = b / b_norm.unsqueeze(-1)

    # [B, N-2, 3]
    n = torch.cross(b, t[:, 1:], dim=-1)

    # [B, N-2, 3, 3]
    tbn = torch.stack([t[:, 1:], b, n], dim=-1)

    # Construct rotation matrices
    rots = []
    for i in range(mask.shape[0]):
        rots_ = torch.eye(3).unsqueeze(0).repeat(mask.shape[1], 1, 1)
        length = torch.sum(mask[i]).int()
        rots_[1:length-1] = tbn[i, :length-2]

        # Handle start of chain
        for j in range(length):
            if j == 0 or chains[i][j] != chains[i][j-1]:
                rots_[j] = rots_[j+1]

        # Handle end of chain
        for j in range(length):
            if j == length - 1 or chains[i][j] != chains[i][j+1]:
                rots_[j] = rots_[j-1]

        # Update
        rots.append(rots_)

    # [B, N, 3, 3]
    rots = torch.stack(rots, dim=0).to(coords.device)

    return rots

def compute_frenet_frames_flat(coords, res_mask, batch, rigids_per_res=3):
    bb_coords = coords[..., 0, :]
    rots = []
    for i in range(batch.max() + 1):
        select = (batch == i)
        _bb_coords = bb_coords[select]
        _rots = compute_frenet_frames(_bb_coords[None], torch.zeros_like(select, dtype=torch.int64)[None], res_mask[select][None])
        rots.append(_rots.squeeze(0)[..., None, :, :].expand(-1, rigids_per_res, -1, -1))

    return torch.cat(rots, dim=0)


class GenieLikeInterpolant:
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
                 rot_sfm_g=0.1,
                 shift_time_scale=False
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
        self.shift_time_scale = shift_time_scale

        self.rigids_per_res = rigids_per_res

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

    @torch.no_grad()
    def corrupt_batch(self, batch: HeteroData):
        res_data = batch["residue"]

        rigids_1 = ru.Rigid.from_tensor_7(res_data["rigids_1"])
        # [N, 5, 3]
        trans_1 = rigids_1.get_trans()

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

        if self.shift_time_scale:
            t = self.time_shift(t, (res_data.batch == 0).float().sum().item())

        nodewise_t = batchwise_to_nodewise(t, res_data.batch)

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
        rotmats_t = compute_frenet_frames_flat(trans_t, res_mask, res_data.batch)
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
        # loss_weighting = 1 / (c_out ** 2)
        loss_weighting = 1 / (3 * 18 * (sig_1 ** 2))
        return {
            "c_skip": c_skip,
            "c_out": c_out,
            "c_in": c_in,
            "loss_weighting": loss_weighting
        }

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