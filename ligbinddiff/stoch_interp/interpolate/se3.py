import tqdm
import torch
from . import so3_utils
from . import utils as du
from scipy.spatial.transform import Rotation
import copy
from functools import partial
from scipy.optimize import linear_sum_assignment, differential_evolution, Bounds, NonlinearConstraint

from ligbinddiff.utils.framediff import all_atom
from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.model.utils.graph import batchwise_to_nodewise

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


def _centered_gaussian(batch, device):
    noise = torch.randn(batch.shape[0], 3, device=device)
    center = scatter(
        noise,
        index=batch,
        dim=0,
        reduce='mean'
    )
    return noise - center[batch]


def _uniform_so3(num_res, device):
    return torch.tensor(
        Rotation.random(num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_res, 3, 3)


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


class SE3Interpolant:
    def __init__(self,
                 cfg,
                 use_batch_ot=False,
                 separate_ot=True,
                 fancy_joint_ot=False,
                 prealign_noise=True,
                 rotate_rots_by_trans_align=False,
                 uniform_rot_noise=False,
                 harmonic_trans_noise=False,
                 sfm=False):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None
        self.use_batch_ot = use_batch_ot
        self.separate_ot = separate_ot
        self.fancy_joint_ot = fancy_joint_ot
        self.prealign_noise = prealign_noise
        self.rotate_rots_by_trans_align = rotate_rots_by_trans_align
        self.uniform_rot_noise = uniform_rot_noise
        self.harmonic_trans_noise = harmonic_trans_noise
        self.sfm = sfm

        print(self.igso3)

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device).float()
        return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t

    def _sample_trans_0(self, batch, device):
        if self.harmonic_trans_noise:
            sample_lens = scatter(
                torch.ones_like(batch),
                batch
            )
            noise = []
            for l in sample_lens.tolist():
                prior = HarmonicPrior(l)
                noise.append(prior.sample().to(device))
            noise = torch.cat(noise, dim=0)
            center = scatter(
                noise,
                index=batch,
                dim=0,
                reduce='mean'
            )
            trans_0 = noise - center[batch]
        else:
            trans_nm_0 = _centered_gaussian(batch, device)
            trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE

        return trans_0.to(device)

    def _corrupt_trans(self, trans_1, trans_0, t, res_mask):
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1

        if self.sfm:
            z_t = self.trans_sfm_noise(trans_t, t)
            trans_t = trans_t + z_t

        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]

    def trans_sfm_noise(self, trans_t, t, g=0.1):
        batch_sig = g**2 * t * (1 - t)
        z_t = torch.randn_like(trans_t) * torch.sqrt(batch_sig)[..., None]
        return z_t

    def _trans_batch_ot(self, trans_1, trans_0, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        if self.prealign_noise:
            aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
                batch_nm_0, batch_nm_1, mask=batch_mask
            )
        else:
            aligned_nm_0 = batch_nm_0
            aligned_nm_1 = batch_nm_1
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)

        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]

    def _sample_rotmats_0(self, rotmats_1):
        num_res = rotmats_1.shape[0]
        if self.uniform_rot_noise:
            rotmats_0 = _uniform_so3(num_res, rotmats_1.device)
        else:
            noisy_rotmats = self.igso3.sample(torch.tensor([1.5]), num_res).to(rotmats_1.device)
            noisy_rotmats = noisy_rotmats.squeeze(0).float()
            rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        return rotmats_0

    def _corrupt_rotmats(self, rotmats_1, rotmats_0, t, res_mask):
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = rotmats_t * res_mask[..., None, None] + identity[None] * (
            ~res_mask[..., None, None]
        )

        if self.sfm:
            g = 0.1
            batch_sig = g**2 * t * (1 - t)
            z_rotmats = self.igso3.sample(batch_sig.sqrt(), 1).to(self._device)
            z_rotmats = z_rotmats.squeeze(1)
            rotmats_t = torch.einsum("...ij,...jk->...ik", rotmats_t, z_rotmats)

        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def rot_sfm_noise(self, num_samples, dt, device, g=0.1):
        z = torch.randn(
            (num_samples, 3), device=device
        )
        dB_skew_sym = so3_utils.rotvec_to_rotmat(g * torch.sqrt(dt).view(1,1).to(device) * z)
        return dB_skew_sym

    def _rot_batch_ot(self, rot_0, rot_1, res_mask):
        num_batch, num_res = rot_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_rot_0 = rot_0[noise_idx]
        batch_rot_1 = rot_1[gt_idx]
        batch_mask = res_mask[gt_idx]

        cost_matrix = torch.sum(
            so3_utils.geodesic_dist(batch_rot_0, batch_rot_1), # num_batch ** 2 x num_res x 3 x 3
            dim=-1
        ) / torch.sum(batch_mask, dim=-1)  # num_batch ** 2
        cost_matrix = cost_matrix.view(num_batch, num_batch)

        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        batch_rot_0 = batch_rot_0.view(num_batch, num_batch, num_res, 3, 3)
        return batch_rot_0[(tuple(gt_perm), tuple(noise_perm))]

    def joint_batch_ot(self, trans_1, trans_0, rot_1, rot_0, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        batch_rot_0 = rot_0[noise_idx]
        batch_rot_1 = rot_1[gt_idx]

        if self.fancy_joint_ot:
            aligned_nm_0, batch_rot_0 = align_rigids(
                batch_nm_1.view(num_batch, num_batch, num_res, 3),
                batch_nm_0.view(num_batch, num_batch, num_res, 3),
                batch_rot_1.view(num_batch, num_batch, num_res, 3, 3),
                batch_rot_0.view(num_batch, num_batch, num_res, 3, 3),
                # batch_mask.view(num_batch, num_batch, num_res)
            )
            batch_rot_0 = batch_rot_0.view(-1, num_res, 3, 3)
            aligned_nm_1 = batch_nm_1.view(num_batch, num_batch, num_res, 3)
        else:
            if self.prealign_noise:
                aligned_nm_0, aligned_nm_1, align_rot = du.batch_align_structures(
                    batch_nm_0, batch_nm_1, mask=batch_mask
                )
            else:
                aligned_nm_0 = batch_nm_0
                aligned_nm_1 = batch_nm_1
                align_rot = None
            aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
            aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)
            if self.rotate_rots_by_trans_align:
                assert align_rot is not None, "need to prealign noise if you want to rotate by trans align"
                rot_0 = rot_0.compose_r(align_rot)

        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        trans_cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        rot_cost_matrix = torch.sum(
            so3_utils.geodesic_dist(batch_rot_0, batch_rot_1), # num_batch ** 2 x num_res x 3 x 3
            dim=-1
        ) / torch.sum(batch_mask, dim=-1).view(-1)  # num_batch ** 2
        rot_cost_matrix = rot_cost_matrix.view(num_batch, num_batch)
        cost_matrix = (trans_cost_matrix**2 + rot_cost_matrix**2)
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        batch_rot_0 = batch_rot_0.view(num_batch, num_batch, num_res, 3, 3)

        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))], batch_rot_0[(tuple(gt_perm), tuple(noise_perm))]

    @torch.no_grad()
    def corrupt_batch(self, batch: HeteroData):
        res_data = batch["residue"]

        rigids_1 = ru.Rigid.from_tensor_7(res_data["rigids_1"])
        # [N, 3]
        trans_1 = rigids_1.get_trans()
        # [N, 3, 3]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()

        # [N]
        res_mask = res_data["res_mask"]
        noising_mask = res_data["res_noising_mask"]
        mask = res_mask & noising_mask

        # [B]
        t = self.sample_t(batch.num_graphs)
        nodewise_t = batchwise_to_nodewise(t, res_data.batch)
        batch["t"] = t

        if "rotmats_0" in res_data:
            rotmats_0 = res_data['rotmats_0']
        else:
            rotmats_0 = self._sample_rotmats_0(rotmats_1)
        if "trans_0" in res_data:
            trans_0 = res_data['trans_0']
        else:
            trans_0 = self._sample_trans_0(res_data.batch, trans_1.device)

        if self.prealign_noise and not self.use_batch_ot:
            # rotate each structure to align as best as possible with noise
            trans_0, _, _ = du.align_structures(trans_0, res_data.batch, trans_1)

        # Apply corruptions
        if self.use_batch_ot:
            sample_lens = scatter(
                torch.ones_like(res_data.batch),
                res_data.batch
            )
            assert (sample_lens[0] == sample_lens).all(), "batch ot can only be used with length batches"
            trans_0 = trans_0.view(-1, sample_lens[0].long().item(), 3)
            trans_1 = trans_1.view(-1, sample_lens[0].long().item(), 3)
            rotmats_0 = rotmats_0.view(-1, sample_lens[0].long().item(), 3, 3)
            rotmats_1 = rotmats_1.view(-1, sample_lens[0].long().item(), 3, 3)
            batch_res_mask = res_mask.view(-1, sample_lens[0].long().item())
            if self.separate_ot:
                trans_0 = self._trans_batch_ot(trans_1, trans_0, batch_res_mask)
                rotmats_0 = self._rot_batch_ot(rotmats_1, rotmats_0, batch_res_mask)
            else:
                trans_0, rotmats_0 = self.joint_batch_ot(trans_1, trans_0, rotmats_1, rotmats_0, batch_res_mask)
            trans_0 = trans_0.contiguous().view(-1, 3)
            trans_1 = trans_1.view(-1, 3)
            rotmats_0 = rotmats_0.contiguous().view(-1, 3, 3)
            rotmats_1 = rotmats_1.view(-1, 3, 3)

        trans_t = self._corrupt_trans(trans_1, trans_0, nodewise_t, mask)
        rotmats_t = self._corrupt_rotmats(rotmats_1, rotmats_0, nodewise_t, mask)
        rigids_t = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t), trans=trans_t
        )
        res_data["rigids_t"] = rigids_t.to_tensor_7()
        res_data["rotmats_t"] = rotmats_t
        res_data["trans_t"] = trans_t
        return batch

    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == "exp":
            return 1 - torch.exp(-t * self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == "linear":
            return t
        else:
            raise ValueError(f"Invalid schedule: {self._rots_cfg.sample_schedule}")

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == "linear":
            # scaling = 1 / (1 - t)
            scaling = 1 / (1 - t.clip(max=0.9))
        elif self._rots_cfg.sample_schedule == "exp":
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f"Unknown sample schedule {self._rots_cfg.sample_schedule}"
            )
        return so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)

# TODO: https://ieeexplore.ieee.org/abstract/document/8718799 for alignment

# I got the so3 part of this from chatgpt
# and emperically i've shown this is not optimal but
# it does consistently reduce the cost so i'm keeping it for now
def align_rigids(trans_1, trans_0, rotmats_1, rotmats_0):
    r3_cov = torch.einsum("...ki,...kj->...ij", trans_0, trans_1)
    rel_rotmat = so3_utils.rot_mult(rotmats_1, rotmats_0.transpose(-1, -2))
    rel_skew_sym = so3_utils.rotmat_to_skew_matrix(rel_rotmat).sum(dim=-3)
    cov = r3_cov + rel_skew_sym

    # Perform singular value decomposition. (all [B x 3 x 3])
    u, _, v_t = torch.linalg.svd(cov)
    # Convenience transposes.
    u_t = u.transpose(-1, -2)
    v = v_t.transpose(-1, -2)

    # Compute rotation matrix correction for ensuring right-handed coordinate system
    # For comparison with other sources: det(AB) = det(A)*det(B) and det(A) = det(A.T)
    sign_correction = torch.sign(torch.linalg.det(
        torch.einsum("...ij,...jk->...ik", v, u_t)
    ))
    # Correct transpose of U: diag(1, 1, sign_correction) @ U.T
    u_t[..., 2, :] = u_t[..., 2, :] * sign_correction[..., None]

    # Compute optimal rotation matrix (R = V @ diag(1, 1, sign_correction) @ U.T).
    rotation_matrices = torch.einsum("...ij,...jk->...ik", v, u_t)

    new_trans_0 = torch.einsum("bcij,bcnj->bcni", rotation_matrices, trans_0)
    new_rotmats_0 = torch.einsum("bcnij,bcjk->bcnik", rotmats_0, rotation_matrices)
    return new_trans_0, new_rotmats_0

# def _cost(rotvecs, trans_1, trans_0, rotmats_1, rotmats_0, batch_mask, num_batch, num_res):
#     rotvec = torch.as_tensor(rotvecs, device=trans_1.device).float().view(num_batch, num_batch, 3)
#     test_rotmat = so3_utils.rotvec_to_rotmat(rotvec)
#     test_rotmat = test_rotmat.view(num_batch, num_batch, 3, 3)
#     aligned_trans_0 = torch.einsum("bcij,bcni->bcnj", test_rotmat, trans_0)
#     aligned_rotmats_0 = torch.einsum("bcnij,bcjk->bcnik", rotmats_0, test_rotmat)
#
#     trans_cost_matrix = torch.sum(
#         torch.linalg.norm(aligned_trans_0 - trans_1, dim=-1), dim=-1
#     ) / torch.sum(batch_mask, dim=-1)
#     rot_cost_matrix = torch.sum(
#         so3_utils.geodesic_dist(
#             aligned_rotmats_0.view(-1, num_res, 3, 3),
#             rotmats_1.view(-1, num_res, 3, 3)
#         ), # num_batch ** 2 x num_res x 3 x 3
#         dim=-1
#     ) / torch.sum(batch_mask, dim=-1).view(-1)  # num_batch ** 2
#     rot_cost_matrix = rot_cost_matrix.view(num_batch, num_batch)
#     cost_matrix = (trans_cost_matrix**2 + rot_cost_matrix**2)
#     cost = cost_matrix.sum()
#     return cost.double().numpy(force=True)
#
# def align_rigids(trans_1, trans_0, rotmats_1, rotmats_0, batch_mask):
#     num_batch, _, num_res = trans_1.shape[:3]
#     batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
#
#     def constraint_fn(x):
#         x = x.reshape(num_batch**2, 3)
#         return np.linalg.norm(x, axis=-1)
#     constraint = NonlinearConstraint(constraint_fn, 0, np.pi)
#
#     result = differential_evolution(
#         func=partial(
#             _cost,
#             trans_1=trans_1,
#             trans_0=trans_0,
#             rotmats_1=rotmats_1,
#             rotmats_0=rotmats_0,
#             batch_mask=batch_mask,
#             num_batch=num_batch,
#             num_res=num_res
#         ),
#         bounds=Bounds(-np.pi * np.ones(3 * num_batch**2), np.pi * np.ones(3 * num_batch ** 2)),
#         # constraints=constraint,
#     )
#     # print(result)
#     best_rotvec = torch.as_tensor(result.x).float().view(num_batch, num_batch, 3)
#     rotation_matrices = so3_utils.rotvec_to_rotmat(best_rotvec)
#     rotation_matrices = rotation_matrices.view(num_batch, num_batch, 3, 3)
#
#     new_trans_0 = torch.einsum("bcij,bcnj->bcni", rotation_matrices, trans_0)
#     new_rotmats_0 = torch.einsum("bcnij,bcjk->bcnik", rotmats_0, rotation_matrices)
#     return new_trans_0, new_rotmats_0
