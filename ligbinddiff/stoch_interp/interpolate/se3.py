import tqdm
import torch
from . import so3_utils
from . import utils as du
from scipy.spatial.transform import Rotation
import copy
from scipy.optimize import linear_sum_assignment

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
                 prealign_noise=True,
                 uniform_rot_noise=False,
                 harmonic_trans_noise=False):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None
        self.use_batch_ot = use_batch_ot
        self.prealign_noise = prealign_noise
        self.uniform_rot_noise = uniform_rot_noise
        self.harmonic_trans_noise = harmonic_trans_noise
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
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask, batch):
        if self.harmonic_trans_noise:
            sample_lens = scatter(
                torch.ones_like(batch),
                batch
            )
            noise = []
            for l in sample_lens.tolist():
                prior = HarmonicPrior(l)
                noise.append(prior.sample().to(self._device))
            noise = torch.cat(noise, dim=0)
            center = scatter(
                noise,
                index=batch,
                dim=0,
                reduce='mean'
            )
            trans_0 = noise - center[batch]
        else:
            trans_nm_0 = _centered_gaussian(batch, self._device)
            trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        if self.use_batch_ot:
            # this requires samples to be length batched
            sample_len = int((batch == 0).sum().item())
            assert batch.numel() % sample_len == 0, (
                f"minibatch OT can only be applied to length batches, "
                f"but you have {batch.numel()} nodes and sample 0 is length {sample_len}")
            trans_0 = self._trans_batch_ot(
                trans_0.view(-1, sample_len, 3),
                trans_1.view(-1, sample_len, 3),
                res_mask.view(-1, sample_len)
            )
            trans_0 = trans_0.view(-1, 3)
        else:
            if self.prealign_noise:
                # rotate each structure to align as best as possible with noise
                trans_0, _, _ = du.align_structures(trans_0, batch, trans_1)
        # trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]

    def _trans_batch_ot(self, trans_0, trans_1, res_mask):
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

    def _corrupt_rotmats(self, rotmats_1, t, res_mask, batch):
        num_res = res_mask.shape[0]
        if self.uniform_rot_noise:
            noisy_rotmats = _uniform_so3(num_res, self._device)
        else:
            noisy_rotmats = self.igso3.sample(torch.tensor([1.5]), num_res).to(self._device)
            noisy_rotmats = noisy_rotmats.squeeze(0)
        if self.use_batch_ot:
            # this requires samples to be length batched
            sample_len = int((batch == 0).sum().item())
            assert batch.numel() % sample_len == 0, (
                f"minibatch OT can only be applied to length batches, "
                f"but you have {batch.numel()} nodes and sample 0 is length {sample_len}")
            noisy_rotmats = self._rot_batch_ot(
                noisy_rotmats.view(-1, sample_len, 3, 3),
                rotmats_1.view(-1, sample_len, 3, 3),
                res_mask.view(-1, sample_len))
            noisy_rotmats = noisy_rotmats.view(-1, 3, 3)
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = rotmats_t * res_mask[..., None, None] + identity[None] * (
            ~res_mask[..., None, None]
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

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
        noising_mask = res_data["noising_mask"]
        mask = res_mask & noising_mask

        # [B]
        t = self.sample_t(batch.num_graphs)
        nodewise_t = batchwise_to_nodewise(t, res_data.batch)
        batch["t"] = t

        # Apply corruptions
        trans_t = self._corrupt_trans(trans_1, nodewise_t, mask, res_data.batch)
        rotmats_t = self._corrupt_rotmats(rotmats_1, nodewise_t, mask, res_data.batch)
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

    def sample(
        self,
        model,
        num_res,
    ):
        total_num_res = sum(num_res)
        data_list = []
        for n in num_res:
            data = HeteroData(
                residue={
                    "res_mask": torch.ones(n, device=self._device),
                    "noising_mask": torch.ones(n, device=self._device),
                    "num_nodes": n
                }
            )
            data_list.append(data)
        batch = Batch.from_data_list(data_list)
        res_data = batch['residue']
        # Set-up initial prior samples
        if self.harmonic_trans_noise:
            noise = []
            for l in num_res:
                prior = HarmonicPrior(l)
                noise.append(prior.sample().to(self._device))
            noise = torch.cat(noise, dim=0)
            center = scatter(
                noise,
                index=res_data.batch,
                dim=0,
                reduce='mean'
            )
            trans_0 = noise - center[res_data.batch]
        else:
            trans_0 = (
                _centered_gaussian(res_data.batch, self._device) * du.NM_TO_ANG_SCALE
            )
        rotmats_0 = _uniform_so3(total_num_res, self._device)

        # Set-up time
        ts = torch.linspace(self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0, torch.zeros((total_num_res, 2), device=self._device))]
        clean_traj = []
        denoiser_out = None
        for t_2 in tqdm.tqdm(ts[1:]):
            # Run model.
            trans_t_1, rotmats_t_1, _ = prot_traj[-1]
            res_data["trans_t"] = trans_t_1
            res_data["rotmats_t"] = rotmats_t_1
            res_data['rigids_t'] = ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_t_1),
                trans=trans_t_1
            ).to_tensor_7()
            t = torch.ones(batch.num_graphs, device=self._device) * t_1
            batch["t"] = t
            with torch.no_grad():
                denoiser_out = model(batch, self_condition=denoiser_out)

            # Process model output.
            pred_rigids = denoiser_out['final_rigids']
            pred_trans_1 = pred_rigids.get_trans()
            pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
            pred_psis = denoiser_out['psi'].detach().cpu()
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu(), pred_psis)
            )

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2, pred_psis))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1, _ = prot_traj[-1]
        res_data["trans_t"] = trans_t_1
        res_data["rotmats_t"] = rotmats_t_1
        res_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        t = torch.ones(batch.num_graphs, device=self._device) * t_1
        batch["t"] = t
        with torch.no_grad():
            denoiser_out = model(batch, self_condition=denoiser_out)

        # Process model output.
        pred_rigids = denoiser_out['final_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
        pred_psis = denoiser_out['psi'].detach().cpu()
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu(), pred_psis)
        )

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrotpsi_to_atom37(prot_traj, res_data.res_mask)
        clean_atom37_traj = all_atom.transrotpsi_to_atom37(clean_traj, res_data.res_mask)
        return atom37_traj, clean_atom37_traj, clean_traj
