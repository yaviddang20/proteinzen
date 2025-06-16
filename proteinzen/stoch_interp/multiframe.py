import tqdm
import torch
import torch.nn.functional as F
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

from . import so3_utils
from . import utils as du

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

class MultiSE3Interpolant:
    def __init__(self,
                 min_t=1e-2,
                 separate_ot=True,
                 prealign_noise=True,
                 trans_preconditioning=False,
                 trans_prior_std=16,
                 rigids_per_res=3,#5,
                 lognorm_t_sched=False,
                 lognorm_mu=0.0,
                 lognorm_sig=1.0,
                 mixed_beta_t_sched=False,
                 beta_p1=1.9,
                 beta_p2=1.0,
                 shift_time_scale=False,
                 use_diffusion_forcing=False,
                 use_stochastic_centering=False,
                 center_on_motif=False,
                 sig_perturb=2.0,
                 use_uniform_rot_noise=False,
                 trans_gamma=0.16,
                 trans_step_size=1.5,
                 rot_gamma=0.16,
                 rot_step_size=1.5,
                 rot_sample_schedule="linear",
                 sampling_noise_mode="churn",
                 sampling_churn=0.4,
                 churn_by_sigma=False,
                 num_timesteps=400,
                 use_euclidean_for_rots=False,
                 rot_sfm=False
    ):
        self._igso3 = None

        self.min_t = min_t
        self.separate_ot = separate_ot
        self.prealign_noise = prealign_noise
        self.trans_preconditioning = trans_preconditioning
        self.trans_prior_std = trans_prior_std
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
        assert rot_sample_schedule in ["linear", "exp"], rot_sample_schedule
        self.rot_sample_schedule = rot_sample_schedule
        assert sampling_noise_mode in ["churn", "euler", None], sampling_noise_mode
        self.sampling_noise_mode = sampling_noise_mode
        self.churn = sampling_churn
        self.churn_by_sigma = churn_by_sigma
        self.num_timesteps = num_timesteps

        print(self.igso3)

        self.rigids_per_res = rigids_per_res
        self.use_diffusion_forcing = use_diffusion_forcing
        self.use_stochastic_centering = use_stochastic_centering
        self.center_on_motif = center_on_motif
        self.sig_perturb = sig_perturb
        self.use_uniform_rot_noise = use_uniform_rot_noise
        self.use_euclidean_for_rots = use_euclidean_for_rots
        self.rot_sfm = rot_sfm

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
                return t * (1 - self.min_t)
            else:
                dist = torch.distributions.beta.Beta(self.beta_p1, self.beta_p2)
                t = dist.sample((num_batch,)).to(self._device)
                return t * (1 - self.min_t)
        else:
            t = torch.rand(num_batch, device=self._device).float()
            return t * (1 - self.min_t)

    def time_shift(self, t, n_res):
        shift_scale = np.sqrt(n_res / 100)
        return t / (t * (1 - shift_scale) + shift_scale)

    def _sample_trans_0(self, batch, device):
        trans_0 = _centered_gaussian(batch, self.rigids_per_res, device)
        trans_0 = trans_0 * self.trans_prior_std
        return trans_0.to(device)

    def _corrupt_trans(self, trans_1, trans_0, t, rigids_mask, diffuse_mask):
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * rigids_mask[..., None]

    def _sample_rotmats_0(self, rotmats_1):
        if self.use_uniform_rot_noise:
            rotmats_0 = _uniform_so3(rotmats_1.shape[0], rotmats_1.shape[1], rotmats_1.device)
        else:
            num_rigids = rotmats_1.shape[0] * rotmats_1.shape[1]
            noisy_rotmats = self.igso3.sample(torch.tensor([1.5]), num_rigids).to(rotmats_1.device)
            noisy_rotmats = noisy_rotmats.view(*rotmats_1.shape[:2], 3, 3).float()
            rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        return rotmats_0

    def _corrupt_rotmats(self, rotmats_1, rotmats_0, t, rigids_mask, diffuse_mask):
        if self.use_euclidean_for_rots:
            rotvecs_1 = so3_utils.rotmat_to_rotvec(rotmats_1)
            # rotvecs_0 = torch.randn_like(rotvecs_1) * torch.pi
            rotvecs_0 = so3_utils.rotmat_to_rotvec(rotmats_0)
            rotvecs_t = (1 - t[..., None]) * rotvecs_0 + t[..., None] * rotvecs_1
            angle_t = torch.linalg.vector_norm(rotvecs_t + 1e-10, dim=-1)
            angle_t = torch.remainder(angle_t, 2 * torch.pi)
            flip_rotvec = angle_t >= torch.pi
            angle_t = (2 * torch.pi - angle_t) * flip_rotvec + angle_t * ~flip_rotvec
            axis_t = F.normalize(rotvecs_t, dim=-1) * (1 - 2 * flip_rotvec)[..., None]
            rotvecs_t = axis_t * angle_t[..., None]
            rotmats_t = so3_utils.rotvec_to_rotmat(rotvecs_t)
        else:
            rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)

        if self.rot_sfm:
            eps_t = torch.sqrt(0.01 * t * (1-t) + 1e-4)
            self.igso3.to(rotmats_1.device)
            noisy_rotmats = self.igso3.sample(eps_t.view(-1), rotmats_1.shape[1]).to(rotmats_1.device)
            rotmats_t = torch.einsum("...ij,...jk->...ik", rotmats_t, noisy_rotmats)

        identity = torch.eye(3, device=t.device)
        rotmats_t = rotmats_t * rigids_mask[..., None, None] + identity[None, None] * (
            ~rigids_mask[..., None, None]
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
        rigidwise_batch = res_data.batch[..., None].expand(-1, self.rigids_per_res)

        # [B]
        rigidwise_t = batch['rigidwise_t']
        if self.shift_time_scale:
            rigidwise_t = self.time_shift(rigidwise_t, (res_data.batch == 0).float().sum().item())

        rotmats_0 = self._sample_rotmats_0(rotmats_1)
        trans_0 = self._sample_trans_0(res_data.batch, trans_1.device)

        if self.center_on_motif:
            # center samples on the center of the fixed region
            # if there is no fixed region, center the whole sample
            center_mask = res_mask[..., None] * (~rigids_noising_mask)
            center_batch = rigidwise_batch[center_mask]
            fixed_trans_1 = trans_1[center_mask]
            fixed_center = scatter(
                fixed_trans_1,
                index=center_batch,
                dim_size=(res_data.batch.max().item() + 1),
                reduce='mean'
            )
            global_center = scatter(
                trans_1.flatten(0, 1),
                index=torch.repeat_interleave(res_data.batch, self.rigids_per_res),
                dim_size=(res_data.batch.max().item() + 1),
                reduce='mean'
            )
            use_fixed_center = scatter(
                torch.ones_like(center_batch, dtype=torch.bool),
                index=center_batch,
                dim_size=(res_data.batch.max().item() + 1),
                reduce='any'
            )
            center = fixed_center * use_fixed_center[..., None] + global_center * (~use_fixed_center[..., None])
        else:
            center = scatter(
                trans_1.flatten(0, 1),
                index=torch.repeat_interleave(res_data.batch, self.rigids_per_res),
                dim_size=(res_data.batch.max().item() + 1),
                reduce='mean'
            )
        # print(center)

        # center = torch.zeros_like(center)

        trans_1 = trans_1 - center[rigidwise_batch]
        # print(trans_1.mean(dim=(0, 1)))

        # this is just so we can calculate atom14 rmsds
        res_data["atom14"] = res_data["atom14"] - center[res_data.batch][..., None,:]
        res_data['atom14'] *= res_data['atom14_mask'][..., None]
        res_data['atom14_gt_positions'] = res_data['atom14_gt_positions'] - center[res_data.batch][..., None, :]
        res_data['atom14_gt_positions'] *= res_data['atom14_mask'][..., None]
        res_data['atom14_alt_gt_positions'] = res_data['atom14_alt_gt_positions'] - center[res_data.batch][..., None, :]
        res_data['atom14_alt_gt_positions'] *= res_data['atom14_mask'][..., None]

        # if True:
        #     from proteinzen.utils.coarse_grain import compute_atom14_from_cg_frames
        #     _rigids = ru.Rigid(rigids_1.get_rots(), trans_1)
        #     _atom14 = compute_atom14_from_cg_frames(
        #         _rigids,
        #         res_mask=res_data['res_mask'],
        #         seq=res_data['seq'],
        #     )
        #     print(_atom14[res_data['atom14_mask'].bool()].mean(dim=0))
        #     print(res_data["atom14"][res_data['atom14_mask'].bool()].mean(dim=0))

        # print("input", batch.name, res_data['atom14_gt_positions'])


        if self.prealign_noise:
            # rotate each structure to align as best as possible with noise

            # aligned_trans_0, _, _ = du.align_structures(
            #     trans_0.flatten(0, 1),
            #     torch.repeat_interleave(res_data.batch, self.rigids_per_res),
            #     trans_1.flatten(0, 1)
            # )
            # trans_0 = aligned_trans_0.view(trans_0.shape)
            align_mask = res_mask[..., None] * rigids_noising_mask
            align_batch = rigidwise_batch[align_mask]

            _, _, align_rot_mats = du.align_structures(
                trans_0[align_mask],
                align_batch,
                trans_1[align_mask]
            )
            trans_0 = torch.einsum("nki,nij->nkj", trans_0, align_rot_mats[res_data.batch])

        if self.use_stochastic_centering:
            stoch_center = torch.randn_like(center) * self.sig_perturb
            trans_0 = trans_0 + stoch_center[rigidwise_batch] # torch.randn_like(trans_0) * self.sig_perturb


        trans_t = self._corrupt_trans(
            trans_1,
            trans_0,
            rigidwise_t,
            res_mask[..., None],
            rigids_noising_mask,
        )
        rotmats_t = self._corrupt_rotmats(
            rotmats_1,
            rotmats_0,
            rigidwise_t,
            res_mask[..., None],
            rigids_noising_mask,
        )
        rigids_t = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t), trans=trans_t
        )
        res_data["rigids_t"] = rigids_t.to_tensor_7()
        res_data["rotmats_t"] = rotmats_t
        res_data["trans_t"] = trans_t

        # we also overwrite the ground truth rigids_1 since we've done some centering
        rigids_1 = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_1), trans=trans_1
        )
        res_data["rigids_1"] = rigids_1.to_tensor_7()

        return batch

    # @torch.no_grad()
    def corrupt_dense_batch(self, batch):
        rigids_data = batch["rigids"]

        rigids_1 = ru.Rigid.from_tensor_7(rigids_data["rigids_1"])
        # [N, 5, 3]
        trans_1 = rigids_1.get_trans()
        # [N, 5, 3, 3]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()

        # [N]
        rigids_mask = rigids_data["rigids_mask"]
        rigids_noising_mask = rigids_data["rigids_noising_mask"]

        rotmats_0 = self._sample_rotmats_0(rotmats_1)
        trans_0 = torch.randn_like(trans_1) * self.trans_prior_std
        trans_0 = trans_0 - trans_0.mean(dim=1)[..., None, :]

        if self.center_on_motif:
            # center samples on the center of the fixed region
            # if there is no fixed region, center the whole sample
            center_mask = rigids_mask * (~rigids_noising_mask)
            fixed_trans_1 = trans_1 * center_mask[..., None]
            fixed_center = fixed_trans_1.sum(dim=1) / center_mask.long().sum(dim=1)[..., None].clip(min=1)
            global_center = (trans_1 * rigids_mask[..., None]).sum(dim=1) / rigids_mask.long().sum(dim=1)[..., None].clip(min=1)
            use_fixed_center = center_mask.any(dim=-1)
            center = fixed_center * use_fixed_center[..., None] + global_center * (~use_fixed_center[..., None])
        else:
            center = (trans_1 * rigids_mask[..., None]).sum(dim=1) / rigids_mask.long().sum(dim=1)[..., None].clip(min=1)

        trans_1 = trans_1 - center[..., None, :]
        # this is just so we can calculate atom14 rmsds
        atom_data = batch['atom']
        atom_data["atom14"] = atom_data["atom14"] - center[..., None, None, :]
        atom_data["atom14"] *= atom_data['atom14_mask'][..., None]
        atom_data['atom14_gt_positions'] = atom_data['atom14_gt_positions'] - center[..., None, None, :]
        atom_data['atom14_gt_positions'] *= atom_data['atom14_mask'][..., None]
        atom_data['atom14_alt_gt_positions'] = atom_data['atom14_alt_gt_positions'] - center[..., None, None, :]
        atom_data['atom14_alt_gt_positions'] *= atom_data['atom14_mask'][..., None]


        if self.prealign_noise:
            # rotate each structure to align as best as possible with noise
            align_mask = (rigids_mask * rigids_noising_mask).bool()
            align_batch = torch.tile(
                torch.arange(rigids_mask.shape[0])[..., None],
                (1, rigids_mask.shape[1])
            ).to(align_mask.device)
            align_batch = align_batch[align_mask]

            _, _, align_rot_mats = du.align_structures(
                trans_0[align_mask],
                align_batch,
                trans_1[align_mask]
            )
            print(trans_0.shape, align_rot_mats.shape)
            trans_0 = torch.einsum("bni,bij->bnj", trans_0, align_rot_mats)

        if self.use_stochastic_centering:
            stoch_center = torch.randn_like(center) * self.sig_perturb
            trans_0 = trans_0 + stoch_center[..., None, :]

        trans_time = batch['trans_t']
        rot_time = batch['rot_t']

        trans_t = self._corrupt_trans(
            trans_1,
            trans_0,
            trans_time,
            rigids_mask,
            rigids_noising_mask.bool(),
        )
        rotmats_t = self._corrupt_rotmats(
            rotmats_1,
            rotmats_0,
            rot_time,
            rigids_mask,
            rigids_noising_mask.bool(),
        )

        rotvecs_t = so3_utils.rotmat_to_rotvec(rotmats_t)
        angle_t = torch.linalg.vector_norm(rotvecs_t + 1e-8, dim=-1)
        axis_t = F.normalize(rotvecs_t, dim=-1)
        rotquats_t = torch.cat([
            torch.cos(angle_t/2)[..., None], torch.sin(angle_t/2)[..., None] * axis_t
        ], dim=-1)
        rigids_t = ru.Rigid(
            rots=ru.Rotation(quats=rotquats_t), trans=trans_t
        )

        rigids_data["rigids_t"] = rigids_t.to_tensor_7()
        rigids_data["rotmats_t"] = rotmats_t
        rigids_data["trans_t"] = trans_t

        # we also overwrite the ground truth rigids_1 since we've done some centering
        rigids_1 = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_1), trans=trans_1
        )
        rigids_data["rigids_1"] = rigids_1.to_tensor_7()

        return batch

    def g_t(self, t):
        return (1 - t) / (t + 0.1) ** 2

    def g_t_inv(self, s):
        return (- (0.2 * s + 1) + ((0.2 * s + 1) ** 2 - 4 * s * (0.01 * s - 1)) ** 0.5) / (2 * s)

    def _center_trans(self, trans_t, batch, trans_noising_mask=None):
        if trans_noising_mask is not None:
            fixed_trans = trans_t[~trans_noising_mask]
            fixed_batch = batch[..., None].expand(-1, trans_t.shape[-2])
            fixed_batch = fixed_batch[~trans_noising_mask]
            center = scatter(
                fixed_trans,
                index=fixed_batch,
                dim=0,
                reduce='mean',
                dim_size=int(batch.max().item() + 1)
            )
        else:
            center = scatter(
                trans_t,
                index=batch,
                dim=0,
                reduce='mean'
            ).mean(dim=-2)
        return trans_t - center[batch][..., None, :], center

    def _trans_churn(self, d_t, t, trans_t, noising_mask):
        if self.sampling_noise_mode == "churn":
            if self.churn_by_sigma:
                curr_sigma = self.g_t(t)
                new_sigma = (1 + self.churn) * curr_sigma
                t_hat = self.g_t_inv(new_sigma)
                t_hat = torch.clamp(t_hat, min=0)
                d_t_hat = d_t + (t - t_hat)
            else:
                t_hat = torch.clamp(t - self.churn * d_t, min=0)
                d_t_hat = d_t * (1 + self.churn)
            noise_scale = torch.sqrt(
                2 * self.g_t(t_hat) * self.trans_gamma - 2 * self.g_t(t) * self.trans_gamma
            )
            trans_t_hat = trans_t + torch.randn_like(trans_t) * noise_scale * 10
            trans_t_hat = _trans_diffuse_mask(trans_t_hat, trans_t, noising_mask)
            return t_hat, d_t_hat, trans_t_hat
        else:
            return t, d_t, trans_t

    def _trans_score(self, t, trans_1, trans_t):
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return (t * trans_vf - trans_t) / (1-t)

    def _trans_euler_step(
            self,
            d_t,
            t,
            trans_1,
            trans_t,
            noising_mask,
            vf_scale=1
    ):
        add_noise = (self.sampling_noise_mode == "euler")
        use_score = (self.sampling_noise_mode is not None)
        trans_vf = (trans_1 - trans_t) / (1 - t)

        step_size = self.trans_step_size if use_score else 1
        step_size = step_size * vf_scale
        g_t = self.g_t(t)
        gamma = self.trans_gamma
        dW_t = torch.randn_like(trans_t) * torch.sqrt(d_t) * 10
        trans_score = self._trans_score(t, trans_1, trans_t) * use_score

        # if t > 0.99:
        if (isinstance(t, torch.Tensor) and (t > 0.99).any()) or (not isinstance(t, torch.Tensor) and t > 0.99):  # TODO: fix hack (what happens if not all t > 0.99?)
            trans_next = trans_t + (trans_vf * d_t) * step_size
        else:
            if add_noise:
                noise_t = torch.sqrt(2 * g_t * gamma) * dW_t
                trans_next = trans_t + (trans_vf * d_t + trans_score * g_t * d_t) * step_size + noise_t
            else:
                # print(trans_vf.shape, d_t.shape, trans_score.shape, g_t.shape, step_size.shape)
                trans_next = trans_t + (trans_vf * d_t + trans_score * g_t * d_t) * step_size

        trans_next[~noising_mask] = trans_t[~noising_mask]
        return trans_next


    def _rot_churn(self, d_t, t, rotmats_t, noising_mask):
        if self.sampling_noise_mode == "churn":
            gamma = self.rot_gamma
            if self.churn_by_sigma:
                curr_sigma = self.g_t(t)
                new_sigma = (1 + self.churn) * curr_sigma
                t_hat = self.g_t_inv(new_sigma)
                t_hat = torch.clamp(t_hat, min=0)
                d_t_hat = d_t + (t - t_hat)
            else:
                t_hat = torch.clamp(t - self.churn * d_t, min=0)
                d_t_hat = d_t * (1 + self.churn)

            noise_scale = torch.sqrt(
                2 * self.g_t(t_hat) * gamma - 2 * self.g_t(t) * gamma
            )
            dB_rot = torch.randn(rotmats_t.shape[:-1], device=rotmats_t.device) * noise_scale
            rotmats_t_hat = so3_utils.rot_mult(
                rotmats_t,
                so3_utils.rotvec_to_rotmat(dB_rot)
            )
            rotmats_t_hat = _rots_diffuse_mask(rotmats_t_hat, rotmats_t, noising_mask)
            return t_hat, d_t_hat, rotmats_t_hat
        else:
            return t, d_t, rotmats_t

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
            noising_mask,
            vf_scale=1
    ):
        add_noise = (self.sampling_noise_mode == "euler")
        use_score = (self.sampling_noise_mode is not None)

        if self.rot_sample_schedule == "linear":
            rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1) / (1 - t)
        elif self.rot_sample_schedule == "exp":
            rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
        else:
            raise ValueError(f"unrecognized rot_sample_schedule {self.rot_sample_schedule}")

        # TODO: this is probably numerically unstable for fixed residues
        # i've seen this nan sometimes on fixed residues
        # which leads to some weird behavior downstream
        # current band-aid patch is just to force replace the fixed residues
        if use_score:
            rot_score = self._rots_score(t, rotmats_1, rotmats_t) * use_score
        else:
            rot_score = torch.zeros_like(rot_vf)

        g_t = self.g_t(t)
        step_size = self.rot_step_size * vf_scale
        gamma = self.rot_gamma
        dB_rot = torch.randn_like(rot_vf) * torch.sqrt(d_t)
        # if t > 0.99:
        if (isinstance(t, torch.Tensor) and (t > 0.99).any()) or (not isinstance(t, torch.Tensor) and t > 0.99):  # TODO: fix hack (what happens if not all t > 0.99?)
            mat_t = so3_utils.rotvec_to_rotmat(d_t * rot_vf * step_size)
        else:
            if add_noise:
                rotvec_t = d_t * rot_vf + d_t * g_t * rot_score
                noise_t = torch.sqrt(2 * g_t * gamma) * dB_rot
                mat_t = so3_utils.rotvec_to_rotmat(rotvec_t * step_size + noise_t)
            else:
                rotvec_t = d_t * rot_vf + d_t * g_t * rot_score
                mat_t = so3_utils.rotvec_to_rotmat(step_size * rotvec_t)

        rotmats_next = torch.einsum("...ij,...jk->...ik", rotmats_t, mat_t)

        rotmats_next[~noising_mask] = rotmats_t[~noising_mask]
        return rotmats_next
