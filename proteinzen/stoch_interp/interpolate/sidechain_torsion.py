import tqdm
import torch
from scipy.spatial.transform import Rotation
import copy
from scipy.optimize import linear_sum_assignment

from proteinzen.utils.framediff import all_atom
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.model.utils.graph import batchwise_to_nodewise

from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import scatter
import dataclasses


class SidechainTorsionInterpolant:
    def __init__(self, uniform_rot_noise=False, sigma=1.5):
        self.uniform_rot_noise = uniform_rot_noise
        self.sigma = sigma

    def _corrupt_rots(self, rots_0, rots_1, t, chi_mask):
        angle_0 = torch.atan2(rots_0[..., 1], rots_0[..., 0])
        angle_1 = torch.atan2(rots_1[..., 1], rots_1[..., 0])
        angle_t = t * angle_1 + (1-t) * angle_0
        rots_t = torch.stack([
            torch.cos(angle_t), torch.sin(angle_t)
        ], dim=-1)
        rots_t = rots_t * chi_mask[..., None] + rots_0 * ~chi_mask[..., None]
        return rots_t

    def sample_chi_noise(self, rots_1):
        if self.uniform_rot_noise:
            angle_0 = torch.rand(rots_1.shape[:-1]) * torch.pi * 2
            rots_0 = torch.stack([
                torch.cos(angle_0), torch.sin(angle_0)
            ], dim=-1)
        else:
            angle_perturb = torch.randn(rots_1.shape[:-1]) * self.sigma
            angle_1 = torch.atan2(rots_1[..., 1], rots_1[..., 0])
            angle_0 = angle_1 + angle_perturb
            rots_0 = torch.stack([
                torch.cos(angle_0), torch.sin(angle_0)
            ], dim=-1)
        return rots_0

    @torch.no_grad()
    def corrupt_batch(self, batch: HeteroData):
        res_data = batch["residue"]
        chis_1 = res_data['chis_1']
        flat_chis_1 = chis_1.view(-1, 2)

        # [N]
        chi_mask = res_data["chi_mask"]
        noising_mask = res_data["noising_mask"]
        mask = chi_mask & noising_mask[..., None]

        rand_angle = torch.rand(chis_1.shape[:-1]) * torch.pi * 2
        rand_chis = torch.stack([
            torch.cos(rand_angle), torch.sin(rand_angle)
        ], dim=-1)

        # [B]
        t = batch["t"]
        nodewise_t = batchwise_to_nodewise(t, res_data.batch)
        chiwise_t = nodewise_t[..., None].expand(-1, chis_1.shape[-2])
        flat_chiwise_t = chiwise_t.reshape(-1)

        flat_chis_0 = self.sample_chi_noise(flat_chis_1)
        flat_chi_mask = chi_mask.view(-1)

        flat_chis_t = self._corrupt_rots(flat_chis_0, flat_chis_1, flat_chiwise_t, flat_chi_mask)
        chis_t = flat_chis_t.view(chis_1.shape)
        # chis_t = mask[..., None] * chis_t + ~mask[..., None] * chis_1
        chis_t = chis_t * noising_mask[..., None, None] + chis_1 * ~noising_mask[..., None, None]
        chis_t = chi_mask[..., None] * chis_t + ~chi_mask[..., None] * rand_chis
        res_data['chis_t'] = chis_t

        return batch

    def euler_step(self, d_t, t, chis_1, chis_t):
        angle_t = torch.atan2(chis_t[..., 1], chis_t[..., 0])
        angle_1 = torch.atan2(chis_1[..., 1], chis_1[..., 0])
        omega = (angle_1 - angle_t) / (1-t)

        angle_tp1 = angle_t + omega * d_t
        chis_tp1 = torch.stack([
            torch.cos(angle_tp1), torch.sin(angle_tp1)
        ], dim=-1)
        return chis_tp1


class SidechainMultiTorsionInterpolant:
    def __init__(self, uniform_rot_noise=False, sigma=1.5):
        self.uniform_rot_noise = uniform_rot_noise
        self.sigma = sigma

    def _corrupt_rots(self, chis_0, chis_1, t, chi_mask):
        angle_0 = torch.atan2(chis_0[..., 1], chis_0[..., 0])
        angle_1 = torch.atan2(chis_1[..., 1], chis_1[..., 0])
        angle_t = t * angle_1 + (1-t) * angle_0
        chis_t = torch.stack([
            torch.cos(angle_t), torch.sin(angle_t)
        ], dim=-1)
        chis_t = chis_t * chi_mask[..., None] + chis_0 * ~chi_mask[..., None]
        return chis_t

    def sample_chi_noise(self, chis_1):
        if self.uniform_rot_noise:
            angle_0 = torch.rand(chis_1.shape[:-1]) * torch.pi * 2
            chis_0 = torch.stack([
                torch.cos(angle_0), torch.sin(angle_0)
            ], dim=-1)
        else:
            angle_perturb = torch.randn(chis_1.shape[:-1]) * self.sigma
            angle_1 = torch.atan2(chis_1[..., 1], chis_1[..., 0])
            angle_0 = angle_1 + angle_perturb
            chis_0 = torch.stack([
                torch.cos(angle_0), torch.sin(angle_0)
            ], dim=-1)
        return chis_0

    @torch.no_grad()
    def corrupt_batch(self, batch: HeteroData):
        res_data = batch["residue"]
        chis_1 = res_data['chis_1']

        # [N]
        chi_mask = res_data["chi_mask"]
        noising_mask = res_data["chi_noising_mask"]

        # [B]
        t = batch["t"]
        nodewise_t = batchwise_to_nodewise(t, res_data.batch)
        chiwise_t = nodewise_t[..., None].expand(-1, chis_1.shape[-2])

        chis_0 = self.sample_chi_noise(chis_1)
        chis_t = self._corrupt_rots(chis_0, chis_1, chiwise_t, chi_mask)
        # if we're not noising chis, replace them with gt chis
        chis_t = chis_t * noising_mask[..., None, None] + chis_1 * ~noising_mask[..., None, None]
        # this is to prevent data leakage for non-existant gt chis being 0
        # but it's not clear to me this is actually a problem
        # chis_t = chi_mask[..., None] * chis_t + ~chi_mask[..., None] * chis_0
        res_data['chis_t'] = chis_t
        res_data['chis_0'] = chis_0

        return batch

    def euler_step(self, d_t, t, multi_chis_1, chis_t, seq_probs_1):
        angle_t = torch.atan2(chis_t[..., 1], chis_t[..., 0])  # [..., 4]
        multi_angle_1 = torch.atan2(multi_chis_1[..., 1], multi_chis_1[..., 0]) # [..., 20, 4]
        # in order to minimize the velocity field
        # we compute the min angle delta,
        # with + being counterclockwise and - being clockwise
        angle_diff_cw = torch.abs(multi_angle_1 - angle_t[..., None, :])
        angle_diff_ccw = torch.pi * 2 - angle_diff_cw
        angle_min_disp = torch.where(
            angle_diff_cw > angle_diff_ccw,
            angle_diff_ccw,
            -angle_diff_cw
        )
        # multi_omega = (angle_min_disp) / (1-t)  # [..., 20, 4]
        c = 10
        multi_omega = (angle_min_disp) * c  # [..., 20, 4]
        omega = torch.sum(multi_omega * seq_probs_1[..., None], dim=-2)  # [..., 4]

        angle_tp1 = angle_t + omega * d_t
        chis_tp1 = torch.stack([
            torch.cos(angle_tp1), torch.sin(angle_tp1)
        ], dim=-1)
        return chis_tp1