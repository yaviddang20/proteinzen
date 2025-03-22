import numpy as np
import torch
from torch_geometric.data import HeteroData

from proteinzen.data.datasets.featurize.sidechain import _ideal_virtual_Cb
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.model.utils.graph import batchwise_to_nodewise
from scipy.spatial.transform import Rotation

import torch_geometric.utils as pygu

from . import utils as du

def _diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None, None] + trans_1 * (~diffuse_mask[..., None, None])


def _uniform_so3(num_res, device):
    return torch.tensor(
        Rotation.random(num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_res, 3, 3)


def center_random_aug(trans, mask, batch):
    masked_batch = batch[mask]
    center = pygu.scatter(
        trans[mask],
        index=masked_batch,
        dim=0,
        dim_size=int(batch.max().item() + 1),
        reduce='sum'
    )
    div_factor = pygu.scatter(
        mask[mask].float(),
        index=masked_batch,
        dim=0,
        dim_size=int(batch.max().item() + 1),
        reduce='sum'
    )
    center = center / div_factor.clip(min=1)[..., None]
    trans = trans - center[batch]
    rand_rot = _uniform_so3(batch.shape[0], batch.device)
    trans = torch.einsum("...ij,...j->...i", rand_rot, trans) * mask[..., None]
    return trans, rand_rot

class Atom14Interpolant:
    def __init__(self,
                 min_t=1e-2,
                 prior_std=16,
                 sig_data=16,
                 lognorm_t_sched=False,
                 num_timesteps=200,
        ):
        self.min_t = min_t
        self.prior_std = prior_std
        self.sig_data = sig_data
        self.lognorm_t_sched = lognorm_t_sched
        self.num_timesteps = num_timesteps

    def sample_t(self, num_batch, device):
        if self.lognorm_t_sched:
            lognorm_mu, lognorm_sig = -1.2, 1.5
            ln_sig = lognorm_mu + torch.randn(num_batch, device=device).float() * lognorm_sig
            t = 1 / (1 + torch.exp(ln_sig))
            return t
        else:
            t = torch.rand(num_batch, device=device).float()
            return t * (1 - 2 * self.min_t) + self.min_t

    def var_scaling_factors(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)

        sig_1 = self.sig_data
        sig_0 = self.prior_std
        sig_signal = sig_1 * t
        sig_noise = sig_0 * (1-t)
        var_t = sig_signal ** 2 + sig_noise ** 2
        return {
            "c_skip": t * sig_1**2 / (var_t),
            "c_out": (1-t) * sig_1 * sig_0 / torch.sqrt(var_t),
            "c_in": 1 / torch.sqrt(var_t),
            "c_data": self.prior_std,
            "loss_weighting": (var_t) / ((1-t) * sig_1 * sig_0)**2
        }

    def _corrupt_x(self, x_1, t, diffuse_mask, batch):
        x_0 = torch.randn_like(x_1) * self.prior_std

        # we align noise against data
        flat_x_0 = x_0.flatten(0, 1)
        flat_x_1 = x_1.flatten(0, 1)
        atom_batch = batch[..., None].expand(-1, 14).reshape(-1)
        _, _, align_rot_mats = du.align_structures(
            flat_x_0,
            atom_batch,
            flat_x_1,
        )
        align_rot_mats = align_rot_mats[atom_batch]
        flat_x_0 = torch.einsum("...ij,...j->...i", align_rot_mats, flat_x_0)
        x_0 = flat_x_0.view(x_0.shape)

        x_t = (1 - t[..., None, None]) * x_0 + t[..., None, None] * x_1
        x_t = _diffuse_mask(x_t, x_1, diffuse_mask)
        return x_t

    @torch.no_grad()
    def corrupt_batch(self, batch: HeteroData):
        res_data = batch["residue"]

        # [N]
        res_mask = res_data["res_mask"]
        res_noising_mask = res_data["res_noising_mask"]

        atom14 = res_data['atom14']

        # [B]
        t = self.sample_t(batch.num_graphs, atom14.device)
        batch["t"] = t
        nodewise_t = batchwise_to_nodewise(t, res_data.batch)

        # Apply corruptions
        noised_atom14 = self._corrupt_x(atom14, nodewise_t, res_noising_mask, batch=res_data.batch)
        noised_atom14 *= res_mask[..., None, None]
        atom_batch = res_data.batch[..., None].expand(-1, 14)
        atom_batch = atom_batch.reshape(-1)
        noised_atom14, aug_rot = center_random_aug(
            noised_atom14.flatten(-3, -2),
            mask=res_mask[atom_batch],
            batch=atom_batch
        )
        noised_atom14 = noised_atom14.reshape(atom14.shape)
        res_data['noised_atom14'] = noised_atom14
        res_data['aug_rot'] = aug_rot[res_data.batch]

        var_scaling_dict = self.var_scaling_factors(t)
        # print(var_scaling_dict)
        batch['c_skip'] = var_scaling_dict['c_skip']
        batch['c_in'] = var_scaling_dict['c_in']
        batch['c_out'] = var_scaling_dict['c_out']
        batch['c_data'] = self.prior_std
        batch['loss_weighting'] = var_scaling_dict['loss_weighting']

        return batch

    def _euler_step(self, d_t, t, x_1, x_t):
        x_vf = (x_1 - x_t) / (1 - t)
        return x_t + x_vf * d_t
