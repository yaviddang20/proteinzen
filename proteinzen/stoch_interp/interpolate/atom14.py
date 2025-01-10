import numpy as np
import torch
from torch_geometric.data import HeteroData

from proteinzen.data.datasets.featurize.sidechain import _ideal_virtual_Cb
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.model.utils.graph import batchwise_to_nodewise


def _diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None, None] + trans_1 * (~diffuse_mask[..., None, None])


class Atom14Interpolant:
    def __init__(self,
                 min_t=1e-2,
                 prior_std=16,
                 sig_data=16,
                 lognorm_t_sched=False
        ):
        self.min_t = min_t
        self.prior_std = prior_std
        self.sig_data = sig_data
        self.lognorm_t_sched = lognorm_t_sched

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
            "loss_weighting": (var_t) / ((1-t) * sig_1 * sig_0)**2
        }

    def _corrupt_x(self, x_1, t, diffuse_mask):
        x_0 = torch.randn_like(x_1) * self.prior_std
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
        noised_atom10_local = self._corrupt_x(atom14, nodewise_t, res_noising_mask)
        noised_atom10_local *= res_mask[..., None, None]
        res_data['noised_atom14'] = noised_atom10_local.float()
        res_data['atom14_vf_scaling'] = self._scaling(t)

        var_scaling_dict = self.var_scaling_factors(t)
        # print(var_scaling_dict)
        batch['c_skip'] = var_scaling_dict['c_skip']
        batch['c_in'] = var_scaling_dict['c_in']
        batch['c_out'] = var_scaling_dict['c_out']
        batch['loss_weighting'] = var_scaling_dict['loss_weighting']

        return batch

    def _scaling(self, t):
        scaling = 1 / (1 - t)
        return scaling

    def _euler_step(self, d_t, t, x_1, x_t):
        x_vf = (x_1 - x_t) / (1 - t)
        return x_t + x_vf * d_t
