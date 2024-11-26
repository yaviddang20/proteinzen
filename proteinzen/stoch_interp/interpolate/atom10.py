import numpy as np
import torch
from torch_geometric.data import HeteroData

from proteinzen.data.datasets.featurize.sidechain import _ideal_virtual_Cb
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.model.utils.graph import batchwise_to_nodewise


def _diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None, None] + trans_1 * (~diffuse_mask[..., None, None])


class Atom10Interpolant:
    def __init__(self,
                 min_t=1e-2,
                 self_condition=True,
                 one_m_exp_c=None,
                 sigmoid_c=None,
                 emperical_mean_offset=False,
                 smarter_prior=False,
                 smarter_prior_std=3,
                 prior_std=1,
                 sig_data=1,
                 nonlocal_prior=False
        ):
        self.min_t = min_t
        self.self_condition = self_condition
        self.one_m_exp_c = one_m_exp_c
        self.sigmoid_c = sigmoid_c
        self.emperical_mean_offset = emperical_mean_offset
        self.smarter_prior = smarter_prior
        self.smarter_prior_std = smarter_prior_std
        assert not (smarter_prior and emperical_mean_offset)
        assert not ((one_m_exp_c is not None) and (sigmoid_c is not None))
        self.prior_std = prior_std
        self.prior_offset = torch.tensor([-0.6, -0.6, -0.6])
        self.cb_local = torch.tensor([-5.537e-01, -7.773e-01, -1.201e+00])
        self.sig_data = sig_data

        self.nonlocal_prior = nonlocal_prior
        if nonlocal_prior:
            self.prior_std = 10



    def set_device(self, device):
        self._device = device

    def get_prior_offset(self, device):
        if self.smarter_prior:
            return self.cb_local.to(device)
        else:
            return self.prior_offset.to(device)

    def var_scaling_factors(self, t):
        assert self.one_m_exp_c is None and self.sigmoid_c is None
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)

        sig_1 = self.sig_data
        if self.smarter_prior:
            sig_0 = self.smarter_prior_std
        else:
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
        if self.emperical_mean_offset:
            # i calculated this from data
            # and rounded to a nice-ish number
            # mean: [-0.5745, -0.6288, -0.6132]
            # var: [1.3639, 1.2443, 1.0186]
            x_0 = x_0 + self.prior_offset[None, None].to(x_0.device)

        if self.one_m_exp_c is not None:
            # technically should be 1-e^(c*(t-1)) but since we've been using 1-t
            # for linear sched i'm doing it this way
            t = torch.exp(self.one_m_exp_c * (t - 1))
        elif self.sigmoid_c is not None:
            t = 1 - torch.sigmoid(self.sigmoid_c * (t - 0.5))
        x_t = (1 - t[..., None, None]) * x_0 + t[..., None, None] * x_1
        x_t = _diffuse_mask(x_t, x_1, diffuse_mask)
        return x_t

    def _corrupt_x_smarter(self, x_1, t, diffuse_mask):
        cb = self.cb_local.to(x_1.device)
        x_0 = torch.randn_like(x_1) * self.smarter_prior_std + cb[None, None]

        if self.one_m_exp_c is not None:
            # technically should be 1-e^(c*(t-1)) but since we've been using 1-t
            # for linear sched i'm doing it this way
            t = torch.exp(self.one_m_exp_c * (t - 1))
        elif self.sigmoid_c is not None:
            t = 1 - torch.sigmoid(self.sigmoid_c * (t - 0.5))
        x_t = (1 - t[..., None, None]) * x_0 + t[..., None, None] * x_1
        x_t = _diffuse_mask(x_t, x_1, diffuse_mask)
        return x_t

    def _corrupt_x_nonlocal(self, x_1, t, rigids_t, diffuse_mask):
        x_0 = torch.randn_like(x_1) * self.prior_std
        x_t = (1 - t[..., None, None]) * x_0 + t[..., None, None] * x_1
        x_t = rigids_t[..., None].invert_apply(x_t)
        x_t = _diffuse_mask(x_t, x_1, diffuse_mask)
        return x_t

    @torch.no_grad()
    def corrupt_batch(self, batch: HeteroData):
        res_data = batch["residue"]

        # [N]
        res_mask = res_data["res_mask"]
        res_noising_mask = res_data["res_noising_mask"]

        atom10 = res_data['atom14'][..., 4:, :]
        rigids = ru.Rigid.from_tensor_7(res_data['rigids_1'])
        atom10_local = rigids[..., None].invert_apply(atom10)
        atom10_local *= res_data['atom14_mask'][..., 4:][..., None]

        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'])

        # [B]
        t = batch["t"]
        nodewise_t = batchwise_to_nodewise(t, res_data.batch)

        # Apply corruptions
        if self.nonlocal_prior:
            noised_atom10_local = self._corrupt_x_nonlocal(atom10_local, nodewise_t, rigids_t, res_noising_mask)
        elif self.smarter_prior:
            noised_atom10_local = self._corrupt_x_smarter(atom10_local, nodewise_t, res_noising_mask)
        else:
            noised_atom10_local = self._corrupt_x(atom10_local, nodewise_t, res_noising_mask)
        noised_atom10_local *= res_mask[..., None, None]
        res_data['noised_atom10_local'] = noised_atom10_local.float()
        res_data['atom10_vf_scaling'] = self._scaling(t)

        if self.one_m_exp_c is None and self.sigmoid_c is None:
            var_scaling_dict = self.var_scaling_factors(t)
            # print(var_scaling_dict)
            batch['atom10_c_skip'] = var_scaling_dict['c_skip']
            batch['atom10_c_in'] = var_scaling_dict['c_in']
            batch['atom10_c_out'] = var_scaling_dict['c_out']
            batch['atom10_loss_weighting'] = var_scaling_dict['loss_weighting']
            if self.smarter_prior:
                batch['atom10_prior_offset'] = self.cb_local.to(noised_atom10_local.device)
            else:
                batch['atom10_prior_offset'] = self.prior_offset.to(noised_atom10_local.device)

        return batch

    def _scaling(self, t):
        if self.one_m_exp_c is not None:
            c = self.one_m_exp_c
            if isinstance(t, torch.Tensor):
                t = t.clip(max=0.9)
                scaling = -c * torch.exp(c*t) / (torch.exp(c*t) - np.exp(c))
            else:
                t = min(t, 0.9)
                scaling = -c * np.exp(c*t) / (np.exp(c*t) - np.exp(c))
        elif self.sigmoid_c is not None:
            c = self.sigmoid_c
            if isinstance(t, torch.Tensor):
                scaling = -c * torch.exp(c*(t-0.5)) / (torch.exp(c*(t-0.5)) + 1)
            else:
                scaling = -c * np.exp(c*(t-0.5)) / (np.exp(c*(t-0.5)) + 1)
        else:
            scaling = 1 / (1 - t)
        return scaling

    def _euler_step(self, d_t, t, x_1, x_t):
        if self.one_m_exp_c is not None:
            scaling = self._scaling(t)
            x_vf = (x_1 - x_t) * scaling
        elif self.sigmoid_c is not None:
            scaling = self._scaling(t)
            x_vf = (x_1 - x_t) * scaling
        else:
            x_vf = (x_1 - x_t) / (1 - t)
            # x_vf = (x_1 - x_t) * 10
        return x_t + x_vf * d_t
