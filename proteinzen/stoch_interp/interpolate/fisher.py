import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere

from . import sphere_utils

torch.set_default_dtype(torch.float32)


class MonotonicLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((in_dim, out_dim)))
        self.bias = nn.Parameter(torch.randn((out_dim,)))

    def forward(self, x):
        # print(self.weight)
        ret = torch.matmul(
            x,
            # torch.abs(self.weight).clamp(min=1e-12)
            torch.exp(self.weight)
        )
        # print(ret)
        ret = ret + self.bias
        # print(ret)
        return ret


class LearnableSchedule(nn.Module):
    def __init__(self, h_hidden=1024):
        super().__init__()
        self.h_hidden = h_hidden
        self.l1 = MonotonicLinear(1, 1)
        self.l2 = MonotonicLinear(1, h_hidden)
        self.l3 = MonotonicLinear(h_hidden, 1)

    def _forward(self, t):
        h = self.l1(t)
        _h = 2. * (t - .5)  # scale input to [-1, +1]
        _h = self.l2(_h)
        _h = 2 * (torch.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
        _h = self.l3(_h) / self.h_hidden
        h = h + _h
        return h

    def forward(self, t):
        device = t.device
        t = t.to(self.l1.weight.device)
        out_0 = self._forward(torch.zeros_like(t))
        out_1 = self._forward(torch.ones_like(t))
        out = self._forward(1-t)
        # print(out_0[0], out_1[1])
        ret = (out - out_0) / (out_1 - out_0).clamp(min=1e-12)
        return ret.to(device)


# implemented from https://arxiv.org/abs/2405.14664
class FisherFlow(nn.Module):
    sched_options = ['linear', 'exp', '1m_exp', 'sigmoid', 'learned']
    def __init__(self,
                 D=20,
                 prior="hypersphere",
                 train_sched="linear",
                 train_c=1,
                 sample_sched="linear",
                 sample_c=1,
        ):
        super().__init__()
        self.D = D
        self.sphere = Hypersphere(dim=D-1)
        assert prior in ['dirichlet', 'hypersphere']
        self.prior = prior
        assert train_sched in self.sched_options
        if train_sched == 'learned':
            sample_sched = 'learned'
            self.learned_sched = LearnableSchedule()
        else:
            self.learned_sched = None

        assert sample_sched in self.sched_options
        self.train_sched = train_sched
        self.train_c = train_c
        self.sample_sched = train_sched
        self.sample_c = sample_c

    def kappa(self, t: torch.Tensor):
        assert self.learned_sched is not None
        if t.dim() == 1:
            _t = t[..., None]
        else:
            _t = t
        # pylint: disable=unbalanced-tuple-unpacking
        kappa, dkappa = torch.func.jvp(
            self.learned_sched,
            (_t,),
            (torch.ones_like(_t),)
        )
        # print(kappa, dkappa)
        if t.dim() == 1:
            kappa = kappa.squeeze(-1)
            dkappa = dkappa.squeeze(-1)
        return kappa, dkappa

    def sample_prior(self, num_nodes, device):
        if self.prior == 'dirichlet':
            # just a way to sample uniform from the simplex
            x0_simplex = torch.distributions.Dirichlet(
                torch.ones(num_nodes, self.D).float()
            ).sample()
        elif self.prior == 'hypersphere':
            x0_hs = self.sphere.random_uniform(num_nodes)
            x0_pos_octant = torch.abs(x0_hs)
            x0_simplex = x0_pos_octant.square()
        return x0_simplex.to(device)

    def vf_scale(self, t, c, schedule, t_clip=None):
        """ Generate vf scaling factor based off of schedule. This is derived by -d(log K(t)) / dt as in https://arxiv.org/abs/2302.03660

        linear: K(t) = 1-t, so scaling factor is 1/(1-t)
        exp: K(t) = e^(-c*t), so scaling factor is c
        neg_exp: K(t) = 1-e^(c*(x-1)), so scaling factor is -c*e^(c*x))/(e^(c*x) - e^c)
        sigmoid: K(t) = 1 / (1+e^(c * (x-0.5))), so scaling factor is (c*e^(c*(x - 0.5)))/(e^(c*(x - 0.5)) + 1)


        Args:
            t (_type_): _description_
            schedule (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if t_clip is not None:
            t = t.clip(max=t_clip)

        if schedule == 'linear':
            return 1 / (1-t)
        elif schedule == 'exp':
            return torch.as_tensor(c, device=t.device)
        elif schedule == '1m_exp':
            return -c * torch.exp(c*t) / (torch.exp(c*t) - np.exp(c))
        elif schedule == 'sigmoid':
            return - c * torch.exp(c*(t-0.5)) / (torch.exp(c*(t-0.5)) + 1)
        elif schedule == 'learned':
            kappa, dkappa = self.kappa(t)
            return - dkappa / kappa
        else:
            raise ValueError("schedule is not recognized")


    def _geodesic_interpolant(self, x0_simplex, x1_simplex, t, schedule="linear", schedule_c=1):
        """
        Unlike FrameFlow and following (FM on Riemannian manifolds), we interpolated based at x1 towards x0

        Args:
            x0_simplex (_type_): _description_
            x1_simplex (_type_): _description_
            t (_type_): _description_
            schedule (str, optional): _description_. Defaults to "linear".
            schedule_c (int, optional): _description_. Defaults to 1.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if schedule == 'linear':
            t = 1 - t
        elif schedule == 'exp':
            t = torch.exp(- schedule_c * t)
        elif schedule == '1m_exp':
            t = 1 - torch.exp(schedule_c * (t-1))
        elif schedule == 'sigmoid':
            t = 1 / (1 + torch.exp(schedule_c * (t - 0.5)))
        elif schedule == 'learned':
            t, _ = self.kappa(t)
            # t = t.detach()
        else:
            raise ValueError("schedule is not recognized")

        x0_hs = torch.sqrt(x0_simplex)
        x1_hs = torch.sqrt(x1_simplex)
        with torch.device(x0_hs.device):
            xt_hs = self.sphere.metric.exp(
                t[..., None] * sphere_utils.log(self.sphere.metric, x0_hs, x1_hs),
                x1_hs
            )
        xt_simplex = torch.square(xt_hs)
        return xt_simplex

    def _corrupt_seq(self, seq, t):
        x1_simplex = F.one_hot(seq, num_classes=self.D).float()
        x0_simplex = self.sample_prior(seq.shape[0], x1_simplex.device)

        xt_simplex = self._geodesic_interpolant(x0_simplex, x1_simplex, t, schedule=self.train_sched, schedule_c=self.train_c)
        return xt_simplex, x1_simplex

    def corrupt_batch(self, model, batch):
        if self.train_sched == 'learned':
            assert hasattr(model, "sched")
            self.learned_sched = model.sched
        res_data = batch["residue"]
        t = batch['t']
        nodewise_t = t[res_data.batch]

        # [N]
        seq_mask = res_data['seq_mask']
        noising_mask = res_data["seq_noising_mask"]

        seq = res_data['seq']
        seq[~seq_mask] = 0
        noised_probs, gt_probs = self._corrupt_seq(seq, nodewise_t)
        gt_probs[~seq_mask] = 0
        noised_probs[~seq_mask] = 0
        noised_probs[~noising_mask] = gt_probs[~noising_mask]

        res_data['seq_probs_t'] = noised_probs
        res_data['seq_probs_1'] = gt_probs
        if self.train_sched == 'learned':
            kappa, dkappa = self.kappa(t)
            batch['kappa_t'] = kappa
            batch['dkappa_t'] = dkappa

        return batch

    # note: this returns the vf on the hypersphere, not the simplex
    def train_vf(self, t, seq_probs_t, seq_probs_1, t_clip=0.9):
        device = seq_probs_t.device
        scaling = self.vf_scale(t, self.train_c, self.train_sched, t_clip=t_clip)
        # this business is to avoid having 0s pass through the sqrt
        seq_t_hs = F.normalize(seq_probs_t.clip(min=1e-12).sqrt(), dim=-1)
        seq_1_hs = F.normalize(seq_probs_1.clip(min=1e-12).sqrt(), dim=-1)
        with torch.device(device):
            hs_vf = sphere_utils.log(self.sphere.metric, seq_1_hs, seq_t_hs)
        return scaling[..., None] * hs_vf


    def euler_step(self, d_t, t, seq_probs_1, seq_probs_t, batch):
        device = seq_probs_t.device
        seq_t_hs = seq_probs_t.sqrt()# .cpu()
        seq_1_hs = seq_probs_1.sqrt()# .cpu()
        with torch.device(device):
            hs_vf = self.sphere.metric.log(seq_1_hs, seq_t_hs)
            scaling = self.vf_scale(t, self.sample_c, self.sample_sched).to(hs_vf.device)
            seq_t_1_hs = self.sphere.metric.exp(scaling[..., None] * d_t * hs_vf, seq_t_hs)
        seq_t_1 = seq_t_1_hs.square()
        return seq_t_1