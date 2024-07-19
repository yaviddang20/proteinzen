import numpy as np
import torch
import torch.nn.functional as F
import os
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere

from . import sphere_utils

torch.set_default_dtype(torch.float32)



# implemented from https://arxiv.org/abs/2405.14664
class FisherFlow:
    def __init__(self,
                 D=20,
                 prior="hypersphere",
                 train_sched="linear",
                 train_c=1,
                 sample_sched="exp",
                 sample_c=10,
        ):
        self.D = D
        self.sphere = Hypersphere(dim=D-1)
        assert prior in ['dirichlet', 'hypersphere']
        self.prior = prior
        assert train_sched in ['linear', 'exp', '1m_exp', 'sigmoid']
        assert sample_sched in ['linear', 'exp', '1m_exp', 'sigmoid']
        self.train_sched = train_sched
        self.train_c = train_c
        self.sample_sched = train_sched
        self.sample_c = sample_c

    def sample_prior(self, num_nodes):
        if self.prior == 'dirichlet':
            # just a way to sample uniform from the simplex
            x0_simplex = torch.distributions.Dirichlet(
                torch.ones(num_nodes, self.D).float()
            ).sample()
        elif self.prior == 'hypersphere':
            x0_hs = self.sphere.random_uniform(num_nodes)
            x0_pos_octant = torch.abs(x0_hs)
            x0_simplex = x0_pos_octant.square()
        return x0_simplex

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
            return c * torch.exp(c*(t-0.5)) / (torch.exp(c*(t-0.5)) + 1)
        else:
            raise ValueError("schedule is not recognized")


    def _geodesic_interpolant(self, x0_simplex, x1_simplex, t, schedule="linear", schedule_c=1):
        """

        In contrast to FrameFlow, we compute the interpolant based at x0 towards x1

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
            t = t
        elif schedule == 'exp':
            t = torch.exp(- schedule_c * t)
        elif schedule == '1m_exp':
            t = 1 - torch.exp(schedule_c * (t-1))
        elif schedule == 'sigmoid':
            t = 1 / (1 + torch.exp(schedule_c * (t - 0.5)))
        else:
            raise ValueError("schedule is not recognized")

        x0_hs = torch.sqrt(x0_simplex)
        x1_hs = torch.sqrt(x1_simplex)
        xt_hs = self.sphere.metric.exp(
            t[..., None] * self.sphere.metric.log(x0_hs, x1_hs),
            x1_hs
        )
        xt_simplex = torch.square(xt_hs)
        return xt_simplex

    def _corrupt_seq(self, seq, t):
        x1_simplex = F.one_hot(seq, num_classes=self.D).float()
        x0_simplex = self.sample_prior(seq.shape[0]).to(x1_simplex.device)

        xt_simplex = self._geodesic_interpolant(x0_simplex, x1_simplex, t, schedule=self.train_sched, schedule_c=self.train_c)
        return xt_simplex, x1_simplex

    def corrupt_batch(self, batch):
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


    def euler_step(self, d_t, t, seq_probs_t, seq_probs_1, batch):
        device = seq_probs_t.device
        seq_t_hs = seq_probs_t.sqrt()# .cpu()
        seq_1_hs = seq_probs_1.sqrt()# .cpu()
        with torch.device(device):
            hs_vf = self.sphere.metric.log(seq_1_hs, seq_t_hs)
            scaling = self.vf_scale(t, self.sample_c, self.sample_sched).to(hs_vf.device)
            seq_t_1_hs = self.sphere.metric.exp(scaling[..., None] * d_t * hs_vf, seq_t_hs)
        seq_t_1 = seq_t_1_hs.square()
        return seq_t_1