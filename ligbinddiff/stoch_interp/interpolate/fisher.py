import numpy as np
import torch
import torch.nn.functional as F

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere

torch.set_default_dtype(torch.float32)


# implemented from https://arxiv.org/abs/2405.14664
class FisherFlow:
    def __init__(self,
                 D=20,
                 prior="dirichlet"):
        self.D = D
        self.sphere = Hypersphere(dim=D-1)
        assert prior in ['dirichlet', 'hypersphere']
        self.prior = prior

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


    def _geodesic_interpolant(self, x0_simplex, x1_simplex, t):
        x0_hs = torch.sqrt(x0_simplex)
        x1_hs = torch.sqrt(x1_simplex)
        xt_hs = self.sphere.metric.exp(
            t[..., None] * self.sphere.metric.log(x1_hs, x0_hs),
            x0_hs
        )
        xt_simplex = torch.square(xt_hs)
        return xt_simplex

    def _corrupt_seq(self, seq, t):
        x1_simplex = F.one_hot(seq, num_classes=self.D)
        x0_simplex = self.sample_prior(seq.shape[0]).to(x1_simplex.device)

        xt_simplex = self._geodesic_interpolant(x0_simplex, x1_simplex, t)
        return xt_simplex, x1_simplex

    def corrupt_batch(self, batch):
        res_data = batch["residue"]
        t = batch['t']
        nodewise_t = t[res_data.batch]

        # [N]
        res_mask = res_data["res_mask"]
        seq_mask = res_data['seq_mask']
        noising_mask = res_data["noising_mask"]
        mask = res_mask & seq_mask

        seq = res_data['seq']
        seq[~seq_mask] = 0
        noised_probs, gt_probs = self._corrupt_seq(seq, nodewise_t)
        noised_probs[~mask] = 0
        gt_probs[~mask] = 0

        res_data['seq_probs_t'] = noised_probs
        res_data['seq_probs_1'] = gt_probs
        return batch

    def euler_step(self, d_t, t, seq_probs_t, seq_probs_1, batch, scaling=None):
        device = seq_probs_t.device
        seq_t_hs = seq_probs_t.sqrt().cpu()
        seq_1_hs = seq_probs_1.sqrt().cpu()
        hs_vf = self.sphere.metric.log(seq_1_hs, seq_t_hs)
        if scaling is None:
            scaling = 1 / (1-t)[..., None].cpu()
        seq_t_1_hs = self.sphere.metric.exp(scaling * d_t * hs_vf, seq_t_hs)
        seq_t_1 = seq_t_1_hs.square()
        return seq_t_1.to(device)