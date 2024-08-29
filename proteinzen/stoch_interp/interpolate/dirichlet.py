import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.special

import warnings

# adapted from https://github.com/HannesStark/dirichlet-flow-matching/blob/main/utils/flow_utils.py
class DirichletConditionalFlow:
    def __init__(self, K=20, alpha_min=1, alpha_max=100, alpha_spacing=0.01, t_max=8, polyn_sched_coeff=1):
        self.t_max = t_max

        # if not exp_t and t_scale < 8:
        #     warnings.warn("dirichlet conditional flow is using time with a linear scheduler but t_max<8. if results r wack try t_max>8 based on the dirichlet fm paper")

        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)
        self.beta_cdfs = []
        self.bs = np.linspace(0, 1, 1000)
        for alph in self.alphas:
            self.beta_cdfs.append(scipy.special.betainc(alph, K-1, self.bs))
        self.beta_cdfs = np.array(self.beta_cdfs)
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / alpha_spacing
        self.K = K
        self.polyn_schedule_coeff = polyn_sched_coeff

    def c_factor(self, probs, t, batch, use_torch=False, eps=1e-4):
        # we need double precision for this, floats will have overflow
        if use_torch:
            device = probs.device
            dtype = probs.dtype
            probs = probs.double().numpy(force=True)
            t = t.double().numpy(force=True)
        alpha = t + 1
        alpha_expand = np.tile(alpha[:, None], (1, self.K))
        out1 = scipy.special.beta(alpha_expand, self.K - 1)
        out2 = np.where(probs < 1 - eps, out1 / ((1 - probs) ** (self.K - 1)), 0)
        out = np.where(probs > 0 + eps, out2 / (probs ** (alpha_expand - 1)), 0)

        interp = []
        for i in range(batch.max().item() + 1):
            subset = (batch == i).numpy(force=True)
            alpha_i = alpha[subset][0].item()
            I_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha_i - self.alphas))]
            interp.append(-np.interp(probs[subset], self.bs, I_func))
        interp = np.concatenate(interp, axis=0)
        final = interp * out
        if use_torch:
            final = torch.as_tensor(final, device=device, dtype=dtype)
        return final

    def sample_prior(self, total_num_res, device):
        alphas = torch.ones((total_num_res, self.K), device=device)
        seq_probs_0 = torch.distributions.Dirichlet(alphas).sample()
        return seq_probs_0

    def sample_cond_prob_path(self, seq, t, noising_mask, seq_mask):
        L = seq.shape[0]
        t = t.clip(max=self.t_max)

        seq = seq * seq_mask
        seq_one_hot = F.one_hot(seq, num_classes=self.K)
        alphas_ = torch.ones((L, self.K), device=seq.device)
        alphas_ = alphas_ + seq_one_hot * t[:,None]
        xt = torch.distributions.Dirichlet(alphas_).sample()
        xt = xt * noising_mask[..., None] + seq_one_hot * (~noising_mask[..., None])

        xt = xt * seq_mask[..., None]
        seq_one_hot = seq_one_hot * seq_mask[..., None]

        return xt, seq_one_hot

    def kappa_t(self, t):
        return (t ** self.polyn_schedule_coeff) * self.t_max

    def dkappa_t(self, t, dt):
        return self.kappa_t(t + dt)  - self.kappa_t(t)

    def corrupt_batch(self, model, batch):
        res_data = batch["residue"]
        t = self.kappa_t(batch['t'])
        nodewise_t = t[res_data.batch]

        # [N]
        res_mask = res_data["res_mask"]
        seq_mask = res_data['seq_mask']
        noising_mask = res_data["noising_mask"]
        mask = res_mask & seq_mask

        seq = res_data['seq']
        seq[~seq_mask] = 0
        noised_probs, gt_probs = self.sample_cond_prob_path(seq, nodewise_t, noising_mask, mask)
        noised_probs[~mask] = 0
        gt_probs[~mask] = 0

        res_data['seq_probs_t'] = noised_probs
        res_data['seq_probs_1'] = gt_probs
        return batch

    def get_normalized_t(self, t):
        return torch.clip(t / self.t_max, max=1.0)

    def euler_step(self, d_t, t, seq_probs_1, seq_probs_t, batch):
        t = self.kappa_t(t)
        d_t = self.dkappa_t(t, d_t)
        c_factor = self.c_factor(seq_probs_t, t, batch, use_torch=True)
        eye = torch.eye(self.K, device=c_factor.device).view(1, self.K, self.K)
        u_i = c_factor[..., None, :] * (eye - seq_probs_t[..., None])
        u = torch.sum(seq_probs_1[..., None, :] * u_i, dim=-1)
        return seq_probs_t + u * d_t[..., None]