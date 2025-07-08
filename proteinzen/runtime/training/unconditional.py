import torch
import numpy as np

from .task import TrainingTask

class UnconditionalGeneration(TrainingTask):
    name: str = "unconditional"
    def __init__(
        self,
        prob=0.0,
        t_sched='lognorm',
        lognorm_mu=0.0,
        lognorm_sig=1.0,
        beta_p1=1.9,
        beta_p2=1.0,
        shift_time_scale=False,
        t_min=0.01,
        t_max=0.99,
    ):
        assert t_sched in ['lognorm', 'mixed_beta', 'uniform']
        self.prob = prob
        self.t_sched = t_sched
        self.lognorm_mu = lognorm_mu
        self.lognorm_sig = lognorm_sig
        self.beta_p1 = beta_p1
        self.beta_p2 = beta_p2
        self.t_min = t_min
        self.t_max = t_max
        self.shift_time_scale = shift_time_scale

    def sample_t_and_mask(self, data):
        rigids_1 = data.rigids['tensor7']
        # device = rigids_1.device
        device = 'cpu'
        if self.t_sched == 'lognorm':
            ln_sig = self.lognorm_mu + torch.randn(1, device=device).float() * self.lognorm_sig
            t = torch.sigmoid(ln_sig)
        elif self.t_sched == 'mixed_beta':
            u = torch.rand(1)
            if u < 0.02:
                t = torch.rand(1, device=device).float()
            else:
                dist = torch.distributions.beta.Beta(self.beta_p1, self.beta_p2)
                t = dist.sample((1,)).to(device)
        elif self.t_sched == 'uniform':
            t = torch.rand(1, device=device).float()
        else:
            raise ValueError(f"self.t_sched={self.t_sched} not recognized")

        rigids_noising_mask = np.ones(rigids_1.shape[:-1], dtype=bool)
        seq_noising_mask = np.ones(data.tokens['token_idx'].shape, dtype=bool)

        return {
            "t": t.numpy(force=True),
            "rigids_noising_mask": rigids_noising_mask,
            "seq_noising_mask": seq_noising_mask,
            "copy_indexed_token_mask": None,
            "copy_unindexed_token_mask": None,
        }

    def max_added_tokens(self, _):
        return 0
