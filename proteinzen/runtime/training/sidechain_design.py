import torch

from .task import TrainingTask

class SidechainDesign(TrainingTask):
    name: str = "sidechain_design"
    def __init__(
        self,
        t_sched='lognorm',
        lognorm_mu=0.0,
        lognorm_sig=1.0,
        beta_p1=1.9,
        beta_p2=1.0,
        shift_time_scale=False,
        t_min=0.01,
        t_max=0.99,
    ):
        assert t_sched in ['lognorm', 'mixed_beta']
        self.t_sched = t_sched
        self.lognorm_mu = lognorm_mu
        self.lognorm_sig = lognorm_sig
        self.beta_p1 = beta_p1
        self.beta_p2 = beta_p2
        self.t_min = t_min
        self.t_max = t_max
        self.shift_time_scale = shift_time_scale

    def sample_t_and_mask(self, batch):
        rigids_1 = batch['residue']['rigids_1']
        device = rigids_1.device
        # num_batch = rigids_t.shape[0]
        num_batch = batch.num_graphs # rigids_t.shape[0]
        rigids_1 = rigids_1.unflatten(0, (num_batch, -1))
        if self.t_sched == 'lognorm':
            ln_sig = self.lognorm_mu + torch.randn(num_batch, device=device).float() * self.lognorm_sig
            t = torch.sigmoid(ln_sig)
        elif self.t_sched == 'mixed_beta':
            u = torch.rand(1)
            if u < 0.02:
                t = torch.rand(num_batch, device=device).float()
            else:
                dist = torch.distributions.beta.Beta(self.beta_p1, self.beta_p2)
                t = dist.sample((num_batch,)).to(device)
        else:
            raise ValueError(f"self.t_sched={self.t_sched} not recognized")

        t = t.view(-1, *[1 for _ in rigids_1.shape[1:-1]]) * torch.ones(rigids_1.shape[:-1], device=device)
        t[..., 0] = 1
        rigids_noising_mask = torch.ones_like(t, dtype=torch.bool)
        rigids_noising_mask[..., 0] = False
        t = t.flatten(0, 1)
        rigids_noising_mask = rigids_noising_mask.flatten(0, 1)
        seq_noising_mask = torch.ones_like(rigids_noising_mask[:, 0])
        return {
            "t": t,
            "rigids_noising_mask": rigids_noising_mask,
            "seq_noising_mask": seq_noising_mask
        }

