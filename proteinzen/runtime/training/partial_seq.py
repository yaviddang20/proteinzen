import torch
import numpy as np

from .task import TrainingTask

class PartialSequenceConditioned(TrainingTask):
    name: str = "partial_seq"
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
        residues = data.residues
        atoms = data.atoms

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

        if np.random.random() < 0.8:
            mask_rate = torch.distributions.beta.Beta(3, 9).sample((1,))
        else:
            mask_rate = np.random.random()

        res_type_noising_mask = torch.rand(residues.shape[0], dtype=torch.float32) < mask_rate
        res_type_noising_mask = res_type_noising_mask.numpy(force=True)

        atom_noising_mask = np.ones(atoms.shape[0], dtype=bool)
        copy_indexed_residue_mask = np.zeros_like(res_type_noising_mask)
        copy_unindexed_residue_mask = np.zeros_like(res_type_noising_mask)
        copy_atomized_residue_mask = np.zeros_like(res_type_noising_mask)

        return {
            "t": t.numpy(force=True),
            "atom_noising_mask": atom_noising_mask,
            "res_type_noising_mask": res_type_noising_mask,
            "copy_indexed_residue_mask": copy_indexed_residue_mask,
            "copy_unindexed_residue_mask": copy_unindexed_residue_mask,
            "copy_atomized_residue_mask": copy_atomized_residue_mask,
        }

    def max_added_tokens(self, _):
        return 0
