from functools import partialmethod

import math
import numpy as np
import torch

from .task import TrainingTask

# inspired by Genie2
class MotifScaffolding(TrainingTask):
    name: str = "motif_scaffolding"
    def __init__(self,
                 t_sched='lognorm',
                 mode='full_residue',
                 lognorm_mu=0.0,
                 lognorm_sig=1.0,
                 beta_p1=1.9,
                 beta_p2=1.0,
                 shift_time_scale=False,
                 t_min=0.01,
                 t_max=0.99,
                 max_frac_res=0.5,
                 max_num_res=40,
                 p_is_unindexed=0.8,
    ):
        assert t_sched in ['lognorm', 'mixed_beta', 'uniform']
        assert mode in ['backbone', 'full_residue', 'inv_rotamer', 'mixed']
        self.t_sched = t_sched
        self.mode = mode
        self.lognorm_mu = lognorm_mu
        self.lognorm_sig = lognorm_sig
        self.beta_p1 = beta_p1
        self.beta_p2 = beta_p2
        self.t_min = t_min
        self.t_max = t_max
        self.shift_time_scale = shift_time_scale

        self.max_frac_res = max_frac_res
        self.max_num_res = max_num_res
        self.p_is_unindexed = p_is_unindexed

    def generate_motif_mask(self, batch):
        masks = []
        N = batch['residue'].num_nodes // batch.num_graphs
        for i in range(batch.num_graphs):
            num_segments = np.random.randint(1, 5)
            # we switch this from Genie2
            # to increase the probability of sampling minimal motifs
            # we also give the option to lower the maximum number of residues sampled
            res_cap = min(math.ceil(self.max_frac_res * N), self.max_num_res)
            res_cap = max(res_cap, num_segments) + 1
            num_res = np.random.randint(num_segments, res_cap) + 1
            # num_res = np.random.randint(math.floor(0.05 * N), math.ceil(0.5 * N) + 1)
            # # when N < 80 it's possible for num_segments > num_res
            # num_res = max(num_segments + 1, num_res)
            B = np.random.choice(num_res - 1, size=num_segments, replace=False) + 1
            B = np.sort(
                np.concatenate([[0], B, [num_res]], axis=0)
            )
            L = B[1:] - B[:-1]
            permute_list = [[0] for _ in range(N - num_res)] + [np.ones((l,)) for l in L]
            permutation = np.random.permutation(len(permute_list))
            permuted_list = [permute_list[i] for i in permutation]
            M = np.concatenate(permuted_list, axis=0)
            masks.append(torch.as_tensor(M, dtype=bool))
        return torch.stack(masks, dim=0)

    def _generate_motif_mask(self, N):
        num_segments = np.random.randint(1, 5)
        # we switch this from Genie2
        # to increase the probability of sampling minimal motifs
        # we also give the option to lower the maximum number of residues sampled
        res_cap = min(math.ceil(self.max_frac_res * N), self.max_num_res)
        res_cap = max(res_cap, num_segments) + 1
        num_res = np.random.randint(num_segments, res_cap) + 1
        # num_res = np.random.randint(math.floor(0.05 * N), math.ceil(0.5 * N) + 1)
        # # when N < 80 it's possible for num_segments > num_res
        # num_res = max(num_segments + 1, num_res)
        B = np.random.choice(num_res - 1, size=num_segments, replace=False) + 1
        B = np.sort(
            np.concatenate([[0], B, [num_res]], axis=0)
        )
        L = B[1:] - B[:-1]
        permute_list = [[0] for _ in range(N - num_res)] + [np.ones((l,)) for l in L]
        permutation = np.random.permutation(len(permute_list))
        permuted_list = [permute_list[i] for i in permutation]
        M = np.concatenate(permuted_list, axis=0)
        return torch.as_tensor(M, dtype=torch.bool)

    def sample_t_and_mask(self, batch):
        rigids_1 = batch['residue']['rigids_1']
        device = rigids_1.device
        num_batch = batch.num_graphs
        rigids_1 = rigids_1.unflatten(0, (num_batch, -1))
        # num_batch = rigids_1.shape[0]
        if self.t_sched == 'lognorm':
            ln_sig = self.lognorm_mu + torch.randn(num_batch, device=device).float() * self.lognorm_sig
            t = torch.sigmoid(ln_sig)
        elif self.t_sched == 'uniform':
            # t = torch.rand(num_batch, device=device).float()
            t = torch.ones(num_batch, device=device).float() * 0.99
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
        rigids_noising_mask = torch.ones_like(t, dtype=bool)

        motif_mask = self.generate_motif_mask(batch)
        t = t.flatten(0, 1)
        rigids_noising_mask = rigids_noising_mask.flatten(0, 1)
        motif_mask = motif_mask.flatten(0, 1)
        seq_noising_mask = ~motif_mask
        if self.mode == 'backbone':
            # t[motif_mask, 0] = 1
            rigids_noising_mask[motif_mask, 0] = False
            seq_noising_mask = torch.ones_like(motif_mask)
        elif self.mode == 'full_residue':
            # t[motif_mask] = 1
            rigids_noising_mask[motif_mask] = False
        elif self.mode == 'inv_rotamer':
            # t[motif_mask, 1:] = 1
            rigids_noising_mask[motif_mask, 1:] = False
        elif self.mode == 'mixed':
            mask_bb_only = torch.rand_like(rigids_noising_mask[..., 0], dtype=torch.float32) < 0.5
            rigids_noising_mask[motif_mask & mask_bb_only, 0] = False
            rigids_noising_mask[motif_mask & (~mask_bb_only)] = False
            seq_noising_mask[motif_mask & mask_bb_only] = True

        res_is_unindexed_mask = torch.rand_like(seq_noising_mask, dtype=torch.float32) < self.p_is_unindexed

        return {
            "t": t,
            "rigids_noising_mask": rigids_noising_mask,
            "seq_noising_mask": seq_noising_mask,
            "res_is_unindexed_mask": res_is_unindexed_mask
        }

    def _sample_t_and_mask(self, data):
        rigids_1 = data['residue']['rigids_1']
        device = rigids_1.device
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
        else:
            raise ValueError(f"self.t_sched={self.t_sched} not recognized")
        rigids_noising_mask = torch.ones(rigids_1.shape[:-1], dtype=bool, device=device)

        motif_mask = self.generate_motif_mask(rigids_1.shape[0])
        seq_noising_mask = ~motif_mask
        if self.mode == 'backbone':
            raise NotImplementedError()
            # t[motif_mask, 0] = 1
            rigids_noising_mask[motif_mask, 0] = False
            seq_noising_mask = torch.ones_like(motif_mask)
        elif self.mode == 'full_residue':
            # t[motif_mask] = 1
            rigids_noising_mask[motif_mask] = False
        elif self.mode == 'inv_rotamer':
            raise NotImplementedError()
            # t[motif_mask, 1:] = 1
            rigids_noising_mask[motif_mask, 1:] = False

        res_is_unindexed_mask = torch.rand_like(seq_noising_mask, dtype=torch.float32) < self.p_is_unindexed
        res_is_atomized_mask = torch.zeros_like(res_is_unindexed_mask, dtype=torch.bool)

        return {
            "t": t,
            "rigids_noising_mask": rigids_noising_mask,
            "seq_noising_mask": seq_noising_mask,
            "res_is_unindexed_mask": res_is_unindexed_mask,
            "res_is_atomized_mask": res_is_atomized_mask
        }


class BackboneMotifScaffolding(MotifScaffolding):
    name: str = "bb_motif_scaffolding"

    __init__ = partialmethod(MotifScaffolding.__init__, mode='backbone')


class ResidueMotifScaffolding(MotifScaffolding):
    name: str = "res_motif_scaffolding"
    __init__ = partialmethod(MotifScaffolding.__init__, mode='full_residue')


class InverseRotamerMotifScaffolding(MotifScaffolding):
    name: str = "inv_rot_motif_scaffolding"

    __init__ = partialmethod(MotifScaffolding.__init__, mode='inv_rotamer')


class MixedMotifScaffolding(MotifScaffolding):
    name: str = "mixed_motif_scaffolding"
    __init__ = partialmethod(MotifScaffolding.__init__, mode='mixed')