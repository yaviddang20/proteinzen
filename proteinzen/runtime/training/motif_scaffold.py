from functools import partialmethod

import math
import numpy as np
import torch

from proteinzen.boltz.data import const

from proteinzen.data.constants import coarse_grain as cg
from proteinzen.data.featurize.tokenize import convert_atom_str_to_tuple

from .task import TrainingTask


def rigid_noise_to_atom_noise(residue, atoms, rigid_noising_mask):
    res_name = residue['name']

    bb_group = ['N', 'CA', 'C', 'O', 'CB']
    # bb_frame = ['C', 'CA', 'N']
    group2 = cg.coarse_grain_sidechain_groups[res_name][2]
    group3 = cg.coarse_grain_sidechain_groups[res_name][3]
    # construct dummy frames as necessary
    # use bb frame if frame2 doesn't exist
    if len(group2) == 0:
        group2 = bb_group
    # use frame2 frame if frame3 doesn't exist
    if len(group3) == 0:
        group3 = group2

    atom_noise_mapping = {}
    frame_atom_groups = [bb_group, group2, group3]
    for i, atom_groups in enumerate(frame_atom_groups):
        noise_atom = rigid_noising_mask[i]
        for atom_name in atom_groups:
            atom_id = convert_atom_str_to_tuple(atom_name)
            atom_noise_mapping[atom_id] = noise_atom

    atom_noising_mask = []
    for atom in atoms:
        atom_name_tuple = tuple(atom["name"])
        if atom_name_tuple in atom_noise_mapping:
            atom_noising_mask.append(
                atom_noise_mapping[atom_name_tuple]
            )
        else:
            atom_noising_mask.append(True)

    return np.array(atom_noising_mask)


# inspired by Genie2
class MotifScaffolding(TrainingTask):
    name: str = "motif_scaffolding"
    def __init__(self,
                 prob=0.0,
                 t_sched='lognorm',
                 mode='mixed',
                 lognorm_mu=0.0,
                 lognorm_sig=1.0,
                 beta_p1=1.9,
                 beta_p2=1.0,
                 shift_time_scale=False,
                 t_min=0.01,
                 t_max=0.99,
                 max_frac_res=0.5,
                 max_num_res=40,
                 max_num_segments=4,
                 p_is_unindexed=0.8,
                 name_override=None,
                 p_fix_tip=0.0,
                 p_fix_bb=0.5,
                 p_fix_all=0.5
    ):
        assert t_sched in ['lognorm', 'mixed_beta', 'uniform']
        assert mode in ['mixed']
        self.prob = prob
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
        self.max_num_segments = max_num_segments
        self.p_is_unindexed = p_is_unindexed

        self.p_fix_tip = p_fix_tip
        self.p_fix_bb = p_fix_bb
        self.p_fix_all = p_fix_all
        assert np.isclose(sum([p_fix_tip, p_fix_bb, p_fix_all]), 1.0)

        if name_override is not None:
            self.name = name_override

    def generate_motif_mask(self, N):
        num_segments = np.random.randint(1, self.max_num_segments + 1)
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

        resolved_mask = data.residues['is_present']
        num_resolved_residues = resolved_mask.sum()
        motif_mask = self.generate_motif_mask(num_resolved_residues)
        residue_noising_mask = np.ones_like(resolved_mask)
        residue_noising_mask[resolved_mask] = ~motif_mask
        res_type_noising_mask = residue_noising_mask.copy()

        atom_noising_mask = []
        if self.mode == 'mixed':
            noise_select = torch.rand_like(torch.as_tensor(residue_noising_mask), dtype=torch.float32).numpy(force=True)
            fix_tip = noise_select < self.p_fix_tip
            noise_sidechain_only = noise_select > (1 - self.p_fix_bb)
            res_type_noising_mask[~residue_noising_mask & fix_tip] = False
            res_type_noising_mask[~residue_noising_mask & noise_sidechain_only] = True

            for chain in data.chains:
                # Get residue indices
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]
                is_protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]

                for i, residue in enumerate(data.residues[res_start:res_end]):
                    atom_idx = residue['atom_idx']
                    atom_num = residue['atom_num']
                    atoms = data.atoms[atom_idx:atom_idx+atom_num]
                    if residue["is_standard"] and (residue['name'] != 'UNK') and is_protein:
                        if fix_tip[i] & (~residue_noising_mask[i]):
                            rigid_noise_mask = [True, False, True]
                        elif noise_sidechain_only[i] & (~residue_noising_mask[i]):
                            rigid_noise_mask = [False, True, True]
                        elif ~residue_noising_mask[i]:
                            rigid_noise_mask = [False for _ in range(3)]
                        else:
                            rigid_noise_mask = [True for _ in range(3)]
                        atom_noising_mask.append(rigid_noise_to_atom_noise(residue, atoms, rigid_noise_mask))
                    else:
                        atom_noising_mask.append([True for _ in atoms])

        else:
            raise ValueError()
        atom_noising_mask = np.concatenate(atom_noising_mask)

        is_unindexed_residue = torch.rand_like(torch.as_tensor(residue_noising_mask), dtype=torch.float32) < self.p_is_unindexed
        is_unindexed_residue = is_unindexed_residue.numpy(force=True)

        copy_indexed_residue_mask = ~residue_noising_mask & ~is_unindexed_residue
        copy_unindexed_residue_mask = ~residue_noising_mask & is_unindexed_residue
        copy_atomized_residue_mask = np.zeros_like(res_type_noising_mask)

        return {
            "t": t.numpy(force=True),
            "atom_noising_mask": atom_noising_mask,
            "res_type_noising_mask": res_type_noising_mask,
            "copy_indexed_residue_mask": copy_indexed_residue_mask,
            "copy_unindexed_residue_mask": copy_unindexed_residue_mask,
            "copy_atomized_residue_mask": copy_atomized_residue_mask,
        }

    def max_added_tokens(self, N):
        res_cap = min(math.ceil(self.max_frac_res * N), self.max_num_res)
        return res_cap


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


# inspired by Genie2
class MinimalMotifScaffolding(TrainingTask):
    name: str = "min_motif_scaffolding"
    def __init__(self,
                 prob=0.0,
                 t_sched='lognorm',
                 mode='mixed',
                 lognorm_mu=0.0,
                 lognorm_sig=1.0,
                 beta_p1=1.9,
                 beta_p2=1.0,
                 shift_time_scale=False,
                 t_min=0.01,
                 t_max=0.99,
                 max_res_islands=8,
                 max_res_per_island=3,
                 p_is_unindexed=0.8,
                 name_override=None,
                 p_fix_tip=0.35,
                 p_fix_bb=0.35,
                 p_fix_all=0.30
    ):
        assert t_sched in ['lognorm', 'mixed_beta', 'uniform']
        assert mode in ['mixed']
        self.prob = prob
        self.t_sched = t_sched
        self.mode = mode
        self.lognorm_mu = lognorm_mu
        self.lognorm_sig = lognorm_sig
        self.beta_p1 = beta_p1
        self.beta_p2 = beta_p2
        self.t_min = t_min
        self.t_max = t_max
        self.shift_time_scale = shift_time_scale

        self.max_res_islands = max_res_islands
        self.max_res_per_island = max_res_per_island
        self.p_is_unindexed = p_is_unindexed

        self.p_fix_tip = p_fix_tip
        self.p_fix_bb = p_fix_bb
        self.p_fix_all = p_fix_all
        assert np.isclose(sum([p_fix_tip, p_fix_bb, p_fix_all]), 1.0)

        if name_override is not None:
            self.name = name_override

    def generate_motif_mask(self, N):
        num_segments = np.random.randint(1, self.max_res_islands+1)
        # we switch this from Genie2
        # to increase the probability of sampling minimal motifs
        # we also give the option to lower the maximum number of residues sampled
        # num_res = np.random.randint(math.floor(0.05 * N), math.ceil(0.5 * N) + 1)
        # # when N < 80 it's possible for num_segments > num_res
        # num_res = max(num_segments + 1, num_res)
        L = np.random.choice(self.max_res_per_island, size=num_segments) + 1
        num_res = np.sum(L)
        permute_list = [[0] for _ in range(N - num_res)] + [np.ones((l,)) for l in L]
        permutation = np.random.permutation(len(permute_list))
        permuted_list = [permute_list[i] for i in permutation]
        M = np.concatenate(permuted_list, axis=0)
        return torch.as_tensor(M, dtype=torch.bool)

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

        resolved_mask = data.residues['is_present']
        num_resolved_residues = resolved_mask.sum()
        motif_mask = self.generate_motif_mask(num_resolved_residues)
        residue_noising_mask = np.ones_like(resolved_mask)
        residue_noising_mask[resolved_mask] = ~motif_mask
        res_type_noising_mask = residue_noising_mask.copy()

        atom_noising_mask = []
        if self.mode == 'mixed':
            noise_select = torch.rand_like(torch.as_tensor(residue_noising_mask), dtype=torch.float32).numpy(force=True)
            fix_tip = noise_select < self.p_fix_tip
            noise_sidechain_only = noise_select > (1 - self.p_fix_bb)
            res_type_noising_mask[~residue_noising_mask & fix_tip] = False
            res_type_noising_mask[~residue_noising_mask & noise_sidechain_only] = True

            for chain in data.chains:
                # Get residue indices
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]
                is_protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]

                for i, residue in enumerate(data.residues[res_start:res_end]):
                    atom_idx = residue['atom_idx']
                    atom_num = residue['atom_num']
                    atoms = data.atoms[atom_idx:atom_idx+atom_num]
                    if residue["is_standard"] and (residue['name'] != 'UNK') and is_protein:
                        if fix_tip[i] & (~residue_noising_mask[i]):
                            rigid_noise_mask = [True, False, True]
                        elif noise_sidechain_only[i] & (~residue_noising_mask[i]):
                            rigid_noise_mask = [False, True, True]
                        elif ~residue_noising_mask[i]:
                            rigid_noise_mask = [False for _ in range(3)]
                        else:
                            rigid_noise_mask = [True for _ in range(3)]
                        atom_noising_mask.append(rigid_noise_to_atom_noise(residue, atoms, rigid_noise_mask))
                    else:
                        atom_noising_mask.append([True for _ in atoms])

        else:
            raise ValueError()
        atom_noising_mask = np.concatenate(atom_noising_mask)

        is_unindexed_residue = torch.rand_like(torch.as_tensor(residue_noising_mask), dtype=torch.float32) < self.p_is_unindexed
        is_unindexed_residue = is_unindexed_residue.numpy(force=True)

        copy_indexed_residue_mask = ~residue_noising_mask & ~is_unindexed_residue
        copy_unindexed_residue_mask = ~residue_noising_mask & is_unindexed_residue
        copy_atomized_residue_mask = np.zeros_like(res_type_noising_mask)

        return {
            "t": t.numpy(force=True),
            "atom_noising_mask": atom_noising_mask,
            "res_type_noising_mask": res_type_noising_mask,
            "copy_indexed_residue_mask": copy_indexed_residue_mask,
            "copy_unindexed_residue_mask": copy_unindexed_residue_mask,
            "copy_atomized_residue_mask": copy_atomized_residue_mask,
        }

    def max_added_tokens(self, N):
        res_cap = self.max_res_per_island * self.max_res_islands
        return res_cap
