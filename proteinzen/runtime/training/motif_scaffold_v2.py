from functools import partialmethod

import math
import numpy as np
import torch

import torch_cluster

from .task import TrainingTask

# inspired by Genie2
class MotifScaffoldingV2(TrainingTask):
    name: str = "motif_scaffolding_v2"
    def __init__(self,
                 prob=0.0,
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
                 name_override=None
    ):
        assert t_sched in ['lognorm', 'mixed_beta', 'uniform']
        assert mode in ['backbone', 'full_residue', 'inv_rotamer', 'mixed']
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
        self.p_is_unindexed = p_is_unindexed

        if name_override is not None:
            self.name = name_override

    def generate_motif_mask(self, N):
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

    def sample_t_and_mask(self, data):
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
        elif self.t_sched == 'uniform':
            t = torch.rand(1, device=device).float()
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
        elif self.mode == 'mixed':
            mask_bb_only = torch.rand_like(rigids_noising_mask[..., 0], dtype=torch.float32) < 0.5
            rigids_noising_mask[motif_mask & mask_bb_only, 0] = False
            rigids_noising_mask[motif_mask & (~mask_bb_only)] = False
            seq_noising_mask[motif_mask & mask_bb_only] = True
        else:
            raise ValueError()


        res_is_unindexed_mask = torch.rand_like(seq_noising_mask, dtype=torch.float32) < self.p_is_unindexed
        res_is_atomized_mask = torch.zeros_like(res_is_unindexed_mask, dtype=torch.bool)

        return {
            "t": t,
            "rigids_noising_mask": rigids_noising_mask,
            "seq_noising_mask": seq_noising_mask,
            "res_is_unindexed_mask": res_is_unindexed_mask,
            "res_is_atomized_mask": res_is_atomized_mask
        }

class ResidueMotifScaffoldingV2(MotifScaffoldingV2):
    name: str = "res_motif_scaffolding_v2"
    __init__ = partialmethod(MotifScaffoldingV2.__init__, mode='full_residue')

class MixedMotifScaffoldingV2(MotifScaffoldingV2):
    name: str = "mixed_motif_scaffolding_v2"
    __init__ = partialmethod(MotifScaffoldingV2.__init__, mode='mixed')


# inspired by Genie2
class ResidueBallMotifScaffoldingV2(TrainingTask):
    name: str = "motif_scaffolding_v2"
    def __init__(self,
                 prob=0.0,
                 t_sched='lognorm',
                 mode='full_residue',
                 lognorm_mu=0.0,
                 lognorm_sig=1.0,
                 beta_p1=1.9,
                 beta_p2=1.0,
                 shift_time_scale=False,
                 t_min=0.01,
                 t_max=0.99,
                 p_is_unindexed=0.8,
                 name_override=None,
                 extend_radius_min=5,
                 extend_radius_max=10,
                 max_motif_seg=4,
                 max_motif_seg_len=8,
                 min_motif_seg_seq_dist=5,
                 residue_frame_center_idx=0,
                 and_for_radius=False
    ):
        assert t_sched in ['lognorm', 'mixed_beta', 'uniform']
        assert mode in ['backbone', 'full_residue', 'inv_rotamer', 'mixed']
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
        self.extend_radius_min = extend_radius_min
        self.extend_radius_max = extend_radius_max
        self.max_motif_seg = max_motif_seg
        self.max_motif_seg_len = max_motif_seg_len
        self.min_motif_seg_seq_dist = min_motif_seg_seq_dist
        self.p_is_unindexed = p_is_unindexed
        self.residue_frame_center_idx = residue_frame_center_idx
        self.and_for_radius = and_for_radius

        if name_override is not None:
            self.name = name_override

    def _generate_motif_seg_lens(self):
        num_segments = np.random.randint(1, self.max_motif_seg + 1)
        if num_segments == 1:
            # make the minimum size of a one-seg motif 3
            L = np.random.choice(self.max_motif_seg_len - 3, size=num_segments) + 3
        else:
            L = np.random.choice(self.max_motif_seg_len - 1, size=num_segments) + 1
        permute_len_list = [l for l in L]
        permutation = np.random.permutation(len(permute_len_list))
        permuted_len_list = [permute_len_list[i] for i in permutation]
        return permuted_len_list

    def generate_motif_mask(self, coords):
        N = coords.shape[0]
        device = coords.device
        # compute motif lens
        motif_lens = self._generate_motif_seg_lens()
        # compute the seq and spatial distance between residues
        spatial_dist_mat = torch.cdist(coords[None], coords[None]).squeeze(0)
        seq_dist_mat = torch.arange(N, device=device)[:, None] - torch.arange(N, device=device)[None]
        # seq_dist_mat = torch.abs(seq_dist_mat)

        # compute which residues belong to the motif
        motif_idx = []
        seed_motif_seg_len = motif_lens[0]
        seed_motif_seg_N_len = np.random.choice(seed_motif_seg_len + 1)
        seed_motif_seg_C_len = seed_motif_seg_len - seed_motif_seg_N_len
        # first, seed the motif by selecting a random center for the first segment
        # we try to choose a "good seed first"
        # e.g. one that could accept an additional motif segment
        spatial_select = spatial_dist_mat < self.extend_radius_min
        # negative seq dist means the candidate residue is more C terminal than the segment
        # so we need to account for the "n terminal length" of the segment
        N_seq_dist_select = (seq_dist_mat < -(self.min_motif_seg_seq_dist + seed_motif_seg_N_len))
        # positive seq dist means the candidate residue is more N terminal than the segment
        # so we need to account for the "c terminal length" of the segment
        C_seq_dist_select = (seq_dist_mat > self.min_motif_seg_seq_dist + seed_motif_seg_C_len)
        suitable_candidates = spatial_select & (N_seq_dist_select | C_seq_dist_select)

        suitable_candidates[:seed_motif_seg_N_len] = False
        if seed_motif_seg_C_len > 0:
            suitable_candidates[-seed_motif_seg_C_len:] = False

        if suitable_candidates.any():
            candidates = suitable_candidates.any(dim=-1)
            candidate_idx = torch.arange(N, device=device)[candidates]
            select_idx = np.random.choice(candidate_idx.shape[0])
            seed_idx = candidate_idx[select_idx]
        else:
            # we fallback on randomly selecting a seed idx
            seed_idx = np.random.choice(N - seed_motif_seg_len) + seed_motif_seg_N_len
        seed_motif_idx = torch.arange(seed_motif_seg_len, device=device) + seed_idx - seed_motif_seg_N_len
        motif_idx.append(seed_motif_idx)

        for i in range(1, len(motif_lens)):
            # for every following segment, attempt to find segments where
            # the next motif center which is within the expansion radius
            # but far enough away in sequence space
            # if this is possible, then expand around that to the motif segment length
            # otherwise, we terminate and return the motif
            seg_len = motif_lens[i]
            seg_N_len = np.random.choice(seg_len + 1)
            seg_C_len = seg_len - seg_N_len
            extend_radius = torch.rand(1, device=device) * (self.extend_radius_max - self.extend_radius_min) + self.extend_radius_min
            keep_based_on_seq = torch.ones(N, dtype=torch.bool, device=device)
            keep_based_on_seq[:seg_N_len] = False
            if seg_C_len > 0:
                keep_based_on_seq[-seg_C_len:] = False

            if self.and_for_radius:
                keep_based_on_radius = keep_based_on_seq.clone()
            else:
                keep_based_on_radius = torch.zeros_like(keep_based_on_seq)

            for motif_seg_idx in motif_idx:
                within_radius = (spatial_dist_mat[motif_seg_idx] < extend_radius).any(dim=0)
                # negative seq dist means the candidate residue is more C terminal than the segment
                # so we need to account for the "n terminal length" of the segment
                outside_N_seq_dist = (seq_dist_mat[motif_seg_idx] < -(self.min_motif_seg_seq_dist + seg_N_len))
                # positive seq dist means the candidate residue is more N terminal than the segment
                # so we need to account for the "c terminal length" of the segment
                outside_C_seq_dist = (seq_dist_mat[motif_seg_idx] > self.min_motif_seg_seq_dist + seg_C_len)
                outside_seq_dist = (outside_N_seq_dist | outside_C_seq_dist).all(dim=0)
                keep_based_on_seq = keep_based_on_seq & outside_seq_dist
                if self.and_for_radius:
                    keep_based_on_radius = keep_based_on_radius & within_radius
                else:
                    keep_based_on_radius = keep_based_on_radius | within_radius
                # print(i, len(motif_lens), seg_len, keep_based_on_seq, keep_based_on_radius, outside_seq_dist)

            keep = keep_based_on_seq & keep_based_on_radius

            if keep.sum() > 0:
                potential_center_idxs = torch.arange(N, device=device)[keep]
                _select = np.random.choice(potential_center_idxs.shape[0])
                center_idx = potential_center_idxs[_select]
                seg_motif_idx = torch.arange(seg_len, device=device) + center_idx - seg_N_len
                motif_idx.append(seg_motif_idx)
            else:
                continue

        motif_mask = torch.zeros(N, dtype=torch.bool, device=coords.device)
        # print(motif_idx)
        for _idx in motif_idx:
            motif_mask[_idx] = True
        return motif_mask

    def sample_t_and_mask(self, data):
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
        elif self.t_sched == 'uniform':
            t = torch.rand(1, device=device).float()
        else:
            raise ValueError(f"self.t_sched={self.t_sched} not recognized")
        rigids_noising_mask = torch.ones(rigids_1.shape[:-1], dtype=bool, device=device)

        motif_mask = self.generate_motif_mask(rigids_1[:, self.residue_frame_center_idx, 4:])
        # print(data['name'])

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
        elif self.mode == 'mixed':
            mask_bb_only = torch.rand_like(rigids_noising_mask[..., 0], dtype=torch.float32) < 0.5
            rigids_noising_mask[motif_mask & mask_bb_only, 0] = False
            rigids_noising_mask[motif_mask & (~mask_bb_only)] = False
            seq_noising_mask[motif_mask & mask_bb_only] = True
        else:
            raise ValueError()


        res_is_unindexed_mask = torch.rand_like(seq_noising_mask, dtype=torch.float32) < self.p_is_unindexed
        res_is_atomized_mask = torch.zeros_like(res_is_unindexed_mask, dtype=torch.bool)

        return {
            "t": t,
            "rigids_noising_mask": rigids_noising_mask,
            "seq_noising_mask": seq_noising_mask,
            "res_is_unindexed_mask": res_is_unindexed_mask,
            "res_is_atomized_mask": res_is_atomized_mask
        }