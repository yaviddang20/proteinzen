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

    rigid_noise_entry = [0]

    bb_group = ['N', 'CA', 'C', 'O', 'CB']
    # bb_frame = ['C', 'CA', 'N']
    group2 = cg.coarse_grain_sidechain_groups[res_name][2]
    group3 = cg.coarse_grain_sidechain_groups[res_name][3]
    # construct dummy frames as necessary
    # use bb frame if frame2 doesn't exist
    if len(group2) == 0:
        group2 = bb_group
        rigid_noise_entry.append(0)
    else:
        rigid_noise_entry.append(1)
    # use frame2 frame if frame3 doesn't exist
    if len(group3) == 0:
        group3 = group2
        rigid_noise_entry.append(rigid_noise_entry[-1])
    else:
        rigid_noise_entry.append(2)

    atom_noise_mapping = {}
    frame_atom_groups = [bb_group, group2, group3]
    for i, atom_groups in zip(rigid_noise_entry, frame_atom_groups):
        noise_atom = rigid_noising_mask[i]
        for atom_name in atom_groups:
            atom_id = convert_atom_str_to_tuple(atom_name)
            atom_noise_mapping[atom_id] = noise_atom

    # this makes backbone override sidechain for noising CB
    # which is necessary otherwise most sidechains will
    # overwrite backbone denoising via CB
    # TODO: probably a cleaner way of doing this?
    for i, atom_groups in zip([0], frame_atom_groups[:1]):
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
class SidechainRedesign(TrainingTask):
    name: str = "sidechain_redesign"
    def __init__(self,
                 prob=0.0,
                 t_sched='lognorm',
                 lognorm_mu=0.0,
                 lognorm_sig=1.0,
                 mixed_beta_p_uniform=0.02,
                 beta_p1=1.9,
                 beta_p2=1.0,
                 shift_time_scale=False,
                 t_min=0.01,
                 t_max=0.99,
                 name_override=None,
    ):
        assert t_sched in ['lognorm', 'mixed_beta', 'uniform']
        self.prob = prob
        self.t_sched = t_sched
        self.lognorm_mu = lognorm_mu
        self.lognorm_sig = lognorm_sig
        self.mixed_beta_p_uniform = mixed_beta_p_uniform
        self.beta_p1 = beta_p1
        self.beta_p2 = beta_p2
        self.t_min = t_min
        self.t_max = t_max
        self.shift_time_scale = shift_time_scale

        if name_override is not None:
            self.name = name_override


    def sample_t_and_mask(self, data):
        atoms = data.atoms

        device = 'cpu'
        if self.t_sched == 'lognorm':
            ln_sig = self.lognorm_mu + torch.randn(1, device=device).float() * self.lognorm_sig
            t = torch.sigmoid(ln_sig)
        elif self.t_sched == 'mixed_beta':
            u = torch.rand(1)
            if u < self.mixed_beta_p_uniform:
                t = torch.rand(1, device=device).float()
            else:
                dist = torch.distributions.beta.Beta(self.beta_p1, self.beta_p2)
                t = dist.sample((1,)).to(device)
        elif self.t_sched == 'uniform':
            t = torch.rand(1, device=device).float()
        else:
            raise ValueError(f"self.t_sched={self.t_sched} not recognized")

        resolved_mask = data.residues['is_present']
        percent_residues_noised = 1 # torch.rand(1).float()
        percent_sequence_noised = 0 #torch.rand(1).float()
        residue_noising_mask = torch.rand_like(torch.as_tensor(resolved_mask), dtype=torch.float32) < percent_residues_noised
        res_type_noising_mask = torch.rand_like(torch.as_tensor(resolved_mask), dtype=torch.float32) < percent_sequence_noised
        # only noise the residue type of residues that are being noised
        res_type_noising_mask = res_type_noising_mask & residue_noising_mask

        atom_noising_mask = []
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
                    if residue_noising_mask[i]:
                        rigid_noise_mask = [False, True, True]
                    else:
                        rigid_noise_mask = [False for _ in range(3)]
                    atom_noising_mask.append(rigid_noise_to_atom_noise(residue, atoms, rigid_noise_mask))
                    # print(residue, residue_noising_mask[i], rigid_noise_to_atom_noise(residue, atoms, rigid_noise_mask))
                else:
                    atom_noising_mask.append([True for _ in atoms])

        atom_noising_mask = np.concatenate(atom_noising_mask)
        copy_indexed_residue_mask = np.zeros_like(resolved_mask, dtype=bool)
        copy_unindexed_residue_mask = np.zeros_like(resolved_mask, dtype=bool)
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
        return 0


class PocketPLACERTraining(SidechainRedesign):
    """PLACER-inspired training task: fix backbone, noise sidechains + ligand.

    Identical to SidechainRedesign for the noising mask (backbone fixed, sidechains
    and ligand noised), but sets use_placer_centering=True so the corrupter centers
    each noised rigid's noise at its nearest fixed backbone rigid rather than the
    global origin. Intended for use with plinder_pocket_processed data.
    """
    name: str = "pocket_placer"
    use_placer_centering: bool = True

    def __init__(self, side_chain_trans_prior_std=3.0, lig_trans_prior_std=3.0, atomize_sidechains=False, **kwargs):
        super().__init__(**kwargs)
        self.side_chain_trans_prior_std = side_chain_trans_prior_std
        self.lig_trans_prior_std = lig_trans_prior_std
        self.atomize_sidechains = atomize_sidechains