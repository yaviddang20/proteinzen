from functools import partialmethod
from typing import Tuple, Optional

import math
import numpy as np
import torch
import random
from itertools import combinations

from scipy.spatial.distance import cdist
import scipy

from proteinzen.boltz.data import const

from proteinzen.data.constants import coarse_grain as cg
from proteinzen.data.featurize.tokenize import convert_atom_str_to_tuple
from proteinzen.data.featurize.cropper import pick_random_token

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

def np_nearest(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    r"""Clusters points in :obj:`x` together which are nearest to a given query
    point in :obj:`y`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.

    :rtype: :class:`NDArray`
    """
    assert x.ndim == 2
    assert y.ndim == 2

    min_xy = min(x.min().item(), y.min().item())
    x, y = x - min_xy, y - min_xy

    max_xy = max(x.max().item(), y.max().item())
    x /= max_xy
    y /= max_xy

    return scipy.cluster.vq.vq(x, y)[0]


class BinderDesignWithBindingMotif(TrainingTask):
    name: str = "binder_design_with_binding_motif"
    def __init__(
        self,
        noise_target,
        prob=0.0,
        t_sched='lognorm',
        lognorm_mu=0.0,
        lognorm_sig=1.0,
        mixed_lognorm_mix_rate=0.02,
        mixed_beta_mix_rate=0.02,
        beta_p1=1.9,
        beta_p2=1.0,
        t_min=0.01,
        t_max=0.99,
        motif_is_unindexed=True,
        max_num_motif_res=40,
        motif_redesign_seq_rate=0,
        motif_repack_rate=0,
        name_override=None,
        p_binding=0.15,
        p_nonbinding=0.075,
        p_both=0.075,
        p_none=0.70,
        max_noised_chains=None,
        interface_num_res_cap=None,
        prefilter_possible_interfaces=False,
        repack_target=True,
    ):
        assert t_sched in ['lognorm', 'mixed_lognorm', 'mixed_beta', 'uniform']
        self.noise_target = noise_target
        self.motif_is_unindexed = motif_is_unindexed
        self.motif_redesign_seq_rate = motif_redesign_seq_rate
        self.motif_repack_rate = motif_repack_rate
        self.max_num_motif_res = max_num_motif_res
        self.prob = prob
        self.t_sched = t_sched
        self.lognorm_mu = lognorm_mu
        self.lognorm_sig = lognorm_sig
        self.mixed_lognorm_mix_rate = mixed_lognorm_mix_rate
        self.mixed_beta_mix_rate = mixed_beta_mix_rate
        self.beta_p1 = beta_p1
        self.beta_p2 = beta_p2
        self.t_min = t_min
        self.t_max = t_max
        self.p_binding = p_binding
        self.p_nonbinding = p_nonbinding
        self.p_both = p_both
        self.p_none = p_none
        self.max_noised_chains = max_noised_chains
        self.interface_num_res_cap = interface_num_res_cap
        self.prefilter_possible_interfaces = prefilter_possible_interfaces
        self.repack_target = repack_target

        if name_override is not None:
            self.name = name_override

    def _return_fallback(self, data):
        residue_noising_mask = np.ones(data.residues['is_present'].shape, dtype=bool)
        chain_noising_mask = np.ones(len(data.chains), dtype=bool)
        residue_hotspot_type = np.zeros(data.residues['is_present'].shape)
        seed_interface = None
        return chain_noising_mask, residue_hotspot_type, seed_interface, residue_noising_mask

    def _get_chain_scope(self, data):
        # we select valid chains as those that are both valid and involved in an interface
        valid_interface_chain_ids = set(data.interfaces['chain_1'].tolist() + data.interfaces['chain_2'].tolist())
        valid_interface_chain_ids = np.array(sorted(valid_interface_chain_ids))
        valid_chain_mask = data.mask.copy()
        valid_interface_chains_mask = np.zeros(len(data.chains), dtype=bool)
        valid_interface_chains_mask[valid_interface_chain_ids] = True
        valid_chain_mask &= valid_interface_chains_mask
        valid_chains = data.chains[valid_chain_mask]

        # select a random subset of chains to be noised
        internal_chain_idx = np.arange(len(valid_chains))
        if self.max_noised_chains is not None:
            num_redesign_chains = np.random.randint(1, self.max_noised_chains+1)
        else:
            num_redesign_chains = np.random.randint(1, len(valid_chains))

        select_redesign_chains = np.random.choice(
            internal_chain_idx,
            size=num_redesign_chains,
            replace=False
        )
        chain_noising_mask = np.zeros(len(data.chains), dtype=bool)
        valid_chain_noising_mask = np.zeros(len(valid_chains), dtype=bool)
        valid_chain_noising_mask[select_redesign_chains] = True
        chain_noising_mask[valid_chain_mask] = valid_chain_noising_mask

        # based on which chains are noised, choose a seed interface for cropping
        valid_interfaces = data.interfaces
        valid_interfaces = valid_interfaces[chain_noising_mask[valid_interfaces['chain_1']] | chain_noising_mask[valid_interfaces['chain_2']]]
        seed_interface = np.random.choice(
            valid_interfaces
        )
        seed_interface = tuple(sorted(
            [seed_interface['chain_1'], seed_interface['chain_2']]
        ))

        return valid_chains, select_redesign_chains, chain_noising_mask, seed_interface


    def select_interface(self, data, hotspot_mode):
        # inspired by boltzgen
        assert hotspot_mode in ["binding", "nonbinding", "both", "none"]

        mask = data.mask
        residues = data.residues

        residue_noising_mask = np.ones(data.residues['is_present'].shape, dtype=bool)

        if len(data.interfaces) < 1:
            return self._return_fallback(data)

        valid_chains, select_redesign_chains, chain_noising_mask, seed_interface = self._get_chain_scope(data)
        # figure out which residues are interface residues
        # we'll do this at the atomic scale to account for residue size
        atoms_fixed = []
        atoms_redesign = []
        atoms_fixed_parent = []
        atoms_redesign_parent = []
        atoms_fixed_chain = []
        atoms_redesign_chain = []

        for i, chain_i in enumerate(valid_chains):
            res_start = chain_i['res_idx']
            res_end = res_start + chain_i['res_num']
            chain_residues = residues[res_start:res_end]

            if i in select_redesign_chains:
                for j, _res in enumerate(chain_residues):
                    if data.residues['is_present'][res_start + j]:
                        atoms_redesign_parent.append(
                            np.full((_res['atom_num'],), res_start + j)
                        )
                        atoms_redesign_chain.append(
                            np.full((_res['atom_num'],), chain_i['asym_id'])
                        )
                        res_atom_start = _res['atom_idx']
                        res_atom_end = res_atom_start + _res["atom_num"]
                        atoms_redesign.append(data.atoms[res_atom_start:res_atom_end])
            else:
                for j, _res in enumerate(chain_residues):
                    if data.residues['is_present'][res_start + j]:
                        atoms_fixed_parent.append(
                            np.full((_res['atom_num'],), res_start + j)
                        )
                        atoms_fixed_chain.append(
                            np.full((_res['atom_num'],), chain_i['asym_id'])
                        )
                        res_atom_start = _res['atom_idx']
                        res_atom_end = res_atom_start + _res["atom_num"]
                        atoms_fixed.append(data.atoms[res_atom_start:res_atom_end])
                residue_noising_mask[res_start:res_end] = False

        atoms_fixed = np.concatenate(atoms_fixed)
        atoms_redesign = np.concatenate(atoms_redesign)
        atoms_fixed_parent = np.concatenate(atoms_fixed_parent)
        atoms_redesign_parent = np.concatenate(atoms_redesign_parent)

        atoms_fixed_chain = np.concatenate(atoms_fixed_chain)
        atoms_redesign_chain = np.concatenate(atoms_redesign_chain)

        atoms_fixed_parent = atoms_fixed_parent[atoms_fixed['is_present']]
        atoms_fixed_chain = atoms_fixed_chain[atoms_fixed['is_present']]
        atoms_redesign_parent = atoms_redesign_parent[atoms_redesign['is_present']]
        atoms_redesign_chain = atoms_redesign_chain[atoms_redesign['is_present']]
        atoms_fixed = atoms_fixed[atoms_fixed['is_present']]
        atoms_redesign = atoms_redesign[atoms_redesign['is_present']]

        # dists = cdist(atoms_fixed["coords"], atoms_redesign["coords"])
        nearest_key_idx = np_nearest(atoms_fixed["coords"], atoms_redesign["coords"])
        dists = np.linalg.vector_norm(atoms_fixed["coords"] - atoms_redesign["coords"][nearest_key_idx], axis=-1)
        interface_cutoff = const.atom_interface_cutoff + abs(np.random.randn()) # np.random.randn()
        cutoff = dists < interface_cutoff

        select_interface_fixed = cutoff
        interface_res_idx_fixed = np.unique(atoms_fixed_parent[select_interface_fixed])
        # print(len(interface_res_idx_fixed))

        if self.interface_num_res_cap is not None:
            proceed_to_binder_design = len(interface_res_idx_fixed) >= self.interface_num_res_cap
            if not proceed_to_binder_design:
                print(f"unconditional design because we have {len(interface_res_idx_fixed)} < {self.interface_num_res_cap} interface residues")
        else:
            proceed_to_binder_design = cutoff.any()

        if proceed_to_binder_design:
            nonbinding_res_idx_fixed = np.setdiff1d(
                np.arange(len(residues)),
                interface_res_idx_fixed
            )
            residue_hotspot_type = np.zeros(data.residues['is_present'].shape)

            if hotspot_mode in ['binding', 'both']:
                interface_size = len(interface_res_idx_fixed)
                subset_interface_res_idx = np.random.choice(interface_res_idx_fixed, size=np.random.randint(1, interface_size + 1))
                # print("WARNING: SELECTING ALL INTERFACE RESIDUES (this is debug code you should not be training)")
                # subset_interface_res_idx = interface_res_idx_fixed
                residue_hotspot_type[subset_interface_res_idx] = 1
            if hotspot_mode in ['nonbinding', 'both']:
                nonbinding_size = len(nonbinding_res_idx_fixed)
                subset_nonbinding_res_idx = np.random.choice(nonbinding_res_idx_fixed, size=np.random.randint(1, nonbinding_size + 1))
                residue_hotspot_type[subset_nonbinding_res_idx] = 2

            residues_fixed_idx = np.unique(atoms_fixed_parent)
            residues_redesign_idx = np.unique(atoms_redesign_parent)
            # TODO this is extrememly roundabout
            residues_fixed = data.atoms[
                data.residues[residues_fixed_idx]["atom_center"]
            ]["coords"]
            residues_redesign = data.atoms[
                data.residues[residues_redesign_idx]["atom_center"]
            ]["coords"]
            nearest_residue_idx = np_nearest(
                residues_redesign,
                residues_fixed,
            )
            res_dists = np.linalg.vector_norm(residues_redesign - residues_fixed[nearest_residue_idx], axis=-1)
            res_dists_idx_sort = np.argsort(res_dists)

            num_motif_res = min(
                np.random.randint(1, res_dists_idx_sort.shape[0]),
                self.max_num_motif_res
            )
            # print(residues_fixed_idx.shape, residues_redesign_idx.shape, nearest_residue_idx.shape, res_dists_idx_sort.shape)
            residues_redesign_motif = residues_redesign_idx[res_dists_idx_sort[:num_motif_res]]
            residue_noising_mask[residues_redesign_motif] = False

            return chain_noising_mask, residue_hotspot_type, seed_interface, residue_noising_mask

        else:
            return self._return_fallback(data)


    def _sample_t(self):
        device = 'cpu'
        if self.t_sched == 'lognorm':
            ln_sig = self.lognorm_mu + torch.randn(1, device=device).float() * self.lognorm_sig
            t = torch.sigmoid(ln_sig)
        elif self.t_sched == 'mixed_lognorm':
            u = torch.rand(1)
            if u < self.mixed_lognorm_mix_rate:
                t = torch.rand(1, device=device).float()
            else:
                ln_sig = self.lognorm_mu + torch.randn(1, device=device).float() * self.lognorm_sig
                t = torch.sigmoid(ln_sig)
        elif self.t_sched == 'mixed_beta':
            u = torch.rand(1)
            if u < self.mixed_beta_mix_rate:
                t = torch.rand(1, device=device).float()
            else:
                dist = torch.distributions.beta.Beta(self.beta_p1, self.beta_p2)
                t = dist.sample((1,)).to(device)
        elif self.t_sched == 'uniform':
            t = torch.rand(1, device=device).float()
        else:
            raise ValueError(f"self.t_sched={self.t_sched} not recognized")

        return t

    def sample_t_and_mask(self, data):
        atoms = data.atoms
        t = self._sample_t()

        hotspot_mode = np.random.choice(
            ["binding", "nonbinding", "both", "none"],
            p=[
                self.p_binding,
                self.p_nonbinding,
                self.p_both,
                self.p_none
            ]
        )
        chain_noising_mask, residue_hotspot_type, seed_interface, residue_noising_mask = self.select_interface(data, hotspot_mode)

        atom_noising_mask = []
        res_type_noising_mask = np.ones(residue_hotspot_type.shape, dtype=bool)
        is_unindexed_residue = np.zeros(res_type_noising_mask.shape, dtype=bool)
        copy_indexed_residue_mask = is_unindexed_residue.copy()
        copy_unindexed_residue_mask = is_unindexed_residue.copy()

        motif_repack_mask = np.random.rand(len(residue_hotspot_type)) < self.motif_repack_rate
        motif_redesign_seq_mask = np.random.rand(len(residue_hotspot_type)) < self.motif_redesign_seq_rate
        # only redesign residues which were meant to be repacked in the first place
        # this makes the net redesign rate motif_repack_rate * motif_redesign_seq_rate
        motif_redesign_seq_mask = motif_redesign_seq_mask & motif_repack_mask

        # torch.set_printoptions(threshold=10000000000000)
        # print(residue_noising_mask)

        for i_chain, chain in enumerate(data.chains):
            # Get residue indices
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            is_protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]

            noise_chain = chain_noising_mask[i_chain]

            for i, residue in enumerate(data.residues[res_start:res_end]):
                atom_idx = residue['atom_idx']
                atom_num = residue['atom_num']
                atoms = data.atoms[atom_idx:atom_idx+atom_num]

                noise_residue = residue_noising_mask[res_start + i]
                is_motif_residue = noise_chain and not noise_residue

                redesign_seq = motif_redesign_seq_mask[res_start+i]
                repack_motif_sidechain = motif_repack_mask[res_start+i]

                # noise any atom which is in a noised residue
                # TODO: what if the noised residue is a motif?
                if residue["is_standard"] and (residue['name'] != 'UNK') and is_protein:
                    if noise_residue:
                        atom_noising_mask.append([True for _ in atoms])
                        # print([True for _ in atoms])
                    elif is_motif_residue:
                        atom_noising_mask.append(
                            rigid_noise_to_atom_noise(
                                residue, atoms,
                                [False, redesign_seq or repack_motif_sidechain, redesign_seq or repack_motif_sidechain]
                            )
                        )
                    else:
                        atom_noising_mask.append(
                            rigid_noise_to_atom_noise(
                                residue, atoms,
                                [False, self.repack_target, self.repack_target]
                            )
                        )
                else:
                    atom_noising_mask.append([True for _ in atoms])

                # noise the restype of any residue which is physically noised
                if redesign_seq:
                    res_type_noising_mask[res_start + i] = True
                else:
                    res_type_noising_mask[res_start + i] = noise_residue

                if is_motif_residue:
                    if self.motif_is_unindexed:
                        copy_unindexed_residue_mask[res_start + i] = True
                    else:
                        copy_indexed_residue_mask[res_start + i] = True

        atom_noising_mask = np.concatenate(atom_noising_mask)
        copy_atomized_residue_mask = np.zeros_like(res_type_noising_mask)

        return {
            "t": t.numpy(force=True),
            "atom_noising_mask": atom_noising_mask,
            "res_type_noising_mask": res_type_noising_mask,
            "res_hotspot_type": residue_hotspot_type,
            "copy_indexed_residue_mask": copy_indexed_residue_mask,
            "copy_unindexed_residue_mask": copy_unindexed_residue_mask,
            "copy_atomized_residue_mask": copy_atomized_residue_mask,
            "seed_interface": seed_interface
        }

    def max_added_tokens(self, N):
        del N
        return self.max_num_motif_res

