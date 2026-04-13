from collections import deque

import numpy as np
import torch

from .task import TrainingTask


class MolSequentialScaffolding(TrainingTask):
    """Sequential two-group denoising task for small molecules.

    Randomly partitions atoms into two groups. Both groups are noised at the
    same timestep t. Training performs two forward passes:
      Pass 1: both groups noised, loss on group 1 only, backward to free graph.
      Pass 2: group 1 fixed at its denoised output from pass 1, group 2 still
              noised, loss on group 2.

    This trains the model for sequential inference where group 1 is generated
    first and then used as context to generate group 2.
    """
    name: str = "mol_sequential_scaffolding"

    def __init__(
        self,
        prob=0.0,
        t_sched='lognorm',
        lognorm_mu=0.0,
        lognorm_sig=1.0,
        t_min=0.01,
        t_max=0.99,
        min_frac_atoms=0.1,
        max_frac_atoms=0.5,
    ):
        assert t_sched in ['lognorm', 'uniform']
        self.prob = prob
        self.t_sched = t_sched
        self.lognorm_mu = lognorm_mu
        self.lognorm_sig = lognorm_sig
        self.t_min = t_min
        self.t_max = t_max
        self.min_frac_atoms = min_frac_atoms
        self.max_frac_atoms = max_frac_atoms

    def sample_t_and_mask(self, data):
        atoms = data.atoms
        residues = data.residues
        n_atoms = atoms.shape[0]
        n_res = residues.shape[0]

        if self.t_sched == 'lognorm':
            ln_sig = self.lognorm_mu + torch.randn(1).float() * self.lognorm_sig
            t = torch.sigmoid(ln_sig)
        else:
            t = torch.rand(1).float()

        # Random fraction assigned to group 1
        min_k = max(1, int(self.min_frac_atoms * n_atoms))
        max_k = max(min_k, int(self.max_frac_atoms * n_atoms))
        k = np.random.randint(min_k, max_k + 1)

        perm = np.random.permutation(n_atoms)
        group1_atom_mask = np.zeros(n_atoms, dtype=bool)
        group1_atom_mask[perm[:k]] = True

        # Both groups are noised
        atom_noising_mask = np.ones(n_atoms, dtype=bool)

        res_type_noising_mask = np.zeros(n_res, dtype=bool)
        copy_indexed_residue_mask = np.zeros(n_res, dtype=bool)
        copy_unindexed_residue_mask = np.zeros(n_res, dtype=bool)
        copy_atomized_residue_mask = np.zeros(n_res, dtype=bool)

        return {
            "t": t.numpy(force=True),
            "group1_atom_mask": group1_atom_mask,
            "atom_noising_mask": atom_noising_mask,
            "res_type_noising_mask": res_type_noising_mask,
            "copy_indexed_residue_mask": copy_indexed_residue_mask,
            "copy_unindexed_residue_mask": copy_unindexed_residue_mask,
            "copy_atomized_residue_mask": copy_atomized_residue_mask,
        }

    def max_added_tokens(self, _):
        return 0
