from collections import deque

import numpy as np
import torch

from .task import TrainingTask


class MolFragmentScaffolding(TrainingTask):
    """Partial-noising training task for small molecules.

    Randomly selects a connected fragment of heavy atoms (via BFS from a
    random seed) as the scaffold motif. Those atoms are kept at their ground
    truth coordinates; the rest are noised normally. Analogous to
    MotifScaffolding but operating at the atom level within a single molecule.

    The loss is automatically restricted to noised atoms because
    multiframe.py already gates on rigids_noising_mask.
    """
    name: str = "mol_fragment_scaffolding"

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

    def _select_fragment(self, n_atoms, bonds):
        """BFS from a random seed atom to collect a connected fragment.

        Returns a boolean mask (length n_atoms), True = atom is in the motif.
        Falls back gracefully for molecules with no bonds (single atom, etc.).
        """
        adj = [[] for _ in range(n_atoms)]
        for bond in bonds:
            i, j = int(bond['atom_1']), int(bond['atom_2'])
            if 0 <= i < n_atoms and 0 <= j < n_atoms:
                adj[i].append(j)
                adj[j].append(i)

        min_k = max(1, int(self.min_frac_atoms * n_atoms))
        max_k = max(min_k, int(self.max_frac_atoms * n_atoms))
        k = np.random.randint(min_k, max_k + 1)

        seed = np.random.randint(n_atoms)
        visited = {seed}
        frontier = deque([seed])
        while frontier and len(visited) < k:
            v = frontier.popleft()
            neighbors = list(adj[v])
            np.random.shuffle(neighbors)
            for w in neighbors:
                if w not in visited and len(visited) < k:
                    visited.add(w)
                    frontier.append(w)

        mask = np.zeros(n_atoms, dtype=bool)
        mask[list(visited)] = True
        return mask

    def sample_t_and_mask(self, data):
        atoms = data.atoms
        residues = data.residues
        bonds = data.bonds
        n_atoms = atoms.shape[0]
        n_res = residues.shape[0]

        if self.t_sched == 'lognorm':
            ln_sig = self.lognorm_mu + torch.randn(1).float() * self.lognorm_sig
            t = torch.sigmoid(ln_sig)
        else:
            t = torch.rand(1).float()

        fragment_mask = self._select_fragment(n_atoms, bonds)
        atom_noising_mask = ~fragment_mask  # True = noise, False = keep ground truth

        # res_type_noising_mask is irrelevant for non-standard residues
        # (tokenize.py hardcodes seq_noising_mask=False for atomized tokens)
        # but we set it correctly anyway
        res_type_noising_mask = np.zeros(n_res, dtype=bool)

        # No copy tokens needed: fixed atoms are already visible at ground
        # truth coords in the noisy input via atom_noising_mask=False
        copy_indexed_residue_mask = np.zeros(n_res, dtype=bool)
        copy_unindexed_residue_mask = np.zeros(n_res, dtype=bool)
        copy_atomized_residue_mask = np.zeros(n_res, dtype=bool)

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
