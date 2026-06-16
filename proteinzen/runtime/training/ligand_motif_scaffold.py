import math
import numpy as np
import torch

from proteinzen.boltz.data import const

from .task import TrainingTask
from .chain_conditioning import _make_t
from .motif_scaffold import rigid_noise_to_atom_noise


class JointProteinLigandGeneration(TrainingTask):
    """Jointly generate protein and ligand from scratch — noise everything."""
    name: str = "joint_protein_ligand_generation"

    def __init__(
        self,
        prob=0.0,
        t_sched="lognorm",
        lognorm_mu=0.0,
        lognorm_sig=1.0,
        beta_p1=1.9,
        beta_p2=1.0,
        t_min=0.01,
        t_max=0.99,
    ):
        assert t_sched in ["lognorm", "mixed_beta", "uniform"]
        self.prob = prob
        self.t_sched = t_sched
        self.lognorm_mu = lognorm_mu
        self.lognorm_sig = lognorm_sig
        self.beta_p1 = beta_p1
        self.beta_p2 = beta_p2
        self.t_min = t_min
        self.t_max = t_max

    def sample_t_and_mask(self, data):
        t = _make_t(
            self.t_sched, self.lognorm_mu, self.lognorm_sig,
            self.beta_p1, self.beta_p2, self.t_min, self.t_max,
        )
        n_atoms = data.atoms.shape[0]
        n_residues = data.residues.shape[0]
        return {
            "t": t,
            "atom_noising_mask": np.ones(n_atoms, dtype=bool),
            "res_type_noising_mask": np.ones(n_residues, dtype=bool),
            "copy_indexed_residue_mask": np.zeros(n_residues, dtype=bool),
            "copy_unindexed_residue_mask": np.zeros(n_residues, dtype=bool),
            "copy_atomized_residue_mask": np.zeros(n_residues, dtype=bool),
        }

    def max_added_tokens(self, _):
        return 0


class LigandConditionedMotifScaffolding(TrainingTask):
    """Fix ligand + protein binding-site motif, generate rest of protein.

    The ligand is held fixed via the noising mask (no copy mechanism).
    A random subset of protein residues within ``binding_site_radius`` Å of
    the ligand are selected as the motif and also fixed; the remainder of the
    protein is generated from noise.
    """
    name: str = "ligand_conditioned_motif_scaffolding"

    def __init__(
        self,
        prob=0.0,
        t_sched="lognorm",
        lognorm_mu=0.0,
        lognorm_sig=1.0,
        beta_p1=1.9,
        beta_p2=1.0,
        t_min=0.01,
        t_max=0.99,
        binding_site_radius=6.0,
        max_num_motif_res=20,
        p_is_unindexed=0.8,
    ):
        assert t_sched in ["lognorm", "mixed_beta", "uniform"]
        self.prob = prob
        self.t_sched = t_sched
        self.lognorm_mu = lognorm_mu
        self.lognorm_sig = lognorm_sig
        self.beta_p1 = beta_p1
        self.beta_p2 = beta_p2
        self.t_min = t_min
        self.t_max = t_max
        self.binding_site_radius = binding_site_radius
        self.max_num_motif_res = max_num_motif_res
        self.p_is_unindexed = p_is_unindexed

    def sample_t_and_mask(self, data):
        t = _make_t(
            self.t_sched, self.lognorm_mu, self.lognorm_sig,
            self.beta_p1, self.beta_p2, self.t_min, self.t_max,
        )

        residues = data.residues
        atoms = data.atoms
        chains = data.chains

        n_atoms = atoms.shape[0]
        n_residues = residues.shape[0]

        atom_noising_mask = np.ones(n_atoms, dtype=bool)
        res_type_noising_mask = np.ones(n_residues, dtype=bool)
        # tracks protein motif residues only (ligand NOT included)
        residue_noising_mask = np.ones(n_residues, dtype=bool)

        ligand_mol_type = const.chain_type_ids["NONPOLYMER"]
        protein_mol_type = const.chain_type_ids["PROTEIN"]

        # Fix ligand chains
        ligand_atom_coords = []
        for chain in chains:
            if int(chain["mol_type"]) != ligand_mol_type:
                continue
            res_start = int(chain["res_idx"])
            res_end = res_start + int(chain["res_num"])
            for res in residues[res_start:res_end]:
                atom_idx = int(res["atom_idx"])
                atom_num = int(res["atom_num"])
                atom_noising_mask[atom_idx:atom_idx + atom_num] = False
                for atom in atoms[atom_idx:atom_idx + atom_num]:
                    if atom["is_present"]:
                        ligand_atom_coords.append(atom["coords"].copy())
            res_type_noising_mask[res_start:res_end] = False

        if len(ligand_atom_coords) == 0:
            raise ValueError("LigandConditionedMotifScaffolding requires a NONPOLYMER chain but none was found")

        ligand_coords = np.array(ligand_atom_coords)  # (L, 3)

        # Find protein residues within binding_site_radius of any ligand atom
        potential_binding_site = np.zeros(n_residues, dtype=bool)
        for chain in chains:
            if int(chain["mol_type"]) != protein_mol_type:
                continue
            res_start = int(chain["res_idx"])
            res_end = res_start + int(chain["res_num"])
            for i in range(res_end - res_start):
                res_idx = res_start + i
                res = residues[res_idx]
                if not res["is_present"] or not res["is_standard"] or str(res["name"]).strip() == "UNK":
                    continue
                atom_idx = int(res["atom_idx"])
                atom_num = int(res["atom_num"])
                present_coords = [
                    atoms[atom_idx + j]["coords"].copy()
                    for j in range(atom_num)
                    if atoms[atom_idx + j]["is_present"]
                ]
                if not present_coords:
                    continue
                centroid = np.mean(present_coords, axis=0)
                min_dist = np.linalg.norm(ligand_coords - centroid[None], axis=-1).min()
                if min_dist < self.binding_site_radius:
                    potential_binding_site[res_idx] = True

        num_potential = int(potential_binding_site.sum())

        copy_indexed_residue_mask = np.zeros(n_residues, dtype=bool)
        copy_unindexed_residue_mask = np.zeros(n_residues, dtype=bool)
        copy_atomized_residue_mask = np.zeros(n_residues, dtype=bool)

        # Fixed ligand residues are atomized copy tokens
        for chain in chains:
            if int(chain["mol_type"]) != ligand_mol_type:
                continue
            res_start = int(chain["res_idx"])
            res_end = res_start + int(chain["res_num"])
            copy_atomized_residue_mask[res_start:res_end] = True

        if num_potential > 0:
            num_motif = np.random.randint(1, min(num_potential, self.max_num_motif_res) + 1)
            potential_indices = np.where(potential_binding_site)[0]
            motif_indices = np.random.choice(potential_indices, size=num_motif, replace=False)

            for res_idx in motif_indices:
                res = residues[res_idx]
                atom_idx = int(res["atom_idx"])
                atom_num = int(res["atom_num"])
                res_atoms = atoms[atom_idx:atom_idx + atom_num]
                per_atom = rigid_noise_to_atom_noise(res, res_atoms, [False, False, False])
                atom_noising_mask[atom_idx:atom_idx + atom_num] = per_atom
                res_type_noising_mask[res_idx] = False
                residue_noising_mask[res_idx] = False

            is_unindexed = np.random.rand(n_residues) < self.p_is_unindexed
            copy_indexed_residue_mask = ~residue_noising_mask & ~is_unindexed
            copy_unindexed_residue_mask = ~residue_noising_mask & is_unindexed

        return {
            "t": t,
            "atom_noising_mask": atom_noising_mask,
            "res_type_noising_mask": res_type_noising_mask,
            "copy_indexed_residue_mask": copy_indexed_residue_mask,
            "copy_unindexed_residue_mask": copy_unindexed_residue_mask,
            "copy_atomized_residue_mask": copy_atomized_residue_mask,
        }

    def max_added_tokens(self, _):
        return self.max_num_motif_res
