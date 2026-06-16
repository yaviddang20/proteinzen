"""Chain-conditioning tasks for protein-ligand training.

ProteinConditioned: fix protein, generate ligand (de novo drug design)
LigandConditioned:  fix ligand, generate protein pocket (pocket design)
"""
import numpy as np
import torch

from proteinzen.boltz.data import const

from .task import TrainingTask


def _make_t(t_sched, lognorm_mu, lognorm_sig, beta_p1, beta_p2, t_min, t_max):
    if t_sched == "lognorm":
        ln_sig = lognorm_mu + torch.randn(1).float() * lognorm_sig
        t = torch.sigmoid(ln_sig)
    elif t_sched == "uniform":
        t = torch.rand(1).float()
    elif t_sched == "mixed_beta":
        u = torch.rand(1)
        if u < 0.02:
            t = torch.rand(1).float()
        else:
            dist = torch.distributions.beta.Beta(beta_p1, beta_p2)
            t = dist.sample((1,))
    else:
        raise ValueError(f"Unknown t_sched: {t_sched}")
    t = t.clamp(t_min, t_max)
    return t.numpy(force=True)


class _ChainConditioningBase(TrainingTask):
    """Base for chain-level conditioning tasks.

    Parameters
    ----------
    condition_mol_type : str
        Chain type to keep fixed ("PROTEIN" or "NONPOLYMER").
    """

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
        shift_time_scale=False,
        condition_mol_type="PROTEIN",
        p_is_unindexed=0.8,
        max_num_res=40,
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
        self.condition_mol_type = const.chain_type_ids[condition_mol_type]
        self.p_is_unindexed = p_is_unindexed
        self.max_num_res = max_num_res

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
        copy_indexed_residue_mask = np.zeros(n_residues, dtype=bool)
        copy_unindexed_residue_mask = np.zeros(n_residues, dtype=bool)
        copy_atomized_residue_mask = np.zeros(n_residues, dtype=bool)

        is_unindexed = np.random.rand(n_residues) < self.p_is_unindexed
        protein_mol_type = const.chain_type_ids["PROTEIN"]

        # Collect all conditioning residue indices first, then subsample
        conditioning_res_indices = []
        for chain in chains:
            if int(chain["mol_type"]) != self.condition_mol_type:
                continue
            res_start = int(chain["res_idx"])
            res_end = res_start + int(chain["res_num"])
            conditioning_res_indices.extend(range(res_start, res_end))

        if len(conditioning_res_indices) > self.max_num_res:
            conditioning_res_indices = np.random.choice(
                conditioning_res_indices, size=self.max_num_res, replace=False
            ).tolist()

        conditioning_res_set = set(conditioning_res_indices)

        for res_idx in conditioning_res_indices:
            res = residues[res_idx]
            atom_idx = int(res["atom_idx"])
            atom_num = int(res["atom_num"])
            atom_noising_mask[atom_idx:atom_idx + atom_num] = False
            res_type_noising_mask[res_idx] = False

            if self.condition_mol_type == protein_mol_type:
                copy_indexed_residue_mask[res_idx] = ~is_unindexed[res_idx]
                copy_unindexed_residue_mask[res_idx] = is_unindexed[res_idx]
            else:
                copy_atomized_residue_mask[res_idx] = True

        return {
            "t": t,
            "atom_noising_mask": atom_noising_mask,
            "res_type_noising_mask": res_type_noising_mask,
            "copy_indexed_residue_mask": copy_indexed_residue_mask,
            "copy_unindexed_residue_mask": copy_unindexed_residue_mask,
            "copy_atomized_residue_mask": copy_atomized_residue_mask,
        }

    def max_added_tokens(self, _):
        return self.max_num_res


class ProteinConditionedGenerateLigand(_ChainConditioningBase):
    """Fix protein pocket, generate ligand."""
    name: str = "protein_conditioned_generate_ligand"

    def __init__(self, **kwargs):
        kwargs.setdefault("condition_mol_type", "PROTEIN")
        super().__init__(**kwargs)


class LigandConditionedGenerateProtein(_ChainConditioningBase):
    """Fix ligand, generate protein pocket."""
    name: str = "ligand_conditioned_generate_protein"

    def __init__(self, **kwargs):
        kwargs.setdefault("condition_mol_type", "NONPOLYMER")
        super().__init__(**kwargs)
