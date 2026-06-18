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
        sig_perturb=None,
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
        if sig_perturb is not None:
            self.sig_perturb = sig_perturb

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
    """Fix entire protein crop, generate ligand."""
    name: str = "protein_conditioned_generate_ligand"

    def __init__(self, **kwargs):
        kwargs.setdefault("condition_mol_type", "PROTEIN")
        super().__init__(**kwargs)

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

        protein_mol_type = const.chain_type_ids["PROTEIN"]
        for chain in chains:
            if int(chain["mol_type"]) != protein_mol_type:
                continue
            res_start = int(chain["res_idx"])
            res_end = res_start + int(chain["res_num"])
            for res_idx in range(res_start, res_end):
                res = residues[res_idx]
                atom_idx = int(res["atom_idx"])
                atom_num = int(res["atom_num"])
                atom_noising_mask[atom_idx:atom_idx + atom_num] = False
                res_type_noising_mask[res_idx] = False

        return {
            "t": t,
            "atom_noising_mask": atom_noising_mask,
            "res_type_noising_mask": res_type_noising_mask,
            "copy_indexed_residue_mask": np.zeros(n_residues, dtype=bool),
            "copy_unindexed_residue_mask": np.zeros(n_residues, dtype=bool),
            "copy_atomized_residue_mask": np.zeros(n_residues, dtype=bool),
        }

    def max_added_tokens(self, _):
        return 0


class LigandConditionedGenerateProtein(_ChainConditioningBase):
    """Fix ligand, generate protein pocket.

    When ``interface_condition=True`` (default), the entire ligand is fixed and
    converted to atomized copy tokens, and a random subset of the protein
    residues within ``atom_interface_cutoff`` of the ligand are additionally
    provided as indexed/unindexed copy tokens.  This is the protein-ligand
    analogue of BinderDesign.

    When ``interface_condition=False``, falls back to the base-class behaviour
    (randomly sample up to ``max_num_res`` ligand atoms as copy tokens).
    """
    name: str = "ligand_conditioned_generate_protein"

    def __init__(
        self,
        interface_condition: bool = True,
        max_num_interface_protein_res: int = 15,
        motif_is_unindexed: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("condition_mol_type", "NONPOLYMER")
        super().__init__(**kwargs)
        self.interface_condition = interface_condition
        self.max_num_interface_protein_res = max_num_interface_protein_res
        self.motif_is_unindexed = motif_is_unindexed

    def sample_t_and_mask(self, data):
        if not self.interface_condition:
            return super().sample_t_and_mask(data)
        return self._sample_interface_conditioned(data)

    def _sample_interface_conditioned(self, data):
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

        protein_mol_type = const.chain_type_ids["PROTEIN"]
        nonpolymer_mol_type = const.chain_type_ids["NONPOLYMER"]

        # ── 1. Fix entire ligand; collect its atom coordinates ──
        ligand_atom_coords = []
        for chain in chains:
            if int(chain["mol_type"]) != nonpolymer_mol_type:
                continue
            res_start = int(chain["res_idx"])
            res_end = res_start + int(chain["res_num"])
            for res_idx in range(res_start, res_end):
                res = residues[res_idx]
                a_start = int(res["atom_idx"])
                a_end = a_start + int(res["atom_num"])
                atom_noising_mask[a_start:a_end] = False
                res_type_noising_mask[res_idx] = False
                copy_atomized_residue_mask[res_idx] = True
                present = atoms[a_start:a_end]["is_present"].astype(bool)
                if present.any():
                    ligand_atom_coords.append(atoms[a_start:a_end]["coords"][present])

        # ── 2. Find protein interface residues ──
        seed_interface = None
        if len(ligand_atom_coords) > 0:
            lig_coords = np.concatenate(ligand_atom_coords, axis=0)
            cutoff = const.atom_interface_cutoff + abs(np.random.randn())

            protein_res_idx_list = []
            protein_res_min_dists = []
            for chain in chains:
                if int(chain["mol_type"]) != protein_mol_type:
                    continue
                res_start = int(chain["res_idx"])
                res_end = res_start + int(chain["res_num"])
                for res_idx in range(res_start, res_end):
                    res = residues[res_idx]
                    if not res["is_present"]:
                        continue
                    a_start = int(res["atom_idx"])
                    a_end = a_start + int(res["atom_num"])
                    res_atoms = atoms[a_start:a_end]
                    present_coords = res_atoms["coords"][res_atoms["is_present"].astype(bool)]
                    if len(present_coords) == 0:
                        continue
                    min_dist = np.linalg.norm(
                        present_coords[:, None, :] - lig_coords[None, :, :], axis=-1
                    ).min()
                    protein_res_idx_list.append(res_idx)
                    protein_res_min_dists.append(min_dist)

            if len(protein_res_idx_list) > 0:
                protein_res_idx_arr = np.array(protein_res_idx_list)
                interface_mask = np.array(protein_res_min_dists) < cutoff
                interface_res = protein_res_idx_arr[interface_mask]

                if len(interface_res) > 0:
                    interface_dists = np.array(protein_res_min_dists)[interface_mask]
                    dist_order = np.argsort(interface_dists)
                    interface_res_sorted = interface_res[dist_order]
                    n_copy = min(
                        np.random.randint(1, len(interface_res_sorted) + 1),
                        self.max_num_interface_protein_res,
                    )
                    selected = interface_res_sorted[:n_copy]
                    for res_idx in selected:
                        res = residues[res_idx]
                        a_start = int(res["atom_idx"])
                        a_end = a_start + int(res["atom_num"])
                        atom_noising_mask[a_start:a_end] = False
                        res_type_noising_mask[res_idx] = False
                        if self.motif_is_unindexed:
                            copy_unindexed_residue_mask[res_idx] = True
                        else:
                            copy_indexed_residue_mask[res_idx] = True

            # ── 3. Seed interface for cropper ──
            if len(data.interfaces) > 0:
                for iface in data.interfaces:
                    c1, c2 = int(iface["chain_1"]), int(iface["chain_2"])
                    t1 = int(chains[c1]["mol_type"]) if c1 < len(chains) else -1
                    t2 = int(chains[c2]["mol_type"]) if c2 < len(chains) else -1
                    if nonpolymer_mol_type in (t1, t2):
                        seed_interface = tuple(sorted((c1, c2)))
                        break
                if seed_interface is None:
                    c1 = int(data.interfaces[0]["chain_1"])
                    c2 = int(data.interfaces[0]["chain_2"])
                    seed_interface = tuple(sorted((c1, c2)))

        result = {
            "t": t,
            "atom_noising_mask": atom_noising_mask,
            "res_type_noising_mask": res_type_noising_mask,
            "copy_indexed_residue_mask": copy_indexed_residue_mask,
            "copy_unindexed_residue_mask": copy_unindexed_residue_mask,
            "copy_atomized_residue_mask": copy_atomized_residue_mask,
        }
        if seed_interface is not None:
            result["seed_interface"] = seed_interface
        return result

    def max_added_tokens(self, N):
        if not self.interface_condition:
            return self.max_num_res
        # ligand copy tokens (bounded by max_num_res) + protein interface copy tokens
        return self.max_num_res + self.max_num_interface_protein_res
