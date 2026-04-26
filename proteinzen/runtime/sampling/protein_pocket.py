"""Protein-pocket-conditioned ligand sampling task.

Loads a protein-ligand complex from a proteinzen npz file (e.g. from the Plinder
dataset), fixes all protein chain atoms, and generates the ligand from pure noise.
This is the inference counterpart to the ProteinConditioned training task.
"""
import numpy as np
from dataclasses import replace
from pathlib import Path

from proteinzen.boltz.data import const
from proteinzen.boltz.data.types import Structure, Connection

from proteinzen.data.featurize.assembler import featurize
from proteinzen.data.featurize.sampling import sample_noise_from_struct_template
from proteinzen.data.featurize.tokenize import Tokenized
from proteinzen.data.datasets.datamodule import strip_h_from_structure

from .task import SamplingTask


def load_structure_from_npz(npz_path: str, include_h: bool = False) -> Structure:
    """Load a Structure from a proteinzen npz file."""
    data = np.load(npz_path, allow_pickle=False)

    chains = data["chains"]
    if "cyclic_period" not in chains.dtype.names:
        new_dtype = chains.dtype.descr + [("cyclic_period", "i4")]
        new_chains = np.empty(chains.shape, dtype=new_dtype)
        for name in chains.dtype.names:
            new_chains[name] = chains[name]
        new_chains["cyclic_period"] = 0
        chains = new_chains

    atoms = data["atoms"]
    if "chirality" not in atoms.dtype.names:
        new_dtype = atoms.dtype.descr + [("chirality", "i1")]
        new_atoms = np.empty(atoms.shape, dtype=new_dtype)
        for name in atoms.dtype.names:
            new_atoms[name] = atoms[name]
        new_atoms["chirality"] = 0
        atoms = new_atoms

    struct = Structure(
        atoms=atoms,
        bonds=data["bonds"],
        residues=data["residues"],
        chains=chains,
        connections=data["connections"].astype(Connection),
        interfaces=data["interfaces"],
        mask=data["mask"],
    )

    if not include_h:
        struct = strip_h_from_structure(struct)

    return struct


class ProteinPocketConditionedSampling(SamplingTask):
    """Fix protein pocket, generate ligand from noise.

    Parameters
    ----------
    npz_path : str
        Path to a proteinzen npz structure file containing both a PROTEIN chain
        and a NONPOLYMER (ligand) chain.
    num_samples : int
        Number of ligand conformers to generate.
    trans_std : float
        Std of the isotropic Gaussian noise applied to ligand atom translations.
    include_h : bool
        Whether to keep hydrogen atoms in the structure.
    """

    task_name: str = "protein_pocket_conditioned"

    def __init__(
        self,
        npz_path: str,
        num_samples: int,
        trans_std: float = 16.0,
        include_h: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.npz_path = npz_path
        self.num_samples = num_samples
        self.trans_std = trans_std
        self.include_h = include_h

    def sample_data(self):
        struct = load_structure_from_npz(self.npz_path, include_h=self.include_h)

        protein_id = const.chain_type_ids["PROTEIN"]
        nonpolymer_id = const.chain_type_ids["NONPOLYMER"]

        active_chains = struct.chains[struct.mask]
        chain_types = {int(c["mol_type"]) for c in active_chains}
        if protein_id not in chain_types:
            raise ValueError(f"No PROTEIN chain in {self.npz_path}")
        if nonpolymer_id not in chain_types:
            raise ValueError(f"No NONPOLYMER (ligand) chain in {self.npz_path}")

        # Build masks: protein atoms fixed, everything else noised.
        # Ligand (NONPOLYMER) residues are always fully noised in
        # sample_noise_from_struct_template regardless of atom_noising_mask
        # (they fall into the atomized per-atom branch), so only the protein
        # mask entries actually matter.
        n_atoms = len(struct.atoms)
        n_residues = len(struct.residues)
        atom_noising_mask = np.ones(n_atoms, dtype=bool)
        res_type_noising_mask = np.ones(n_residues, dtype=bool)

        for chain in active_chains:
            if int(chain["mol_type"]) != protein_id:
                continue
            res_start = int(chain["res_idx"])
            res_end = res_start + int(chain["res_num"])
            for res in struct.residues[res_start:res_end]:
                atom_start = int(res["atom_idx"])
                atom_end = atom_start + int(res["atom_num"])
                atom_noising_mask[atom_start:atom_end] = False
            res_type_noising_mask[res_start:res_end] = False

        task_masks = {
            "atom_noising_mask": atom_noising_mask,
            "res_type_noising_mask": res_type_noising_mask,
            "residue_is_unindexed_mask": np.zeros(n_residues, dtype=bool),
        }

        task_name = self.kwargs.get("name", self.task_name)

        for _ in range(self.num_samples):
            token_data, rigid_data, token_bonds, fixed_com = sample_noise_from_struct_template(
                struct,
                task_masks=task_masks,
                trans_std=self.trans_std,
            )

            # Center atom coords on the protein pocket COM to match rigid centering
            atoms_centered = struct.atoms.copy()
            atoms_centered["coords"] -= fixed_com[None]
            struct_centered = replace(struct, atoms=atoms_centered)

            data = Tokenized(
                tokens=token_data,
                rigids=rigid_data,
                bonds=token_bonds,
                structure=struct_centered,
            )
            task_data = {"t": np.array([0.0], dtype=float)}

            yield featurize(data, task_data, task_name=task_name, smiles=None)
