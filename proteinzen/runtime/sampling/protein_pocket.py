"""Protein-pocket-conditioned ligand sampling task.

Loads a protein-ligand complex from a proteinzen npz file (e.g. from the Plinder
dataset), fixes all protein chain atoms, and generates the ligand from pure noise.
This is the inference counterpart to the ProteinConditioned training task.
"""
import numpy as np
from dataclasses import replace

from scipy.spatial.distance import cdist

from proteinzen.boltz.data import const
from proteinzen.boltz.data.types import Structure, Connection

from proteinzen.data.featurize.assembler import featurize
from proteinzen.data.featurize.sampling import sample_noise_from_struct_template
from proteinzen.data.featurize.tokenize import Tokenized

from .task import SamplingTask


def _strip_h(struct: Structure) -> Structure:
    """Remove hydrogen atoms (element=1), reindexing residues/chains/bonds."""
    atoms = struct.atoms
    heavy_mask = atoms["element"] != 1
    old_to_new = np.full(len(atoms), -1, dtype=np.int32)
    old_to_new[np.where(heavy_mask)[0]] = np.arange(heavy_mask.sum(), dtype=np.int32)

    new_atoms = atoms[heavy_mask]

    bonds = struct.bonds
    if len(bonds) > 0:
        valid = (old_to_new[bonds["atom_1"]] >= 0) & (old_to_new[bonds["atom_2"]] >= 0)
        new_bonds = bonds[valid].copy()
        new_bonds["atom_1"] = old_to_new[new_bonds["atom_1"]]
        new_bonds["atom_2"] = old_to_new[new_bonds["atom_2"]]
    else:
        new_bonds = bonds.copy()

    cum = np.concatenate([[0], np.cumsum(heavy_mask)])
    new_residues = struct.residues.copy()
    for i, res in enumerate(new_residues):
        s, e = int(res["atom_idx"]), int(res["atom_idx"]) + int(res["atom_num"])
        new_residues[i]["atom_idx"] = cum[s]
        new_residues[i]["atom_num"] = cum[e] - cum[s]

    new_chains = struct.chains.copy()
    for i, chain in enumerate(new_chains):
        s, e = int(chain["atom_idx"]), int(chain["atom_idx"]) + int(chain["atom_num"])
        new_chains[i]["atom_idx"] = cum[s]
        new_chains[i]["atom_num"] = cum[e] - cum[s]

    return replace(struct, atoms=new_atoms, bonds=new_bonds, residues=new_residues, chains=new_chains)


def _crop_protein_to_pocket(struct: Structure, max_protein_residues: int) -> Structure:
    """Keep all ligand (NONPOLYMER) residues + the closest protein residues to the ligand.

    Protein residues are ranked by their minimum heavy-atom distance to any
    ligand atom and kept greedily nearest-first up to max_protein_residues.
    """
    protein_id = const.chain_type_ids["PROTEIN"]
    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]
    active_chains = struct.chains[struct.mask]

    # Collect present ligand atom coords
    lig_coords = []
    for chain in active_chains:
        if int(chain["mol_type"]) != nonpolymer_id:
            continue
        a0 = int(chain["atom_idx"])
        atoms = struct.atoms[a0:a0 + int(chain["atom_num"])]
        coords = atoms["coords"][atoms["is_present"].astype(bool)]
        coords = coords[~np.any(np.isnan(coords), axis=1)]
        if len(coords):
            lig_coords.append(coords)

    if not lig_coords:
        return struct
    lig_coords = np.concatenate(lig_coords, axis=0)

    # Score each protein residue by min distance to any ligand atom
    prot_res_dists = []
    for chain in active_chains:
        if int(chain["mol_type"]) != protein_id:
            continue
        r0 = int(chain["res_idx"])
        for ri in range(r0, r0 + int(chain["res_num"])):
            res = struct.residues[ri]
            a0 = int(res["atom_idx"])
            atoms = struct.atoms[a0:a0 + int(res["atom_num"])]
            coords = atoms["coords"][atoms["is_present"].astype(bool)]
            coords = coords[~np.any(np.isnan(coords), axis=1)]
            dist = cdist(coords, lig_coords).min() if len(coords) else np.inf
            prot_res_dists.append((dist, ri))

    prot_res_dists.sort(key=lambda x: x[0])
    kept_prot_res = {ri for _, ri in prot_res_dists[:max_protein_residues]}

    # Build per-residue keep mask (all ligand + selected protein)
    n_res = len(struct.residues)
    keep_res = np.zeros(n_res, dtype=bool)
    for ci, chain in enumerate(struct.chains):
        if not struct.mask[ci]:
            continue
        r0, rn = int(chain["res_idx"]), int(chain["res_num"])
        mol = int(chain["mol_type"])
        if mol == nonpolymer_id:
            keep_res[r0:r0 + rn] = True
        elif mol == protein_id:
            for ri in range(r0, r0 + rn):
                if ri in kept_prot_res:
                    keep_res[ri] = True

    if keep_res.all():
        return struct

    # Atom keep mask
    n_atoms = len(struct.atoms)
    keep_atom = np.zeros(n_atoms, dtype=bool)
    for ri in np.where(keep_res)[0]:
        res = struct.residues[ri]
        a0 = int(res["atom_idx"])
        keep_atom[a0:a0 + int(res["atom_num"])] = True

    # Old-to-new atom index mapping
    old_to_new_atom = np.full(n_atoms, -1, dtype=np.int64)
    old_to_new_atom[keep_atom] = np.arange(keep_atom.sum())

    new_atoms = struct.atoms[keep_atom]

    bonds = struct.bonds
    if len(bonds):
        valid = (old_to_new_atom[bonds["atom_1"]] >= 0) & (old_to_new_atom[bonds["atom_2"]] >= 0)
        new_bonds = bonds[valid].copy()
        new_bonds["atom_1"] = old_to_new_atom[new_bonds["atom_1"]]
        new_bonds["atom_2"] = old_to_new_atom[new_bonds["atom_2"]]
    else:
        new_bonds = bonds.copy()

    # Rebuild residues
    kept_res_idx = np.where(keep_res)[0]
    new_residues = struct.residues[kept_res_idx].copy()
    for i, ri in enumerate(kept_res_idx):
        res = struct.residues[ri]
        new_residues[i]["atom_idx"] = old_to_new_atom[int(res["atom_idx"])]

    old_res_to_new = np.full(n_res, -1, dtype=np.int64)
    old_res_to_new[kept_res_idx] = np.arange(len(kept_res_idx))

    # Rebuild chains (only those that still have residues)
    new_chains_list = []
    new_mask_list = []
    for ci, chain in enumerate(struct.chains):
        r0, rn = int(chain["res_idx"]), int(chain["res_num"])
        kept = [ri for ri in range(r0, r0 + rn) if keep_res[ri]]
        if not kept:
            continue
        c = chain.copy()
        c["res_idx"] = old_res_to_new[kept[0]]
        c["res_num"] = len(kept)
        c["atom_idx"] = old_to_new_atom[int(struct.residues[kept[0]]["atom_idx"])]
        c["atom_num"] = int(sum(struct.residues[ri]["atom_num"] for ri in kept))
        new_chains_list.append(c)
        new_mask_list.append(bool(struct.mask[ci]))

    new_chains = np.array(new_chains_list, dtype=struct.chains.dtype)
    new_mask = np.array(new_mask_list, dtype=bool)

    return replace(struct, atoms=new_atoms, bonds=new_bonds,
                   residues=new_residues, chains=new_chains, mask=new_mask)


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
        struct = _strip_h(struct)

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
    max_protein_residues : int
        Crop the protein to this many residues closest to the ligand. Set to
        None to use the full protein.
    """

    task_name: str = "protein_pocket_conditioned"

    def __init__(
        self,
        npz_path: str,
        num_samples: int,
        trans_std: float = 16.0,
        include_h: bool = False,
        max_protein_residues: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.npz_path = npz_path
        self.num_samples = num_samples
        self.trans_std = trans_std
        self.include_h = include_h
        self.max_protein_residues = max_protein_residues

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

        if self.max_protein_residues is not None:
            struct = _crop_protein_to_pocket(struct, self.max_protein_residues)
            active_chains = struct.chains[struct.mask]

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
