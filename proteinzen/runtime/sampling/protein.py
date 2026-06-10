import copy
import string
import functools as fn
from dataclasses import astuple
import warnings

from rdkit import Chem

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure as BPStructure

from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import tree
import numpy as np

from proteinzen.boltz.data.types import Structure, Atom, Bond, Residue, Chain, SamplingResidue
from proteinzen.boltz.data import const

from proteinzen.openfold.data import residue_constants
from proteinzen.data.constants import coarse_grain as cg
from proteinzen.openfold.data import data_transforms
from proteinzen.openfold.utils import rigid_utils as ru
from proteinzen.data.featurize.tokenize import Tokenized, convert_atom_str_to_tuple
from proteinzen.data.featurize.sampling import ResidueData, AtomData, ChainData, generate_protein_structure_template

from .biomolecule import Biomolecule

CHAIN_ALPHABET = string.ascii_uppercase + string.ascii_lowercase


def expand_contigs(contigs, residues_per_chain):
    contigs = [c.strip() for c in contigs.split(",")]
    residues = []

    frag_index = 0
    for contig in contigs:
        chain = contig[0]
        assert chain in CHAIN_ALPHABET
        if len(contig) == 1:
            # select the entire chain
            for resid in residues_per_chain[chain]:
                residues.append((chain, resid))
        else:
            # select a residue span
            if "-" in contig:
                start, end = [int(i) for i in contig[1:].split("-")]
                for resid in range(start, end+1):
                    residues.append((chain, resid))
            # select a single
            else:
                resid = int(contig[1:])
                residues.append((chain, resid))
    return residues


def infer_frag_index(res_keys):
    frag_index = 0
    last_chain, last_res_idx = res_keys[0]
    entity_ids = []
    for chain, res_idx in res_keys:
        if chain == last_chain and abs(res_idx - last_res_idx) < 2:
            entity_ids.append(frag_index)
        else:
            frag_index += 1
            entity_ids.append(frag_index)

        last_chain, last_res_idx = chain, res_idx
    return entity_ids


def biopython_to_boltz(
    residue,
    res_idx,
    atom_idx,
    noise_bb=True,
    noise_tip=True,
    noise_sidechain=True,
    is_copy=True
):
    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    res_name = residue.get_resname()
    res_ref_atoms = const.ref_atoms[res_name]
    ordered_atom_list = []

    atom_noising_mask = []

    for atom_name in res_ref_atoms:
        if atom_name in residue:
            atom = residue[atom_name]
            element_idx = periodic_table.GetAtomicNumber(atom.element)

            if atom.is_disordered():
                atom.disordered_select("A")

            atom_data = AtomData(
                name=np.array(convert_atom_str_to_tuple(atom_name)),
                element=element_idx,
                charge=0,  # TODO: probs should get this from a reference
                coords=np.array(atom.coord),
                conformer=np.array((0.0, 0.0, 0.0)),  # not used by proteinzen
                is_present=atom.is_disordered() == 0,
                chirality=0  # TODO: probs should get this from a reference
            )
            ordered_atom_list.append(
                astuple(atom_data)
            )
            if atom.is_disordered():
                atom_noising_mask.append(True)
            else:
                if atom_name in ['N', 'CA', 'C', 'O', 'CB']:
                    atom_noising_mask.append(noise_bb)
                else:
                    if atom_name in cg.coarse_grain_sidechain_groups[res_name][2]:
                        atom_noising_mask.append(noise_sidechain and noise_tip)
                    else:
                        atom_noising_mask.append(noise_sidechain)

        else:
            element_idx = periodic_table.GetAtomicNumber(atom_name[0])
            atom_data = AtomData(
                name=np.array(convert_atom_str_to_tuple(atom_name)),
                element=element_idx,
                charge=0, # TODO: probs should get this from a reference
                coords=np.array((0.0, 0.0, 0.0)),
                conformer=np.array((0.0, 0.0, 0.0)),  # not used by proteinzen
                is_present=False,
                chirality=0  # TODO: probs should get this from a reference
            )
            ordered_atom_list.append(
                astuple(atom_data)
            )
            atom_noising_mask.append(True)

    res_data = ResidueData(
        name=res_name,
        res_type=const.token_ids[res_name],
        res_idx=res_idx,
        atom_idx=atom_idx,
        atom_num=len(res_ref_atoms),
        atom_center=1,
        atom_disto=1,
        is_standard=True,
        is_present=True,
        is_copy=is_copy
    )
    new_atom_idx = atom_idx + len(res_ref_atoms)
    return astuple(res_data), ordered_atom_list, atom_noising_mask, new_atom_idx


def contig_str_to_struct(
    bp_structure,
    contig_residues,
    residue_metadata,
    output_chain_name,
):

    # this is a little confusing
    # so contig_internal_idx is "where we are in the contig currently"
    # and unindexed_res_idx is "what res_idx we should assign a new unindexed residue"
    # we then create "output_res_idx" which is the actual res_idx assigned to the residue
    # based on whether or not the residue is indexed

    # frag_idx assumes that residues adjacent to each other in index are part of the same fragment
    # TODO: i believe this assumption should generally hold but there may be some edge cases

    contig_internal_idx = 0
    unindexed_res_idx = 0
    curr_atom_idx = 0
    atoms = []
    residues = []
    atom_noising_mask = []
    res_type_noising_mask = []
    res_is_unindexed_mask = []
    res_hotspot_type = []
    token_entity_ids = []
    chain_data = {}

    for chain_id, resid in contig_residues:
        metadata = residue_metadata[(chain_id, resid)]
        chain = bp_structure[0][chain_id]
        if resid in chain:
            residue = chain[resid]

            redesign_seq = metadata['noise_seq']
            redesign_bb = metadata['noise_bb']
            hotspot_type = metadata['hotspot_type']

            is_motif_residue = metadata['is_motif']
            is_unindexed_residue = (metadata['res_index'] == -1)
            token_entity_id = metadata['entity_id']

            if is_motif_residue:
                if is_unindexed_residue:
                    output_res_idx = unindexed_res_idx
                    unindexed_res_idx += 1
                else:
                    output_res_idx = metadata['res_index']
            else:
                output_res_idx = contig_internal_idx

            repack_residue = metadata['repack']

            if repack_residue:
                res_data, atom_data, _atom_noising_mask, new_atom_idx = biopython_to_boltz(
                    residue, output_res_idx, curr_atom_idx,
                    noise_bb=False,
                    noise_tip=True,
                    noise_sidechain=True,
                    is_copy=is_motif_residue
                )
                res_type_noising_mask.append(False)
            else:
                res_data, atom_data, _atom_noising_mask, new_atom_idx = biopython_to_boltz(
                    residue, output_res_idx, curr_atom_idx,
                    noise_bb=redesign_bb,
                    noise_tip=redesign_seq,
                    noise_sidechain=redesign_seq,
                    is_copy=is_motif_residue
                )
                res_type_noising_mask.append(redesign_seq)

            residues.append(res_data)
            atoms.extend(atom_data)
            atom_noising_mask.extend(_atom_noising_mask)
            res_is_unindexed_mask.append(is_unindexed_residue)
            res_hotspot_type.append(hotspot_type)
            token_entity_ids.append(token_entity_id)

            contig_internal_idx += 1
            curr_atom_idx = new_atom_idx
        else:
            warnings.warn(f"We did not find residue {resid} in chain {chain_id} (it may be unresolved), skipping")

    atoms = np.array(atoms, dtype=Atom)
    residues = np.array(residues, dtype=SamplingResidue)

    # # reset the chain res_idx such that the first chain starts at res_idx=0
    # min_chain_res_idx = min([c['res_idx'] for c in chain_data.values()])
    # for c in chain_data.values():

    chain_data[output_chain_name] = {
        "name": output_chain_name,
        "mol_type": const.chain_type_ids["PROTEIN"],
        "entity_id": 0,
        "sym_id": 0,
        "asym_id": 0,
        "cyclic_period": 0,
        "atom_idx": curr_atom_idx,
        "atom_num": len(atoms),
        "res_idx": 0,
        "res_num": len(residues)
    }
    chains = np.array([astuple(ChainData(**c)) for c in chain_data.values()], dtype=Chain)

    struct = Structure(
        atoms=atoms,
        bonds=np.array([], dtype=Bond),
        residues=residues,
        chains=chains,
        connections=np.array([]),
        interfaces=np.array([]),
        mask=np.array([True], dtype=bool)
    )
    res_type_noising_mask = np.array(res_type_noising_mask)
    atom_noising_mask = np.array(atom_noising_mask)
    res_is_unindexed_mask = np.array(res_is_unindexed_mask)
    res_hotspot_type = np.array(res_hotspot_type)
    token_entity_ids = np.array(token_entity_ids)

    return struct, atom_noising_mask, res_type_noising_mask, res_is_unindexed_mask, res_hotspot_type, token_entity_ids


class ReferenceChain(Biomolecule):
    def __init__(
        self,
        pdb_path,
        pdb_contigs,
        chain_name,
        is_motif,
        redesign_seq_contigs=None,
        unindexed_contigs=None,
        hotspot_contigs=None,
        antihotspot_contigs=None,
        repack=False
    ):
        self.chain_name = chain_name
        parser = PDBParser()

        # a little roundabout but this makes mypy happy
        structure = parser.get_structure("", pdb_path)
        assert isinstance(structure, BPStructure)
        self.structure: BPStructure = structure

        residues_per_chain = {
            chain.id: []
            for chain in self.structure[0].get_chains()
        }
        for chain in self.structure[0].get_chains():
            for residue in chain.get_residues():
                residues_per_chain[chain.id].append(
                    residue.id[1]
                )

        self.residue_metadata = {}
        if redesign_seq_contigs is not None:
            redesign_seq_residues = expand_contigs(redesign_seq_contigs, residues_per_chain)
        else:
            redesign_seq_residues = []

        if unindexed_contigs == "all":
            unindexed_residues = expand_contigs(pdb_contigs, residues_per_chain)
        elif unindexed_contigs is not None:
            unindexed_residues = expand_contigs(unindexed_contigs, residues_per_chain)
        else:
            unindexed_residues = []

        if hotspot_contigs is not None:
            hotspot_residues = expand_contigs(hotspot_contigs, residues_per_chain)
        else:
            hotspot_residues = []

        if antihotspot_contigs is not None:
            antihotspot_residues = expand_contigs(antihotspot_contigs, residues_per_chain)
        else:
            antihotspot_residues = []

        overspecified_hotspot_type = set(hotspot_residues).intersection(antihotspot_residues)
        assert len(overspecified_hotspot_type) == 0, f"you've designated {overspecified_hotspot_type} as both hotspots and antihotspots!"

        pdb_residues = expand_contigs(pdb_contigs, residues_per_chain)
        inferred_frag_idx = infer_frag_index(pdb_residues)

        for i, res_key in enumerate(pdb_residues):
            if is_motif:
                res_index = -1 if res_key in unindexed_residues else i
            else:
                res_index = res_key[1]

            # we only derive inferred entity ids if its a unindexed motif residue
            entity_id = inferred_frag_idx[i] if (res_index == -1) else 0

            if res_key in hotspot_residues:
                hotspot_type = 1
            elif res_key in antihotspot_residues:
                hotspot_type = 2
            else:
                hotspot_type = 0

            self.residue_metadata[res_key] = {
                'is_motif': is_motif,
                'res_index': res_index,
                'noise_seq': res_key in redesign_seq_residues,
                'noise_bb': False,
                'hotspot_type': hotspot_type,
                'repack': repack if not is_motif else False,
                'entity_id': entity_id
            }

        (
            self.boltz_struct,
            self.atom_noising_mask,
            self.res_type_noising_mask,
            self.res_is_unindexed_mask,
            self.res_hotspot_type,
            self.residue_entity_ids
        ) = contig_str_to_struct(
            self.structure,
            pdb_residues,
            self.residue_metadata,
            chain_name
        )

    def sample(self):
        return {
            "struct": copy.deepcopy(self.boltz_struct),
            "atom_noising_mask": self.atom_noising_mask.copy(),
            "res_type_noising_mask": self.res_type_noising_mask.copy(),
            "res_is_unindexed_mask": self.res_is_unindexed_mask.copy(),
            "res_hotspot_type": self.res_hotspot_type.copy(),
            "residue_entity_ids": self.residue_entity_ids.copy(),
        }


class DesignChain(Biomolecule):
    def __init__(
        self,
        chain_length,
        chain_name,
    ):
        if isinstance(chain_length, str):
            if "-" in chain_length:
                min_len, max_len = [int(i) for i in chain_length.strip().split("-")]
                self.chain_length = list(range(min_len, max_len+1))
            else:
                self.chain_length = [int(chain_length.strip())]
        else:
            assert isinstance(chain_length, int), "chain_length must either be a single int or a string specifying a single or range of lengths"
            self.chain_length = [chain_length]

        self.chain_name = chain_name

    def sample(self):
        chain_length = np.random.choice(self.chain_length)
        struct_template = generate_protein_structure_template(
            {self.chain_name: chain_length}
        )
        return {
            "struct": struct_template,
            "atom_noising_mask": np.zeros(0, dtype=bool),
            "res_type_noising_mask": np.ones(chain_length, dtype=bool),
            "res_is_unindexed_mask": np.zeros(chain_length, dtype=bool),
            "residue_entity_ids": np.zeros(chain_length, dtype=int),
        }


class PartiallyNoisedChain(Biomolecule):
    def __init__(
        self,
        pdb_path,
        pdb_contigs,
        chain_name,
        t_noise: float,
    ):
        self.chain_name = chain_name
        self.t_noise = t_noise
        parser = PDBParser()

        # a little roundabout but this makes mypy happy
        structure = parser.get_structure("", pdb_path)
        assert isinstance(structure, BPStructure)
        self.structure: BPStructure = structure

        residues_per_chain = {
            chain.id: []
            for chain in self.structure[0].get_chains()
        }
        for chain in self.structure[0].get_chains():
            for residue in chain.get_residues():
                residues_per_chain[chain.id].append(
                    residue.id[1]
                )

        self.residue_metadata = {}

        pdb_residues = expand_contigs(pdb_contigs, residues_per_chain)

        for i, res_key in enumerate(pdb_residues):
            res_index = res_key[1]

            self.residue_metadata[res_key] = {
                'is_motif': False,
                'res_index': res_index,
                'noise_seq': True,
                'noise_bb': True,
                'hotspot_type': 0,
                'repack': False,
                'entity_id': 0
            }

        (
            self.boltz_struct,
            self.atom_noising_mask,
            self.res_type_noising_mask,
            self.res_is_unindexed_mask,
            self.res_hotspot_type,
            self.residue_entity_ids
        ) = contig_str_to_struct(
            self.structure,
            pdb_residues,
            self.residue_metadata,
            chain_name
        )

    def sample(self):
        return {
            "struct": copy.deepcopy(self.boltz_struct),
            "atom_noising_mask": self.atom_noising_mask.copy(),
            "res_type_noising_mask": self.res_type_noising_mask.copy(),
            "res_is_unindexed_mask": self.res_is_unindexed_mask.copy(),
            "res_hotspot_type": self.res_hotspot_type.copy(),
            "residue_entity_ids": self.residue_entity_ids.copy(),
        }