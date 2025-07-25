import copy
from dataclasses import astuple, asdict, dataclass, replace
from typing import Tuple, Optional

from rdkit import Chem

import numpy as np
import torch
import networkx as nx
from scipy.spatial.transform import Rotation

from boltz.data import const
from boltz.data.types import (
    Residue, Atom, Chain, Structure
)

# from proteinzen.data.datasets.featurize.tokenize import ChainData, ResidueData
from proteinzen.data.datasets.featurize.tokenize import RES_TO_AA, Tokenized, convert_atom_name
from proteinzen.data.openfold import residue_constants as rc
from proteinzen.data.constants import coarse_grain as cg
from proteinzen.utils import coarse_grain as cg_utils
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.utils.openfold.rigid_utils import rot_to_quat

from .tokenize import Token, Rigid, TokenBond, TokenData, RigidData, Tokenized, standard_residue_to_frames, arbitrary_atom_to_frame, IDENTITY_TENSOR7

@dataclass(frozen=True)
class AtomData:
    """Tokenized datatype."""

    name: np.ndarray
    element: int
    charge: int
    coords: np.ndarray
    conformer: np.ndarray
    is_present: bool
    chirality: int


@dataclass(frozen=True)
class ResidueData:
    """Tokenized datatype."""

    name: str
    res_type: int
    res_idx: int
    atom_idx: int
    atom_num: int
    atom_center: int
    atom_disto: int
    is_standard: bool
    is_present: bool


@dataclass(frozen=True)
class ChainData:
    """Tokenized datatype."""

    name: str
    mol_type: int
    entity_id: int
    sym_id: int
    asym_id: int
    atom_idx: int
    atom_num: int
    res_idx: int
    res_num: int
    cyclic_period: int



def generate_protein_structure_template(  # noqa: C901, PLR0915
    chain_lens: dict[str, int],
    extra_mols: Optional[Structure] = None
) -> Structure:
    """Tokenize a structure.

    Parameters
    ----------
    struct : Structure
        The structure to tokenize.

    Returns
    -------
    np.ndarray
        The tokenized data.
    np.ndarray
        The tokenized bonds.

    """
    if extra_mols is not None:
        extra_mols = copy.deepcopy(extra_mols)

    chain_names = list(chain_lens.keys())
    if extra_mols is not None:
        chain_names.extend(extra_mols.chains['name'].tolist())
    chain_names = sorted(set(chain_names))

    chain_data = {}
    # if extra_mols is not None:
    #     for chain in extra_mols.chains:
    #         _chain_data = {
    #             key: chain[key]
    #             for key in chain.dtype.names
    #         }
    #         chain_data[chain['name']] = _chain_data
    residue_data = []
    res_idx = 0
    # if extra_mols is not None:
    #     res_idx = extra_mols.residues.shape[0]
    # else:
    #     res_idx = 0


    for chain, chain_len in chain_lens.items():
        chain_idx = chain_names.index(chain)
        if chain not in chain_data:
            chain_data[chain] = {
                "name": chain,
                "mol_type": const.chain_type_ids["PROTEIN"],
                "entity_id": chain_idx,
                "sym_id": chain_idx,
                "asym_id": chain_idx,
                "atom_idx": 0,
                "atom_num": 0,
                "res_idx": res_idx,
                "res_num": chain_len,
                "cyclic_period": 0,
            }
        else:
            chain_data[chain]["res_num"] = chain_data[chain]["res_num"] + chain_len

        for _ in range(chain_len):
            res = ResidueData(
                name=const.unk_token["PROTEIN"],
                res_type=const.unk_token_ids["PROTEIN"],
                res_idx=res_idx,
                atom_idx=0,  # placeholder
                atom_num=0,  # placeholder
                atom_center=1,
                atom_disto=1,
                is_standard=True,
                is_present=True
            )
            residue_data.append(astuple(res))
            res_idx += 1

    residues = np.array(residue_data, dtype=Residue)
    chain_data = [astuple(ChainData(**d)) for d in chain_data.values()]
    chains = np.array(chain_data, dtype=Chain)
    atoms = np.array([])
    bonds = np.array([])

    if extra_mols is not None:
        # # we prepend here so that we don't mess up any of the bond information
        # residues = np.concatenate([extra_mols.residues, residues], axis=0)
        # chains = np.concatenate([extra_mols.chains, chains], axis=0)
        extra_mols.residues['res_idx'] += len(residues)
        extra_mols.chains['res_idx'] += len(residues)
        residues = np.concatenate([residues, extra_mols.residues], axis=0)
        chains = np.concatenate([chains, extra_mols.chains], axis=0)
        atoms = extra_mols.atoms
        bonds = extra_mols.bonds

    struct = Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        connections=np.array([]),
        interfaces=np.array([]),
        mask=np.ones(chains.shape[0], dtype=bool),
    )
    return struct


def sample_noise_from_struct_template(  # noqa: C901, PLR0915
    struct: Structure,
    task_masks: Optional[dict[str, np.ndarray]] = None,
    trans_std=16,
) -> Tuple[np.ndarray, np.array, np.ndarray, np.ndarray]:
    """Tokenize a structure.

    Parameters
    ----------
    struct : Structure
        The structure to tokenize.

    Returns
    -------
    np.ndarray
        The tokenized data.
    np.ndarray
        The tokenized bonds.

    """
    bond_graph = nx.Graph([(bond["atom_1"], bond["atom_2"]) for bond in struct.bonds])

    # Create token data and rigid data
    token_data = []
    rigid_data = []

    # Keep track of atom_idx, rigid_idx to token_idx
    token_idx = 0
    rigid_idx = 0
    atom_to_rigid = {}
    rigid_to_token = {}

    # Filter to valid chains only
    chains = struct.chains[struct.mask]

    for chain in chains:
        # Get residue indices
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        is_protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]

        for i, res in enumerate(struct.residues[res_start:res_end]):
            # Get atom indices
            atom_start = res["atom_idx"]
            atom_end = res["atom_idx"] + res["atom_num"]

            # Standard residues are tokens
            if res["is_standard"] and (res['name'] == 'UNK') and is_protein:
                # Token is present if centers are
                is_present = res["is_present"]

                rigid_trans = np.random.randn(3, 3) * trans_std
                rigid_quat = Rotation.random(3).as_quat(canonical=True)
                rigid_tensor7 = np.concatenate([rigid_quat, rigid_trans], axis=-1)
                rigid_mask = np.ones(3, dtype=bool)

                # Create token
                token = TokenData(
                    token_idx=token_idx,
                    rigid_idx=rigid_idx,
                    rigid_num=3,
                    res_idx=res["res_idx"],
                    res_type=res["res_type"],
                    res_name=res["name"],
                    sym_id=chain["sym_id"],
                    asym_id=chain["asym_id"],
                    entity_id=chain["entity_id"],
                    mol_type=chain["mol_type"],
                    resolved_mask=is_present,
                    center_coords=rigid_tensor7[0, 4:],
                    is_copy=False,
                    is_unindexed=False,
                    is_atomized=False,
                    seq_noising_mask=True
                )
                token_data.append(astuple(token))

                # Update rigid_idx to token_idx
                for j in range(3):
                    rigid = RigidData(
                        rigid_idx=rigid_idx,
                        token_idx=token_idx,
                        sidechain_idx=j,
                        is_atomized=False,
                        element=-1,
                        charge=0,
                        tensor7=rigid_tensor7[j],
                        is_present=rigid_mask[j],
                        rigids_noising_mask=True
                    )
                    rigid_data.append(astuple(rigid))
                    rigid_to_token[rigid_idx] = token_idx

                    rigid_idx += 1

                token_idx += 1

            # Standard residues are tokens
            elif res["is_standard"] and (res['name'] != 'UNK') and is_protein:
                # If we're provided a standard token, we're currently assuming its a copy
                # since we're not doing multi-chain sampling
                # TODO: we should have a separate mask for this at some point

                # Token is present if centers are
                is_present = res["is_present"]

                # Get frame atoms
                atoms = struct.atoms[atom_start:atom_end]
                assert task_masks is not None
                atom_noising_mask = task_masks['atom_noising_mask']
                res_type_noising_mask = task_masks['res_type_noising_mask']
                residue_is_unindexed_mask = task_masks['residue_is_unindexed_mask']

                is_unindexed = residue_is_unindexed_mask[res_start+i]
                noise_seq = res_type_noising_mask[res_start + i]

                rigid_tensor7, rigid_mask, dummy_rigid_idx = standard_residue_to_frames(
                    res, atoms
                )

                # Create token
                token = TokenData(
                    token_idx=token_idx,
                    rigid_idx=rigid_idx,
                    rigid_num=3,
                    res_idx=res["res_idx"],
                    res_type=res["res_type"],
                    res_name=res["name"],
                    sym_id=chain["sym_id"],
                    asym_id=chain["asym_id"],
                    entity_id=chain["entity_id"],
                    mol_type=chain["mol_type"],
                    resolved_mask=is_present,
                    center_coords=rigid_tensor7[0, 4:],
                    is_copy=True,
                    is_unindexed=bool(is_unindexed),
                    is_atomized=False,
                    seq_noising_mask=bool(noise_seq),
                )
                token_data.append(astuple(token))

                noise_atoms = atom_noising_mask[atom_start:atom_end]

                atom_name_to_cg_idx = {}
                cg_idx_to_atom_idx = {0: [], 1: [], 2: []}
                atom_order = rc.restype_name_to_atom14_names[res["name"]]

                for atom_name in ['N', 'CA', 'C', 'O', 'CB']:
                    atom_name_to_cg_idx[atom_name] = 0
                    if atom_name == 'CB' and res["name"] == "GLY":
                        continue
                    cg_idx_to_atom_idx[0].append(atom_order.index(atom_name))

                for atom_name in cg.coarse_grain_sidechain_groups[res["name"]][2]:
                    atom_name_to_cg_idx[atom_name] = 1
                    cg_idx_to_atom_idx[1].append(atom_order.index(atom_name))
                for atom_name in cg.coarse_grain_sidechain_groups[res["name"]][3]:
                    atom_name_to_cg_idx[atom_name] = 2
                    cg_idx_to_atom_idx[2].append(atom_order.index(atom_name))

                for j, atom in enumerate(atoms):
                    atom_name = atom['name']
                    atom_name = [chr(c + 32) for c in atom_name if c != 0]
                    atom_name = "".join(atom_name)
                    atom_to_rigid[atom_start + j] = rigid_idx + atom_name_to_cg_idx[atom_name]

                _noise_rigid = []
                # Update rigid_idx to token_idx
                for j in range(3):
                    cg_atom_idxs = cg_idx_to_atom_idx[j]
                    # we figure out if we're noising the rigid
                    # by checking if any of its component atoms are being noised
                    # we also need to check if its a dummy rigid
                    # and if so, copy the noising status of its corresponding non-dummy rigid
                    if len(cg_atom_idxs) > 0:
                        noise_rigid = noise_atoms[cg_atom_idxs].any()
                    else:
                        noise_rigid = _noise_rigid[dummy_rigid_idx[j]]
                    _noise_rigid.append(noise_rigid)

                    rigid = RigidData(
                        rigid_idx=rigid_idx,
                        token_idx=token_idx,
                        sidechain_idx=j,
                        is_atomized=False,
                        element=-1,
                        charge=0,
                        tensor7=rigid_tensor7[j],
                        is_present=rigid_mask[j],
                        rigids_noising_mask=bool(noise_rigid)
                    )
                    rigid_data.append(astuple(rigid))
                    rigid_to_token[rigid_idx] = token_idx

                    rigid_idx += 1

                token_idx += 1


            # Non-standard are tokenized per atom
            else:
                # We use the unk protein token as res_type
                unk_token = const.unk_token["PROTEIN"]
                unk_id = const.token_ids[unk_token]

                # Get atom coordinates
                atom_data = struct.atoms[atom_start:atom_end]
                atom_coords = atom_data["coords"]

                valid_neighbors = [i for i in range(res["atom_num"]) if atom_data[i]["is_present"]]
                valid_neighbor_coords = atom_coords[np.array(valid_neighbors, dtype=int)]

                # Tokenize each atom
                for j, atom in enumerate(atom_data):
                    # Token is present if atom is
                    is_present = res["is_present"] & atom["is_present"]
                    atom_idx = atom_start + j

                    atom_trans = np.random.randn(3) * trans_std
                    atom_quat = Rotation.random().as_quat(canonical=True)
                    atom_tensor7 = np.concatenate([atom_quat, atom_trans], axis=-1)

                    # Create token
                    token = TokenData(
                        token_idx=token_idx,
                        rigid_idx=rigid_idx,
                        rigid_num=1,
                        res_idx=res["res_idx"],
                        res_type=unk_id,
                        res_name=res["name"],
                        sym_id=chain["sym_id"],
                        asym_id=chain["asym_id"],
                        entity_id=chain["entity_id"],
                        mol_type=chain["mol_type"],
                        resolved_mask=is_present,
                        center_coords=atom_tensor7[4:],
                        is_copy=False,
                        is_unindexed=False,
                        is_atomized=True,
                        seq_noising_mask=False
                    )
                    token_data.append(astuple(token))

                    # Update atom_idx to token_idx
                    atom_to_rigid[atom_start + j] = rigid_idx
                    rigid = RigidData(
                        rigid_idx=rigid_idx,
                        token_idx=token_idx,
                        sidechain_idx=0,
                        is_atomized=True,
                        element=atom["element"],
                        charge=atom["charge"],
                        tensor7=atom_tensor7,
                        is_present=atom["is_present"],
                        rigids_noising_mask=True
                    )
                    rigid_data.append(astuple(rigid))
                    rigid_to_token[rigid_idx] = token_idx

                    rigid_idx += 1
                    token_idx += 1

    # Create token bonds
    token_bonds = []

    # Add atom-atom bonds from ligands
    for bond in struct.bonds:
        atom1 = bond["atom_1"]
        atom2 = bond["atom_2"]
        if atom1 not in atom_to_rigid or atom2 not in atom_to_rigid:
            continue
        rigid1 = atom_to_rigid[atom1]
        rigid2 = atom_to_rigid[atom2]
        if rigid1 not in rigid_to_token or rigid2 not in rigid_to_token:
            continue
        token_bond = (
            rigid_to_token[rigid1],
            rigid_to_token[rigid2],
            bond["type"] + 1,
        )
        token_bonds.append(token_bond)

    token_data = np.array(token_data, dtype=Token)
    token_bonds = np.array(token_bonds, dtype=TokenBond)
    rigid_data = np.array(rigid_data, dtype=Rigid)

    # center the motifs on the COM of the fixed rigids
    fixed_rigids_mask = ~rigid_data['rigids_noising_mask']
    if fixed_rigids_mask.any():
        fixed_rigids = rigid_data[fixed_rigids_mask]
        fixed_rigids_com = fixed_rigids['tensor7'][:, 4:].mean(axis=0)
        tensor7_edit = np.zeros((rigid_data.shape[0], 7))
        tensor7_edit[fixed_rigids_mask, 4:] = fixed_rigids_com[None]
        rigid_data['tensor7'] -= tensor7_edit
    else:
        fixed_rigids_com = np.zeros(3)
    # fixed_rigids_com = np.zeros(3)

    return token_data, rigid_data, token_bonds, fixed_rigids_com


def construct_atoms(
    data: Tokenized,
    struct: Structure,
):
    token_data = data.tokens
    rigids_data = data.rigids

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    atom_data = []
    tokens_to_residues = {}
    new_residues = []

    # res_idx = 1
    atom_idx = 0

    for token in token_data:
        # # we discard any tokens which are copies
        # if token['is_copy']:
        #     continue

        res_idx = token['res_idx']
        if res_idx not in tokens_to_residues:
            tokens_to_residues[res_idx] = [token]
        else:
            tokens_to_residues[res_idx].append(token)

    # print(tokens_to_residues)
    # print(struct.residues)

    for residue in struct.residues:
        res_idx = residue['res_idx']
        token_set = tokens_to_residues[res_idx]
        if len(token_set) == 1:
            token = token_set[0]
            token_is_standard_protein = (
                token['mol_type'] == const.chain_type_ids["PROTEIN"]
                and
                token['rigid_num'] == 3  ## TODO: we should probably use an is_atomized field instead
            )
            assert token_is_standard_protein

            if token['seq_noising_mask']:
                res_type = token['res_type']
                res_name = const.tokens[res_type]
            else:
                res_type = residue['res_type']
                res_name = residue['name']
            atom_num = len(rc.residue_atoms[res_name])


            res = ResidueData(
                name=res_name,
                res_type=res_type,
                res_idx=res_idx,
                atom_idx=atom_idx,
                atom_num=atom_num,
                atom_center=1,
                atom_disto=1,
                is_standard=True,
                is_present=True
            )
            new_residues.append(astuple(res))

            # grab proper rigids
            rigid_start = token['rigid_idx']
            rigid_end = rigid_start + token['rigid_num']
            tensor7 = rigids_data['tensor7'][rigid_start:rigid_end].copy()
            tensor7 = torch.as_tensor(tensor7)
            rigids_for_atom14 = ru.Rigid.from_tensor_7(tensor7)
            # compute atom14
            tensor7_seq = torch.as_tensor([RES_TO_AA[token['res_type']]])
            dummy_mask = torch.ones_like(tensor7_seq, dtype=torch.bool)
            atom14, atom14_mask = cg_utils.compute_atom14_from_cg_frames(rigids_for_atom14, dummy_mask, tensor7_seq, return_atom_mask=True)

            # keep the atoms which are not null in the residue
            num_atoms = len(rc.residue_atoms[res_name])
            atom14 = atom14.squeeze()
            atom14_mask = atom14_mask.squeeze().bool()
            res_atom14, res_atom14_mask = atom14, atom14_mask
            # atom_coords = res_atom14[res_atom14_mask].numpy(force=True)
            # atom_is_present = res_atom14_mask[res_atom14_mask].numpy(force=True)
            atom_coords = res_atom14[:num_atoms].numpy(force=True)
            atom_is_present = res_atom14_mask[:num_atoms].numpy(force=True)


            # construct the rest of the atom data
            res_atom_names = [i for i in rc.restype_name_to_atom14_names[res_name] if len(i) > 0]
            for i, res_atom_name in enumerate(res_atom_names):
                atom_name = np.array(convert_atom_name(res_atom_name))
                atom_element = res_atom_name[0]  # TODO: im pretty sure this works but might be good to have smth less error prone
                atom_element = periodic_table.GetAtomicNumber(atom_element)
                atom_charge = 0
                atom_conformer = np.zeros_like(atom_coords)
                atom_chirality = 0
                coords = atom_coords[i]
                is_present = atom_is_present[i]
                atom_conformer = np.zeros_like(coords)

                atom = AtomData(
                    name=atom_name,
                    element=atom_element,
                    charge=atom_charge,
                    coords=coords,
                    conformer=atom_conformer,
                    is_present=is_present,
                    chirality=atom_chirality
                )
                atom_data.append(astuple(atom))
                atom_idx += 1

        else:
            raise NotImplementedError()
            res_name = 'LIG'
            atom_num = len(token_set)

            res = ResidueData(
                name=res_name,
                res_type=token_set[0]['res_type'],
                res_idx=res_idx,
                atom_idx=atom_idx,
                atom_num=atom_num,
                atom_center=1,
                atom_disto=1,
                is_standard=True,
                is_present=True
            )
            # construct the rest of the atom data
            for i, token in enumerate(token_set):
                rigid = rigids_data[token['rigid_idx']]
                atom_name = np.array(convert_atom_name("X"))
                atom_element = res_atom_name[0]  # TODO: im pretty sure this works but might be good to have smth less error prone
                atom_element = periodic_table.GetAtomicNumber(atom_element)
                atom_charge = 0
                atom_conformer = np.zeros_like(atom_coords)
                atom_chirality = 0
                coords = atom_coords[i]
                is_present = atom_is_present[i]
                atom_conformer = np.zeros_like(coords)

                atom = AtomData(
                    name=atom_name,
                    element=atom_element,
                    charge=atom_charge,
                    coords=coords,
                    conformer=atom_conformer,
                    is_present=is_present,
                    chirality=atom_chirality
                )
                atom_data.append(astuple(atom))
                atom_idx += 1

    struct = Structure(
        atoms=np.array(atom_data, dtype=Atom),
        bonds=np.array([]),
        residues=np.array(new_residues, dtype=Residue),
        chains=struct.chains,
        connections=np.array([]),
        interfaces=np.array([]),
        mask=struct.mask,
    )
    # print(struct.residues)
    return struct


def infer_structure(
    data: Tokenized,
    ref_tokens=None
):
    token_data = data.tokens
    rigids_data = data.rigids

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    atom_data = []
    tokens_to_residues = {}
    residues = []
    ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    chain_data = {}

    # res_idx = 1
    atom_idx = 0

    for token in token_data:
        res_idx = token['res_idx']
        if res_idx not in tokens_to_residues:
            tokens_to_residues[res_idx] = [token]
        else:
            tokens_to_residues[res_idx].append(token)
        chain_id = token['asym_id']
        if ALPHABET[chain_id] not in chain_data:
            chain_data[ALPHABET[chain_id]] = {
                "name": ALPHABET[chain_id],
                "mol_type": token['mol_type'],
                "entity_id": token['entity_id'],
                "sym_id": token['sym_id'],
                "asym_id": token['asym_id'],
                "cyclic_period": 0
            }

    for res_idx, token_set in tokens_to_residues.items():
        if len(token_set) == 1:
            token = token_set[0]
            token_is_standard_protein = (
                token['mol_type'] == const.chain_type_ids["PROTEIN"]
                and
                token['rigid_num'] == 3  ## TODO: we should probably use an is_atomized field instead
            )
            assert token_is_standard_protein

            res_name = const.tokens[token['res_type']]
            atom_num = len(rc.residue_atoms[res_name])

            res = ResidueData(
                name=res_name,
                res_type=token['res_type'],
                res_idx=res_idx,
                atom_idx=atom_idx,
                atom_num=atom_num,
                atom_center=1,
                atom_disto=1,
                is_standard=True,
                is_present=True
            )
            residues.append(astuple(res))

            # grab proper rigids
            rigid_start = token['rigid_idx']
            rigid_end = rigid_start + token['rigid_num']
            tensor7 = rigids_data['tensor7'][rigid_start:rigid_end].copy()
            tensor7 = torch.as_tensor(tensor7)
            rigids_for_atom14 = ru.Rigid.from_tensor_7(tensor7)
            # compute atom14
            tensor7_seq = torch.as_tensor([RES_TO_AA[token['res_type']]])
            dummy_mask = torch.ones_like(tensor7_seq, dtype=torch.bool)
            atom14, atom14_mask = cg_utils.compute_atom14_from_cg_frames(rigids_for_atom14, dummy_mask, tensor7_seq, return_atom_mask=True)

            # keep the atoms which are not null in the residue
            num_atoms = len(rc.residue_atoms[res_name])
            atom14 = atom14.squeeze()
            atom14_mask = atom14_mask.squeeze().bool()
            res_atom14, res_atom14_mask = atom14, atom14_mask
            # atom_coords = res_atom14[res_atom14_mask].numpy(force=True)
            # atom_is_present = res_atom14_mask[res_atom14_mask].numpy(force=True)
            atom_coords = res_atom14[:num_atoms].numpy(force=True)
            atom_is_present = res_atom14_mask[:num_atoms].numpy(force=True)


            # construct the rest of the atom data
            res_atom_names = [i for i in rc.restype_name_to_atom14_names[res_name] if len(i) > 0]
            for i, res_atom_name in enumerate(res_atom_names):
                atom_name = np.array(convert_atom_name(res_atom_name))
                atom_element = res_atom_name[0]  # TODO: im pretty sure this works but might be good to have smth less error prone
                atom_element = periodic_table.GetAtomicNumber(atom_element)
                atom_charge = 0
                atom_conformer = np.zeros_like(atom_coords)
                atom_chirality = 0
                coords = atom_coords[i]
                is_present = atom_is_present[i]
                atom_conformer = np.zeros_like(coords)

                atom = AtomData(
                    name=atom_name,
                    element=atom_element,
                    charge=atom_charge,
                    coords=coords,
                    conformer=atom_conformer,
                    is_present=is_present,
                    chirality=atom_chirality
                )
                atom_data.append(astuple(atom))
                atom_idx += 1

            chain_idx = token['asym_id']
            chain = ALPHABET[chain_idx]
            if "res_idx" not in chain_data[chain]:
                chain_data[chain]["res_idx"] = res_idx
                chain_data[chain]["res_num"] = 0
            else:
                chain_data[chain]["res_num"] += 1

            if "atom_idx" not in chain_data[chain]:
                chain_data[chain]["atom_idx"] = atom_idx
                chain_data[chain]["atom_num"] = num_atoms
            else:
                chain_data[chain]["atom_num"] += num_atoms

        else:
            raise NotImplementedError("ill do this later")
            res_name = 'LIG'
            atom_num = len(token_set)

            res = ResidueData(
                name=res_name,
                res_type=token_set[0]['res_type'],
                res_idx=res_idx,
                atom_idx=atom_idx,
                atom_num=atom_num,
                atom_center=1,
                atom_disto=1,
                is_standard=True,
                is_present=True
            )
            # construct the rest of the atom data
            for i, token in enumerate(token_set):
                rigid = rigids_data[token['rigid_idx']]
                atom_name = np.array(convert_atom_name("X"))
                atom_element = res_atom_name[0]  # TODO: im pretty sure this works but might be good to have smth less error prone
                atom_element = periodic_table.GetAtomicNumber(atom_element)
                atom_charge = 0
                atom_conformer = np.zeros_like(atom_coords)
                atom_chirality = 0
                coords = atom_coords[i]
                is_present = atom_is_present[i]
                atom_conformer = np.zeros_like(coords)

                atom = AtomData(
                    name=atom_name,
                    element=atom_element,
                    charge=atom_charge,
                    coords=coords,
                    conformer=atom_conformer,
                    is_present=is_present,
                    chirality=atom_chirality
                )
                atom_data.append(astuple(atom))
                atom_idx += 1

    chain_data = [astuple(ChainData(**d)) for d in chain_data.values()]

    struct = Structure(
        atoms=np.array(atom_data, dtype=Atom),
        bonds=np.array([]),
        residues=np.array(residues, dtype=Residue),
        chains=np.array(chain_data, dtype=Chain),
        connections=np.array([]),
        interfaces=np.array([]),
        mask=np.ones(len(chain_data), dtype=bool),
    )
    return struct


