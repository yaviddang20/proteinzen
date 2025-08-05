import copy
from dataclasses import astuple, asdict, dataclass, replace
from typing import Tuple, Optional


import numpy as np
import torch
import networkx as nx
from scipy.spatial.transform import Rotation

from proteinzen.boltz.data import const
from proteinzen.boltz.data.types import (
    Structure
)

from proteinzen.openfold.data import residue_constants as rc
from proteinzen.data.constants import coarse_grain as cg
from proteinzen.data.constants.atomize import get_standard_protein_residue_bonds
from proteinzen.utils import coarse_grain as cg_utils
from proteinzen.openfold.utils import rigid_utils as ru
from proteinzen.openfold.utils.rigid_utils import rot_to_quat

Token = [
    ("token_idx", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_type", np.dtype("i4")),
    ("res_name", np.dtype("<U8")),
    ("rigid_idx", np.dtype("i4")),
    ("rigid_num", np.dtype("i4")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("entity_id", np.dtype("i4")),
    ("mol_type", np.dtype("i4")),  # the total bytes need to be divisible by 4
    # ("rep_rigid_idx", np.dtype("i4")),
    ("resolved_mask", np.dtype("?")),
    ("center_coords", np.dtype("3f4")),
    ("is_copy", np.dtype("?")),
    ("is_unindexed", np.dtype("?")),
    ("is_atomized", np.dtype("?")),
    ("seq_noising_mask", np.dtype("?")),
]

Rigid = [
    ("rigid_idx", np.dtype("i4")),
    ("token_idx", np.dtype("i4")),
    ("sidechain_idx", np.dtype("i1")),
    ("is_atomized", np.dtype("?")),
    ("element", np.dtype("i1")),
    ("charge", np.dtype("i1")),
    ("tensor7", np.dtype("7f4")),
    ("is_present", np.dtype("?")),
    ("rigids_noising_mask", np.dtype("?")),
]


TokenBond = [
    ("token_1", np.dtype("i4")),
    ("token_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]

IDENTITY_TENSOR7 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

RES_TO_AA = {}
for i, aa in enumerate(rc.resnames):
    RES_TO_AA[const.token_ids[aa]] = i
AA_TO_RES = {j: i for i, j in RES_TO_AA.items()}

@dataclass(frozen=True)
class Tokenized:
    """Tokenized datatype."""

    tokens: np.ndarray
    rigids: np.ndarray
    bonds: np.ndarray
    structure: Structure

# TODO: probably move mmcif.py from data processing into proteinzen so i can just import this
def convert_atom_name(name: str) -> tuple[int, int, int, int]:
    """Convert an atom name to a standard format.

    Parameters
    ----------
    name : str
        The atom name.

    Returns
    -------
    tuple[int, int, int, int]
        The converted atom name.

    """
    name = name.strip()
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (4 - len(name))
    return tuple(name)


@dataclass
class TokenData:
    """TokenData datatype."""

    token_idx: int
    res_idx: int
    res_type: int
    res_name: str
    rigid_idx: int
    rigid_num: int
    sym_id: int
    asym_id: int
    entity_id: int
    mol_type: int
    resolved_mask: bool
    center_coords: np.array
    is_copy: bool
    is_unindexed: bool
    is_atomized: bool
    seq_noising_mask: bool


@dataclass
class RigidData:
    """RigidData datatype."""

    rigid_idx: int
    token_idx: int
    sidechain_idx: int
    is_atomized: bool
    element: int
    charge: int
    tensor7: np.ndarray
    is_present: bool
    rigids_noising_mask: bool


def compute_frame(
    n: np.ndarray,
    ca: np.ndarray,
    c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the frame for a residue.

    Parameters
    ----------
    n : np.ndarray
        The N atom.
    ca : np.ndarray
        The C atom.
    c : np.ndarray
        The CA atom.

    Returns
    -------
    np.ndarray
        The frame.

    """
    v1 = c - ca
    v2 = n - ca
    e1 = v1 / (np.linalg.norm(v1) + 1e-10)
    u2 = v2 - e1 * np.dot(e1.T, v2)
    e2 = u2 / (np.linalg.norm(u2) + 1e-10)
    e3 = np.cross(e1, e2)
    rot = np.column_stack([e1, e2, e3])
    t = ca
    return rot, t


def get_unk_token(chain: np.ndarray) -> int:
    """Get the unk token for a residue.

    Parameters
    ----------
    chain : np.ndarray
        The chain.

    Returns
    -------
    int
        The unk token.

    """
    if chain["mol_type"] == const.chain_type_ids["DNA"]:
        unk_token = const.unk_token["DNA"]
    elif chain["mol_type"] == const.chain_type_ids["RNA"]:
        unk_token = const.unk_token["RNA"]
    else:
        unk_token = const.unk_token["PROTEIN"]

    res_id = const.token_ids[unk_token]
    return res_id


def standard_residue_to_frames(residue, atoms):
    res_name = residue['name']
    atoms = atoms[atoms['is_present']]  # only select present atoms
    dummy_rigid_idx = [0]

    bb_frame = ['N', 'CA', 'C']
    # bb_frame = ['C', 'CA', 'N']
    frame2 = cg.coarse_grain_sidechain_axes[res_name][2]
    frame3 = cg.coarse_grain_sidechain_axes[res_name][3]
    # construct dummy frames as necessary
    # use bb frame if frame2 doesn't exist
    if len(frame2) == 0:
        frame2 = bb_frame
        dummy_rigid_idx.append(dummy_rigid_idx[-1])
    else:
        dummy_rigid_idx.append(dummy_rigid_idx[-1]+1)
    # use frame2 frame if frame3 doesn't exist
    if len(frame3) == 0:
        frame3 = frame2
        dummy_rigid_idx.append(dummy_rigid_idx[-1])
    else:
        dummy_rigid_idx.append(dummy_rigid_idx[-1]+1)


    frame_atom_names = [bb_frame, frame2, frame3]
    # figure out which frames are resolved enough for us to model
    # this is a little roundabout because the atom ids are stored as arrays of 4 ints
    frame_atom_ids = [
        np.array([convert_atom_name(c) for c in atom_set])  # shape (3, 4)
        for atom_set in frame_atom_names
    ]
    atom_ids = atoms["name"]  # shape of (n_atom, 4)

    rigid_mask = []
    rigid_tensor7 = []
    for i, atom_id_set in enumerate(frame_atom_ids):
        atom_match = (atom_ids[..., None, :] == atom_id_set[None]).all(axis=-1)  # shape (n_atom, 3)
        frame_exists = atom_match.any(axis=0).all()
        rigid_mask.append(frame_exists)
        if frame_exists:
            frame_atom_coords = atoms["coords"][..., None, :] * atom_match[..., None]
            frame_atom_coords = frame_atom_coords.sum(axis=0)
            # print(frame_atom_coords)
            # TODO: i'm not sure why i need to do this, this is really jenk...
            if i == 0:
                frame_rot, frame_trans = compute_frame(
                    frame_atom_coords[0],
                    frame_atom_coords[1],
                    frame_atom_coords[2],
                )
                rigid = ru.Rigid(
                    rots=ru.Rotation(torch.as_tensor(frame_rot)),
                    trans=torch.as_tensor(frame_trans)
                )
            else:
                rigid = ru.Rigid.from_3_points(
                    torch.as_tensor(frame_atom_coords[0]),
                    torch.as_tensor(frame_atom_coords[1]),
                    torch.as_tensor(frame_atom_coords[2]),
                )
            tensor7 = rigid.to_tensor_7().numpy(force=True)
            rigid_tensor7.append(tensor7)
        else:
            rigid_tensor7.append(IDENTITY_TENSOR7.copy())

    return np.stack(rigid_tensor7, axis=0), np.array(rigid_mask), dummy_rigid_idx


def arbitrary_atom_to_frame(
    atom,
    atom_idx,
    valid_neighbors: list[int],
    valid_neighbor_coords: np.ndarray,
    neighbor_graph: nx.Graph
):
    if not atom["is_present"]:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # atom_idx = atom["atom_idx"]
    # print(atom_idx, neighbor_graph.nodes)
    if atom_idx in neighbor_graph.nodes:
        neighbors = [n for n in neighbor_graph.neighbors(atom_idx) if n in valid_neighbors]
    else:
        atom_name = atom['name']
        atom_name = [chr(c + 32) for c in atom_name if c != 0]
        atom_name = "".join(atom_name)
        # print(atom, atom_name)
        neighbors = []

    np.random.shuffle(neighbors)

    if len(neighbors) == 0:
        quat = Rotation.random().as_quat(canonical=True)
        trans = atom["coords"]
        return np.concatenate([quat, trans], axis=0)
    elif len(neighbors) == 1:
        try:
            neighbor_idx = neighbors[0]
            neighbor_coord = valid_neighbor_coords[valid_neighbors.index(neighbor_idx)]
            x_axis = neighbor_coord - atom["coords"]
            x_axis = x_axis / np.linalg.norm(x_axis + 1e-6)
            # sample y vecs until we get one which is suitable to make an axis from
            while True:
                y_vec = np.random.randn(3)
                y_vec = y_vec / np.linalg.norm(y_vec)
                if np.dot(x_axis, y_vec) < 1 - 1e-6:
                    break
            y_axis = y_vec - np.dot(x_axis, y_vec) * x_axis
            y_axis = y_axis / np.linalg.norm(y_axis)
            point3 = y_axis + atom["coords"]
            rot, trans = compute_frame(
                neighbor_coord,
                atom["coords"],
                point3,
            )
            quat = rot_to_quat(torch.as_tensor(rot)).numpy(force=True)
            tensor7 = np.concatenate([quat, trans], axis=-1)
            if np.isnan(tensor7).any():
                raise ValueError("encountered nan in computing semi-random rotation")
            return tensor7
        except Exception as e:
            print(f"Caught exception '{e}', replacing with random rotation")
            quat = Rotation.random().as_quat(canonical=True)
            trans = atom["coords"]
            return np.concatenate([quat, trans], axis=0)
    else:
        neighbor1_idx, neighbor2_idx = neighbors[:2]
        neighbor1_coord = valid_neighbor_coords[valid_neighbors.index(neighbor1_idx)]
        neighbor2_coord = valid_neighbor_coords[valid_neighbors.index(neighbor2_idx)]
        rot, trans = compute_frame(
            neighbor1_coord,
            atom["coords"],
            neighbor2_coord,
        )
        quat = rot_to_quat(torch.as_tensor(rot)).numpy(force=True)
        tensor7 = np.concatenate([quat, trans], axis=-1)
        return tensor7

# TODO: this is such a crazy function i should probably modularize it
# also some of the logic here feels really clunky
def tokenize_structure(  # noqa: C901, PLR0915
    struct: Structure,
    task_data: dict[str, np.ndarray],
    shuffle_chains: bool = False,
    shuffle_copied_fragments: bool = True
) -> Tuple[np.ndarray, np.array, np.ndarray]:
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
    atom_noising_mask = task_data['atom_noising_mask']
    res_type_noising_mask = task_data['res_type_noising_mask']

    # Create token data and rigid data
    token_data = []
    rigid_data = []
    # Create token bonds
    token_bonds = []
    atomized_bond_store = {}

    # Keep track of atom_idx, rigid_idx to token_idx
    token_idx = 0
    rigid_idx = 0
    atom_to_rigid = {}
    rigid_to_token = {}

    copy_data = []
    copy_indexed_residue_mask = task_data['copy_indexed_residue_mask']
    copy_unindexed_residue_mask = task_data['copy_unindexed_residue_mask']
    copy_atomized_residue_mask = task_data['copy_atomized_residue_mask']

    # Filter to valid chains only
    chains = struct.chains[struct.mask]

    if shuffle_chains:
        np.random.shuffle(chains)

    def _get_standard_protein_residue_data(
        res,
        atom_to_rigid,
        rigid_to_token,
        _token_idx,
        _rigid_idx,
        noise_seq
    ):
        # Get atom indices
        atom_start = res["atom_idx"]
        atom_end = res["atom_idx"] + res["atom_num"]

        ret_rigids = []
        # Token is present if centers are
        is_present = res["is_present"]

        # If protein, compute frame, only used for templates
        rigid_tensor7 = np.stack([IDENTITY_TENSOR7.copy() for _ in range(3)], axis=0)
        rigid_mask = np.zeros(3, dtype=bool)

        # Get frame atoms
        atoms = struct.atoms[atom_start:atom_end]
        noise_atoms = atom_noising_mask[atom_start:atom_end]

        rigid_tensor7, rigid_mask, dummy_rigid_idx = standard_residue_to_frames(
            res, atoms
        )

        # Create token
        token = TokenData(
            token_idx=_token_idx,
            rigid_idx=_rigid_idx,
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
            seq_noising_mask=noise_seq
        )

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

        for i, atom in enumerate(atoms):
            atom_name = atom['name']
            atom_name = [chr(c + 32) for c in atom_name if c != 0]
            atom_name = "".join(atom_name)
            atom_to_rigid[atom_start + i] = _rigid_idx + atom_name_to_cg_idx[atom_name]

        _noise_rigid = []
        # Update rigid_idx to token_idx
        for i in range(3):
            cg_atom_idxs = cg_idx_to_atom_idx[i]
            # we figure out if we're noising the rigid
            # by checking if any of its component atoms are being noised
            # we also need to check if its a dummy rigid
            # and if so, copy the noising status of its corresponding non-dummy rigid
            if len(cg_atom_idxs) > 0:
                noise_rigid = noise_atoms[cg_atom_idxs].any()
            else:
                noise_rigid = _noise_rigid[dummy_rigid_idx[i]]
            _noise_rigid.append(noise_rigid)

            rigid = RigidData(
                rigid_idx=_rigid_idx,
                token_idx=_token_idx,
                sidechain_idx=i,
                is_atomized=False,
                element=-1,
                charge=0,
                tensor7=rigid_tensor7[i],
                is_present=rigid_mask[i],
                rigids_noising_mask=noise_rigid
            )
            ret_rigids.append(rigid)
            rigid_to_token[_rigid_idx] = _token_idx

            _rigid_idx = _rigid_idx + 1
        _token_idx = _token_idx + 1
        return token, ret_rigids, _token_idx, _rigid_idx

    def _get_nonstandard_residue_data(
        res,
        atom_to_rigid,
        rigid_to_token,
        _token_idx,
        _rigid_idx,
        bond_graph_override=None
    ):
        if bond_graph_override is None:
            bond_graph_override = bond_graph

        ret_tokens = []
        ret_rigids = []
        # Get atom indices
        atom_start = res["atom_idx"]
        atom_end = res["atom_idx"] + res["atom_num"]

        # We use the unk protein token as res_type
        unk_token = const.unk_token["PROTEIN"]
        unk_id = const.token_ids[unk_token]

        # Get atom coordinates
        atom_data = struct.atoms[atom_start:atom_end]
        atom_coords = atom_data["coords"]

        valid_neighbors = [i for i in range(res["atom_num"]) if atom_data[i]["is_present"]]
        valid_neighbor_coords = atom_coords[np.array(valid_neighbors, dtype=int)]

        # Tokenize each atom
        for i, atom in enumerate(atom_data):
            # Token is present if atom is
            is_present = res["is_present"] & atom["is_present"]
            atom_idx = atom_start + i

            atom_tensor7 = arbitrary_atom_to_frame(
                atom,
                atom_idx,
                valid_neighbors,
                valid_neighbor_coords,
                bond_graph_override
            )

            # Create token
            token = TokenData(
                token_idx=_token_idx,
                rigid_idx=_rigid_idx,
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
            ret_tokens.append(token)

            # Update atom_idx to token_idx
            atom_to_rigid[atom_start + i] = _rigid_idx
            rigid = RigidData(
                rigid_idx=_rigid_idx,
                token_idx=_token_idx,
                sidechain_idx=0,
                is_atomized=True,
                element=atom["element"],
                charge=atom["charge"],
                tensor7=atom_tensor7,
                is_present=atom["is_present"],
                rigids_noising_mask=atom_noising_mask[atom_start + i]
            )
            ret_rigids.append(rigid)
            rigid_to_token[_rigid_idx] = _token_idx

            _rigid_idx = _rigid_idx + 1
            _token_idx = _token_idx + 1

        return ret_tokens, ret_rigids, _token_idx, _rigid_idx


    for chain in chains:
        # Get residue indices
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        is_protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]

        for i, res in enumerate(struct.residues[res_start:res_end]):
            # Standard residues are tokens
            if res["is_standard"] and (res['name'] != 'UNK') and is_protein:
                token, ret_rigids, token_idx, rigid_idx = _get_standard_protein_residue_data(
                    res,
                    atom_to_rigid,
                    rigid_to_token,
                    token_idx,
                    rigid_idx,
                    noise_seq=res_type_noising_mask[res_start + i]
                )

                if copy_indexed_residue_mask[res_start + i]:
                    if copy_atomized_residue_mask[res_start + i]:
                        aa_bond_data = get_standard_protein_residue_bonds(res['name'], atom_idx=0)
                        _bond_graph_override = nx.Graph([(bond["atom_1"], bond["atom_2"]) for bond in aa_bond_data])
                        ret_tokens, _ret_rigids, _, _ = _get_nonstandard_residue_data(res, {}, {}, 0, 0, bond_graph_override=_bond_graph_override)

                        atomized_bond_store[(chain['asym_id'], res['res_idx'])] = aa_bond_data

                        copy_data.extend([
                            {"token": t, "rigids": [r], "indexed?": True, "atomized?": True, "res_internal_idx": i}
                            for i, (t, r) in enumerate(zip(ret_tokens, _ret_rigids))
                        ])
                    else:
                        copy_token = copy.deepcopy(token)
                        copy_data.append({
                            "token": copy_token, "rigids": ret_rigids, "indexed?": True, "atomized?": False, "res_internal_idx": 0
                        })
                    token = replace(token, seq_noising_mask=True)
                    ret_rigids = [replace(r, rigids_noising_mask=True) for r in ret_rigids]

                elif copy_unindexed_residue_mask[res_start + i]:
                    if copy_atomized_residue_mask[res_start + i]:
                        aa_bond_data = get_standard_protein_residue_bonds(res['name'], atom_idx=0)
                        _bond_graph_override = nx.Graph([(bond["atom_1"], bond["atom_2"]) for bond in aa_bond_data])
                        ret_tokens, _ret_rigids, _, _ = _get_nonstandard_residue_data(res, {}, {}, 0, 0, bond_graph_override=_bond_graph_override)

                        atomized_bond_store[(chain['asym_id'], res['res_idx'])] = aa_bond_data

                        copy_data.extend([
                            {"token": t, "rigids": [r], "indexed?": False, "atomized?": True, "res_internal_idx": i}
                            for i, (t, r) in enumerate(zip(ret_tokens, _ret_rigids))
                        ])
                    else:
                        copy_token = copy.deepcopy(token)
                        copy_data.append({
                            "token": copy_token, "rigids": ret_rigids, "indexed?": False, "atomized?": False, "res_internal_idx": 0
                        })
                    token = replace(token, seq_noising_mask=True)
                    ret_rigids = [replace(r, rigids_noising_mask=True) for r in ret_rigids]

                token_data.append(astuple(token))
                rigid_data.extend([astuple(r) for r in ret_rigids])

            # Non-standard are tokenized per atom
            else:
                ret_tokens, ret_rigids, token_idx, rigid_idx = _get_nonstandard_residue_data(
                    res,
                    atom_to_rigid,
                    rigid_to_token,
                    token_idx,
                    rigid_idx
                )
                token_data.extend([astuple(t) for t in ret_tokens])
                rigid_data.extend([astuple(r) for r in ret_rigids])


    if shuffle_copied_fragments and len(copy_data) > 0:
        # we swap around the order of copied segments
        # since the ordering of these segments in the model inputs
        # could potentially serve to leak the ground truth ordering of these segments
        frag_idx = 0
        token_0 = asdict(copy_data[0]['token'])
        last_res_idx = token_0['res_idx']
        last_chain_id = token_0['asym_id']
        frag_mapping = {}
        for i, copy_dict in enumerate(copy_data):
            token = asdict(copy_dict['token'])
            chain_id = token['asym_id']
            if abs(last_res_idx - token['res_idx']) > 1 or chain_id != last_chain_id:
                frag_idx += 1

            if frag_idx not in frag_mapping:
                frag_mapping[frag_idx] = [copy_dict]
            else:
                frag_mapping[frag_idx].append(copy_dict)

            last_res_idx = token['res_idx']
            last_chain_id = chain_id

        frag_order = np.random.permutation(frag_idx+1)
        _new_copy_data = []
        for i in frag_order:
            _new_copy_data.extend(frag_mapping[i])
        copy_data = _new_copy_data

    copy_atomized_bonds = []
    # append our copied residues onto the tokenized data
    # this preserves res_idx but assigns the correct token_idx and rigid_idx
    # as well as specifying proper bookkeeping values
    for copy_dict in copy_data:
        # copy_dict = copy.deepcopy(copy_dict)  # TODO: idek if this is necessary
        token = asdict(copy_dict['token'])
        rigids = [asdict(r) for r in copy_dict['rigids']]

        token['is_copy'] = True
        token['token_idx'] = token_idx
        token['rigid_idx'] = rigid_idx
        token['is_unindexed'] = not copy_dict["indexed?"]
        token['is_atomized'] = copy_dict['atomized?']
        token_data.append(astuple(TokenData(**token)))

        token_res_tag = (token['asym_id'], token['res_idx'])
        if token_res_tag in atomized_bond_store and copy_dict['res_internal_idx'] == 0:
            copy_bond_data = atomized_bond_store[token_res_tag]
            copy_bond_data['atom_1'] += token_idx
            copy_bond_data['atom_2'] += token_idx
            copy_bond_data['type'] += 1
            copy_atomized_bonds.append(copy_bond_data.astype(TokenBond))

        for r in rigids:
            r['token_idx'] = token_idx
            r['rigid_idx'] = rigid_idx
            rigid_data.append(astuple(RigidData(**r)))
            rigid_idx += 1

        token_idx += 1

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

    if len(copy_atomized_bonds) > 0:
        token_bonds = np.concatenate([token_bonds] + copy_atomized_bonds)

    return token_data, rigid_data, token_bonds


def frames_to_atom14(res, res_tensor7):
    res_type = res["res_type"]
    aa_type = RES_TO_AA[int(res_type)]
    rigids = ru.Rigid.from_tensor_7(torch.as_tensor(res_tensor7))

    dummy_mask = torch.tensor([True])
    seq = torch.tensor([aa_type]).long()

    return cg_utils.compute_atom14_from_cg_frames(rigids, dummy_mask, seq, return_atom_mask=True)


def update_protein_sequence(
    struct: Structure,
    token_seq,
    update_token_seq,
    res_type_input=False
):
    """ Update a Structure object from tokens/rigid data

    Args:
        token_data (_type_): _description_
        rigid_data (_type_): _description_
    """
    struct = copy.deepcopy(struct)
    if not res_type_input:
        token_res_type = np.vectorize(lambda x: AA_TO_RES[x])(token_seq)
    else:
        token_res_type = token_seq

    chains = struct.chains[struct.mask]
    token_idx = 0

    for chain in chains:
        # Get residue indices
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        is_protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]

        for res in struct.residues[res_start:res_end]:
            if res["is_standard"] and is_protein:
                if update_token_seq[token_idx]:
                    res_type_i = token_res_type[token_idx]
                    res["res_type"] = res_type_i
                    res["name"] = const.tokens[res_type_i]
                token_idx += 1
            else:
                token_idx += res["atom_num"]

    return struct


def update_structure(
    struct: Structure,
    rigid_tensor7,
):
    """ Update a Structure object from tokens/rigid data

    Args:
        token_data (_type_): _description_
        rigid_data (_type_): _description_
    """
    struct = copy.deepcopy(struct)
    chains = struct.chains[struct.mask]

    token_idx = 0
    rigid_idx = 0

    for chain in chains:
        # Get residue indices
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        is_protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]

        tensor7_for_atom14 = []
        tensor7_idx = []
        tensor7_seq = []
        _rigid_idx = rigid_idx
        _token_idx = token_idx
        for i, res in enumerate(struct.residues[res_start:res_end]):
            if res["is_standard"] and is_protein:
                rigid_start = _rigid_idx
                rigid_end = _rigid_idx + 3
                tensor7 = rigid_tensor7[rigid_start:rigid_end].copy()
                tensor7_for_atom14.append(tensor7)
                tensor7_idx.append(i)

                res_type = res["res_type"]
                aa_type = RES_TO_AA[int(res_type)]
                tensor7_seq.append(aa_type)

                _rigid_idx += 3
                _token_idx += 1
            else:
                _rigid_idx += res["atom_num"]
                _token_idx += res["atom_num"]

        if len(tensor7_idx) > 0:
            tensor7_for_atom14 = np.stack(tensor7_for_atom14, axis=0)
            tensor7_for_atom14 = torch.as_tensor(tensor7_for_atom14)
            rigids_for_atom14 = ru.Rigid.from_tensor_7(tensor7_for_atom14)
            tensor7_seq = torch.as_tensor(tensor7_seq)
            dummy_mask = torch.ones_like(tensor7_seq, dtype=torch.bool)
            atom14, atom14_mask = cg_utils.compute_atom14_from_cg_frames(rigids_for_atom14, dummy_mask, tensor7_seq, return_atom_mask=True)
            atom14 = atom14.squeeze()
            atom14_mask = atom14_mask.squeeze().bool()
        else:
            tensor7_for_atom14 = []
            tensor7_for_atom14 = torch.as_tensor(tensor7_for_atom14)
            rigids_for_atom14 = None
            tensor7_seq = None
            dummy_mask = None
            atom14 = None
            atom14_mask = None

        for i, res in enumerate(struct.residues[res_start:res_end]):
            # Get atom indices
            atom_start = res["atom_idx"]
            atom_end = res["atom_idx"] + res["atom_num"]

            # Standard residues are tokens
            if res["is_standard"] and is_protein:
                assert atom14 is not None
                assert atom14_mask is not None
                # Token is present if centers are
                is_present = res["is_present"]

                # Get frame atoms
                atoms = struct.atoms[atom_start:atom_end]

                # rigid_start = rigid_idx
                # rigid_end = rigid_idx + 3
                # res_tensor7 = rigid_tensor7[rigid_start:rigid_end].copy()
                # atom14, atom14_mask = frames_to_atom14(res, res_tensor7)
                # atom14 = atom14.squeeze()
                # atom14_mask = atom14_mask.squeeze()
                # atom14_mask = atom14_mask.bool()
                _idx = tensor7_idx.index(i)
                res_atom14, res_atom14_mask = atom14[_idx], atom14_mask[_idx]
                atoms["coords"] = res_atom14[res_atom14_mask].numpy(force=True)
                atoms["is_present"] = res_atom14_mask[res_atom14_mask].numpy(force=True)

                token_idx += 1
                rigid_idx += 3

            # Non-standard are tokenized per atom
            else:
                # Get atom coordinates
                atom_data = struct.atoms[atom_start:atom_end]

                # Tokenize each atom
                for i, atom in enumerate(atom_data):
                    # Token is present if atom is
                    is_present = res["is_present"] & atom["is_present"]
                    atom_data[i]["coords"] = rigid_tensor7[rigid_idx, 4:]

                    rigid_idx += 1
                    token_idx += 1

    return struct


def sample_noise_tokenized_structure(  # noqa: C901, PLR0915
    chain_lens: dict[str, int],
    trans_std=16,
) -> Tuple[np.ndarray, np.array, np.ndarray]:
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
    # Create token data and rigid data
    token_data = []
    rigid_data = []

    # Keep track of atom_idx, rigid_idx to token_idx
    token_idx = 0
    rigid_idx = 0
    res_idx = 0

    # Filter to valid chains only

    for chain_idx, (chain, chain_len) in enumerate(chain_lens.items()):
        for _res_idx in range(chain_len):
            rigid_trans = np.random.randn(3, 3) * trans_std
            rigid_quat = Rotation.random(3).as_quat(canonical=True)
            rigid_tensor7 = np.concatenate([rigid_quat, rigid_trans], axis=-1)
            rigid_mask = np.ones(3, dtype=bool)

            # Create token
            token = TokenData(
                token_idx=token_idx,
                rigid_idx=rigid_idx,
                rigid_num=3,
                res_idx=res_idx,
                res_type=const.unk_token_ids['PROTEIN'],
                res_name=const.unk_token['PROTEIN'],
                sym_id=chain_idx,
                asym_id=chain_idx,
                entity_id=chain_idx,
                mol_type=const.chain_type_ids["PROTEIN"],
                resolved_mask=True,
                center_coords=rigid_tensor7[0, 4:],
                is_copy=False,
                is_unindexed=False,
                is_atomized=False,
                seq_noising_mask=True
            )
            token_data.append(astuple(token))

            # Update rigid_idx to token_idx
            for i in range(3):
                rigid = RigidData(
                    rigid_idx=rigid_idx,
                    token_idx=token_idx,
                    sidechain_idx=i,
                    is_atomized=False,
                    element=-1,
                    charge=0,
                    tensor7=rigid_tensor7[i],
                    is_present=rigid_mask[i],
                    rigids_noising_mask=True
                )
                rigid_data.append(astuple(rigid))

                rigid_idx += 1

            token_idx += 1
            res_idx += 1

    # Create token bonds
    token_bonds = []

    token_data = np.array(token_data, dtype=Token)
    token_bonds = np.array(token_bonds, dtype=TokenBond)
    rigid_data = np.array(rigid_data, dtype=Rigid)

    return token_data, rigid_data, token_bonds


