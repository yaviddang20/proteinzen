import copy
from dataclasses import astuple, asdict, dataclass, replace, fields
from typing import Tuple, List, Optional, Union
import functools as fn


import numpy as np
import torch
import networkx as nx
from scipy.spatial.transform import Rotation

from proteinzen.boltz.data import const
from proteinzen.boltz.data.types import (
    Structure, Atom, Bond, Chain, Connection, Interface, Residue
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
    ("resolved_mask", np.dtype("?")),
    ("center_coords", np.dtype("3f4")),
    ("is_copy", np.dtype("?")),
    ("is_unindexed", np.dtype("?")),
    ("is_atomized", np.dtype("?")),
    ("seq_noising_mask", np.dtype("?")),
    ("hotspot_type", np.dtype("i1")),
]

Rigid = [
    ("rigid_idx", np.dtype("i4")),
    ("token_idx", np.dtype("i4")),
    ("sidechain_idx", np.dtype("i1")),
    ("is_atomized", np.dtype("?")),
    ("element", np.dtype("i1")),
    ("charge", np.dtype("i1")),
    ("chirality", np.dtype("i1")),
    ("tensor7", np.dtype("7f4")),
    ("is_present", np.dtype("?")),
    ("rigids_noising_mask", np.dtype("?")),
    ("num_real_input_axes", np.dtype("i1")),
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

# TODO: probably move mmcif.py from data processing into proteinzen so i can just import this
def convert_atom_str_to_tuple(name: str) -> tuple[int, int, int, int]:
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
    name_tuple = [ord(c) - 32 for c in name]
    name_tuple = name_tuple + [0] * (4 - len(name))
    return tuple(name_tuple)

def convert_atom_tuple_to_str(name: tuple[int, int, int, int]) -> str:
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
    atom_name = [chr(c + 32) for c in name if c != 0]
    atom_name = "".join(atom_name)
    return atom_name

@dataclass(frozen=True)
class Tokenized:
    """Tokenized datatype."""

    tokens: np.ndarray
    rigids: np.ndarray
    bonds: np.ndarray
    structure: Structure

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
    hotspot_type: int


@dataclass
class RigidData:
    """RigidData datatype."""

    rigid_idx: int
    token_idx: int
    sidechain_idx: int
    is_atomized: bool
    element: int
    charge: int
    chirality: int
    tensor7: np.ndarray
    is_present: bool
    rigids_noising_mask: bool
    num_real_input_axes: int


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
    """ Generate the frame presentation for a canonical protein residue.

    Parameters
    ==========
    residue: proteinzen.boltz.data.types.Residue
        The residue to featurize
    atoms: np.ndarray, dtype=proteinzen.boltz.data.types.Atom
        The atoms in the residue to be featurized

    Returns
    =======
    rigids_tensor7: np.ndarray
        Stack of residue frames in tensor7 format
    rigids_mask: np.ndarray, dtype=bool
        Mask specifying if all the atoms needed to define each frame exists
    dummy_rigid_idx:  list[int]
        Specifies the source frame for any duplicate frames generated. If all frames
        are unique, this is [0, 1, 2].
    """
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
        np.array([convert_atom_str_to_tuple(c) for c in atom_set])  # shape (3, 4)
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


def is_colinear(point1, point2, point3, tol=1e-2):
    """ Check if three points are colinear"""
    v1 = point1 - point2
    v2 = point3 - point1
    e1 = v1 / (np.linalg.norm(v1) + 1e-10)
    e2 = v2 / (np.linalg.norm(v2) + 1e-10)
    e3 = np.cross(e1, e2)
    return np.linalg.norm(e3) < tol


def select_axes(atom_coord, neighbors, valid_neighbors, valid_neighbor_coords):
    np.random.shuffle(neighbors)
    for i, neighbor1 in enumerate(neighbors[:-1]):
        for neighbor2 in neighbors[i+1:]:
            coord1 = valid_neighbor_coords[valid_neighbors.index(neighbor1)]
            coord2 = valid_neighbor_coords[valid_neighbors.index(neighbor2)]
            if not is_colinear(coord1, atom_coord, coord2):
                return (neighbor1, neighbor2)
    return None

def gen_rand_rot_frame(trans):
    quat = Rotation.random().as_quat(canonical=True, scalar_first=True)
    return np.concatenate([quat, trans], axis=0), 0

def gen_semirand_rot_frame(center, x_axis_point):
    x_axis = x_axis_point - center
    x_axis = x_axis / np.linalg.norm(x_axis + 1e-6)
    # sample y vecs until we get one which is suitable to make an axis from
    while True:
        y_vec = np.random.randn(3)
        y_vec = y_vec / np.linalg.norm(y_vec)
        if np.dot(x_axis, y_vec) < 1 - 1e-6:
            break
    y_axis = y_vec - np.dot(x_axis, y_vec) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)
    y_axis_point = y_axis + center
    rot, trans = compute_frame(
        x_axis_point,
        center,
        y_axis_point,
    )
    quat = rot_to_quat(torch.as_tensor(rot)).numpy(force=True)
    tensor7 = np.concatenate([quat, trans], axis=-1)
    if np.isnan(tensor7).any():
        raise ValueError("encountered nan in computing semi-random rotation")
    return tensor7, 1

def gen_det_rot_frame(center, point1, point2):
    rot, trans = compute_frame(
        point1,
        center,
        point2,
    )
    quat = rot_to_quat(torch.as_tensor(rot)).numpy(force=True)
    tensor7 = np.concatenate([quat, trans], axis=-1)
    return tensor7, 2


def arbitrary_atom_to_frame(
    atom,
    atom_idx,
    valid_neighbors: list[int],
    valid_neighbor_coords: np.ndarray,
    neighbor_graph: nx.Graph,
    use_identity_rot: bool = True,
):
    """ For an arbitrary atom, compute a frame to use for that atom. This function will try its best to
    construct a frame from real axes given the particular input chemical graph. If it cannot, it'll either
    construct a semi-random rotation (one axis is fixed) or sample a random rotation for the frame.

    Parameters
    ==========
    atom: proteinzen.boltz.data.types.Atom
        Data about the atom to featurize
    atom_idx: int
        The identifier of `atom` in this particular chemical subgraph.
    valid_neighbors: list[int]
        A list of atom ids which are valid connections to `atom` to use for frame construction.
    valid_neighbor_coords: np.ndarray
        The coordinates for every atom in `valid_neighbors`.
    neighbor_graph: nx.Graph
        A networkX graph which specifies the connectivity between atoms in this particular chemical subgraph.

    Returns
    =======
    tensor7: np.ndarray
        The frame for the atom, represented in tensor7 format.
    num_real_input_axes: int
        The number of real axes used to construct the frame. Possible values {0, 1, 2}.
    """
    if not atom["is_present"]:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0

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

    _select_axes = fn.partial(select_axes, valid_neighbors=valid_neighbors, valid_neighbor_coords=valid_neighbor_coords)
    if use_identity_rot:
        quat = Rotation.identity().as_quat(canonical=True, scalar_first=True)
        trans = atom["coords"]
        return np.concatenate([quat, trans], axis=0), 0
    else:
        if len(neighbors) == 0:
            quat = Rotation.random().as_quat(canonical=True, scalar_first=True)
            trans = atom["coords"]
            return np.concatenate([quat, trans], axis=0), 0
        elif len(neighbors) == 1:
            neighbor_idx = neighbors[0]
            neighbor_neighbors = [n for n in neighbor_graph.neighbors(neighbor_idx) if n in valid_neighbors and n != atom_idx]
            # print(atom, atom_idx, neighbor_idx, neighbor_neighbors, list(neighbor_graph.edges))
            if len(neighbor_neighbors) > 0:
                # if we can get a second hop neighbor to define the frame, use that
                axes = _select_axes(atom["coords"], [neighbor_idx] + neighbor_neighbors)
                if axes is not None:
                    # print(atom, 2)
                    return gen_det_rot_frame(atom['coords'], *axes)
            try:
                neighbor_coord = valid_neighbor_coords[valid_neighbors.index(neighbor_idx)]
                # print(atom, 1)
                return gen_semirand_rot_frame(atom['coords'], neighbor_coord)
            except Exception as e:
                print(f"Caught exception '{e}', replacing with random rotation")
                # print(atom, 0)
                return gen_rand_rot_frame(atom['coords'])
        else:
            axes = _select_axes(atom["coords"], neighbors)
            if axes is not None:
                return gen_det_rot_frame(atom["coords"], *axes)

            for neighbor_idx in neighbors:
                try:
                    neighbor_coord = valid_neighbor_coords[valid_neighbors.index(neighbor_idx)]
                    return gen_semirand_rot_frame(atom['coords'], neighbor_coord)
                except Exception:
                    pass
            print("Error in featurizing rotation, replacing with random rotation")
            return gen_rand_rot_frame(atom['coords'])


def generate_copy_structure(
    struct: Structure,
    task_data: dict[str, np.ndarray],
    shuffle_copied_fragments: bool = True
):
    # Filter to valid chains only
    chains = struct.chains[struct.mask]

    copy_indexed_residue_mask = task_data['copy_indexed_residue_mask']
    copy_unindexed_residue_mask = task_data['copy_unindexed_residue_mask']
    assert not task_data['copy_atomized_residue_mask'].any(), "generate_copy_structure currently doesn't support copying atomized residues"

    res_copied_indices = []
    atom_copied_indices = []

    copy_chains = {}
    copy_residues = []
    copy_atoms = []
    frag_ids = []
    curr_frag_id = 0
    last_res_idx = None
    last_chain_id = None

    max_res_idx = 0
    max_atom_idx = 0

    # parse out the copy structure elements
    for chain in chains:
        # Get residue indices
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        is_protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]

        for i, res in enumerate(struct.residues[res_start:res_end]):
            # Standard residues are tokens
            if res["is_standard"] and (res['name'] != 'UNK') and is_protein:
                res_idx = res_start + i #res['res_idx']
                is_indexed = copy_indexed_residue_mask[res_idx]
                is_unindexed = copy_unindexed_residue_mask[res_idx]
                assert not (is_indexed & is_unindexed), f"residue copy of {res_idx} cannot be both indexed and unindexed!"

                if not (is_indexed | is_unindexed):
                    continue

                # add a new chain entry if it doesn't exist yet
                chain_name = chain['name']
                if chain_name not in copy_chains:
                    chain_copy = chain.copy()
                    chain_copy['atom_idx'] = max_atom_idx
                    chain_copy['atom_num'] = 0
                    chain_copy['res_idx'] = max_res_idx
                    chain_copy['res_num'] = 0
                    copy_chains[chain_name] = chain_copy

                # update the chain entry associated with this copied residue
                copy_chains[chain_name]['res_num'] += 1
                copy_chains[chain_name]['atom_num'] += res['atom_num']
                # copy the residue atoms
                atom_start = res['atom_idx']
                atom_end = atom_start + res['atom_num']
                copy_atoms.append(struct.atoms[atom_start:atom_end].copy())
                # update the residue-to-atoms mapping
                res_copy = res.copy()
                res_copy['atom_idx'] = max_atom_idx
                copy_residues.append(res_copy)

                max_res_idx += 1
                max_atom_idx += res['atom_num']

                # mark which residue was copied
                res_copied_indices.append(res_start + i)
                atom_copied_indices.append(list(range(atom_start, atom_end)))
                # mark what fragment this residue belongs to
                if last_res_idx is None:
                    last_res_idx = res['res_idx']
                if last_chain_id is None:
                    last_chain_id = chain['asym_id']
                # if this residue is adjacent to the previous residue, we keep the frag idx the same
                # else we increment to the next one
                is_adj_to_last_res = (abs(last_res_idx - res['res_idx']) < 2 and last_chain_id == chain['asym_id'])
                if not is_adj_to_last_res:
                    curr_frag_id += 1
                frag_ids.append(curr_frag_id)
                last_res_idx = res['res_idx']
                last_chain_id = chain['asym_id']

    copy_chains = list(copy_chains.values())

    if shuffle_copied_fragments:
        # if specified, we shuffle the linear ordering of copied fragments within each copy chain
        # then shuffle the order of the copy chains themselves

        # assign a chain idx to each copied residue
        chain_frag_mapping = {}
        res_partition = np.repeat(
            np.arange(len(copy_chains)),
            [c['res_num'] for c in copy_chains]
        )

        # structure the residues into chains as collection of fragments
        # being sure to record where each copy residue originally came from
        for i, residue in enumerate(copy_residues):
            res_chain_idx = res_partition[i]
            res_frag_idx = frag_ids[i]
            if res_chain_idx not in chain_frag_mapping:
                chain_frag_mapping[res_chain_idx] = {
                    res_frag_idx: {
                        "residues": [],
                        "atoms": [],
                        "original_res_copy_idx": [],
                        "original_atom_copy_idx": [],
                    }
                }
            else:
                if res_frag_idx not in chain_frag_mapping[res_chain_idx]:
                    chain_frag_mapping[res_chain_idx][res_frag_idx] = {
                        "residues": [],
                        "atoms": [],
                        "original_res_copy_idx": [],
                        "original_atom_copy_idx": [],
                    }
            chain_frag = chain_frag_mapping[res_chain_idx][res_frag_idx]
            chain_frag['residues'].append(residue)
            chain_frag['atoms'].append(copy_atoms[i])
            chain_frag['original_res_copy_idx'].append(res_copied_indices[i])
            chain_frag['original_atom_copy_idx'].append(atom_copied_indices[i])

        # compute the new order of the copy chains
        chain_order = np.random.permutation(len(copy_chains))

        # shuffle the fragments by shuffling fragments within chains
        # in order of chain_order
        shuffled_chains = []
        shuffled_residues = []
        shuffled_atoms = []
        shuffled_res_copied_indices = []
        shuffled_atom_copied_indices = []
        shuffled_frag_ids = []
        num_res = 0
        num_atoms = 0
        for chain_idx in chain_order:
            chain_frag_list = list(chain_frag_mapping[chain_idx].keys())
            selected_chain = copy_chains[chain_idx]
            # renumber the chain metadata
            # (we don't need to change the counter fields bc we keep the same number of residues)
            selected_chain['res_idx'] = num_res
            selected_chain['atom_idx'] = num_atoms
            shuffled_chains.append(selected_chain)

            frag_order = np.random.permutation(chain_frag_list)

            for frag_idx in frag_order:
                frag_dict = chain_frag_mapping[chain_idx][frag_idx]
                frag_residues = frag_dict['residues']

                frag_atom_start = frag_residues[0]['atom_idx']
                # reindex the residue-to-atoms mapping
                for residue in frag_residues:
                    residue['atom_idx'] -= frag_atom_start
                    residue['atom_idx'] += num_atoms
                    shuffled_residues.append(residue)
                    shuffled_frag_ids.append(frag_idx)

                num_res += len(frag_residues)
                for atoms in frag_dict['atoms']:
                    shuffled_atoms.append(atoms)
                    num_atoms += len(atoms)
                shuffled_res_copied_indices.extend(frag_dict['original_res_copy_idx'])
                shuffled_atom_copied_indices.extend(frag_dict['original_atom_copy_idx'])

        copy_chains = shuffled_chains
        copy_residues = shuffled_residues
        copy_atoms = shuffled_atoms
        res_copied_indices = shuffled_res_copied_indices
        atom_copied_indices = shuffled_atom_copied_indices
        frag_ids = shuffled_frag_ids

    copy_struct = Structure(
        atoms=np.concatenate(copy_atoms, axis=-1),
        bonds=np.array([], dtype=Bond),
        residues=np.stack(copy_residues, axis=0),
        chains=np.stack(copy_chains, axis=0),
        connections=np.array([], dtype=Connection),
        interfaces=np.array([], dtype=Interface),
        mask=np.array([True for _ in copy_chains])
    )

    new_copy_struct = copy.deepcopy(copy_struct)
    new_copy_struct.chains['res_idx'] += len(struct.residues)
    new_copy_struct.chains['atom_idx'] += len(struct.atoms)
    new_copy_struct.residues['atom_idx'] += len(struct.atoms)

    struct_new = Structure(
        **{
            key: np.concatenate([
                getattr(struct, key),
                getattr(new_copy_struct, key)
            ], axis=-1)
        for key in [f.name for f in fields(Structure)]
        }
    )

    # flatten atom_copied_indices
    _atom_copied_indices = []
    for l in atom_copied_indices:
        _atom_copied_indices.extend(l)
    atom_copied_indices = _atom_copied_indices

    # create new task data masks
    new_task_data = {}
    def extract_mask(mask, indices, replace_value):
        source_mask = mask.copy()
        copy_mask = source_mask[indices]
        if replace_value is not None:
            source_mask[indices] = replace_value
        return np.concatenate([source_mask, copy_mask], axis=-1)

    new_task_data['res_type_noising_mask'] = extract_mask(
        task_data['res_type_noising_mask'],
        res_copied_indices,
        True
    )
    new_task_data['is_unindexed_residue_mask'] = extract_mask(
        task_data['copy_unindexed_residue_mask'],
        res_copied_indices,
        False
    )
    new_task_data['is_indexed_residue_mask'] = extract_mask(
        task_data['copy_indexed_residue_mask'],
        res_copied_indices,
        False
    )
    new_task_data['is_atomized_residue_mask'] = extract_mask(
        task_data['copy_atomized_residue_mask'],
        res_copied_indices,
        False
    )
    if 'res_hotspot_type' in task_data:
        new_task_data['res_hotspot_type'] = extract_mask(
            task_data['res_hotspot_type'],
            res_copied_indices,
            None
        )
    new_task_data['atom_noising_mask'] = extract_mask(
        task_data['atom_noising_mask'],
        atom_copied_indices,
        True
    )

    residue_entity_ids = []
    for chain in struct.chains:
        residue_entity_ids.extend([chain['entity_id'] for _ in range(chain['res_num'])])
    max_entity_id = max(residue_entity_ids)
    residue_entity_ids.extend([i + max_entity_id + 1 for i in frag_ids])

    new_task_data['residue_entity_id'] = np.array(residue_entity_ids)

    return copy_struct, struct_new, new_task_data


class StructureTokenizer:
    """ A class object for tokenization, mainly to help code organization"""
    def __init__(
        self,
        struct: Structure,
        task_data: dict[str, np.ndarray],
        shuffle_chains: bool = False,
        shuffle_copied_fragments: bool = True,
        use_identity_rot: bool = True,
    ):
        self.struct = struct
        self.task_data = task_data
        self.shuffle_chains = shuffle_chains
        self.shuffle_copied_fragments = shuffle_copied_fragments
        self.use_identity_rot = use_identity_rot
        self.bond_graph = nx.Graph([(bond["atom_1"], bond["atom_2"]) for bond in struct.bonds])

        self.token_idx = 0
        self.rigid_idx = 0
        self.atom_to_rigid = {}
        self.rigid_to_token = {}

        self.copy_data = []
        self.atomized_bond_store = {}

        self.processed = False

    def _get_standard_protein_residue_data(
        self,
        chain,
        res,
        noise_seq,
        hotspot_type,
        entity_id
    ):
        # Get atom indices
        atom_start = res["atom_idx"]
        atom_end = res["atom_idx"] + res["atom_num"]
        # store returned rigids
        ret_rigids = []
        # Token is present if centers are
        is_present = res["is_present"]

        # If protein, compute frame, only used for templates
        rigid_tensor7 = np.stack([IDENTITY_TENSOR7.copy() for _ in range(3)], axis=0)
        rigid_mask = np.zeros(3, dtype=bool)

        # Get frame atoms
        atoms = self.struct.atoms[atom_start:atom_end]
        atom_noising_mask = self.task_data['atom_noising_mask']
        noise_atoms = atom_noising_mask[atom_start:atom_end]
        # get residue frames
        rigid_tensor7, rigid_mask, dummy_rigid_idx = standard_residue_to_frames(
            res, atoms
        )

        # Create token
        token = TokenData(
            token_idx=self.token_idx,
            rigid_idx=self.rigid_idx,
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
            seq_noising_mask=noise_seq,
            hotspot_type=hotspot_type
        )

        # compute a mapping for which rigids correspond to which atoms
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
        # update the global atom_to_rigid mapping
        for i, atom in enumerate(atoms):
            atom_name = atom['name']
            atom_name = [chr(c + 32) for c in atom_name if c != 0]
            atom_name = "".join(atom_name)
            self.atom_to_rigid[atom_start + i] = self.rigid_idx + atom_name_to_cg_idx[atom_name]

        _noise_rigid : List[bool] = []
        # Update rigid_idx to token_idx
        for i in range(3):
            cg_atom_idxs = cg_idx_to_atom_idx[i]
            # we figure out if we're noising the rigid
            # by checking if any of its component atoms are being noised
            # we also need to check if its a dummy rigid
            if len(cg_atom_idxs) > 0:
                noise_rigid = bool(noise_atoms[cg_atom_idxs].any())
            else:
                # if rigid is a dummy rigid, copy the noising status of its corresponding non-dummy rigid
                noise_rigid = _noise_rigid[dummy_rigid_idx[i]]
            _noise_rigid.append(noise_rigid)

            rigid = RigidData(
                rigid_idx=self.rigid_idx,
                token_idx=self.token_idx,
                sidechain_idx=i,
                is_atomized=False,
                element=-1,
                charge=0,
                chirality=0,
                tensor7=rigid_tensor7[i],
                is_present=rigid_mask[i],
                rigids_noising_mask=noise_rigid,
                num_real_input_axes=2
            )
            ret_rigids.append(rigid)
            self.rigid_to_token[self.rigid_idx] = self.token_idx

            self.rigid_idx = self.rigid_idx + 1
        self.token_idx = self.token_idx + 1
        return token, ret_rigids

    def _get_nonstandard_residue_data(
        self,
        chain,
        res,
        hotspot_type,
        entity_id,
    ):
        token_idx = self.token_idx
        rigid_idx = self.rigid_idx
        atom_to_rigid = self.atom_to_rigid
        rigid_to_token = self.rigid_to_token

        ret_tokens = []
        ret_rigids = []
        # Get atom indices
        atom_start = res["atom_idx"]
        atom_end = res["atom_idx"] + res["atom_num"]

        # We use the unk protein token as res_type
        unk_token = const.unk_token["PROTEIN"]
        unk_id = const.token_ids[unk_token]

        # Get atom coordinates
        atom_data = self.struct.atoms[atom_start:atom_end]
        atom_coords = atom_data["coords"]
        atom_noising_mask = self.task_data['atom_noising_mask']

        valid_neighbors = [i for i in range(res["atom_num"]) if atom_data[i]["is_present"]]
        valid_neighbor_coords = atom_coords[np.array(valid_neighbors, dtype=int)]

        # Tokenize each atom
        for i, atom in enumerate(atom_data):
            # Token is present if atom is
            is_present = res["is_present"] & atom["is_present"]
            atom_idx = atom_start + i
            # print(bond_graph_override.edges.data(), i)

            atom_tensor7, num_real_input_axes = arbitrary_atom_to_frame(
                atom,
                atom_idx,
                valid_neighbors,
                valid_neighbor_coords,
                self.bond_graph,
                use_identity_rot=self.use_identity_rot,
            )

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
                entity_id=entity_id,
                mol_type=chain["mol_type"],
                resolved_mask=is_present,
                center_coords=atom_tensor7[4:],
                is_copy=False,
                is_unindexed=False,
                is_atomized=True,
                seq_noising_mask=False,
                hotspot_type=hotspot_type
            )
            ret_tokens.append(token)

            # Update atom_idx to token_idx
            atom_to_rigid[atom_start + i] = rigid_idx
            rigid = RigidData(
                rigid_idx=rigid_idx,
                token_idx=token_idx,
                sidechain_idx=0,
                is_atomized=True,
                element=atom["element"],
                charge=atom["charge"],
                chirality=atom["chirality"],
                tensor7=atom_tensor7,
                is_present=atom["is_present"],
                rigids_noising_mask=bool(atom_noising_mask[atom_start + i]),
                num_real_input_axes=num_real_input_axes
            )
            ret_rigids.append(rigid)
            rigid_to_token[rigid_idx] = token_idx

            rigid_idx = rigid_idx + 1
            token_idx = token_idx + 1

        self.token_idx = token_idx
        self.rigid_idx = rigid_idx

        return ret_tokens, ret_rigids

    def tokenize(
        self,
    ):
        assert not self.processed, (
            "im surprised you got here, "
            "this is a one-time use function, and we have already tokenized this data. "
            "please recreate this object if you wish to re-tokenize this data"
        )
        # Create token data and rigid data
        token_data = []
        rigid_data = []
        # Create token bonds
        token_bonds = []

        res_type_noising_mask = self.task_data['res_type_noising_mask']
        # res_hotspot_type = self.task_data['res_hotspot_type']

        # Filter to valid chains only
        chains = self.struct.chains[self.struct.mask]

        if self.shuffle_chains:
            np.random.shuffle(chains)

        for chain in chains:
            # Get residue indices
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            is_protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]

            for i, res in enumerate(self.struct.residues[res_start:res_end]):
                if "res_hotspot_type" in self.task_data:
                    res_hotspot_type = self.task_data['res_hotspot_type'][res_start + i]
                else:
                    res_hotspot_type = 0

                is_unindexed_motif = self.task_data['is_unindexed_residue_mask'][res_start + i]
                is_indexed_motif = self.task_data['is_indexed_residue_mask'][res_start + i]
                is_atomized = self.task_data['is_atomized_residue_mask'][res_start + i]
                entity_id = self.task_data['residue_entity_id'][res_start + i]

                # Standard residues are tokens
                if res["is_standard"] and (res['name'] != 'UNK') and is_protein:

                    if not is_atomized:
                        token, ret_rigids = self._get_standard_protein_residue_data(
                            chain, res,
                            noise_seq=res_type_noising_mask[res_start + i],
                            hotspot_type=res_hotspot_type,
                            entity_id=entity_id
                        )

                        token = replace(
                            token,
                            is_copy=is_unindexed_motif or is_indexed_motif,
                            is_unindexed=is_unindexed_motif,
                        )

                        token_data.append(astuple(token))
                        rigid_data.extend([astuple(r) for r in ret_rigids])
                    else:
                        # add atomized residue bond data to bond graph
                        aa_bond_data = get_standard_protein_residue_bonds(
                            res['name'],
                            atom_idx=res['atom_idx']
                        )
                        aa_bond_graph = nx.Graph([(bond["atom_1"], bond["atom_2"]) for bond in aa_bond_data])
                        self.bond_graph = nx.compose(self.bond_graph, aa_bond_graph)

                        tokens, rigids = self._get_nonstandard_residue_data(
                            chain, res, hotspot_type=res_hotspot_type,
                            entity_id=entity_id
                        )

                        tokens_new = []
                        for t in tokens:
                            _t = replace(
                                t,
                                is_copy=is_unindexed_motif or is_indexed_motif,
                                is_unindexed=is_unindexed_motif,
                                is_atomized=True
                            )
                            tokens_new.append(_t)

                        token_data.extend([astuple(t) for t in tokens_new])
                        rigid_data.extend([astuple(r) for r in rigids])

                # Non-standard are tokenized per atom
                else:
                    ret_tokens, ret_rigids = self._get_nonstandard_residue_data(
                        chain, res, hotspot_type=res_hotspot_type,
                        entity_id=entity_id
                    )
                    token_data.extend([astuple(t) for t in ret_tokens])
                    rigid_data.extend([astuple(r) for r in ret_rigids])

        # Add atom-atom bonds from ligands
        for bond in self.struct.bonds:
            atom1 = bond["atom_1"]
            atom2 = bond["atom_2"]
            if atom1 not in self.atom_to_rigid or atom2 not in self.atom_to_rigid:
                continue
            rigid1 = self.atom_to_rigid[atom1]
            rigid2 = self.atom_to_rigid[atom2]
            if rigid1 not in self.rigid_to_token or rigid2 not in self.rigid_to_token:
                continue
            token_bond = (
                self.rigid_to_token[rigid1],
                self.rigid_to_token[rigid2],
                bond["type"] + 1,
            )
            token_bonds.append(token_bond)

        token_data = np.array(token_data, dtype=Token)
        token_bonds = np.array(token_bonds, dtype=TokenBond)
        rigid_data = np.array(rigid_data, dtype=Rigid)

        # mark that we've run this function already
        self.processed = True
        return token_data, rigid_data, token_bonds


def tokenize_structure(  # noqa: C901, PLR0915
    struct: Structure,
    task_data: dict[str, np.ndarray],
    shuffle_chains: bool = False,
    shuffle_copied_fragments: bool = True,
    use_identity_rot: bool = True,
    copy_indexed_residues: bool = True
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
        The rigids data.
    np.ndarray
        The tokenized bonds.

    """

    chain_residue_mask = []
    for i, chain in enumerate(struct.chains):
        chain_residue_mask.extend([struct.mask[i] for _ in range(chain['res_num'])])
    chain_residue_mask = np.array(chain_residue_mask)
    unindexed_copy = (task_data['copy_unindexed_residue_mask'] & struct.residues['is_present'] & chain_residue_mask)
    indexed_copy = (task_data['copy_indexed_residue_mask'] & struct.residues['is_present'] & chain_residue_mask)
    atomized_copy = (task_data['copy_atomized_residue_mask'] & struct.residues['is_present'] & chain_residue_mask)

    is_unk_standard = (struct.residues['name'] == 'UNK')
    # prevent copying a nonstandard residues
    # TODO: in theory we should allow copying of nonstandard residues if they are atomized
    unindexed_copy[~struct.residues['is_standard']] = False
    unindexed_copy[is_unk_standard] = False
    indexed_copy[~struct.residues['is_standard']] = False
    indexed_copy[is_unk_standard] = False
    # unindexed_copy[~struct.residues['is_standard']] &= atomized_copy[~struct.residues['is_standard']]
    # indexed_copy[~struct.residues['is_standard']] &= atomized_copy[~struct.residues['is_standard']]

    perform_copy = (
        unindexed_copy.any()
        or (indexed_copy.any() and copy_indexed_residues)
        or atomized_copy.any()
    )

    if perform_copy:
        copy_struct, struct_new, new_task_data = generate_copy_structure(
            struct,
            task_data,
            shuffle_copied_fragments
        )
    else:
        struct_new = struct

        # rename task mask keys to new key names
        new_task_data = {}
        new_task_data['res_type_noising_mask'] = task_data['res_type_noising_mask']
        assert not unindexed_copy.any()
        new_task_data['is_unindexed_residue_mask'] = task_data['copy_unindexed_residue_mask']
        assert not atomized_copy.any()
        new_task_data['is_atomized_residue_mask'] = task_data['copy_atomized_residue_mask']
        if copy_indexed_residues:
            assert not indexed_copy.any()
        new_task_data['is_indexed_residue_mask'] = task_data['copy_indexed_residue_mask']
        if 'res_hotspot_type' in task_data:
            new_task_data['res_hotspot_type'] = task_data['res_hotspot_type']
        new_task_data['atom_noising_mask'] = task_data['atom_noising_mask']

        # add entity ids
        residue_entity_ids = []
        for chain in struct.chains:
            residue_entity_ids.extend([chain['entity_id'] for _ in range(chain['res_num'])])
        new_task_data['residue_entity_id'] = np.array(residue_entity_ids)

    tokenizer = StructureTokenizer(
        struct_new,
        new_task_data,
        shuffle_chains,
        use_identity_rot=use_identity_rot
    )
    return tokenizer.tokenize()
