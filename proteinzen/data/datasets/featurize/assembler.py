import random
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, from_numpy
from torch.nn.functional import one_hot

from boltz.data import const
from boltz.data.feature.symmetry import (
    get_amino_acids_symmetries,
    get_chain_symmetries,
    get_ligand_symmetries,
)
from boltz.data.pad import pad_dim
from boltz.data.types import (
    Structure,
)


@dataclass(frozen=True)
class Tokenized:
    """Tokenized datatype."""

    tokens: np.ndarray
    rigids: np.ndarray
    bonds: np.ndarray
    structure: Structure


def process_token_features(
    data: Tokenized,
    copy_unindexed_token_mask: Optional[np.ndarray] = None,
    copy_indexed_token_mask: Optional[np.ndarray] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Tensor]:
    """Get the token features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_tokens : int
        The maximum number of tokens.

    Returns
    -------
    dict[str, Tensor]
        The token features.

    """
    # Token data
    token_data = data.tokens
    token_bonds = data.bonds

    seq_index = torch.arange(len(token_data), dtype=torch.long)
    if copy_indexed_token_mask is not None:
        copy_tokens = token_data[copy_indexed_token_mask]
        copy_tokens["token_idx"] = np.arange(len(copy_tokens)) + len(token_data)
        token_data = np.concatenate([
            token_data, copy_indexed_tokens,
        ], dim=0)
        seq_index = torch.cat([
            seq_index,
            seq_index[from_numpy(copy_indexed_token_mask)],
        ], dim=0)

    if copy_unindexed_token_mask is not None:
        copy_tokens = token_data[copy_unindexed_token_mask]
        copy_tokens["token_idx"] = np.arange(len(copy_tokens)) + len(token_data)
        token_data = np.concatenate([
            token_data, copy_unindexed_tokens,
        ], dim=0)
        seq_index = torch.cat([
            seq_index,
            torch.arange(len(copy_tokens)) + len(seq_index),
        ], dim=0)

    # Token core features
    token_index = torch.arange(len(token_data), dtype=torch.long)
    residue_index = from_numpy(token_data["res_idx"].copy()).long()
    asym_id = from_numpy(token_data["asym_id"].copy()).long()
    entity_id = from_numpy(token_data["entity_id"].copy()).long()
    sym_id = from_numpy(token_data["sym_id"].copy()).long()
    mol_type = from_numpy(token_data["mol_type"].copy()).long()
    res_type = from_numpy(token_data["res_type"].copy()).long()
    token_to_rep_rigid = from_numpy(token_data["rigid_idx"]).long()

    # Token mask features
    pad_mask = torch.ones(len(token_data), dtype=torch.float)
    resolved_mask = from_numpy(token_data["resolved_mask"].copy()).float()

    # Token bond features
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        num_tokens = max_tokens if pad_len > 0 else len(token_data)
    else:
        num_tokens = len(token_data)

    # TODO: this does not preserve bonds for copied residues
    # i think this is not a problem for now but might need to consider it in the future
    tok_to_idx = {tok["token_idx"]: idx for idx, tok in enumerate(token_data)}
    bonds = torch.zeros(num_tokens, num_tokens, dtype=torch.float)
    for token_bond in token_bonds:
        token_1 = tok_to_idx[token_bond["token_1"]]
        token_2 = tok_to_idx[token_bond["token_2"]]
        bonds[token_1, token_2] = 1
        bonds[token_2, token_1] = 1

    bonds = bonds.unsqueeze(-1)

    # Pad to max tokens if given
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        if pad_len > 0:
            token_index = pad_dim(token_index, 0, pad_len)
            seq_index = pad_dim(seq_index, 0, pad_len)
            residue_index = pad_dim(residue_index, 0, pad_len)
            asym_id = pad_dim(asym_id, 0, pad_len)
            entity_id = pad_dim(entity_id, 0, pad_len)
            sym_id = pad_dim(sym_id, 0, pad_len)
            mol_type = pad_dim(mol_type, 0, pad_len)
            res_type = pad_dim(res_type, 0, pad_len)
            pad_mask = pad_dim(pad_mask, 0, pad_len)
            resolved_mask = pad_dim(resolved_mask, 0, pad_len)
            token_to_rep_rigid = pad_dim(token_to_rep_rigid, 0, pad_len)

    token_features = {
        "token_idx": token_index,
        "residue_idx": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "res_type": res_type,
        "token_to_rep_rigid": token_to_rep_rigid,
        "token_bonds": bonds,
        "token_pad_mask": pad_mask,
        "token_resolved_mask": resolved_mask,
    }
    return token_features


def process_rigid_features(
    data: Tokenized,
    copy_unindexed_token_mask: Optional[np.ndarray] = None,
    copy_indexed_token_mask: Optional[np.ndarray] = None,
    rigids_per_window_queries: int = 16,
    max_atoms: Optional[int] = None,
) -> dict[str, Tensor]:
    """Get the atom features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_atoms : int, optional
        The maximum number of atoms.

    Returns
    -------
    dict[str, Tensor]
        The atom features.

    """
    # Filter to tokens' atoms
    token_data = data.tokens
    rigid_data = data.rigids
    ref_space_uid = []
    rigid_to_token = []

    if copy_indexed_token_mask is not None:
        copy_tokens = token_data[copy_indexed_token_mask]
        copy_tokens["token_idx"] = np.arange(len(copy_tokens)) + len(token_data)
        token_data = np.concatenate([
            token_data, copy_indexed_tokens,
        ], dim=0)

    if copy_unindexed_token_mask is not None:
        copy_tokens = token_data[copy_unindexed_token_mask]
        copy_tokens["token_idx"] = np.arange(len(copy_tokens)) + len(token_data)
        token_data = np.concatenate([
            token_data, copy_unindexed_tokens,
        ], dim=0)


    chain_res_ids = {}
    for token_id, token in enumerate(token_data):
        # # Get the chain residue ids
        # chain_idx, res_id = token["asym_id"], token["res_idx"]

        # if (chain_idx, res_id) not in chain_res_ids:
        #     new_idx = len(chain_res_ids)
        #     chain_res_ids[(chain_idx, res_id)] = new_idx
        # else:
        #     new_idx = chain_res_ids[(chain_idx, res_id)]

        # # Map atoms to token indices
        # ref_space_uid.extend([new_idx] * token["rigid_num"])
        rigid_to_token.extend([token_id] * token["rigid_num"])

    # Compute features
    ref_element = from_numpy(rigid_data["element"]).long()
    ref_charge = from_numpy(rigid_data["charge"])
    sidechain_idx = from_numpy(rigid_data["sidechain_idx"])
    # ref_space_uid = from_numpy(ref_space_uid)
    tensor7 = from_numpy(rigid_data["tensor7"])
    resolved_mask = from_numpy(rigid_data["is_present"])
    is_atom_mask = from_numpy(rigid_data["is_atomized"])
    pad_mask = torch.ones(len(rigid_data), dtype=torch.float)
    rigid_to_token = torch.tensor(rigid_to_token, dtype=torch.long)

    # Compute padding and apply
    if max_atoms is not None:
        assert max_atoms % rigids_per_window_queries == 0
        pad_len = max_atoms - len(rigid_data)
    else:
        pad_len = (
            (len(rigid_data) - 1) // rigids_per_window_queries + 1
        ) * rigids_per_window_queries - len(rigid_data)

    if pad_len > 0:
        pad_mask = pad_dim(pad_mask, 0, pad_len)
        resolved_mask = pad_dim(resolved_mask, 0, pad_len)
        ref_element = pad_dim(ref_element, 0, pad_len)
        ref_charge = pad_dim(ref_charge, 0, pad_len)
        # ref_space_uid = pad_dim(ref_space_uid, 0, pad_len)
        rigid_to_token = pad_dim(rigid_to_token, 0, pad_len)
        sidechain_idx = pad_dim(sidechain_idx, 0, pad_len)

        tensor7_pad = torch.zeros((pad_len, 7), device=tensor7.device, dtype=tensor7.dtype)
        tensor7_pad[:, 0] = 1
        tensor7 = torch.cat([
            tensor7, tensor7_pad
        ], dim=0)

    return {
        "rigid_sidechain_idx": sidechain_idx,
        "rigid_is_atom_mask": is_atom_mask,
        "rigid_ref_element": ref_element,
        "rigid_ref_charge": ref_charge,
        "rigid_resolved_mask": resolved_mask,
        "rigid_pad_mask": pad_mask,
        "rigid_to_token": rigid_to_token,
        "rigid_tensor7": tensor7,
        # "ref_space_uid": ref_space_uid,
    }