"""Reward scoring utilities for endpoint-based steering."""

from __future__ import annotations

import copy
import inspect
import os
import tempfile
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from proteinzen.stoch_interp.steering.rewards.base_reward import TOTAL_REWARD_KEY
from proteinzen.stoch_interp.steering.utils import get_batch_size


RewardFn = Callable[[Mapping[str, Any], Mapping[str, Any]], torch.Tensor]


class EndpointRewardScorer:
    """Score fully rolled-out ProteinZen endpoints.

    The default path mirrors ProteinZen's prediction writer: convert the final
    denoiser output back through the original ``input_data`` structure, write a
    temporary PDB, then call the copied Proteina-Complexa-style reward model.
    A custom ``reward_fn`` can be supplied for in-memory rewards or tests.
    """

    def __init__(
        self,
        *,
        reward_model: Any | None = None,
        reward_fn: RewardFn | None = None,
        reward_kwargs: Mapping[str, Any] | None = None,
        work_dir: str | None = None,
        keep_pdbs: bool = False,
        pass_sequence: bool | None = None,
        pass_structure: bool | None = None,
        drop_copy_inputs: bool = True,
    ) -> None:
        if reward_model is None and reward_fn is None:
            raise ValueError("EndpointRewardScorer requires reward_model or reward_fn")
        self.reward_model = reward_model
        self.reward_fn = reward_fn
        self.reward_kwargs = dict(reward_kwargs or {})
        self.work_dir = work_dir
        self.keep_pdbs = keep_pdbs
        self.pass_sequence = pass_sequence
        self.pass_structure = pass_structure
        self.drop_copy_inputs = drop_copy_inputs
        if self.work_dir is not None:
            os.makedirs(self.work_dir, exist_ok=True)

    def score_batch(
        self,
        batch: Mapping[str, Any],
        denoiser_out: Mapping[str, Any],
        *,
        tags: list[str] | None = None,
    ) -> torch.Tensor:
        """Return one scalar reward per sample in ``batch``."""
        if self.reward_fn is not None:
            rewards = self.reward_fn(batch, denoiser_out)
            return torch.as_tensor(rewards, dtype=torch.float32)

        batch_size = get_batch_size(batch)
        if tags is None:
            tags = [f"sample_{i}" for i in range(batch_size)]
        if len(tags) != batch_size:
            raise ValueError(f"Expected {batch_size} tags, got {len(tags)}")

        rewards = []
        tmp_ctx = None
        if self.work_dir is None:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="proteinzen_steer_")
            out_dir = tmp_ctx.name
        else:
            out_dir = self.work_dir

        try:
            for i in range(batch_size):
                endpoint = self._endpoint_for_sample(batch, denoiser_out, i, out_dir, tags[i])
                score_kwargs = self._score_kwargs(endpoint)
                result = self.reward_model.score(**score_kwargs)
                total_reward = result[TOTAL_REWARD_KEY]
                rewards.append(torch.as_tensor(total_reward, dtype=torch.float32).reshape(()))
                if not self.keep_pdbs:
                    self._cleanup_endpoint(endpoint)
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

        return torch.stack(rewards, dim=0)

    def _score_kwargs(self, endpoint: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(self.reward_kwargs)
        kwargs["pdb_path"] = endpoint["pdb_path"]
        kwargs.setdefault("requires_grad", False)

        pass_sequence = self.pass_sequence
        pass_structure = self.pass_structure
        protocol = getattr(self.reward_model, "protocol", None)
        if pass_sequence is None:
            pass_sequence = protocol == "hallucination"
        if pass_structure is None:
            pass_structure = protocol == "hallucination"

        if pass_sequence and endpoint.get("sequence") is not None:
            kwargs.setdefault("sequence", endpoint["sequence"])
        if pass_structure and endpoint.get("structure") is not None:
            kwargs.setdefault("structure", endpoint["structure"])
        return kwargs

    def _endpoint_for_sample(
        self,
        batch: Mapping[str, Any],
        denoiser_out: Mapping[str, Any],
        sample_idx: int,
        out_dir: str,
        tag: str,
    ) -> dict[str, Any]:
        if "input_data" not in batch:
            raise ValueError(
                "Default endpoint scoring requires batch['input_data']. "
                "Pass a custom reward_fn to score tensors directly."
            )

        from proteinzen.stoch_interp.steering.rewards.pdb_utils import from_pdb_string

        input_data = copy.deepcopy(batch["input_data"][sample_idx])
        num_rigids = input_data["rigids"]["tensor7"].shape[0]
        num_tokens = input_data["tokens"]["token_idx"].shape[0]

        pred_tensor7 = denoiser_out["denoised_rigids"].to_tensor_7()[sample_idx, :num_rigids]
        pred_seq = denoiser_out["pred_seq"][sample_idx, :num_tokens]
        input_data["rigids"]["tensor7"] = pred_tensor7.detach().cpu().numpy()
        input_data["tokens"]["res_type"] = pred_seq.detach().cpu().numpy()

        sequence_input_data = copy.deepcopy(input_data)
        pdb_input_data = copy.deepcopy(input_data)
        if self.drop_copy_inputs:
            copy_mask = self._copy_mask_for_sample(batch, input_data, sample_idx, num_tokens)
            sequence_input_data = self._drop_copy_numpy_inputs(sequence_input_data, copy_mask)
            pdb_input_data = self._drop_copy_numpy_inputs(pdb_input_data, copy_mask)

        pdb_str = self._pdb_string_from_input_data(pdb_input_data)
        if not self._pdb_has_atoms(pdb_str):
            raise ValueError(
                "Copy-filtered reward endpoint produced a PDB with no atoms. "
                "Check token_is_copy_mask/token['is_copy'] or set drop_copy_inputs=False."
            )
        self._validate_reward_pdb_for_model(pdb_str)
        pdb_path = os.path.join(out_dir, f"{self._safe_tag(tag)}.pdb")
        with open(pdb_path, "w") as handle:
            handle.write(pdb_str)

        endpoint: dict[str, Any] = {"pdb_path": pdb_path}
        sequence_pdb_str = self._pdb_string_from_input_data(sequence_input_data)
        if not self._pdb_has_atoms(sequence_pdb_str):
            raise ValueError(
                "Copy-filtered reward sequence/structure endpoint has no atoms. "
                "Check token_is_copy_mask/token['is_copy'] or set drop_copy_inputs=False."
            )
        protein = from_pdb_string(sequence_pdb_str)
        endpoint["structure"] = torch.as_tensor(protein.atom_positions, dtype=torch.float32)
        aatype = torch.as_tensor(protein.aatype, dtype=torch.long).clamp(max=19)
        endpoint["sequence"] = F.one_hot(aatype, num_classes=20).float() * 1e9
        return endpoint

    @staticmethod
    def _pdb_string_from_input_data(input_data: Mapping[str, Any]) -> str:
        from proteinzen.boltz.data.types import Structure
        from proteinzen.data.featurize.sampling import construct_atoms, update_structure
        from proteinzen.data.featurize.tokenize import Tokenized
        from proteinzen.data.write.pdb import to_pdb

        input_data = copy.deepcopy(input_data)
        if "structure" in input_data and "mask" in input_data["structure"]:
            input_data["structure"]["mask"] = input_data["structure"]["mask"].astype(bool)
            input_data["structure"]["mask"][...] = True
        struct = Structure(**input_data["structure"])
        tokenized = Tokenized(
            tokens=input_data["tokens"],
            rigids=input_data["rigids"],
            bonds=input_data["bonds"],
            structure=struct,
        )
        struct = construct_atoms(tokenized, struct)
        struct = update_structure(struct, tokenized.rigids["tensor7"])
        return to_pdb(struct)

    @staticmethod
    def _cleanup_endpoint(endpoint: Mapping[str, Any]) -> None:
        pdb_path = endpoint.get("pdb_path")
        if isinstance(pdb_path, str) and os.path.exists(pdb_path):
            os.remove(pdb_path)

    @staticmethod
    def _copy_mask_for_sample(
        batch: Mapping[str, Any],
        input_data: Mapping[str, Any],
        sample_idx: int,
        num_tokens: int,
    ) -> np.ndarray:
        tokens = input_data["tokens"]
        copy_mask = np.zeros(num_tokens, dtype=bool)
        if tokens.dtype.names is not None and "is_copy" in tokens.dtype.names:
            copy_mask |= tokens["is_copy"][:num_tokens].astype(bool)

        token_batch = batch.get("token", {})
        torch_copy_mask = token_batch.get("token_is_copy_mask")
        if torch_copy_mask is not None:
            copy_mask |= torch_copy_mask[sample_idx, :num_tokens].detach().cpu().numpy().astype(bool)
        return copy_mask

    @classmethod
    def _drop_copy_numpy_inputs(cls, input_data: dict[str, Any], copy_mask: np.ndarray) -> dict[str, Any]:
        if not copy_mask.any():
            return input_data

        tokens = input_data["tokens"]
        rigids = input_data["rigids"]
        bonds = input_data["bonds"]
        keep_tokens = ~copy_mask
        if not keep_tokens.any():
            raise ValueError("Cannot score endpoint after removing copies: no non-copy tokens remain")

        filtered_structure, old_res_pos_to_new, token_residue_pos = cls._drop_copy_structure(
            input_data["structure"],
            tokens,
            copy_mask,
        )

        old_token_ids = tokens["token_idx"].astype(int)
        token_id_to_new = {
            int(old_token_id): new_idx
            for new_idx, old_token_id in enumerate(old_token_ids[keep_tokens])
        }

        keep_token_positions = np.nonzero(keep_tokens)[0]
        tokens_new = tokens[keep_tokens].copy()
        for new_idx, (old_token_pos, token) in enumerate(zip(keep_token_positions, tokens_new)):
            old_res_pos = int(token_residue_pos[old_token_pos])
            token["token_idx"] = new_idx
            if old_res_pos in old_res_pos_to_new:
                token["res_idx"] = old_res_pos_to_new[old_res_pos]

        rigid_keep = np.array(
            [int(token_id) in token_id_to_new for token_id in rigids["token_idx"]],
            dtype=bool,
        )
        rigids_new = rigids[rigid_keep].copy()
        for new_rigid_idx, rigid in enumerate(rigids_new):
            rigid["rigid_idx"] = new_rigid_idx
            rigid["token_idx"] = token_id_to_new[int(rigid["token_idx"])]

        for token_idx, token in enumerate(tokens_new):
            token_rigid_idx = np.nonzero(rigids_new["token_idx"] == token_idx)[0]
            if len(token_rigid_idx) == 0:
                raise ValueError(f"Token {token_idx} has no rigids after removing copy inputs")
            token["rigid_idx"] = int(token_rigid_idx[0])
            token["rigid_num"] = int(len(token_rigid_idx))

        input_data["tokens"] = tokens_new
        input_data["rigids"] = rigids_new
        input_data["bonds"] = cls._remap_token_bonds(bonds, token_id_to_new)
        input_data["structure"] = filtered_structure
        return input_data

    @classmethod
    def _drop_copy_structure(
        cls,
        structure: dict[str, Any],
        tokens: np.ndarray,
        copy_mask: np.ndarray,
    ) -> tuple[dict[str, Any], dict[int, int], np.ndarray]:
        residues = structure["residues"]
        chains = structure["chains"]
        atoms = structure["atoms"]
        chain_mask = structure.get("mask", np.ones(chains.shape[0], dtype=bool)).astype(bool)

        token_residue_pos = cls._map_tokens_to_residue_positions(structure, tokens)
        residue_copy = np.zeros(residues.shape[0], dtype=bool)
        mapped_copy_tokens = copy_mask & (token_residue_pos >= 0)
        residue_copy[token_residue_pos[mapped_copy_tokens]] = True
        if (copy_mask & (token_residue_pos < 0)).any():
            copy_keys = {
                (int(token["asym_id"]), int(token["res_idx"]))
                for token in tokens[copy_mask & (token_residue_pos < 0)]
            }
            for chain in chains[chain_mask]:
                for res_pos in range(int(chain["res_idx"]), int(chain["res_idx"] + chain["res_num"])):
                    key = (int(chain["asym_id"]), int(residues[res_pos]["res_idx"]))
                    residue_copy[res_pos] = residue_copy[res_pos] or key in copy_keys

        new_residues = []
        new_chains = []
        atom_indices: list[int] = []
        old_res_pos_to_new: dict[int, int] = {}

        for chain in chains[chain_mask]:
            old_start = int(chain["res_idx"])
            old_end = old_start + int(chain["res_num"])
            kept_positions = [pos for pos in range(old_start, old_end) if not residue_copy[pos]]
            if not kept_positions:
                continue

            new_chain = chain.copy()
            new_chain["res_idx"] = len(new_residues)
            new_chain["res_num"] = len(kept_positions)
            new_chain["atom_idx"] = len(atom_indices)

            for old_pos in kept_positions:
                old_residue = residues[old_pos]
                new_residue = old_residue.copy()
                new_res_idx = len(new_residues)
                old_atom_start = int(old_residue["atom_idx"])
                old_atom_num = int(old_residue["atom_num"])

                new_residue["res_idx"] = new_res_idx
                new_residue["atom_idx"] = len(atom_indices)
                if new_residue.dtype.names is not None and "is_copy" in new_residue.dtype.names:
                    new_residue["is_copy"] = False
                new_residues.append(new_residue)
                atom_indices.extend(range(old_atom_start, old_atom_start + old_atom_num))
                old_res_pos_to_new[old_pos] = new_res_idx

            new_chain["atom_num"] = len(atom_indices) - int(new_chain["atom_idx"])
            new_chains.append(new_chain)

        if not new_residues:
            raise ValueError("Cannot score endpoint after removing copies: no non-copy residues remain")

        filtered = copy.deepcopy(structure)
        filtered["residues"] = np.array(new_residues, dtype=residues.dtype)
        filtered["chains"] = np.array(new_chains, dtype=chains.dtype)
        filtered["atoms"] = atoms[np.array(atom_indices, dtype=int)].copy()
        filtered["mask"] = np.ones(len(new_chains), dtype=bool)
        filtered["bonds"] = cls._drop_or_remap_atom_bonds(structure.get("bonds"), atom_indices)
        filtered["connections"] = np.array([], dtype=structure["connections"].dtype)
        filtered["interfaces"] = np.array([], dtype=structure["interfaces"].dtype)

        return filtered, old_res_pos_to_new, token_residue_pos

    @staticmethod
    def _map_tokens_to_residue_positions(
        structure: Mapping[str, Any],
        tokens: np.ndarray,
    ) -> np.ndarray:
        residues = structure["residues"]
        chains = structure["chains"]
        chain_mask = structure.get("mask", np.ones(chains.shape[0], dtype=bool)).astype(bool)
        token_residue_pos = np.full(tokens.shape[0], -1, dtype=int)
        token_pos = 0

        for chain in chains[chain_mask]:
            asym_id = int(chain["asym_id"])
            old_start = int(chain["res_idx"])
            old_end = old_start + int(chain["res_num"])
            for res_pos in range(old_start, old_end):
                res_idx = int(residues[res_pos]["res_idx"])
                start_token_pos = token_pos
                while (
                    token_pos < tokens.shape[0]
                    and token_residue_pos[token_pos] < 0
                    and int(tokens[token_pos]["asym_id"]) == asym_id
                    and int(tokens[token_pos]["res_idx"]) == res_idx
                ):
                    token_residue_pos[token_pos] = res_pos
                    token_pos += 1

                if start_token_pos == token_pos:
                    matches = np.nonzero(
                        (token_residue_pos < 0)
                        & (tokens["asym_id"].astype(int) == asym_id)
                        & (tokens["res_idx"].astype(int) == res_idx)
                    )[0]
                    if matches.size == 0:
                        continue
                    match_pos = int(matches[0])
                    while (
                        match_pos < tokens.shape[0]
                        and token_residue_pos[match_pos] < 0
                        and int(tokens[match_pos]["asym_id"]) == asym_id
                        and int(tokens[match_pos]["res_idx"]) == res_idx
                    ):
                        token_residue_pos[match_pos] = res_pos
                        match_pos += 1

        return token_residue_pos

    @staticmethod
    def _remap_token_bonds(bonds: np.ndarray, token_id_to_new: dict[int, int]) -> np.ndarray:
        if bonds.size == 0 or bonds.dtype.names is None:
            return bonds[:0].copy()
        if "token_1" not in bonds.dtype.names or "token_2" not in bonds.dtype.names:
            return bonds[:0].copy()

        remapped = []
        for bond in bonds:
            token_1 = int(bond["token_1"])
            token_2 = int(bond["token_2"])
            if token_1 not in token_id_to_new or token_2 not in token_id_to_new:
                continue
            new_bond = bond.copy()
            new_bond["token_1"] = token_id_to_new[token_1]
            new_bond["token_2"] = token_id_to_new[token_2]
            remapped.append(new_bond)
        return np.array(remapped, dtype=bonds.dtype)

    @staticmethod
    def _drop_or_remap_atom_bonds(bonds: np.ndarray | None, atom_indices: list[int]) -> np.ndarray:
        if bonds is None or bonds.size == 0 or bonds.dtype.names is None:
            return np.array([], dtype=bonds.dtype if bonds is not None else [])
        if "atom_1" not in bonds.dtype.names or "atom_2" not in bonds.dtype.names:
            return bonds[:0].copy()

        atom_id_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(atom_indices)}
        remapped = []
        for bond in bonds:
            atom_1 = int(bond["atom_1"])
            atom_2 = int(bond["atom_2"])
            if atom_1 not in atom_id_to_new or atom_2 not in atom_id_to_new:
                continue
            new_bond = bond.copy()
            new_bond["atom_1"] = atom_id_to_new[atom_1]
            new_bond["atom_2"] = atom_id_to_new[atom_2]
            remapped.append(new_bond)
        return np.array(remapped, dtype=bonds.dtype)

    @staticmethod
    def _pdb_has_atoms(pdb_str: str) -> bool:
        return any(line.startswith(("ATOM", "HETATM")) for line in pdb_str.splitlines())

    def _validate_reward_pdb_for_model(self, pdb_str: str) -> None:
        if getattr(self.reward_model, "protocol", None) != "binder":
            return
        target_chain = self.reward_kwargs.get("target_chain")
        if target_chain is None:
            return
        chain_ids = self._pdb_chain_ids(pdb_str)
        if str(target_chain) not in chain_ids:
            raise ValueError(
                f"AF2 binder target_chain={target_chain!r} has no ATOM/HETATM records "
                "after copy removal. Tokens marked as copies are removed even if they "
                "belong to the configured binder or target chain; ensure the target "
                "chain has at least one non-copy residue for AF2 binder scoring."
            )

    @staticmethod
    def _pdb_chain_ids(pdb_str: str) -> set[str]:
        chain_ids = set()
        for line in pdb_str.splitlines():
            if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
                chain_ids.add(line[21].strip())
        return chain_ids

    @staticmethod
    def _safe_tag(tag: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in tag)


def reward_fn_accepts_endpoint(reward_fn: Callable[..., Any]) -> bool:
    """Return whether ``reward_fn`` advertises an ``endpoint`` keyword."""
    try:
        return "endpoint" in inspect.signature(reward_fn).parameters
    except (TypeError, ValueError):
        return False
