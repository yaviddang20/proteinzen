"""Small tree helpers used by steering integrators."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from scipy.spatial.transform import Rotation as R

from proteinzen.openfold.utils import rigid_utils as ru


def get_batch_size(batch: Mapping[str, Any]) -> int:
    """Infer the leading batch dimension from a ProteinZen sampling batch."""
    if "token" in batch and "seq_noising_mask" in batch["token"]:
        return int(batch["token"]["seq_noising_mask"].shape[0])
    if "rigids" in batch and "rigids_1" in batch["rigids"]:
        return int(batch["rigids"]["rigids_1"].shape[0])
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            return int(value.shape[0])
    raise ValueError("Could not infer batch size from batch")


def clone_tree(value: Any) -> Any:
    """Clone tensors/Rigids and deepcopy everything else."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, ru.Rigid):
        return ru.Rigid.from_tensor_7(value.to_tensor_7().clone())
    if isinstance(value, ru.Rotation):
        return ru.Rotation(rot_mats=value.get_rot_mats().clone())
    if isinstance(value, Mapping):
        return {key: clone_tree(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_tree(val) for val in value)
    if isinstance(value, list):
        return [clone_tree(val) for val in value]
    return copy.deepcopy(value)


def index_tree(value: Any, indices: torch.Tensor | list[int], batch_size: int | None = None) -> Any:
    """Index tensors, Rigids, nested mappings, and batch-sized lists on dim 0."""
    if isinstance(indices, list):
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        indices_list = indices
    else:
        indices_tensor = indices.long()
        indices_list = indices_tensor.detach().cpu().tolist()

    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value.clone()
        if batch_size is not None and value.shape[0] != batch_size:
            return value.clone()
        return value.index_select(0, indices_tensor.to(value.device))
    if isinstance(value, ru.Rigid):
        tensor7 = value.to_tensor_7()
        return ru.Rigid.from_tensor_7(index_tree(tensor7, indices_tensor, batch_size))
    if isinstance(value, ru.Rotation):
        rot_mats = value.get_rot_mats()
        return ru.Rotation(rot_mats=index_tree(rot_mats, indices_tensor, batch_size))
    if isinstance(value, Mapping):
        return {key: index_tree(val, indices_tensor, batch_size) for key, val in value.items()}
    if isinstance(value, tuple):
        return tuple(index_tree(val, indices_tensor, batch_size) for val in value)
    if isinstance(value, list):
        if batch_size is not None and len(value) == batch_size:
            return [clone_tree(value[i]) for i in indices_list]
        return [clone_tree(val) for val in value]
    return copy.deepcopy(value)


def select_batch(batch: Mapping[str, Any], indices: torch.Tensor | list[int]) -> dict[str, Any]:
    """Return a batch with all batch-sized fields selected by ``indices``."""
    return index_tree(batch, indices, get_batch_size(batch))


def expand_batch_interleave(batch: Mapping[str, Any], repeats: int) -> dict[str, Any]:
    """Repeat each sample adjacently: ``s0,s0,...,s1,s1,...``."""
    if repeats <= 1:
        return clone_tree(batch)
    batch_size = get_batch_size(batch)
    device = batch["token"]["seq_noising_mask"].device
    indices = torch.arange(batch_size, device=device).repeat_interleave(repeats)
    return select_batch(batch, indices)

def replace_mapping_inplace(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Mutate ``target`` so callers holding the original dict see ``source``."""
    target.clear()
    target.update(source)


def select_traj_point(
    point: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Select a trajectory tuple on its leading batch dimension."""
    trans, rotmats, seq = point
    idx_trans = indices.to(trans.device)
    idx_rot = indices.to(rotmats.device)
    seq_out = None if seq is None else seq.index_select(0, indices.to(seq.device))
    return (
        trans.index_select(0, idx_trans),
        rotmats.index_select(0, idx_rot),
        seq_out,
    )


def tree_batch_size(value: Any) -> int | None:
    """Find the first leading tensor size in a nested tree."""
    if isinstance(value, torch.Tensor) and value.dim() > 0:
        return int(value.shape[0])
    if isinstance(value, ru.Rigid):
        return int(value.to_tensor_7().shape[0])
    if isinstance(value, Mapping):
        for child in value.values():
            size = tree_batch_size(child)
            if size is not None:
                return size
    if isinstance(value, (tuple, list)):
        for child in value:
            size = tree_batch_size(child)
            if size is not None:
                return size
    return None


def initialize_beam(batch: Mapping[str, Any], repeats: int) -> dict[str, Any]:
    """Repeat each sample adjacently: ``s0,s0,...,s1,s1,...``."""
    expand_batch = expand_batch_interleave(batch, repeats)
    device = batch["token"]["seq_noising_mask"].device
    # this is a little confusing, but is because rigids_1 contains the initial noise sample
    # because the source structure is a noise template
    rigids_0 = expand_batch['rigids']['rigids_1']
    rigids_noising_mask = expand_batch['rigids']['rigids_noising_mask']
    num_noised_rigids = rigids_noising_mask.sum()
    expand_rotquats_0 = torch.as_tensor(
        R.random(num_noised_rigids).as_quat(scalar_first=True),
        device=device
    )
    expand_trans_0 = torch.randn((num_noised_rigids, 3), device=device) * 16
    expand_tensor7_0 = torch.cat([expand_rotquats_0, expand_trans_0], dim=-1)
    rigids_0[rigids_noising_mask] = expand_tensor7_0.float()

    return expand_batch
