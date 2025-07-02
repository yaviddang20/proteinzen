"""Utilities for calculating all atom representations."""
import torch
from proteinzen.utils.openfold import feats, rigid_utils as ru
from proteinzen.utils.openfold.tensor_utils import batched_gather
from proteinzen.data.openfold import residue_constants
from proteinzen.data.constants import coarse_grain

from .framediff.all_atom import adjust_oxygen_pos


Rigid = ru.Rigid
Rotation = ru.Rotation

# Residue Constants from OpenFold/AlphaFold2.
ATOM_MASK = torch.tensor(residue_constants.restype_atom14_mask)

def _gen_version_dict(module):
    return {
        "IDEALIZED_POS37": torch.tensor(module.restype_atom37_cg_group_positions),
        "IDEALIZED_POS37_MASK": torch.any(torch.tensor(module.restype_atom37_cg_group_positions), axis=-1),
        "IDEALIZED_POS": torch.tensor(module.restype_atom14_cg_group_positions),
        "DEFAULT_FRAMES": torch.tensor(module.restype_cg_group_default_frame),
        "GROUP_IDX": torch.tensor(module.restype_atom14_to_cg_group),
    }

cg_version_constants = {
    1: _gen_version_dict(coarse_grain),
}


def frames_to_atom14_pos(
        r: Rigid,
        aatype: torch.Tensor,
        cg_version: int=1
    ):
    """Convert frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 8, 3]
        aatype: Residue types. [..., N]

    Returns:

    """
    GROUP_IDX = cg_version_constants[cg_version]["GROUP_IDX"]
    DEFAULT_FRAMES = cg_version_constants[cg_version]["DEFAULT_FRAMES"]
    IDEALIZED_POS = cg_version_constants[cg_version]["IDEALIZED_POS"]

    # [*, N, 14]
    group_mask = GROUP_IDX.to(aatype.device)[aatype, ...]

    # [*, N, 14, 4]
    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=DEFAULT_FRAMES.shape[-3],
    ).to(r.device)

    # [*, N, 14, 4]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 14, 1]
    frame_atom_mask = ATOM_MASK.to(aatype.device)[aatype, ...].unsqueeze(-1).to(r.device)

    # [*, N, 14, 3]
    frame_null_pos = IDEALIZED_POS.to(aatype.device)[aatype, ...].to(r.device)
    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions, frame_atom_mask


def compute_atom14_from_cg_frames(rigids, res_mask, seq, cg_version=1, return_atom_mask=False):
    rigids = ru.Rigid.cat([
        rigids[..., 0:1],
        rigids[..., 0:1],
        rigids[..., 1:],
    ], dim=-1)
    atom14_pos, atom14_mask = frames_to_atom14_pos(
        rigids,
        seq,
        cg_version
    )
    atom37_bb_pos = torch.zeros(rigids[..., 0].shape + (37, 3), device=atom14_pos.device)
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
    atom37_bb_pos[..., :3, :] = atom14_pos[..., :3, :]
    atom37_bb_pos[..., 3, :] = atom14_pos[..., 4, :]
    atom37_bb_pos[..., 4, :] = atom14_pos[..., 3, :]
    atom37 = adjust_oxygen_pos(atom37_bb_pos.view(-1, 37, 3), res_mask.view(-1)).view(atom37_bb_pos.shape)
    atom14_pos[..., 3, :] = atom37[..., 4, :]

    if return_atom_mask:
        return atom14_pos, atom14_mask
    else:
        return atom14_pos


def compute_atom14_frames_from_cg_frames(rigids, res_mask, seq, cg_version=1):
    rigids_4 = ru.Rigid.cat([
        rigids[..., 0:1],
        rigids[..., 0:1],
        rigids[..., 1:],
    ], dim=-1)
    atom14_pos, _ = frames_to_atom14_pos(
        rigids_4,
        seq,
        cg_version
    )
    atom37_bb_pos = torch.zeros(rigids[..., 0].shape + (37, 3), device=atom14_pos.device)
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
    atom37_bb_pos[..., :3, :] = atom14_pos[..., :3, :]
    atom37_bb_pos[..., 3, :] = atom14_pos[..., 4, :]
    atom37_bb_pos[..., 4, :] = atom14_pos[..., 3, :]
    atom37 = adjust_oxygen_pos(atom37_bb_pos.view(-1, 37, 3), res_mask.view(-1)).view(atom37_bb_pos.shape)
    atom14_pos[..., 3, :] = atom37[..., 4, :]

    GROUP_IDX = cg_version_constants[cg_version]["GROUP_IDX"].to(seq.device)
    atom14_mask = ATOM_MASK.to(seq.device)[seq].bool()
    rigid_quats = rigids_4.get_rots().get_quats()
    atom_quats = batched_gather(
        rigid_quats,
        GROUP_IDX[seq],
        1,
        1
    )
    atom_quats = atom_quats * atom14_mask[..., None]
    atom_quats[atom14_mask, 0] = 1

    atom_rigids = ru.Rigid(
        rots=ru.Rotation(quats=atom_quats),
        trans=atom14_pos,
    )

    return atom_rigids, atom14_mask