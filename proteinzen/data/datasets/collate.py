import copy
from typing import Sequence, Dict, Union
import os

import tqdm
import tree
import torch
import numpy as np
import torch.nn.functional as F

from proteinzen.data.openfold import data_transforms, residue_constants
from proteinzen.utils.openfold import rigid_utils as ru

from .featurize.rigid_assembler import RigidAssembler

def featurize_input(
    task,
    data,
    cg_version=8,
    dummy_rigid_to_sidechain_rigid=True
):
    data = copy.deepcopy(data)
    res_data = data['residue']

    task_data = task.sample_t_and_mask(data)
    t = task_data['t']
    rigids_noising_mask = task_data['rigids_noising_mask']
    seq_noising_mask = task_data['seq_noising_mask']
    res_is_unindexed_mask = task_data['res_is_unindexed_mask']

    data['t'] = t
    res_data['res_is_unindexed_mask'] = res_is_unindexed_mask


    # compute base features
    chain_feats = {
        'aatype': torch.as_tensor(res_data['seq']).long(),
        'all_atom_positions': torch.as_tensor(res_data['atom37']).double(),
        'all_atom_mask': torch.as_tensor(res_data['atom37_mask']).double(),
    }
    chain_feats = data_transforms.atom37_to_cg_frames(chain_feats, cg_version=cg_version)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)

    rigids_1 = ru.Rigid.from_tensor_4x4(chain_feats['cg_groups_gt_frames'])[:, (0, 2, 3)]

    # compute bb frame features
    rigids_mask = chain_feats["cg_groups_gt_exists"][:, (0, 2, 3)]
    rigids_mask *= res_data['res_mask'][..., None]
    rigids_1_tensor_7 = rigids_1.to_tensor_7()

    if dummy_rigid_to_sidechain_rigid:
        seq = res_data['seq']
        mask_AG = (seq == residue_constants.restype_order['G']) | (seq == residue_constants.restype_order['A'])
        dummy_rigid = rigids_1_tensor_7[..., 0, :] * mask_AG[..., None] + rigids_1_tensor_7[..., 1, :] * (~mask_AG[..., None])
        rigids_1_tensor_7 = (
            rigids_1_tensor_7 * rigids_mask[..., None] +
            dummy_rigid[..., None, :] * (1 - rigids_mask[..., None].float())
        )
    else:
        rigids_1_tensor_7 = (
            rigids_1_tensor_7 * rigids_mask[..., None] +
            rigids_1_tensor_7[..., 0:1, :] * (1 - rigids_mask[..., None].float())
        )

    assembler = RigidAssembler(
        cg_rigids=rigids_1_tensor_7,
        res_mask=res_data['res_mask'],
        seq=res_data['seq'],
        cg_version=cg_version
    )
    rigids_data = assembler.assemble(
        rigids_noising_mask=rigids_noising_mask,
        res_atomized_mask=torch.zeros_like(res_data['seq'], dtype=torch.bool),
        res_is_unindexed_mask=res_is_unindexed_mask
    )
    rigids_key_renaming = {
        "rigids": "rigids_1",
        "rigids_noising_mask": "rigids_noising_mask",
        "seq_idx": "rigids_seq_idx",
        "rigid_idx": "rigids_idx",
        "is_atomized_mask": "rigids_is_atomized_mask",
        "is_unindexed_mask": "rigids_is_unindexed_mask",
        "is_center_rigid_mask": "rigids_center_mask",
        "is_ligand_mask": "rigids_is_ligand_mask",
    }
    for rigids_key, res_key in rigids_key_renaming.items():
        res_data[res_key] = rigids_key

    # compute sidechain features
    ## generate data dict
    copy_keys = [
        "atom14_atom_exists",
        "atom14_gt_exists",
        "atom14_gt_positions",
        "atom14_alt_gt_exists",
        "atom14_alt_gt_positions",
    ]
    diff_feats_t = {k: chain_feats[k] for k in copy_keys}
    diff_feats_t['bb'] = diff_feats_t['atom14_gt_positions'][..., :4, :]
    diff_feats_t['atom37'] = chain_feats['all_atom_positions']
    diff_feats_t['atom37_mask'] = chain_feats['all_atom_mask']
    # redundant for convenience
    diff_feats_t['atom14'] = chain_feats['atom14_gt_positions']
    diff_feats_t['atom14_mask'] = chain_feats['atom14_gt_exists']
    diff_feats_t = tree.map_structure(
        lambda x: torch.as_tensor(x),
        diff_feats_t)

    res_data.update(diff_feats_t)

    return data