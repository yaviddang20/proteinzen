import copy
from typing import Sequence, Dict, Union
import os

import tqdm
import tree
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import default_collate

from proteinzen.data.openfold import data_transforms, residue_constants
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.data.constants import coarse_grain as cg

from .featurize.rigid_assembler import RigidAssembler



def featurize_input(
    task,
    data,
    cg_version=1,
    dummy_rigid_to_sidechain_rigid=True,
    promote_full_motif_to_token=False,
):
    data = copy.deepcopy(data)
    res_data = data['residue']

    features = {
        'token': {},
        'rigids': {},
        'ligands': {},
        'atom': {},
        'task': task,
        'name': [data.name]
    }

    # compute base features from raw data
    chain_feats = {
        'aatype': torch.as_tensor(res_data['seq']).long(),
        'all_atom_positions': torch.as_tensor(res_data['atom37']).double(),
        'all_atom_mask': torch.as_tensor(res_data['atom37_mask']).double(),
    }
    chain_feats = data_transforms.atom37_to_cg_frames(chain_feats, cg_version=cg_version)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    rigids_1 = ru.Rigid.from_tensor_4x4(chain_feats['cg_groups_gt_frames'])[:, (0, 2, 3)]

    # compute frame features
    cg_group_mask = [
        cg.cg_group_mask[residue_constants.restype_1to3[resname]]
        for resname in residue_constants.restypes
    ]
    cg_group_mask.append([0.0] * 4)
    cg_group_mask = torch.as_tensor(cg_group_mask, dtype=torch.bool)[:, (0, 2, 3)]
    rigids_mask = chain_feats["cg_groups_gt_exists"][:, (0, 2, 3)].bool()
    rigids_mask *= res_data['res_mask'][..., None]
    rigids_1_tensor_7 = rigids_1.to_tensor_7()
    if dummy_rigid_to_sidechain_rigid:
        seq = res_data['seq']
        rigids_mask_from_seq = cg_group_mask[seq]
        mask_AG = (seq == residue_constants.restype_order['G']) | (seq == residue_constants.restype_order['A'])
        mask_not_X = (seq != residue_constants.restype_order_with_x['X'])
        dummy_rigid = rigids_1_tensor_7[..., 0, :] * (mask_AG & mask_not_X)[..., None] + rigids_1_tensor_7[..., 1, :] * (~mask_AG & mask_not_X)[..., None]
        dummy_rigid_location = (~rigids_mask_from_seq) * mask_not_X[..., None]

        unresolved_rigids = rigids_mask_from_seq & ~rigids_mask
        unresolved_dummy_rigid_mask = (
            (unresolved_rigids[..., 0] * (mask_AG & mask_not_X))
            |
            (unresolved_rigids[..., 1] * (~mask_AG & mask_not_X))
        )
        dummy_rigid_location[unresolved_dummy_rigid_mask] = False

        rigids_1_tensor_7[dummy_rigid_location] = 0
        rigids_1_tensor_7 += dummy_rigid[..., None, :] * dummy_rigid_location[..., None]

        rigids_mask[dummy_rigid_location] = True
        # rigids_1_tensor_7 = (
        #     rigids_1_tensor_7 * rigids_mask[..., None] +
        #     dummy_rigid[..., None, :] * (1 - rigids_mask[..., None].float())
        # )
    else:
        rigids_1_tensor_7 = (
            rigids_1_tensor_7 * rigids_mask[..., None] +
            rigids_1_tensor_7[..., 0:1, :] * (1 - rigids_mask[..., None].float())
        )
    data['residue']['rigids_1'] = rigids_1_tensor_7

    # sample task and associated time/masks
    task_data = task.sample_t_and_mask(data)
    t = task_data['t']
    rigids_noising_mask = task_data['rigids_noising_mask']
    seq_noising_mask = task_data['seq_noising_mask']
    res_is_unindexed_mask = task_data['res_is_unindexed_mask']
    res_is_atomized_mask = task_data['res_is_atomized_mask']

    features['t'] = t
    res_data['seq_noising_mask'] = seq_noising_mask

    # assemble rigid features based off of task masks
    assembler = RigidAssembler(
        cg_rigids=rigids_1_tensor_7,
        res_mask=res_data['res_mask'],
        rigids_mask=rigids_mask,
        seq=res_data['seq'],
        cg_version=cg_version,
        promote_full_motif_to_token=promote_full_motif_to_token
    )
    rigids_data = assembler.assemble(
        rigids_noising_mask=rigids_noising_mask,
        res_atomized_mask=res_is_atomized_mask,
        res_is_unindexed_mask=res_is_unindexed_mask
    )
    rigids_key_renaming = {
        "rigids": "rigids_1",
        "rigids_mask": "rigids_mask",
        "rigids_noising_mask": "rigids_noising_mask",
        "token_uid": "rigids_token_uid",
        "seq_idx": "rigids_seq_idx",
        "rigid_idx": "rigids_idx",
        "is_atomized_mask": "rigids_is_atomized_mask",
        "is_unindexed_mask": "rigids_is_unindexed_mask",
        "is_token_rigid_mask": "rigids_is_token_rigid_mask",
        "is_ligand_mask": "rigids_is_ligand_mask",
        "is_protein_output_mask": "rigids_is_protein_output_mask",
    }
    for rigids_key, res_key in rigids_key_renaming.items():
        features['rigids'][res_key] = rigids_data[rigids_key]

    # derive node features derived from rigids features
    rigids_is_token_rigid_mask = rigids_data['is_token_rigid_mask']
    rigids_to_token_names = {
        "rigids_mask": "token_mask",
        "seq_idx": "token_seq_idx",
        "is_atomized_mask": "token_is_atomized_mask",
        "is_unindexed_mask": "token_is_unindexed_mask",
        "is_ligand_mask": "token_is_ligand_mask",
        "is_protein_output_mask": "token_is_protein_output_mask",
        "token_uid": "token_uid"
    }
    for rigids_key, token_key in rigids_to_token_names.items():
        features['token'][token_key] = rigids_data[rigids_key][rigids_is_token_rigid_mask]
    features['token']['token_gather_idx'] = rigids_data["token_gather_idx"]

    # add features from the original node data
    token_feats = [
        "seq",
        "seq_noising_mask",
        "seq_mask",
        "chain_idx"
    ]
    # duplicate the motif residues and update res_data
    token_seq_idx = features['token']['token_seq_idx']
    for key in token_feats:
        tensor = res_data[key]
        if promote_full_motif_to_token:
            raise NotImplementedError()
            tensor_expand = tensor[..., None].expand(-1, rigids_noising_mask.shape[-1])
            if key == "seq_noising_mask":
                features['token'][key] = torch.cat([torch.ones_like(tensor), tensor_expand[rigids_noising_mask]], dim=0)
            else:
                features['token'][key] = torch.cat([tensor, tensor_expand[rigids_noising_mask]], dim=0)
        else:
            features['token'][key] = tensor[token_seq_idx]
    token_is_protein_output_mask = features['token']['token_is_protein_output_mask']

    unindexed_motif_mask = (~rigids_noising_mask & res_is_unindexed_mask[..., None])
    if unindexed_motif_mask.any():
        seq_noising_mask = features['token']['seq_noising_mask']
        seq_noising_mask[token_is_protein_output_mask] = True

    # compute sidechain features
    ## generate data dict
    atomic_copy_keys = [
        "atom14_atom_exists",
        "atom14_gt_exists",
        "atom14_gt_positions",
        "atom14_alt_gt_exists",
        "atom14_alt_gt_positions",
    ]
    atomic_feats = {k: chain_feats[k] for k in atomic_copy_keys}
    # renaming for convenience
    atomic_feats['atom37'] = chain_feats['all_atom_positions']
    atomic_feats['atom37_mask'] = chain_feats['all_atom_mask']
    atomic_feats['atom14'] = chain_feats['atom14_gt_positions']
    atomic_feats['atom14_mask'] = chain_feats['atom14_gt_exists']
    atomic_feats = tree.map_structure(
        torch.as_tensor,
        atomic_feats
    )

    features['atom'].update(atomic_feats)

    return features


def featurize_sample_input_from_task_data(
    task_data,
    data,
    cg_version=1,
    dummy_rigid_to_sidechain_rigid=True
):
    data = copy.deepcopy(data)
    res_data = data['residue']

    features = {
        'token': {},
        'rigids': {},
        'ligands': {},
        'atom': {}
    }

    # compute base features from raw data
    chain_feats = {
        'aatype': torch.as_tensor(res_data['seq']).long(),
        'all_atom_positions': torch.as_tensor(res_data['atom37']).double(),
        'all_atom_mask': torch.as_tensor(res_data['atom37_mask']).double(),
    }
    chain_feats = data_transforms.atom37_to_cg_frames(chain_feats, cg_version=cg_version)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    rigids_1 = ru.Rigid.from_tensor_4x4(chain_feats['cg_groups_gt_frames'])[:, (0, 2, 3)]

    # compute frame features

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
    data['residue']['rigids_1'] = rigids_1_tensor_7

    # sample task and associated time/masks
    t = task_data['t']
    rigids_noising_mask = task_data['rigids_noising_mask']
    seq_noising_mask = task_data['seq_noising_mask']
    res_is_unindexed_mask = task_data['res_is_unindexed_mask']
    res_is_atomized_mask = task_data['res_is_atomized_mask']

    features['t'] = t
    res_data['seq_noising_mask'] = seq_noising_mask

    # assemble rigid features based off of task masks
    assembler = RigidAssembler(
        cg_rigids=rigids_1_tensor_7,
        res_mask=res_data['res_mask'],
        seq=res_data['seq'],
        cg_version=cg_version,
        promote_full_motif_to_token=False
    )
    rigids_data = assembler.assemble(
        rigids_noising_mask=rigids_noising_mask,
        res_atomized_mask=res_is_atomized_mask,
        res_is_unindexed_mask=res_is_unindexed_mask
    )
    rigids_key_renaming = {
        "rigids": "rigids_1",
        "rigids_mask": "rigids_mask",
        "rigids_noising_mask": "rigids_noising_mask",
        "token_uid": "rigids_token_uid",
        "seq_idx": "rigids_seq_idx",
        "rigid_idx": "rigids_idx",
        "is_atomized_mask": "rigids_is_atomized_mask",
        "is_unindexed_mask": "rigids_is_unindexed_mask",
        "is_token_rigid_mask": "rigids_is_token_rigid_mask",
        "is_ligand_mask": "rigids_is_ligand_mask",
        "is_protein_output_mask": "rigids_is_protein_output_mask",
    }
    for rigids_key, res_key in rigids_key_renaming.items():
        features['rigids'][res_key] = rigids_data[rigids_key]

    # derive node features derived from rigids features
    rigids_is_token_rigid_mask = rigids_data['is_token_rigid_mask']
    rigids_to_token_names = {
        "rigids_mask": "token_mask",
        "seq_idx": "token_seq_idx",
        "is_atomized_mask": "token_is_atomized_mask",
        "is_unindexed_mask": "token_is_unindexed_mask",
        "is_ligand_mask": "token_is_ligand_mask",
        "is_protein_output_mask": "token_is_protein_output_mask",
    }
    for rigids_key, token_key in rigids_to_token_names.items():
        features['token'][token_key] = rigids_data[rigids_key][rigids_is_token_rigid_mask]
    features['token']['token_gather_idx'] = rigids_data["token_gather_idx"]

    # add features from the original node data
    token_feats = [
        "seq",
        "seq_noising_mask",
        "seq_mask",
        "chain_idx"
    ]
    # duplicate the motif residues and update res_data
    for key in token_feats:
        tensor = res_data[key]
        tensor_expand = tensor[..., None].expand(-1, rigids_noising_mask.shape[-1])
        features['token'][key] = torch.cat([tensor, tensor_expand[rigids_noising_mask]], dim=0)

    # compute sidechain features
    ## generate data dict
    atomic_copy_keys = [
        "atom14_atom_exists",
        "atom14_gt_exists",
        "atom14_gt_positions",
        "atom14_alt_gt_exists",
        "atom14_alt_gt_positions",
    ]
    atomic_feats = {k: chain_feats[k] for k in atomic_copy_keys}
    # renaming for convenience
    atomic_feats['atom37'] = chain_feats['all_atom_positions']
    atomic_feats['atom37_mask'] = chain_feats['all_atom_mask']
    atomic_feats['atom14'] = chain_feats['atom14_gt_positions']
    atomic_feats['atom14_mask'] = chain_feats['atom14_gt_exists']
    atomic_feats = tree.map_structure(
        torch.as_tensor,
        atomic_feats
    )

    features['atom'].update(atomic_feats)

    return features


def collate(data_list):
    def pad(data, token_max_len, rigids_max_len, atom_max_len):
        padded_data = {
            't': data['t'],
            'token': {},
            'rigids': {},
            'ligands': {},
            'atom': {}
        }
        ## pad token data
        for key, value in data['token'].items():
            pad_len = token_max_len - value.shape[0]
            if value.dim() == 1:
                padded_data['token'][key] = F.pad(value, (0, pad_len))
            elif value.dim() == 2:
                padded_data['token'][key] = F.pad(value, (0, 0, 0, pad_len))
            else:
                raise ValueError(f"we ran into a token input with dim not in (1, 2): {key} with dim {value.dim()}")

        ## pad rigid data
        for key, value in data['rigids'].items():
            pad_len = rigids_max_len - value.shape[0]
            if value.dim() == 1:
                padded_data['rigids'][key] = F.pad(value, (0, pad_len))
            elif value.dim() == 2:
                padded_data['rigids'][key] = F.pad(value, (0, 0, 0, pad_len))
            else:
                raise ValueError(f"we ran into a rigid input with dim not in (1, 2): {key} with dim {value.dim()}")

            # we're padding tensor_7s so we need to ensure the quat isn't 0
            if key in ('rigids_t', 'rigids_1'):
                if pad_len > 0:
                    padded_data['rigids'][key][..., -pad_len:, 0] = 1

        ## pad atom data
        for key, value in data['atom'].items():
            pad_len = atom_max_len - value.shape[0]
            if value.dim() == 1:
                padded_data['atom'][key] = F.pad(value, (0, pad_len))
            elif value.dim() == 2:
                padded_data['atom'][key] = F.pad(value, (0, 0, 0, pad_len))
            elif value.dim() == 3:
                padded_data['atom'][key] = F.pad(value, (0, 0, 0, 0, 0, pad_len))
            else:
                raise ValueError(f"we ran into an atom input with dim not in (1, 2, 3): {key} with dim {value.dim()}")

        return padded_data

    token_lens = []
    rigids_lens = []
    atom_lens = []
    for data in data_list:
        token_lens.append(
            data['token']['token_seq_idx'].numel()
        )
        rigids_lens.append(
            data['rigids']['rigids_seq_idx'].numel()
        )
        atom_lens.append(
            data['atom']['atom14_mask'].shape[0]
        )
    # print(token_lens, rigids_lens, atom_lens)
    padded_data_list = [
        pad(data, max(token_lens), max(rigids_lens), max(atom_lens))
        for data in data_list
    ]
    batched_data_list = default_collate(padded_data_list)
    batched_data_list['task'] = [data['task'] for data in data_list]
    if 'name' in data_list[0]:
        batched_data_list['name'] = [data['name'] for data in data_list]
    return batched_data_list
