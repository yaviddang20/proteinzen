""" Code for generating unconditional generation inputs

"""
import copy
import string

from Bio.PDB.PDBParser import PDBParser
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import tree
from torch_geometric.data import HeteroData
import numpy as np

from proteinzen.data.openfold import residue_constants, data_transforms
from proteinzen.utils.openfold import rigid_utils as ru
from .task import SamplingTask


def dict_cat(data):
    return tree.map_structure(torch.cat, *data)


def residue_to_atom37(residue):
    residue_37 = torch.full((91, 3), torch.nan)
    aa_3lt = residue.get_resname()
    atoms = list(residue.get_atoms())
    for atom in atoms:
        atom_name = atom.get_name()
        atom_idx = residue_constants.atom_types.index(atom_name)
        residue_37[atom_idx] = atom.get_coord()
    return residue_37, residue_constants.resname_to_idx[aa_3lt]


def _centered_gaussian(num_rigids):
    noise = torch.randn(num_rigids, 3)
    return noise - noise.mean(dim=0, keepdim=True)

def _uniform_so3(num_rigids):
    return torch.as_tensor(
        Rotation.random(num_rigids).as_quat(),
        dtype=torch.float32,
    )


class UnconditionalSampling(SamplingTask):
    rigids_per_res: int = 3
    task_name: str = "unconditional"

    def __init__(
        self,
        sample_length,
        num_samples,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.sample_length = sample_length
        self.num_samples = num_samples

    def _sample_init_data(self):
        seq = torch.full((self.sample_length,), residue_constants.restype_order_with_x['X']).long()

        # replace all the remaining nans with centered noise
        num_noise_rigids = self.sample_length * self.rigids_per_res
        trans_0 = _centered_gaussian(num_noise_rigids) * 16 # * 20 #
        rotquats_0 = _uniform_so3(num_noise_rigids)
        rigids_0_tensor_7 = torch.cat([rotquats_0, trans_0], dim=-1)
        rigids_0_tensor_7 = rigids_0_tensor_7.unflatten(0, (-1, self.rigids_per_res)).contiguous()
        # rigids_0 = ru.Rigid.from_tensor_7(rigids_0_tensor_7)
        rigids_noising_mask = torch.ones(rigids_0_tensor_7.shape[:-1], dtype=torch.bool)

        # construct the input heterodata object
        num_res = seq.numel()
        data = HeteroData(
            residue={
                "res_mask": torch.ones(num_res).bool(),
                "rigids_t": rigids_0_tensor_7,
                "rigids_mask": torch.ones((num_res, 3)).bool(),
                "rigids_noising_mask": rigids_noising_mask.bool(),
                "seq": seq.long(),
                "seq_mask": torch.ones(num_res).bool(),
                "seq_noising_mask": (seq == residue_constants.restype_order_with_x['X']),
                "chain_idx": torch.zeros(num_res),
                "num_res": num_res,
                "num_nodes": num_res
            }
        )
        data['task'] = self
        return data

    def sample_data(self):
        for _ in range(self.num_samples):
            yield self._sample_init_data()

    def pad_data(self, data, n_padding):
        data = copy.copy(data)
        res_data = data['residue']
        res_data["res_mask"] = F.pad(res_data["res_mask"], pad=(0, n_padding), value=False)
        rigids_t = res_data["rigids_t"]
        rigid_padding = ru.Rigid.identity(
            (n_padding, *(rigids_t.shape[1:-1]))
        ).to_tensor_7()
        rigids_t_tensor_7 = torch.cat(
            [
                rigids_t,
                rigid_padding.to(device=rigids_t.device)
            ],
            dim=0
        )
        res_data["rigids_t"] = rigids_t_tensor_7

        res_data["rigids_mask"] = F.pad(res_data["rigids_mask"], pad=(0, 0, 0, n_padding), value=False)
        res_data["rigids_noising_mask"] = F.pad(res_data["rigids_noising_mask"], pad=(0, 0, 0, n_padding), value=False)
        res_data["seq"] = F.pad(res_data["seq"], pad=(0, n_padding), value=residue_constants.restype_order_with_x['X'])
        res_data["seq_mask"] = F.pad(res_data["seq_mask"], pad=(0, n_padding), value=False)
        res_data["seq_noising_mask"] = F.pad(res_data["seq_noising_mask"], pad=(0, n_padding), value=False)
        res_data["chain_idx"] = F.pad(res_data["chain_idx"], pad=(0, n_padding), value=res_data["chain_idx"][-1])
        res_data["num_nodes"] = res_data["num_nodes"] + n_padding

        return data

