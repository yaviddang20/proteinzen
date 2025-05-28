import copy
import string
import functools as fn

from Bio.PDB.PDBParser import PDBParser
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import tree
from torch_geometric.data import HeteroData
import numpy as np

from proteinzen.data.openfold import residue_constants, data_transforms
from proteinzen.data.constants import coarse_grain as cg
from proteinzen.data.datasets.featurize.rigid_assembler import SampleRigidAssembler
from proteinzen.utils.openfold import rigid_utils as ru
from .task import SamplingTask


def dict_cat(data):
    return tree.map_structure(lambda *x: torch.cat(x), *data)


def residue_to_atom37(residue):
    residue_37 = torch.full((37, 3), torch.nan)
    aa_3lt = residue.get_resname()
    atoms = list(residue.get_atoms())
    for atom in atoms:
        atom_name = atom.get_name()
        # we ignore hydrogens
        if str(atom.element).upper() == 'H':
            # print("skipping H")
            continue
        atom_idx = residue_constants.atom_types.index(atom_name)
        residue_37[atom_idx] = torch.as_tensor(atom.get_coord())
    return residue_37, residue_constants.resname_to_idx[aa_3lt]


def _centered_gaussian(num_rigids):
    noise = torch.randn(num_rigids, 3)
    return noise - noise.mean(dim=0, keepdim=True)

def _uniform_so3(num_rigids):
    return torch.as_tensor(
        Rotation.random(num_rigids).as_quat(),
        dtype=torch.float32,
    )


class UnconditionalSamplingV2(SamplingTask):
    chain_alphabet: str = string.ascii_uppercase
    task_name: str = "motif_scaffolding_v2"

    def __init__(
        self,
        num_samples,
        sample_length,
        cg_version=1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.generator = np.random.default_rng()
        self.num_samples = num_samples
        self.cg_version = cg_version
        self.total_len = sample_length

    def _sample_init_data(self):
        noisy_trans = _centered_gaussian(self.total_len * 3).unflatten(0, (-1, 3)) * 16
        noisy_quats = _uniform_so3(self.total_len * 3).unflatten(0, (-1, 3))
        noisy_rigids = torch.cat([noisy_quats, noisy_trans], dim=-1)

        dummy_motif = torch.zeros((0, 7), device=noisy_rigids.device)
        dummy_motif_seq = torch.zeros((0,), device=noisy_rigids.device)

        rigids_data = SampleRigidAssembler.assemble_noise(
            noisy_rigids
        )

        features = {
            'token': {},
            'rigids': {},
            'residue': {},  # only for batching
            'atom': {'atom14_mask': torch.zeros(self.total_len, 14)}
        }

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
            # print(rigids_key)
            features['token'][token_key] = rigids_data[rigids_key][rigids_is_token_rigid_mask]
        features['token']['token_gather_idx'] = rigids_data["token_gather_idx"]

        # add features from the original node data
        res_data = {
            'seq': torch.full((self.total_len,), fill_value=residue_constants.restype_order_with_x['X']).long(),
            'seq_noising_mask': torch.ones(self.total_len, dtype=torch.bool),
            'seq_mask': torch.ones(self.total_len, dtype=torch.bool),
            'chain_idx': torch.zeros(self.total_len),
        }
        # token_data = features['token']
        # token_data['seq'] = torch.cat([res_data['seq'], dummy_motif_seq], dim=-1)
        # token_data['seq_noising_mask'] = torch.cat([res_data['seq_noising_mask'], torch.zeros_like(dummy_motif_seq).bool()], dim=-1)
        # token_data['seq_mask'] = torch.cat([res_data['seq_mask'], torch.ones_like(dummy_motif_seq).bool()], dim=-1)
        # token_data['chain_idx'] = torch.cat([res_data['chain_idx'], torch.full_like(dummy_motif_seq, fill_value=res_data['chain_idx'][-1])], dim=-1)
        token_feats = [
            "seq",
            "seq_noising_mask",
            "seq_mask",
            "chain_idx"
        ]
        # duplicate the motif residues and update res_data
        for key in token_feats:
            features['token'][key] = res_data[key]

        features['residue']['num_nodes'] = torch.as_tensor(features['token']['seq'].numel())
        features['residue']['num_res'] = torch.as_tensor(features['token']['seq'].numel())
        features['t'] = torch.zeros((1,))[None]
        features['task'] = self

        return features

    def sample_data(self):
        for _ in range(self.num_samples):
            yield self._sample_init_data()

    def pad_data(self, data, n_padding):
        return NotImplemented

