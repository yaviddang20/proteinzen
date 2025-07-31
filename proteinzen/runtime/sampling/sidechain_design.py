""" Code for generating motif scaffolding inputs

Conditions are specified with RFDiffusion-inspired strings which use a comma-separated list
to specify conditions along the sequence dimension.
- motif input can take the form ``X##'' to specify a single residue of chain ``X''
  or ``X##-##'' to specify a segment from chain ``X''
- sections of protein to design from scratch can be specified as ``##'' for a fixed number of residues
  or ``##-##'' for a variable range of possible residue spacings. If a variable range is specified,
  the number of residues for that segment will be sampled uniformly at random from the range (inclusive)

The precise form of motif scaffolding will be inferred from the data provided in the input PDB file. More concretely,
- if the motif does not contain the necessary backbone atoms, the backbone frame will be designed from scratch
- if the motif does not contain the necessary sidechain atoms, the missing sidechain frame(s) will be designed from scratch
The residue's identity will also be read from the PDB file and used as conditioning input.
If you wish to redesign the sequence identity as well, modify the input PDB residue identity to be ``UNK''.

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

from proteinzen.openfold.data import residue_constants
from proteinzen.openfold.data import data_transforms
from proteinzen.openfold.utils import rigid_utils as ru
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
        if atom_name.startswith('H'):
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


class SidechainDesignTask(SamplingTask):
    chain_alphabet: str = string.ascii_uppercase
    task_name: str = "sidechain_design"

    def __init__(
        self,
        pdb,
        num_samples,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.generator = np.random.default_rng()
        parser = PDBParser()
        self.structure = parser.get_structure("", pdb)
        self.num_samples = num_samples

    def _get_chain_data(self, chain):
        atom37 = []
        seq = []
        for residue in chain.get_residues():
            res_atom37, res_seq = residue_to_atom37(residue)
            atom37.append(res_atom37)
            seq.append(res_seq)

        atom37 = torch.stack(atom37).double()
        seq = torch.as_tensor(seq).long()
        atom37_mask = torch.isfinite(atom37).all(dim=-1)

        return {
            'aatype': seq,
            'all_atom_positions': atom37,
            'all_atom_mask': atom37_mask,
        }

    def _sample_init_data(self):
        data_chunks = []
        for chain in self.structure[0].get_chains():
            data_chunks.append(self._get_chain_data(chain))

        for chain_idx, d in enumerate(data_chunks):
            seq_len = d['seq'].numel()
            d['chain_idx'] = torch.full((seq_len,), chain_idx)

        chain_feats = dict_cat(data_chunks)
        chain_feats = data_transforms.atom37_to_cg_frames(chain_feats, cg_version=self.cg_version)
        chain_feats = data_transforms.atom37_to_torsion_angles(prefix="")(chain_feats)  # TODO: uncurry this
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        seq = chain_feats['aatype']

        # compute rigids from data
        # a lot of these will be nan
        rigids_1 = ru.Rigid.from_tensor_4x4(chain_feats['cg_groups_gt_frames'])[:, (0, 2, 3)]
        rigids_1_tensor_7 = rigids_1.to_tensor_7()
        rigids_noising_mask = torch.ones(rigids_1.shape, dtype=torch.bool)
        rigids_noising_mask[..., 0] = False

        # replace all the remaining nans with centered noise
        num_noise_rigids = rigids_1.shape[0] * (rigids_1.shape[1] - 1)
        trans_0 = _centered_gaussian(num_noise_rigids) * 16 # 10
        rotquats_0 = _uniform_so3(num_noise_rigids)
        noise_tensor_7 = torch.cat([rotquats_0, trans_0], dim=-1)
        rigids_1_tensor_7[..., 1:, :] = noise_tensor_7

        # construct the input heterodata object
        num_res = seq.numel()
        data = HeteroData(
            residue={
                "res_mask": torch.ones(num_res).bool(),
                "rigids_t": rigids_1_tensor_7,
                "rigids_mask": torch.ones((num_res, 3)).bool(),
                "rigids_noising_mask": rigids_noising_mask.bool(),
                "seq": torch.full((num_res,), residue_constants.restype_order_with_x['X']).bool(),
                "seq_mask": torch.ones(num_res).bool(),
                "seq_noising_mask": torch.ones(num_res).bool(),
                "chain_idx": chain_feats['chain_idx'],
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
        rigid_padding = ru.Rigid.identity(
            (n_padding, *(res_data["rigids_t"].shape[1:-1]))
        ).to_tensor_7()
        rigids_t = res_data["rigids_t"]
        res_data["rigids_t"] = torch.cat(
            [
                rigids_t,
                rigid_padding.to(device=rigids_t.device)
            ],
            dim=0
        )
        res_data["rigids_mask"] = F.pad(res_data["rigids_mask"], pad=(0, 0, 0, n_padding), value=False)
        res_data["rigids_noising_mask"] = F.pad(res_data["rigids_noising_mask"], pad=(0, 0, 0, n_padding), value=False)
        res_data["seq"] = F.pad(res_data["seq"], pad=(0, n_padding), value=residue_constants.restype_order_with_x['X'])
        res_data["seq_mask"] = F.pad(res_data["seq_mask"], pad=(0, n_padding), value=False)
        res_data["seq_noising_mask"] = F.pad(res_data["seq_noising_mask"], pad=(0, n_padding), value=False)
        res_data["chain_idx"] = F.pad(res_data["chain_idx"], pad=(0, n_padding), value=res_data["chain_idx"][-1])
        res_data["num_nodes"] = res_data["num_nodes"] + n_padding

        return data

