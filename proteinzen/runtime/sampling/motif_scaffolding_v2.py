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
import functools as fn

from Bio.PDB.PDBParser import PDBParser
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import tree
from torch_geometric.data import HeteroData
import numpy as np

from proteinzen.data.openfold import residue_constants, data_transforms
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


class MotifScaffoldingTaskV2(SamplingTask):
    chain_alphabet: str = string.ascii_uppercase
    task_name: str = "motif_scaffolding_v2"

    def __init__(
        self,
        pdb_contigs,
        # redesign_contigs,
        contigs_idx,
        pdb,
        num_samples,
        total_length,
        cg_version=1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.generator = np.random.default_rng()
        parser = PDBParser()
        self.structure = parser.get_structure("", pdb)

        self.redesign_contigs = {}
        # if len(redesign_contigs) > 1:
        #     redesign_contigs = [c.strip() for c in redesign_contigs.split(",")]
        #     for contig in redesign_contigs:
        #         chain = contig[0]
        #         if chain not in self.redesign_contigs:
        #             self.redesign_contigs[chain] = set()
        #         if "-" in contig:
        #             start, end = [int(i) for i in contig[1:].split("-")]
        #             self.redesign_contigs[chain].update(set(range(start, end+1)))
        #         else:
        #             resid = int(contig[1:])
        #             self.redesign_contigs[chain].add(resid)

        try:
            self.contigs = self._parse_condition_str(pdb_contigs)
        except Exception as e:
            print(f"error with parsing contigs {pdb_contigs} from {pdb}")
            raise e
        try:
            self.motif_seq_idx, self.motif_res_is_unindexed = self._parse_idx_str(contigs_idx)
        except Exception as e:
            print(f"error with parsing contigs {contigs_idx}")
            raise e
        self.num_samples = num_samples
        self.cg_version = cg_version

        self.total_len = [int(i) for i in total_length.split("-")]
        assert len(self.total_len) in (1, 2)

    def _parse_condition_str(self, contigs_str):
        contigs = [c.strip() for c in contigs_str.split(",")]
        data_chunks = []

        for contig in contigs:
            if contig[0] in self.chain_alphabet:
                chain = contig[0]
                if "-" in contig:
                    motif_start, motif_end = [int(i) for i in contig[1:].split("-")]
                    # print("scaffold", contig, motif_start, motif_end)
                    data_chunks.append(self._get_motif_data(chain, motif_start, motif_end))
                else:
                    motif_resid = int(contig[1:])
                    # print("scaffold", contig, motif_resid)
                    data_chunks.append(self._get_motif_data(chain, motif_resid))
            else:
                raise ValueError()

        return dict_cat(data_chunks)

    def _parse_idx_str(self, idx_str):
        idx_contigs = [c.strip() for c in idx_str.split(",")]
        idxs = []

        for contig in idx_contigs:
            if contig.startswith("[") and contig.endswith("]"):
                # unindexed residue
                for _ in range(int(contig[1:-1])):
                    idxs.append(-1)
            elif "-" in contig:
                # index range
                motif_start, motif_end = [int(i) for i in contig.split("-")]
                for i in range(motif_start, motif_end + 1):
                    idxs.append(i)
            else:
                # there might be some adversarial case where this check holds but this should be fine for now
                assert str(int(contig)) == contig
                motif_resid = int(contig)
                idxs.append(motif_resid)

        motif_seq_idx = torch.as_tensor(idxs, dtype=torch.long)
        motif_res_is_unindexed = (motif_seq_idx == -1)
        motif_seq_idx[motif_res_is_unindexed] = 0

        return motif_seq_idx, motif_res_is_unindexed

    def _get_motif_data(self, chain_id, motif_start, motif_end=None):
        chain = self.structure[0][chain_id]
        atom37 = []
        seq = []
        if motif_end is not None:
            for resid in range(motif_start, motif_end+1):
                res_atom37, res_seq = residue_to_atom37(chain[resid])
                atom37.append(res_atom37)
                seq.append(res_seq)
        else:
            res_atom37, res_seq = residue_to_atom37(chain[motif_start])
            atom37.append(res_atom37)
            seq.append(res_seq)

        atom37 = torch.stack(atom37).double()
        seq = torch.as_tensor(seq).long()
        atom37_mask = torch.isfinite(atom37).all(dim=-1)
        atom37 = atom37_mask * atom37_mask[..., None]

        return {
            'aatype': seq,
            'all_atom_positions': atom37,
            'all_atom_mask': atom37_mask.float(),
        }


    def _get_blank_data(self, lower, upper, override=None):
        if override is not None:
            num_res = override
        else:
            num_res = self.generator.integers(lower, upper, endpoint=True)
        dummy_atom37 = torch.full((num_res, 37, 3), torch.nan)
        atom37_mask = torch.ones((num_res, 37))
        seq = torch.full((num_res,), residue_constants.restype_order_with_x['X'])

        return {
            'aatype': seq.long(),
            'all_atom_positions': dummy_atom37.double(),
            'all_atom_mask': atom37_mask.double(),
        }

    def _sample_init_data(self):
        motif_feats = self.contigs
        motif_feats = data_transforms.atom37_to_cg_frames(motif_feats, cg_version=self.cg_version)
        motif_feats = data_transforms.atom37_to_torsion_angles(prefix="")(motif_feats)  # TODO: uncurry this

        for key, value in motif_feats.items():
            print(key, value.dtype)
            if value.dtype == torch.float64:
                motif_feats[key] = value.float()

        motif_feats = data_transforms.make_atom14_masks(motif_feats)
        motif_feats = data_transforms.make_atom14_positions(motif_feats)
        motif_seq = motif_feats['aatype']
        motif_rigids = ru.Rigid.from_tensor_4x4(motif_feats['cg_groups_gt_frames'])[:, (0, 2, 3)].to_tensor_7()

        # compute the dummy rigids
        rigids_mask = motif_feats["cg_groups_gt_exists"][:, (0, 2, 3)].bool()
        mask_AG = (motif_seq == residue_constants.restype_order['G']) | (motif_seq == residue_constants.restype_order['A'])
        mask_not_X = (motif_seq != residue_constants.restype_order_with_x['X'])
        dummy_rigid = motif_rigids[..., 0, :] * mask_AG[..., None] + motif_rigids[..., 1, :] * (~mask_AG & mask_not_X)[..., None]
        dummy_rigid_location = (~rigids_mask) * mask_not_X[..., None]
        motif_rigids[dummy_rigid_location] = 0
        motif_rigids += dummy_rigid[..., None, :] * dummy_rigid_location[..., None]

        assembler = SampleRigidAssembler(
            motif_rigids=motif_rigids,
            motif_seq=motif_seq,
            promote_full_motif_to_token=False
        )

        if len(self.total_len) == 1:
            num_res = self.total_len[0]
        else:
            num_res = self.generator.integers(self.total_len[0], self.total_len[1], endpoint=True)

        noisy_trans = _centered_gaussian(num_res * 3).unflatten(0, (-1, 3))
        noisy_quats = _uniform_so3(num_res * 3).unflatten(0, (-1, 3))
        noisy_rigids = torch.cat([noisy_quats, noisy_trans], dim=-1)
        motif_noising_mask = torch.zeros(num_res, 3)

        rigids_data = assembler.assemble(
            noisy_rigids,
            motif_noising_mask,
            self.motif_seq_idx,
            self.motif_res_is_unindexed
        )

        features = {
            'token': {},
            'rigids': {},
            'residue': {}  # only for batching
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
            features['token'][token_key] = rigids_data[rigids_key][rigids_is_token_rigid_mask]
        features['token']['token_gather_idx'] = rigids_data["token_gather_idx"]

        # add features from the original node data
        res_data = {
            'seq': torch.full((num_res,), fill_value=residue_constants.restype_order_with_x['X']).long(),
            'seq_noising_mask': torch.ones(num_res, dtype=torch.bool),
            'seq_mask': torch.ones(num_res, dtype=torch.bool),
            'chain_idx': torch.zeros(num_res),
        }
        token_feats = [
            "seq",
            "seq_noising_mask",
            "seq_mask",
            "chain_idx"
        ]
        # duplicate the motif residues and update res_data
        for key in token_feats:
            tensor = res_data[key]
            tensor_expand = tensor[..., None].expand(-1, 3)
            features['token'][key] = torch.cat([tensor, tensor_expand.flatten(0, 1)], dim=0)

        features['residue']['num_nodes'] = features['token']['seq'].numel()
        features['residue']['num_res'] = features['token']['seq'].numel()

        return features

    def sample_data(self):
        for _ in range(self.num_samples):
            yield self._sample_init_data()

    def pad_data(self, data, n_padding):
        return NotImplemented

