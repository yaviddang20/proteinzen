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


class MotifScaffoldingTask(SamplingTask):
    chain_alphabet: str = string.ascii_uppercase
    task_name: str = "motif_scaffolding"

    def __init__(
        self,
        contigs,
        pdb,
        num_samples,
        redesign_contigs,
        total_length=None,
        cg_version=1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.generator = np.random.default_rng()
        parser = PDBParser()
        self.structure = parser.get_structure("", pdb)
        self.num_samples = num_samples
        self.cg_version = cg_version

        if total_length is not None:
            self.total_len = [int(i) for i in total_length.split("-")]
            assert len(self.total_len) in (1, 2)
        else:
            self.total_len = None

        self.redesign_contigs = {}

        if len(redesign_contigs) > 1:
            redesign_contigs = [c.strip() for c in redesign_contigs.split(",")]
            for contig in redesign_contigs:
                chain = contig[0]
                if chain not in self.redesign_contigs:
                    self.redesign_contigs[chain] = set()
                if "-" in contig:
                    start, end = [int(i) for i in contig[1:].split("-")]
                    self.redesign_contigs[chain].update(set(range(start, end+1)))
                else:
                    resid = int(contig[1:])
                    self.redesign_contigs[chain].add(resid)

        try:
            self.contigs = self._parse_condition_str(contigs)
        except Exception as e:
            print(f"error with parsing contigs {contigs} from {pdb}")
            raise e

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
                if "-" in contig:
                    lower, upper = [int(i) for i in contig.split("-")]
                    # print("design", contig, lower, upper)
                    chunk_fn = fn.partial(self._get_blank_data, lower=lower, upper=upper)
                    data_chunks.append(chunk_fn)
                else:
                    length = int(contig)
                    # print("design", contig, length)
                    chunk_fn = fn.partial(self._get_blank_data, lower=length, upper=length)
                    data_chunks.append(chunk_fn)

        return data_chunks

    def _get_motif_data(self, chain_id, motif_start, motif_end=None):
        chain = self.structure[0][chain_id]
        atom37 = []
        seq = []
        redesign_mask = []
        if motif_end is not None:
            for resid in range(motif_start, motif_end+1):
                res_atom37, res_seq = residue_to_atom37(chain[resid])
                if chain.id in self.redesign_contigs and resid in self.redesign_contigs[chain.id]:
                    res_atom37[3:] = torch.nan
                    atom37.append(res_atom37)
                    seq.append(residue_constants.restype_order_with_x['X'])
                    redesign_mask.append(True)
                else:
                    atom37.append(res_atom37)
                    seq.append(res_seq)
                    redesign_mask.append(False)
        else:
            res_atom37, res_seq = residue_to_atom37(chain[motif_start])
            if chain.id in self.redesign_contigs and motif_start in self.redesign_contigs[chain.id]:
                res_atom37[3:] = torch.nan
                atom37.append(res_atom37)
                seq.append(residue_constants.restype_order_with_x['X'])
                redesign_mask.append(True)
            else:
                atom37.append(res_atom37)
                seq.append(res_seq)
                redesign_mask.append(False)

        atom37 = torch.stack(atom37).double()
        seq = torch.as_tensor(seq).long()
        atom37_mask = torch.isfinite(atom37).all(dim=-1)
        redesign_mask = torch.as_tensor(redesign_mask, dtype=torch.bool)

        return {
            'aatype': seq,
            'all_atom_positions': atom37,
            'all_atom_mask': atom37_mask,
            'redesign_mask': redesign_mask
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
            'redesign_mask': torch.zeros_like(seq, dtype=torch.bool)
        }

    def _sample_init_data(self):
        data_chunks = []
        # print(len(self.contigs))
        for i, c in enumerate(self.contigs):
            if isinstance(c, dict):
                data_chunks.append(c)
            elif callable(c):
                if i == len(self.contigs)-1 and self.total_len is not None and len(self.total_len) == 1:
                    curr_len = sum([d['aatype'].numel() for d in data_chunks])
                    min_len = self.total_len[0] - curr_len
                    max_len = self.total_len[-1] - curr_len
                    if min_len == max_len:
                        override_len = max_len
                    else:
                        override_len = self.generator.integers(min_len, max_len, endpoint=True)
                    if override_len > 0:
                        data_chunks.append(c(override=override_len))
                else:
                    data_chunks.append(c())
            else:
                raise ValueError(f"we dont recognize condition {c}")
            # print(i, data_chunks[-1]['aatype'].numel())
            # print({k: v.shape for k,v in data_chunks[-1].items()})
        # print(sum([d['aatype'].numel() for d in data_chunks]))

        chain_feats = dict_cat(data_chunks)

        # TODO: the nans stuff doesn't work unless every atom in the residue is nan
        chain_feats = data_transforms.atom37_to_cg_frames(chain_feats, cg_version=self.cg_version)
        chain_feats = data_transforms.atom37_to_torsion_angles(prefix="")(chain_feats)  # TODO: uncurry this

        for key, value in chain_feats.items():
            if value.dtype == torch.float64:
                chain_feats[key] = value.float()

        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        seq = chain_feats['aatype']
        redesign_mask = chain_feats['redesign_mask']

        # compute rigids from data
        # a lot of these will be nan
        rigids_1 = ru.Rigid.from_tensor_4x4(chain_feats['cg_groups_gt_frames'])[:, (0, 2, 3)]

        # we're gonna do some jenk stuff via the rigid trans
        # bc doing rotations conversions with nans is messy
        rigids_1_isnan = torch.isnan(rigids_1.get_trans()).any(dim=-1)
        rigids_1_finite = rigids_1[~rigids_1_isnan]
        rigids_1_tensor_7 = torch.full((*rigids_1.shape, 7), torch.nan) # rigids_1.to_tensor_7()
        rigids_1_tensor_7[~rigids_1_isnan] = rigids_1_finite.to_tensor_7()

        # TODO: man this is really clunky
        # we copy over dummy rigids where available
        rigids_mask = chain_feats["cg_groups_gt_exists"][:, (0, 2, 3)].bool()
        rigids_mask = rigids_mask * (~rigids_1_isnan)
        mask_AG = (seq == residue_constants.restype_order['G']) | (seq == residue_constants.restype_order['A'])
        mask_not_X = (seq != residue_constants.restype_order_with_x['X'])
        dummy_rigid = rigids_1_tensor_7[..., 0, :] * mask_AG[..., None] + rigids_1_tensor_7[..., 1, :] * (~mask_AG & mask_not_X)[..., None]
        dummy_rigid_location = (~rigids_mask) * mask_not_X[..., None]
        rigids_1_tensor_7[dummy_rigid_location] = 0
        rigids_1_tensor_7 += dummy_rigid[..., None, :] * dummy_rigid_location[..., None]
        rigids_noising_mask = torch.isnan(rigids_1_tensor_7).any(dim=-1)
        rigids_noising_mask[redesign_mask, 1:] = True

        # print(rigids_noising_mask)

        # center the motif rigids
        motif_rigids_1_tensor_7 = rigids_1_tensor_7[~rigids_noising_mask]
        center = motif_rigids_1_tensor_7[..., 4:].mean(dim=0)
        motif_rigids_1_tensor_7[..., 4:] -= center[None]
        rigids_1_tensor_7[~rigids_noising_mask] = motif_rigids_1_tensor_7

        # replace all the remaining nans with centered noise
        num_noise_rigids = int(rigids_noising_mask.long().sum())
        trans_0 = _centered_gaussian(num_noise_rigids) * 16 # 10
        rotquats_0 = _uniform_so3(num_noise_rigids)
        noise_tensor_7 = torch.cat([rotquats_0, trans_0], dim=-1)
        rigids_1_tensor_7[rigids_noising_mask] = noise_tensor_7

        # in theory there should be no more nans
        assert torch.isfinite(rigids_1_tensor_7).all()
        # print(rigids_1_tensor_7[..., 4:].mean(dim=(0, 1)))
        # rigids_1 = ru.Rigid.from_tensor_7(rigids_1_tensor_7)

        # construct the input heterodata object
        num_res = seq.numel()
        data = HeteroData(
            residue={
                "res_mask": torch.ones(num_res).bool(),
                "rigids_t": rigids_1_tensor_7,
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

