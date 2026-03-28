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

from proteinzen.boltz.data.types import Structure

from proteinzen.openfold.data import residue_constants
from proteinzen.openfold.data import data_transforms
from proteinzen.openfold.utils import rigid_utils as ru
from proteinzen.data.featurize.mol.sampling import smiles_to_struct, mol_to_struct
from proteinzen.data.featurize.tokenize import Tokenized
from proteinzen.data.featurize.sampling import sample_noise_from_struct_template
from proteinzen.data.featurize.assembler import featurize


from rdkit import Chem
from .task import SamplingTask

from proteinzen.stoch_interp import so3_utils

from proteinzen.data.featurize.tokenize import tokenize_structure
from proteinzen.data.featurize.assembler import process_rigid_features
from proteinzen.data.write.pdb import to_pdb

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


class UnconditionalSamplingFromSMILES(SamplingTask):
    task_name: str = "unconditional_smiles"

    def __init__(
        self,
        smiles: str,
        num_samples: int,
        trans_std: float = 3,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.smiles = smiles
        self.num_samples = num_samples
        self.trans_std = trans_std

    def igso3(self):
        sigma_grid = torch.linspace(0.1, 1.5, 1000)
        _igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        return _igso3 

    def sample_data(self):
        igso3 = self.igso3()
        struct = smiles_to_struct(self.smiles)

        # pdb_str =to_pdb(struct)
        # with open(f"clean_data.pdb", "w") as f:
        #     f.write(pdb_str)
        # exit()
        
        # clean_task_data= {}
        # atom_noising_mask = np.ones(len(struct.atoms), dtype=bool)
        # res_type_noising_mask = np.ones(len(struct.residues), dtype=bool)
        # clean_task_data["atom_noising_mask"] = atom_noising_mask
        # clean_task_data["res_type_noising_mask"] = res_type_noising_mask
        # clean_token_data, clean_rigid_data, clean_token_bonds = tokenize_structure(struct, clean_task_data)
        for _ in range(self.num_samples):
            token_data, rigid_data, token_bonds, _ = sample_noise_from_struct_template(
                struct,
                igso3=igso3,
                trans_std=self.trans_std
            )
            data = Tokenized(
                tokens=token_data,
                rigids=rigid_data,
                bonds=token_bonds,
                structure=struct
            )
            task_data = {
                "t": np.array([0.0], dtype=float),
            }

            # clean_data = Tokenized(
            #     tokens=clean_token_data,
            #     rigids=clean_rigid_data,
            #     bonds=clean_token_bonds,
            #     structure=struct
            # )


            # task_data['clean_rigids_1'] = process_rigid_features(clean_data)['rigids_1'] 

            # yield featurize_inference(clean_data, task_data, task_name=self.kwargs.get("name", self.task_name))
            yield featurize(data, task_data, task_name=self.kwargs.get("name", self.task_name))

class UnconditionalSamplingFromMol(SamplingTask):
    task_name: str = "unconditional_mol"

    def __init__(
        self,
        mol_pdb_path: str,
        num_samples: int,
        trans_std: float = 3,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.mol = Chem.MolFromPDBFile(mol_pdb_path)
        self.num_samples = num_samples
        self.trans_std = trans_std

    def igso3(self):
        sigma_grid = torch.linspace(0.1, 1.5, 1000)
        _igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        return _igso3 

    def sample_data(self):
        igso3 = self.igso3()
        struct = mol_to_struct(self.mol)

        # pdb_str = to_pdb(struct)
        # with open(f"clean_data_mol.pdb", "w") as f:
        #     f.write(pdb_str)
        # exit()
        # clean_task_data= {}
        # atom_noising_mask = np.ones(len(struct.atoms), dtype=bool)
        # res_type_noising_mask = np.ones(len(struct.residues), dtype=bool)
        # clean_task_data["atom_noising_mask"] = atom_noising_mask
        # clean_task_data["res_type_noising_mask"] = res_type_noising_mask
        # clean_token_data, clean_rigid_data, clean_token_bonds = tokenize_structure(struct, clean_task_data)
        for _ in range(self.num_samples):
            token_data, rigid_data, token_bonds, _ = sample_noise_from_struct_template(
                struct,
                igso3=igso3,
                trans_std=self.trans_std
            )

            # print("token_data", token_data)
            # print("rigid_data", rigid_data)
            # exit()
            data = Tokenized(
                tokens=token_data,
                rigids=rigid_data,
                bonds=token_bonds,
                structure=struct
            )

            # clean_data = Tokenized(
            #     tokens=clean_token_data,
            #     rigids=clean_rigid_data,
            #     bonds=clean_token_bonds,
            #     structure=struct
            # )

            task_data = {
                "t": np.array([0.0], dtype=float),
            }

            yield featurize(data, task_data, task_name=self.kwargs.get("name", self.task_name))