"""Library for parsing different data structures."""
import numpy as np
from typing import List, Tuple

import Bio.PDB as PDB
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Polypeptide import Polypeptide
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from ligbinddiff.data.openfold import residue_constants
from ligbinddiff.data.io import protein


Protein = protein.Protein


def load_list_mmcif(paths: List[str]) -> List[Structure]:
    """ Load a list of mmCIF files """
    parser = PDB.FastMMCIFParser()
    return list(map(parser.get_structure, paths))


def split_ligands(struct: Model) -> Tuple[Polypeptide, List[Residue]]:
    """
    Split a PDB model into protein chains and ligands

    Args
    ----
    struct: Bio.PDB.Model.Model
        Model to be parsed

    Returns
    -------
    chains: Bio.PDB.Polypeptite.Polypeptide
        Model with ligands removed
    ligands: List[Bio.PDB.Residue.Residue]
        List of ligands
    """
    pp_builder = PDB.Polypeptide.CaPPBuilder()
    chains = pp_builder.build_peptides(struct)

    ligands = []
    for chain in struct.get_chains():
        for residue in chain.get_residues():
            if residue.id[0].startswith("H_"):
                ligands.append(residue)

    return chains, ligands


def process_chain(chain: Chain, chain_id: str) -> Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.

    Forked from alphafold.common.protein.from_pdb_string

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.

    Took out lines 110-112 since that would mess up CDR numbering.

    Args:
        chain: Instance of Biopython's chain class.

    Returns:
        Protein object with protein features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num)
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.
            res_b_factors[residue_constants.atom_order[atom.name]
                          ] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors))
