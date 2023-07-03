""" Parsers for data files """
from typing import List, Tuple

import Bio.PDB as PDB
from Bio.PDB.Model import Model
from Bio.PDB.Polypeptide import Polypeptide
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure


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
