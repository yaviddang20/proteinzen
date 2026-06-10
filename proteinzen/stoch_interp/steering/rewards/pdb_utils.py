import io
import dataclasses
from collections.abc import Mapping
from typing import Any, Optional, Sequence

import numpy as np
from Bio.PDB import PDBParser
from proteinzen.openfold.data import residue_constants

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.

@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None

    # Optional remark about the protein. Included as a comment in output PDB
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None

    def __post_init__(self):
        if(len(np.unique(self.chain_index)) > PDB_MAX_CHAINS):
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} "
                "chains because these cannot be written to PDB format"
            )


def from_pdb_file(pdb_file: str, chain_id: str | None = None) -> Protein:
    """Takes a PDB file and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_file: Path to the PDB file
      chain_id: If chain_id is specified (e.g. A), then only that chain is parsed. Otherwise all chains are parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    with open(pdb_file) as f:
        pdb_str = f.read()
    return from_pdb_string(pdb_str=pdb_str, chain_id=chain_id)


def from_pdb_string(pdb_str: str, chain_id: str | None = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If chain_id is specified (e.g. A), then only that chain
        is parsed. Otherwise all chains are parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(f"Only single model PDBs are supported. Found {len(models)} models.")
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. These are not supported."
                )
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            restype_idx = residue_constants.restype_order.get(res_shortname, residue_constants.restype_num)
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
    )
