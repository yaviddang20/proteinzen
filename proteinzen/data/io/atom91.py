""" Utils for loading and editing PDBs """
import os

from Bio.PDB import Atom, Residue, Chain, Model, Structure
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBParser import PDBParser
import numpy as np

from proteinzen.data.openfold.residue_constants import restype_1to3
from proteinzen.utils.atom_reps import sidechain_atoms, atom91_start_end


def build_residue(aa, coords, res_idx, atom_start_idx, segid="A"):
    # coords: n_atom x 3
    aa_3lt = restype_1to3[aa]
    sidechain_atom_labels = sidechain_atoms[aa_3lt]
    global_atom_idx = atom_start_idx
    atoms = []
    for backbone_atom_idx, atom_type in enumerate(['N', 'CA', 'C', 'O']):
        atom = Atom.Atom(name=atom_type,
                         coord=coords[backbone_atom_idx],
                         occupancy=1.,
                         bfactor=0.,  # dummy value
                         altloc=" ",  # no altloc
                         fullname=f" {atom_type} ",
                         serial_number=global_atom_idx
        )
        atoms.append(atom)
        global_atom_idx += 1

    for sidechain_atom_idx, atom_type in enumerate(sidechain_atom_labels):
        local_atom_idx = atom91_start_end[aa_3lt][0] + sidechain_atom_idx
        atom = Atom.Atom(name=atom_type,
                         coord=coords[local_atom_idx],
                         occupancy=1.,
                         bfactor=0.,  # dummy value
                         altloc=" ",  # no altloc
                         fullname=f" {atom_type} ",
                         serial_number=global_atom_idx
        )


        atoms.append(atom)
        global_atom_idx += 1

    res_id = (" ", res_idx, " ")
    res = Residue.Residue(res_id, aa_3lt, segid=segid)
    for atom in atoms:
        res.add(atom)

    return res, global_atom_idx

def atom91_to_chain(seq, atom91, chain_id, x_bb=None):
    if x_bb:
        atom91[..., :4, :] = x_bb
    atom_idx = 1 # 0
    chain = Chain.Chain(chain_id)
    for seq_idx, aa in enumerate(seq):
        coords = atom91[seq_idx]
        if np.isnan(coords).all():
            continue
        res, atom_idx = build_residue(aa, coords, seq_idx+1, atom_idx) # 1-indexed chain for pyrosetta
        chain.add(res)
    return chain

def chains_to_model(chains, model_id=0):
    model = Model.Model(id=model_id)
    for chain in chains:
        model.add(chain)
    return model

def models_to_struct(models, struct_id=0):
    struct = Structure.Structure(struct_id)
    for model in models:
        struct.add(model)
    return struct

def chains_to_struct(chains, model_id=0, struct_id=0):
    model = Model.Model(id=model_id)
    for chain in chains:
        model.add(chain)
    struct = Structure.Structure(struct_id)
    struct.add(model)
    return struct

def save_struct(struct, filename):
    io = PDBIO()
    io.set_structure(struct)
    io.save(filename)

def atom91_to_pdb(seq, atom91, name):
    chain = atom91_to_chain(seq, atom91, "A")
    struct = chains_to_struct([chain])
    save_struct(struct, name + ".pdb")


def chain_to_atom91(chain):
    atom91 = []
    bb_atoms = ["N", "CA", "C", "O"]
    for residue in chain.get_residues():
        residue_91 = np.zeros((91, 3)) * np.nan
        aa_3lt = residue.get_resname()
        sidechain_atom_labels = sidechain_atoms[aa_3lt]
        atoms = list(residue.get_atoms())
        for atom in atoms:
            atom_name = atom.get_name()
            if atom_name in bb_atoms:
                sidechain_atom_idx = bb_atoms.index(atom_name)
                residue_91[sidechain_atom_idx] = atom.get_coord()
            else:
                sidechain_atom_idx = sidechain_atom_labels.index(atom_name)
                local_atom_idx = atom91_start_end[aa_3lt][0] + sidechain_atom_idx
                residue_91[local_atom_idx] = atom.get_coord()
        atom91.append(residue_91)
    return np.stack(atom91, axis=0)


def struct_to_atom91(struct):
    atom91 = []
    for chain in struct.get_chains():
        atom91.append(chain_to_atom91(chain))
    return np.concatenate(atom91, axis=0)


def pdb_to_atom91(pdb_path, silent=False):
    parser = PDBParser(QUIET=silent)
    name = os.path.splitext(
        os.path.basename(pdb_path)
    )[0]
    struct = parser.get_structure(name, pdb_path)
    atom91 = struct_to_atom91(struct)
    atom91_mask = np.isnan(atom91)
    atom91[atom91_mask] = 0
    return atom91, atom91_mask

def pdb_to_structure(pdb_path, silent=False):
    parser = PDBParser(QUIET=silent)
    name = os.path.splitext(
        os.path.basename(pdb_path)
    )[0]
    struct = parser.get_structure(name, pdb_path)
    return struct