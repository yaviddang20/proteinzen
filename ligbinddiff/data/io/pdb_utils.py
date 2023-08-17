""" Utils for loading and editing PDBs """

from Bio.PDB import Atom, Residue, Polypeptide, Chain, Model, Structure
from Bio.PDB.PDBIO import PDBIO
import numpy as np
from ligbinddiff.data.datasets.dataset import ProteinGraphDataset
from ligbinddiff.data.sampler import BatchSampler

from ligbinddiff.utils.atom_reps import restype_1to3, sidechain_atoms, atom91_start_end


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
    atom_idx = 0
    chain = Chain.Chain(chain_id)
    # pp = Polypeptide.Polypeptide()
    for seq_idx, aa in enumerate(seq):
        coords = atom91[seq_idx]
        if np.isnan(coords).all():
            continue
        res, atom_idx = build_residue(aa, coords, seq_idx, atom_idx)
        chain.add(res)
        # pp.append(res)
    # chain.add(pp)
    return chain

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


if __name__ == '__main__':
    from functools import partial

    import torch
    from torch.utils.data import DataLoader, SequentialSampler
    from ligbinddiff.data.datasets.cath import CATHDataset
    from ligbinddiff.utils.atom_reps import num_to_letter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    dataloader = lambda x: DataLoader(x,
                        num_workers=4,  #args.num_workers,
                        batch_sampler=BatchSampler(SequentialSampler(x), batch_size=100),
                        collate_fn=lambda x: x[0])

    torch.set_float32_matmul_precision('medium')

    cath = CATHDataset(path="../../../data/cath/micro_chain_set.jsonl",
                    splits_path="../../../data/cath/micro_chain_set_splits.json")
    n_max = 5
    dataset = partial(ProteinGraphDataset, density_nmax=n_max, channel_atoms=True, bb_density=False)
    trainset, valset, testset = map(dataset,
                                    (cath.train, cath.val, cath.test))

    sample = next(iter(trainset))
    print(sample)
    print(sample.name)
    atom91 = sample.ndata["atom91"]
    seq = sample.ndata["seq"]
    seq = "".join([num_to_letter[i] for i in seq.tolist()])

    atom91_to_pdb(seq, atom91, sample.name)
