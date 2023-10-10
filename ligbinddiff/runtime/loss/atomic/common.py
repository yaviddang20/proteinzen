""" TODO """
import torch
from ligbinddiff.runtime.loss.utils import vec_norm
from ligbinddiff.utils.atom_reps import alphabet, atom91_start_end, restype_1to3


def atoms_to_angles(bond_atom_coords, eps=1e-8):
    # Based on https://en.wikipedia.org/wiki/Dihedral_angle

    # absolute atom positions
    c1 = bond_atom_coords[..., 0, :]
    c2 = bond_atom_coords[..., 1, :]
    c3 = bond_atom_coords[..., 2, :]
    # print("c1", c1)
    # print("c2", c2)
    # print("c3", c3)

    # relative atom positions
    n1 = (c2 - c1) / vec_norm(c2-c1).unsqueeze(-1)
    n2 = (c3 - c2) / vec_norm(c3-c2).unsqueeze(-1)
    # print("n1", n1)
    # print("n2", n2)

    cosX = torch.sum(n1 * n2, dim=-1)
    # print("cosX", cosX)
    sinX_vec = torch.cross(n1, n2)
    # print("sinX_vec", sinX_vec)
    y_axis = torch.cross(sinX_vec, n1)
    # print("y_axis", y_axis)
    sinX_sign = torch.sign(torch.sum(y_axis * n2, dim=-1))
    sinX = vec_norm(sinX_vec, dim=-1) * sinX_sign
    # print("sinX", sinX)

    return torch.stack([cosX, sinX], dim=-1)


def atoms_to_torsions(chi_atom_coords, eps=1e-8):
    # Based on https://en.wikipedia.org/wiki/Dihedral_angle

    # absolute atom positions
    c1 = chi_atom_coords[..., 0, :]
    c2 = chi_atom_coords[..., 1, :]
    c3 = chi_atom_coords[..., 2, :]
    c4 = chi_atom_coords[..., 3, :]

    # relative atom positions
    a1 = c2 - c1
    a2 = c3 - c2
    a3 = c4 - c3

    # backbone normals
    v1 = torch.cross(a1, a2)
    v2 = torch.cross(a2, a3)

    # Angle between normals
    x = torch.sum(v1 * v2, -1)
    a2_norm = vec_norm(a2, mask_nans=False)
    y = torch.sum(a1 * v2, dim=-1) * a2_norm

    angle_vec = torch.stack([x, y], dim=-1)
    norm = vec_norm(angle_vec, mask_nans=False).unsqueeze(-1)

    if (angle_vec / norm).isnan().any():
        print("angle vec", angle_vec)
        print("norm", norm)

    angle_vec = angle_vec / norm
    return angle_vec


def atom91_to_atom14(atom91, seq):
    num_nodes = atom91.shape[0]
    atom91 = atom91.float()
    atom14 = torch.empty((num_nodes, 14, 3), device=atom91.device) * torch.nan
    atom14[:, :4] = atom91[:, :4]

    for i, aa_1lt in enumerate(alphabet):
        select_aa = (seq == i)
        start, end = atom91_start_end[restype_1to3[aa_1lt]]
        l = end - start
        atom14[select_aa, 4:4+l] = atom91[select_aa, start:end]

    atom14_mask = torch.isnan(atom14)
    atom14[atom14_mask] = 0

    return atom14, atom14_mask
