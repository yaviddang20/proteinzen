""" Utils for frame-based operations """

import torch

from ligbinddiff.utils.atom_reps import backbone_frame_atoms


def _normalize(x, dim=-1):
    return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True)


def rigid_from_3_points(x1, x2, x3):
    """
    Construct frames from 3 coordinates

    AlphaFold Algorithm 21, RigidFrom3Points
    """
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = _normalize(v1)
    u2 = v2 - e1 * (e1 * v2).sum(dim=-1)[..., None]
    e2 = _normalize(u2)
    e3 = torch.cross(e1, e2)
    R = torch.stack([e1, e2, e3], dim=-2)
    t = x2
    return (R, t)


def quaternion_to_rotation(q):
    a, b, c, d = q[..., :]
    r1 = torch.stack(
        [a**2 + b**2 - c**2 -d**2, 2*b*c -2*a*d, 2*b*d +2*a*c],
        dim=-1
    )
    r2 = torch.stack(
        [2*b*c + 2*a*d, a**2 - b**2 + c**2 - d**2, 2*c*d - 2*a*b],
        dim=-1
    )
    r3 = torch.stack(
        [2*b*d - 2*a*c, 2*c*d + 2*a*b, a**2 - b**2 - c**2 + d**2],
        dim=-1
    )
    R = torch.stack([r1, r2, r3], dim=-2)
    return R


def update_frame_with_quaternion(frames, quaternion, translation):
    """
    Update a frame by a rotation and translation

    AlphaFold 23, BackboneUpdate
    """
    R_update = quaternion_to_rotation(quaternion)
    R, t = frames
    new_R = torch.bmm(R_update, R)
    new_t = t + translation
    return (new_R, new_t)


def backbone_frames_to_bb_atoms(bb_frames):
    num_nodes = bb_frames.shape[0]
    backbone_ref = torch.as_tensor(backbone_frame_atoms, device=bb_frames.device, dtype=torch.float)
    backbone_ref = backbone_ref[None, ...].expand(num_nodes, -1, -1)
    bb_N = bb_frames.apply(backbone_ref[:, 0])
    bb_CA = bb_frames.apply(backbone_ref[:, 1])
    bb_C = bb_frames.apply(backbone_ref[:, 2])
    bb = torch.stack([bb_N, bb_CA, bb_C], dim=-2)
    return bb
