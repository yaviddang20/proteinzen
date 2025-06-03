""" A differentiable loss inspired by the Rosetta Holes score """

import torch
import torch_geometric.utils as pygu
from torch_geometric.nn import radius, radius_graph, nearest
from e3nn import o3

from proteinzen.data.openfold.residue_constants import restype_atom14_radius
from ..utils import _nodewise_to_graphwise

def calc_overlap_mask(atom_i, atom_q, r_i, r_q, point):
    """_summary_

    Args:
        atom_i (torch.Tensor): (n_atom_pair, 3)
        atom_q (torch.Tensor): (n_atom_pair, 3)
        r_i (torch.Tensor): (n_atom_pair,)
        r_q (torch.Tensor): (n_atom_pair,)
        point (torch.Tensor): surface points for atom_i (n_atom_pair, n_radii, n_points, 3)

    Returns:
        _type_: _description_
    """
    dist_to_atom_i = torch.linalg.vector_norm(point - atom_i[..., None, None, :], dim=-1)
    dist_to_atom_q = torch.linalg.vector_norm(point - atom_q[..., None, None, :], dim=-1)
    # print(point, atom_i, atom_q)
    return (dist_to_atom_i - r_i[..., None, None]) < (dist_to_atom_q - r_q[..., None, None])


def generate_as_ri(radii, resolution=13, debug_2d=False):
    if debug_2d:
        angles = torch.linspace(0, 2*torch.pi, steps=resolution * resolution + 1)[:-1]
        xyz = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        xyz = xyz.to(radii.device)
        xyz = xyz[None, None] * radii[:, :, None, None]
        return xyz
    else:
        alpha, beta = o3.s2_grid(resolution, resolution)
        # (resolution x resolution,)
        alpha_beta_grid = torch.stack(
            [
                torch.tile(alpha[:, None], (1, beta.shape[0])),
                torch.tile(beta[None, :], (alpha.shape[0], 1)),
            ],
            dim=0
        )
        alpha_beta_grid = alpha_beta_grid.flatten(-2, -1)
        # (resolution x resolution, 3)
        xyz = o3.angles_to_xyz(alpha_beta_grid[0], alpha_beta_grid[1]).to(radii.device)
        # (n_atoms, n_radii, resolution x resolution, 3)
        xyz = xyz[None, None] * radii[:, :, None, None]
        return xyz


def calc_csa_ri(
        atoms,
        atom_radii,
        atom_batch,
        radii=torch.linspace(0.1, 3.0, 30),
        resolution=13,
        random_rots=False,
        debug_2d=False
    ):
    radii = radii.to(atoms.device)
    # use a radius graph to reduce computation
    max_radius = radii.max().item()
    max_atom_radius = atom_radii.max().item()

    with torch.no_grad():
        # (n_atoms, n_radii, resolution x resolution, 3)
        shell_radii = radii[None] + atom_radii[:, None]
        point_shells = generate_as_ri(shell_radii, resolution, debug_2d=debug_2d)
        if random_rots:
            rots = o3.rand_matrix(atoms.shape[0], radii.shape[0])
            point_shells = torch.einsum("nsij,nsrj->nsri", rots, point_shells)
        # (n_atoms, n_radii, resolution x resolution)
        points_distances = torch.linalg.vector_norm(point_shells, dim=-1)

    # (n_atoms, n_radii, resolution x resolution, 3)
    atom_points = atoms[..., None, None, :] + point_shells

    with torch.no_grad():
        edge_index = radius_graph(atoms, max_radius*2 + max_atom_radius*2, atom_batch, max_num_neighbors=128)
        # (n_atom_pairs, n_radii, resolution x resolution, 3)
        atom_points_i = atom_points[edge_index[1]]
        # (n_atom_pairs, 3)
        atoms_i = atoms[edge_index[1]]
        atoms_q = atoms[edge_index[0]]
        # (n_atom_pairs)
        atoms_i_radii = atom_radii[edge_index[1]]
        atoms_q_radii = atom_radii[edge_index[0]]
        # (n_atom_pairs, n_radii, resolution x resolution)
        atom_points_masks = calc_overlap_mask(atoms_i, atoms_q, atoms_i_radii, atoms_q_radii, atom_points_i)
        # (n_atoms, n_radii, resolution x resolution)
        atom_points_mask = pygu.scatter(
            atom_points_masks.float(),
            edge_index[1],
            dim_size=points_distances.shape[0],
            reduce='mul'
        ).bool()

        points_distances = points_distances * atom_points_mask
        # (n_atoms, n_radii x resolution x resolution)
        points_distances = points_distances.flatten(-2, -1)
        # (n_atoms, ), (n_atoms, )
        points_max_distance, point_idx = torch.max(points_distances, dim=-1)
        # (n_atoms, )
        points_radius = points_max_distance - atom_radii

    # (n_atoms, n_radii x resolution x resolution, 3)
    flat_atom_points = atom_points.flatten(-3, -2)
    # (n_atoms, 3)
    max_points = torch.gather(flat_atom_points, -2, point_idx[..., None, None].expand([-1, -1, flat_atom_points.shape[-1]])).squeeze(-2)
    cavity_batch = atom_batch

    # in this manner, max_points should have gradients attached but points_radius won't
    return max_points, points_radius, cavity_batch


def generate_water_probes(radii, resolution=13, debug_2d=False):
    if debug_2d:
        angles = torch.linspace(0, 2*torch.pi, steps=resolution * resolution + 1)[:-1]
        xyz = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        xyz = xyz.to(radii.device)
        xyz = xyz[None] * radii[:, None, None]
        return xyz
    else:
        # radii: (n_cavities,)
        alpha, beta = o3.s2_grid(resolution, resolution)
        # (resolution x resolution,)
        alpha_beta_grid = torch.stack(
            [
                torch.tile(alpha[:, None], (1, beta.shape[0])),
                torch.tile(beta[None, :], (alpha.shape[0], 1)),
            ],
            dim=0
        )
        alpha_beta_grid = alpha_beta_grid.flatten(-2, -1)
        # (resolution x resolution, 3)
        xyz = o3.angles_to_xyz(alpha_beta_grid[0], alpha_beta_grid[1]).to(radii.device)
        # (n_cavities, resolution x resolution, 3)
        xyz = xyz[None] * radii[:, None, None]
        return xyz


def is_cavity_buried(
        atoms,
        atom_batch,
        atom_radii,
        cavities,
        cavity_batch,
        cavity_radii,
        resolution=13,
        random_rots=False,
        debug_2d=False
):
    water_probes = generate_water_probes(cavity_radii + 1.4, resolution=resolution, debug_2d=debug_2d) + cavities[..., None, :]
    water_probes_batch = cavity_batch[..., None].tile((1, resolution*resolution))
    source_cavity = torch.arange(cavities.shape[0], device=atoms.device)
    source_cavity_batch = source_cavity[..., None].tile((1, resolution*resolution)).view(-1)

    flat_water_probes = water_probes.view(-1, atoms.shape[-1])
    flat_water_probes_batch = water_probes_batch.view(-1)
    atom_edge_index = radius(atoms, flat_water_probes, r=1.4+atom_radii.max(), batch_x=atom_batch, batch_y=flat_water_probes_batch)
    cavity_edge_index = radius(cavities, flat_water_probes, r=1.4+cavity_radii.max(), batch_x=cavity_batch, batch_y=flat_water_probes_batch)
    # print(atom_edge_index[0].max(), atom_edge_index[1].max())
    # print(cavity_edge_index[0].max(), cavity_edge_index[1].max())

    probe_dist_to_atom = torch.linalg.vector_norm(atoms[atom_edge_index[1]] - flat_water_probes[atom_edge_index[0]], dim=-1)
    probe_dist_to_cavity = torch.linalg.vector_norm(cavities[cavity_edge_index[1]] - flat_water_probes[cavity_edge_index[0]], dim=-1)

    # True if clash, False if no clash
    probe_atom_clash = probe_dist_to_atom < 1.4 + atom_radii[atom_edge_index[1]]
    probe_cavity_clash = probe_dist_to_cavity < 1.4 + cavity_radii[cavity_edge_index[1]]
    # 1 if no clash, 0 if clash
    probe_atom_clash_status = pygu.scatter(
        (~probe_atom_clash).float(),
        index=atom_edge_index[0],
        dim_size=flat_water_probes.shape[0],
        reduce='mul'
    ).bool()
    probe_cavity_clash_status = pygu.scatter(
        (~probe_cavity_clash).float(),
        index=cavity_edge_index[0],
        dim_size=flat_water_probes.shape[0],
        reduce='mul'
    ).bool()
    # 1 if no clash, 0 if clash
    probe_clash_status = probe_atom_clash_status * probe_cavity_clash_status

    # 1 if buried, 0 if exposed
    cavity_buried = pygu.scatter(
        (~probe_clash_status).float(),
        index=source_cavity_batch,
        dim_size=cavities.shape[0],
        reduce='mul'
    ).bool()

    return cavity_buried


def get_buried_cavities(
        atoms,
        atom_radii,
        atom_batch,
        radii=torch.linspace(0.1, 3.0, 30),
        resolution=12,
        random_rots=False,
        n_prune_iter=20,
        debug_2d=False
):
    radii = radii.to(atoms.device)
    cavities, _cavity_radii, cavity_batch = calc_csa_ri(
        atoms,
        atom_radii,
        atom_batch,
        radii=radii,
        resolution=resolution,
        random_rots=random_rots,
        debug_2d=debug_2d
    )
    # print(cavities, _cavity_radii)

    print(_cavity_radii.numel())
    with torch.no_grad():
        for _ in range(n_prune_iter):
            if cavities.numel() == 0:
                break
            cavity_buried_status = is_cavity_buried(
                atoms,
                atom_batch,
                atom_radii,
                cavities,
                cavity_batch,
                _cavity_radii,
                resolution=resolution,
                random_rots=random_rots,
                debug_2d=debug_2d
            )
            mask = (cavity_buried_status == 1)
            if mask.all():
                break

            cavities = cavities[mask]
            cavity_batch = cavity_batch[mask]
            _cavity_radii = _cavity_radii[mask]
            print(cavity_batch.numel())


    if cavities.numel() == 0:
        return None, None, None

    else:
        # we have to filter out the batches without buried cavities
        # so we can use nearest()
        batch_x_no_cavities = set(atom_batch.unique().long().tolist()) - set(cavity_batch.unique().long().tolist())
        print(batch_x_no_cavities)
        remove_batch = list(sorted(batch_x_no_cavities))
        remove_atom_mask = atom_batch[..., None] == torch.as_tensor(remove_batch, device=atom_batch.device)[None]
        remove_atom_mask = remove_atom_mask.any(dim=-1)
        filtered_atoms = atoms[~remove_atom_mask ]
        filtered_atom_radii = atom_radii[~remove_atom_mask]
        filtered_atom_batch = atom_batch[~remove_atom_mask]

        print(filtered_atoms.shape, filtered_atom_batch.shape)

        # now we compute cavity radii with gradients attached
        # so we can use this in a loss function
        dist_edge_index = nearest(cavities, filtered_atoms, batch_x=cavity_batch, batch_y=filtered_atom_batch)
        # print(filtered_atoms, cavities, dist_edge_index)
        cavity_radii = torch.linalg.vector_norm(
            filtered_atoms[dist_edge_index] - cavities,
            dim=-1
        ) - filtered_atom_radii[dist_edge_index]

        return cavities, cavity_radii, cavity_batch


def buried_cavity_loss(
        batch,
        model_outputs
):
    res_data = batch['residue']
    atom14 = res_data['atom14']
    atom14_mask = res_data['atom14_mask'].bool()
    seq = res_data['seq']

    seq_logits = model_outputs['decoded_seq_logits']
    pred_seq = seq_logits.argmax(dim=-1)
    pred_atom14 = model_outputs['decoded_atom14']
    pred_atom14_mask = model_outputs['decoded_atom14_mask'].bool()
    pred_atom14_gt_seq = model_outputs['decoded_atom14_gt_seq']

    atoms = pred_atom14_gt_seq[atom14_mask]
    atom_batch = torch.arange(
        atom14.shape[0],
        device=atom14.device
    )[..., None].tile((1, 14))[atom14_mask]
    atom_radii = torch.as_tensor(restype_atom14_radius, device=atom14.device)[seq][atom14_mask]

    # atoms = atom14[atom14_mask]
    # atom_batch = torch.arange(
    #     atom14.shape[0],
    #     device=atom14.device
    # )[..., None].tile((1, 14))[atom14_mask]
    # atom_radii = torch.as_tensor(restype_atom14_radius, device=atom14.device)[seq][atom14_mask]

    _, cavity_radii, cavity_batch = get_buried_cavities(
        atoms,
        atom_radii,
        atom_batch,
    )

    if cavity_radii is not None and cavity_batch is not None:
        cavity_volume = 4/3 * torch.pi * (cavity_radii ** 3)
        total_cavity_volume = _nodewise_to_graphwise(cavity_volume, cavity_batch, torch.ones_like(cavity_batch, dtype=torch.bool), reduction='sum')
        normed_cavity_volume = total_cavity_volume / pygu.scatter(
            torch.ones_like(res_data.batch),
            res_data.batch,
            reduce='sum'
        )
    else:
        normed_cavity_volume = torch.zeros_like(batch['t'])

    return {
        "cavity_volume_loss": normed_cavity_volume
    }





