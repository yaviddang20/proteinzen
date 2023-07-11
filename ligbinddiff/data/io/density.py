""" Methods for outputting density files for visualization """

import mrcfile
import numpy as np
import torch
import tqdm

import matplotlib.pyplot as plt


ATOM_CHANNELS = ["C", "N", "O", "S"]


def density_to_mrc(rho,
                   voxel_size: float,
                   coord_bounds: tuple,
                   out_prefix="out",
                   overwrite=True,
                   memory_mode=0):
    # sample the density on a grid
    coord_min, coord_max = coord_bounds
    steps = int((coord_max - coord_min) / voxel_size)
    x = torch.linspace(coord_min, coord_max, steps)
    y = torch.linspace(coord_min, coord_max, steps)
    z = torch.linspace(coord_min, coord_max, steps)
    z, y, x = torch.meshgrid(z, y, x)
    zyx = torch.stack([z, y, x], dim=-1)
    xyz = torch.stack([x, y, z], dim=-1)
    if memory_mode == 1:
        print("Iterate over x")
        density = [rho(xyz[i]).numpy(force=True) for i in tqdm.tqdm(range(steps))]
        density = np.array(density)
    elif memory_mode == 2:
        density = []
        print("Iterate over x")
        for i in tqdm.tqdm(range(steps)):
            print("Iterate over y")
            density_row = []
            for j in tqdm.tqdm(range(steps)):
                density_row.append(rho(xyz[i][j]).numpy(force=True))
            density.append(density_row)
        density = np.array(density)
    elif memory_mode == 0:
        density = rho(xyz)
        density = density.numpy(force=True)
    else:
        raise ValueError("memory_mode must be either 0, 1, or 2")
    print(density.shape)

    # fig = plt.figure(figsize=plt.figaspect(1.))
    # ax = fig.add_subplot(projection='3d')

    # nonzero = torch.from_numpy(density != 0)
    # print(z.shape, y.shape, x.shape, nonzero.shape)
    # # p = ax.scatter(z[nonzero], y[nonzero], x[nonzero], c=density[nonzero], alpha=0.5)
    # p = ax.scatter(x[nonzero], y[nonzero], z[nonzero], c=density[nonzero], alpha=0.5)
    # fig.colorbar(p)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    num_channels = density.shape[-1]
    density = density.astype(np.float32)
    for i in range(num_channels):
        if i == 0 and num_channels == 1:
            atom_type = "all"
        else:
            atom_type = ATOM_CHANNELS[i]
        out_file = f"{out_prefix}_{atom_type}.mrc"
        channel_density = density[..., i]
        mrc = mrcfile.new(out_file, overwrite=overwrite)
        mrc.set_data(channel_density)
        mrc.mapc = 1
        mrc.mapr = 2
        mrc.maps = 3
        mrc.voxel_size = voxel_size
        mrc.header["origin"] = coord_min
        mrc.close()

    # return ax

    return zyx, density


if __name__ == '__main__':

    import os, time, gzip, urllib, json
    import mmtf
    from collections import defaultdict

    def download_cached(url, target_location):
        """ Download with caching """
        target_dir = os.path.dirname(target_location)
        if not os.path.isfile(target_location):
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # Use MMTF for speed
            response = urllib.request.urlopen(url)
            size = int(float(response.headers['Content-Length']) / 1e3)
            # print('Downloading {}, {} KB'.format(target_location, size))
            with open(target_location, 'wb') as f:
                f.write(response.read())
        return target_location

    def mmtf_fetch(pdb, cache_dir='cath/mmtf/'):
        """ Retrieve mmtf record from PDB with local caching """
        mmtf_file = cache_dir + pdb + '.mmtf.gz'
        url = 'http://mmtf.rcsb.org/v1.0/full/' + pdb + '.mmtf.gz'
        mmtf_file = download_cached(url, mmtf_file)
        mmtf_record = mmtf.parse_gzip(mmtf_file)
        return mmtf_record

    def mmtf_parse(pdb_id, chain, target_atoms = ['N', 'CA', 'C', 'O'], cache_dir='mmtf/'):
        """ Parse mmtf file to extract C-alpha coordinates """
        # MMTF traversal derived from the specification
        # https://github.com/rcsb/mmtf/blob/master/spec.md
        A = mmtf_fetch(pdb_id, cache_dir=cache_dir)

        # Build a dictionary
        mmtf_dict = {}
        mmtf_dict['seq'] = []
        mmtf_dict['coords'] = {code:[] for code in target_atoms}

        # Get chain of interest from Model 0
        model_ix, chain_ix, group_ix, atom_ix = 0, 0, 0, 0
        target_chain_ix, target_entity = next(
            (i, entity) for entity in A.entity_list for i in entity['chainIndexList']
            if entity['type'] == 'polymer' and A.chain_name_list[i] == chain
        )

        # Traverse chains
        num_chains = A.chains_per_model[model_ix]
        mmtf_dict['num_chains'] = num_chains
        for ii in range(num_chains):
            chain_name = A.chain_name_list[chain_ix]

            # Chain of interest?
            if chain_ix == target_chain_ix:
                mmtf_dict['seq'] = target_entity['sequence']
                coords_null = [[float('nan')] * 3] * len(mmtf_dict['seq'])
                mmtf_dict['coords'] = {code : list(coords_null) for code in target_atoms}

                # Traverse groups, storing data
                chain_group_count = A.groups_per_chain[chain_ix]
                for jj in range(chain_group_count):
                    group = A.group_list[A.group_type_list[group_ix]]

                    # Extend coordinate data
                    seq_ix = A.sequence_index_list[group_ix]
                    for code in target_atoms:
                        if code in group['atomNameList']:
                            A_ix = atom_ix + group['atomNameList'].index(code)
                            xyz = [A.x_coord_list[A_ix], A.y_coord_list[A_ix], A.z_coord_list[A_ix]]
                            mmtf_dict['coords'][code][seq_ix] = xyz

                    group_atom_count = len(group['atomNameList'])
                    atom_ix += group_atom_count
                    group_ix += 1
                chain_ix += 1

            else:
                # Traverse groups
                chain_group_count = A.groups_per_chain[chain_ix]
                for jj in range(chain_group_count):
                    group = A.group_list[A.group_type_list[group_ix]]
                    atom_ix += len(group['atomNameList'])
                    group_ix += 1
                chain_ix += 1

        return mmtf_dict


    entry = mmtf_parse(
        #"2nd3",
        "6A5J",
        "A",
        target_atoms=[
            "N", "CA", "C", "O",
            "CB",
            "CG", "CG1", "CG2", "OG", "OG1", "SG",
            "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
            "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
            "CZ", "CZ2", "CZ3", "NZ",
            "CH2", "NH1", "NH2", "OH"
        ])
    # entry['name'] = "2nd3.A"#"6a5j.A"
    entry['name'] = "6a5j.A"

    from ligbinddiff.utils.atom_reps import atom37_atom_label, atom37_to_atom14, atom14_to_atom91, letter_to_num, atom91_atom_masks

    coords = entry['coords']
    atom37 = list(zip(
        *[coords[atom] for atom in atom37_atom_label]
    ))
    atom14, atom14_mask = atom37_to_atom14(entry['seq'], np.array(atom37))
    entry['coords'] = atom14
    entry['coords_mask'] = atom14_mask

    def ideal_virtual_Cb(X_bb):
        # from ProteinMPNN paper computation of Cb
        N, Ca, C = [X_bb[..., i, :] for i in range(3)]
        b = Ca - N
        c = C- Ca
        a = torch.cross(b, c)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
        return Cb

    def Cb_2x_center(X_bb):
        # from ProteinMPNN paper computation of Cb
        N, Ca, C = [X_bb[..., i, :] for i in range(3)]
        b = Ca - N
        c = C- Ca
        a = torch.cross(b, c)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
        Cb_vec = Cb - Ca
        return Cb + Cb_vec

    from ligbinddiff.utils.zernike import ZernikeTransform

    points = torch.as_tensor(entry['coords'], dtype=torch.float32)
    points_mask = torch.from_numpy(entry['coords_mask']).any(dim=-1)
    zt = ZernikeTransform(5, 8)
    X_ca = torch.from_numpy(entry['coords'][:, 1])
    X_cb = ideal_virtual_Cb(torch.from_numpy(entry["coords"]))
    print(X_cb.shape)
    print(points.shape)
    channel_atoms = np.array([
        atom91_atom_masks[atom] for atom in ['C', 'N', 'O', 'S']
    ])
    channel_atoms = torch.as_tensor(channel_atoms)

    atom91, atom91_mask = atom14_to_atom91(entry['seq'], atom14)
    atom91 = torch.as_tensor(atom91)
    atom91_mask = torch.as_tensor(atom91_mask).any(dim=-1)
    Z = zt.forward_transform(atom91, X_cb, atom91_mask, point_value=channel_atoms)

    from ligbinddiff.utils.type_l import type_l_randn_like, type_l_sub, type_l_apply, type_l_add, type_l_mult
    from ligbinddiff.utils.fiber import compact_fiber_to_nl


    def zernike_coeff_loss(ref_density, pred_density, mask):
        """ Invariant MSE loss on Zernike coeffs
        Average of norms of the difference between all type-l vectors """
        diff = type_l_sub(ref_density, pred_density)
        square_diff = type_l_apply(torch.square, diff)
        loss = 0
        numel = 0
        for (n, l), elems in square_diff.items():
            if (n - l) % 2 == 1:
                continue
            mags = elems.sum(dim=-1).sqrt()
            loss = loss + mags.sum()
            numel += mags.numel()
            # print((n, l), mags.sum()/mags.numel())

        #print(numel)

        return loss / numel

    noise = type_l_randn_like(Z)

    for alpha_t in torch.linspace(0, 1, 100):
        noised_density = type_l_add(
            type_l_mult(1-alpha_t, Z),
            type_l_mult(alpha_t, noise)
        )
        if zernike_coeff_loss(noised_density, Z, mask=None) > 0.3:
            print(zernike_coeff_loss(noised_density, Z, mask=None))
            Z = noised_density
            break

    rho = zt.back_transform(Z, X_cb)

    min_coord = int(points[~points_mask].min() - 1)
    max_coord = int(points[~points_mask].max() + 1)

    density_to_mrc(rho, voxel_size=0.5, coord_bounds=(min_coord, max_coord), memory_mode=1, out_prefix=entry['name'])
    # points = points[~points_mask] # - X_ca
    # print(points)
    # ax.scatter(points[..., 0], points[..., 1], points[..., 2], c="black", alpha=1)
    # plt.show()
