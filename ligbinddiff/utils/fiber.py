""" Utils for dealing with fibers """


import torch
from se3_transformer.model.fiber import Fiber


def gen_nmax_fiber(n_max):
    return Fiber({
        l: n_max - l + 1
        for l in range(n_max+1)
    })


def gen_compact_nmax_fiber(n_max):
    fiber_dict = {l: 0 for l in range(n_max+1)}
    n_levels = {l: [] for l in range(n_max+1)}
    for l in fiber_dict.keys():
        for n in range(n_max+1):
            if n < l: continue
            if (n-l) % 2 == 0:
                fiber_dict[l] = fiber_dict[l] + 1
                n_levels[l].append(n)
    return Fiber(fiber_dict)


def gen_full_n_channel_fiber(l_max, n_channels):
    return Fiber({
        l: n_channels
        for l in range(l_max + 1)
    })


def nl_to_fiber(Z):
    """ Convert from Zernike coeffs to a Fiber dict (no n) """
    fiber_dict = {}
    for (n,l), coeffs in Z.items():
        if l not in fiber_dict:
            fiber_dict[l] = []
        fiber_dict[l].append((n, coeffs))

    for l, coeff_list in fiber_dict.items():
        coeff_list = sorted(coeff_list, key=lambda p: p[0], reverse=True)  # order from largest to smallest n
        coeff_list = list(zip(*coeff_list))[1]  # take only coeffs
        fiber_dict[l] = torch.cat(coeff_list, dim=-2)  # fuse coeffs into one tensor

    return fiber_dict


def nl_to_compact_fiber(Z):
    """ Convert from Zernike coeffs to a Fiber dict (no n) """
    fiber_dict = {}
    for (n,l), coeffs in Z.items():
        if (coeffs == 0.).all():
            continue
        if l not in fiber_dict:
            fiber_dict[l] = []
        fiber_dict[l].append((n, coeffs))

    for l, coeff_list in fiber_dict.items():
        coeff_list = sorted(coeff_list, key=lambda p: p[0], reverse=True)  # order from largest to smallest n
        coeff_list = list(zip(*coeff_list))[1]  # take only coeffs
        fiber_dict[l] = torch.cat(coeff_list, dim=-2)  # fuse coeffs into one tensor

    return fiber_dict


def fiber_to_nl(fiber_dict, n_channels=1):
    """ Convert from Fiber dict (no n) to Zernike coeffs.
        n_max is inferred from the fiber structure """
    n_max_p1_multiple = fiber_dict[0].shape[-2]
    assert n_max_p1_multiple % n_channels == 0
    n_max = (n_max_p1_multiple // n_channels) - 1

    nl_dict = {}
    for l, coeffs in fiber_dict.items():
        for i, n in enumerate(range(n_max, l-1, -1)):
            nl_dict[(n,l)] = coeffs[..., i:i+n_channels, :]

    return nl_dict


def compact_fiber_to_nl(fiber_dict, n_channels=1):
    """ Convert from Fiber dict (no n) to Zernike coeffs.
        n_max is inferred from the fiber structure """
    n_max = max([int(l) for l in fiber_dict.keys()])

    nl_dict = {}
    for l, coeffs in fiber_dict.items():
        for i, n in enumerate(range(n_max, l-1, -1)):
            if (n - l) % 2 == 0:
                nl_dict[(n,l)] = coeffs[..., i:i+n_channels, :]

    return nl_dict


def rand_fiber_density(num_nodes, n_max, device='cpu'):
    fiber = gen_nmax_fiber(n_max)
    density = {}
    for l, num_vecs in fiber.items():
        m_tot = 2*l+1
        density[l] = torch.randn((num_nodes, num_vecs, m_tot), device=device)
    return density


def rand_compact_fiber_density(num_nodes, n_max, device='cpu'):
    fiber = gen_compact_nmax_fiber(n_max)
    density = {}
    for l, num_vecs in fiber.items():
        m_tot = 2*l+1
        density[l] = torch.randn((num_nodes, num_vecs, m_tot), device=device)
    return density
