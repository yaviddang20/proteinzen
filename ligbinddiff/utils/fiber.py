""" Utils for dealing with fibers """


import torch
from se3_transformer.model.fiber import Fiber


def gen_nmax_fiber(n_max):
    return Fiber({
        l: n_max - l + 1
        for l in range(n_max+1)
    })

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
        fiber_dict[l] = torch.stack(coeff_list, dim=-2)  # fuse coeffs into one tensor

    return fiber_dict


def fiber_to_nl(fiber_dict):
    """ Convert from Fiber dict (no n) to Zernike coeffs.
        n_max is inferred from the fiber structure """
    n_max = fiber_dict[0].shape[-2]
    nl_dict = {}
    for l, coeffs in fiber_dict.items():
        for i, n in enumerate(range(n_max, l-1, -1)):
            nl_dict[(n,l)] = coeffs[..., i, :]

    return nl_dict


def rand_fiber_density(num_nodes, n_max, device='cpu'):
    fiber = gen_nmax_fiber(n_max)
    density = {}
    for l, num_vecs in fiber.items():
        m_tot = 2*l+1
        density[l] = torch.randn((num_nodes, num_vecs, m_tot), device=device)
    return density
