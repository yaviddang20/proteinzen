""" utils for SO3 embeddings """
import torch

from ligbinddiff.model.modules.equiformer_v2.so3 import SO3_Embedding


def type_l_to_so3(type_l):
    dtype = list(type_l.values())[0].dtype
    device = list(type_l.values())[0].device
    num_nodes = list(type_l.values())[0].shape[0]
    max_channels = max([vecs.shape[-2] for vecs in type_l.values()])
    lmax = max(type_l.keys())
    so3 = SO3_Embedding(
        0,
        [lmax],
        max_channels,
        dtype=dtype,
        device=device
    )

    embedding = torch.zeros(
        num_nodes,
        int((lmax + 1)**2),
        max_channels,
        device=device,
        dtype=dtype
    )
    offset = 0
    for l in range(lmax+1):
        num_coeffs = 2*l+1
        if l in type_l.keys():
            vecs = type_l[l]  # n_nodes x n_channels x n_m
            num_channels = vecs.shape[-2]
            embedding[:, offset:offset + num_coeffs, :num_channels] = vecs.transpose(-1, -2)
        offset += int(num_coeffs)
    so3.set_embedding(embedding)
    return so3

def density_to_so3(density, num_channels=4):
    l0 = density[0]
    nmax = max(density.keys())
    lmax_list = list(range(nmax, -1, -2))
    dtype = l0.dtype
    device = l0.device
    num_nodes = l0.shape[0]
    so3 = SO3_Embedding(
        0,
        lmax_list,
        1,
        dtype=dtype,
        device=device
    )

    embedding = torch.zeros(
        num_nodes,
        sum([int((lmax + 1)**2) for lmax in lmax_list]),
        num_channels,
        device=device,
        dtype=dtype
    )
    offset = 0
    for idx, lmax in enumerate(lmax_list):
        for l in range(lmax+1):
            num_coeffs = 2*l+1
            vecs = density[l]  # n_nodes x n_channels x n_m
            select_vec = vecs[:, idx:idx+num_channels]  # n_nodes x 1 x n_m
            embedding[:, offset:offset + num_coeffs] = select_vec.transpose(-1, -2)
            offset += int(num_coeffs)
    so3.set_embedding(embedding)
    return so3

def so3_to_density(so3):
    lmax_list = so3.lmax_list
    density = {l: [] for l in range(max(lmax_list)+1)}
    offset = 0
    for lmax in lmax_list:
        for l in range(lmax+1):
            num_coeffs = 2*l+1
            select_vec = so3.embedding[:, offset:offset + num_coeffs].transpose(-1, -2)
            density[l].append(select_vec)  # n_nodes x 1 x n_m
            offset += int(num_coeffs)
    for l in density.keys():
        density[l] = torch.cat(density[l], dim=-2)

    return density
