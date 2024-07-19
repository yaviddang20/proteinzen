from functools import partial

import torch
import torch.nn.functional as F

from torch_cluster import radius

from proteinzen.runtime.loss.atomic.common import atoms_to_angles, atoms_to_torsions
from proteinzen.runtime.loss.utils import vec_norm, _nodewise_to_graphwise
from proteinzen.model.utils.graph import get_data_lens


def compute_bb_n_h(batch_bb,
                   num_nodes,
                   batch_x_mask,
                   ideal_n_h_length=1.008):  # from CHARMM force field
    split_bb = batch_bb.split(num_nodes, dim=0)
    split_x_mask = batch_x_mask.split(num_nodes, dim=0)
    hydrogens = []
    h_masks = []
    for bb, x_mask in zip(split_bb, split_x_mask):
        N = bb[:, 0][1:]  # TODO: we'll ignore the n terminus for now
        CA = bb[:, 1][1:]
        C = bb[:, 2][:-1]
        mask = x_mask[1:] & x_mask[:-1]

        ca2n = (N - CA) / vec_norm(N - CA)[..., None]
        c2n = (N - C) / vec_norm(N - C)[..., None]
        bisector = (ca2n + c2n) / vec_norm(ca2n + c2n)[..., None]
        hydrogen = bisector * ideal_n_h_length + N
        hydrogen[~mask] = 0
        hydrogen = F.pad(hydrogen, (0, 0, 1, 0))
        hydrogens.append(hydrogen)

        h_mask = F.pad(mask, (1, 0), value=False)
        h_masks.append(h_mask)

    return torch.cat(hydrogens, dim=0), torch.cat(h_masks, dim=0)


def get_hbond_params(D, H, A, AB, R):
    """ Compute hbond parameters delta_HA, Theta, Psi, X """
    delta_HA = vec_norm(H - A)
    Theta = atoms_to_angles(
        torch.stack([D, H, A], dim=-2)
    )
    Psi = atoms_to_angles(
        torch.stack([H, A, AB], dim=-2)
    )
    X = atoms_to_torsions(
        torch.stack([H, A, AB, R], dim=-2)
    )
    return delta_HA, Theta, Psi, X


def bb_hbond_loss(batch,
                  denoiser_outputs,
                  distance_cutoff=2.5):  # TODO: this is pretty arbitrary https://proteopedia.org/wiki/index.php/Hydrogen_bonds
    """ Add MSE loss on hbond parameters delta_HA, Theta, Psi, X for bb hbonds which aren't helix-helix"""
    pred_bb = denoiser_outputs['denoised_bb']
    res_data = batch['residue']
    ref_bb = res_data['atom37'][:, (0, 1, 2, 4)]
    num_nodes = get_data_lens(batch, hetero_key='residue')
    res_mask = res_data['res_mask']
    is_helix = res_data['dssp_helix']
    pred_H, _ = compute_bb_n_h(pred_bb, num_nodes, res_mask)
    ref_H, ref_h_mask = compute_bb_n_h(ref_bb, num_nodes, res_mask)


    # ref_N = ref_bb[:, 0]
    # ref_CA = ref_bb[:, 1]
    # ref_C = ref_bb[:, 2]
    # ref_O = ref_bb[:, 3]
    # print(vec_norm(ref_H - ref_N))
    # print(vec_norm(ref_H - ref_CA))
    # print(vec_norm(ref_H - ref_C))
    # print(vec_norm(ref_H - ref_O))

    # print(x_mask)
    # print(ref_h_mask)
    # # torch.set_printoptions(threshold=100000)
    # print(ref_H)
    # ref_N = ref_bb[:, 0]
    # print(ref_N)

    ref_O = ref_bb[:, 3]
    ref_O[~res_mask] = torch.inf

    ref_H[~ref_h_mask] = torch.inf
    edge_index = radius(ref_O, ref_H,
                        r=distance_cutoff,
                        batch_x=res_data.batch,
                        batch_y=res_data.batch,)
                        #max_num_neighbors=3)
    # ref_N = ref_bb[:, 0]
    # dist_mat = vec_norm(ref_O[:, None] - ref_N[None])
    # print(dist_mat.shape)
    # print(dist_mat.sort(dim=-1)[0][:, 0])
    assert edge_index.numel() > 0
    edge_index = torch.stack([edge_index[1], edge_index[0]])  # O is 0, H is 1
    # print(edge_index.shape)
    dst, src = edge_index
    print(is_helix.shape, edge_index.max())
    helix_helix_hbond = is_helix[src] & is_helix[dst]
    edge_index = edge_index[:, ~helix_helix_hbond]
    if edge_index.numel() == 0:
        zeros = torch.zeros(res_data.batch.max() + 1, device=is_helix.device)
        return {
            "delta_mse": zeros,
            "theta_mse": zeros,
            "psi_mse": zeros,
            "X_mse": zeros
        }
    dst, src = edge_index

    data_lens = [0] + num_nodes
    partitions = torch.cumsum(torch.as_tensor(data_lens, device=res_mask.device), dim=0)
    edge_index_batch_id = torch.bucketize(edge_index[1], partitions, right=True) - 1

    ref_N, ref_CA, ref_C, ref_O = ref_bb[src, 0], ref_bb[dst, 1], ref_bb[dst, 2], ref_bb[dst, 3]
    ref_H = ref_H[src]

    pred_N, pred_CA, pred_C, pred_O = pred_bb[src, 0], pred_bb[dst, 1], pred_bb[dst, 2], pred_bb[dst, 3]
    pred_H = pred_H[src]

    h_mask = ref_h_mask[src]


    pred_delta_HA, pred_Theta, pred_Psi, pred_X = get_hbond_params(
        D=pred_N,
        H=pred_H,
        A=pred_O,
        AB=pred_C,
        R=pred_CA
    )
    ref_delta_HA, ref_Theta, ref_Psi, ref_X = get_hbond_params(
        D=ref_N,
        H=ref_H,
        A=ref_O,
        AB=ref_C,
        R=ref_CA
    )

    # print(
    #     list(map(lambda x: torch.isnan(x).any(),
    #     [pred_delta_HA, pred_Theta, pred_Psi, pred_X]
    #     ))
    # )
    # print(
    #     list(map(lambda x: torch.isnan(x).any(),
    #     [ref_delta_HA, ref_Theta, ref_Psi, ref_X]
    #     ))
    # )

    mse_delta_HA = torch.square(pred_delta_HA - ref_delta_HA)
    mse_Theta = vec_norm(pred_Theta - ref_Theta)
    mse_Psi = vec_norm(pred_Psi - ref_Psi)
    mse_X = vec_norm(pred_X - ref_X)

    # print(
    #     list(map(lambda x: torch.isnan(x).any(),
    #     [mse_delta_HA, mse_Theta, mse_Psi, mse_X]
    #     ))
    # )

    mse_delta_HA, mse_Theta, mse_Psi, mse_X = map(
        partial(
            _nodewise_to_graphwise,
            batch=edge_index_batch_id,
            node_elem_mask=h_mask
        ),
        [mse_delta_HA, mse_Theta, mse_Psi, mse_X]
    )
    # print(h_mask)

    # print(
    #     list(map(lambda x: torch.isnan(x).any(),
    #     [mse_delta_HA, mse_Theta, mse_Psi, mse_X]
    #     ))
    # )

    return {
        "delta_mse": mse_delta_HA,
        "theta_mse": mse_Theta,
        "psi_mse": mse_Psi,
        "X_mse": mse_X
    }
