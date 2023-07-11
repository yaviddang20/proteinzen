""" Inference loops and related """
import tqdm

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from ligbinddiff.utils.type_l import type_l_sub, type_l_apply


def zernike_coeff_loss(ref_density, pred_density, mask):
    """ Invariant MSE loss on Zernike coeffs
    Average of norms of the difference between all type-l vectors """

    # only compute residues where all atoms are present
    ref_density = type_l_apply(lambda x: x[~mask], ref_density)
    pred_density = type_l_apply(lambda x: x[~mask], pred_density)

    diff = type_l_sub(ref_density, pred_density)
    square_diff = type_l_apply(torch.square, diff)
    loss = 0
    numel = 0
    for _, elems in square_diff.items():
        tmp = elems.sum(dim=-1).sqrt()
        loss = loss + tmp.sum()
        numel += elems.numel()

    #print(numel)

    return loss / numel


def seq_recov_loss(ref_seq, seq_logits, mask):
    """ CCE loss on logits """
    ref_seq = ref_seq[~mask]
    seq_logits = seq_logits[~mask]
    return F.cross_entropy(seq_logits, ref_seq)


def seq_recov(ref_seq, seq_logits, mask):
    """ CCE loss on logits """
    ref_seq = ref_seq[~mask]
    seq_logits = seq_logits[~mask]

    pred_seq = seq_logits.argmax(dim=-1)
    return (pred_seq == ref_seq).type(torch.float).mean()


def atom91_rmsd_loss(ref_atom91, pred_atom91, atom91_mask):
    """ RMSD on relevant atom91 atoms """
    diff = ref_atom91[~atom91_mask] - pred_atom91[~atom91_mask]
    sd = torch.square(diff)
    return sd.mean().sqrt()


def cath_inference_loop(diffuser, dataloader, num_steps, device):
    epoch_denoising_loss = []
    epoch_seq_loss = []
    epoch_atom91_rmsd = []
    epoch_seq_recov = []

    for batch in (pbar := tqdm.tqdm(dataloader)):
        batch = batch.to(device)
        node_s = batch.ndata['backbone_s']
        node_v = batch.ndata['backbone_v']
        edge_s = batch.edata['edge_s']
        edge_v = batch.edata['edge_v']
        seq = batch.ndata['seq']
        x_mask = batch.ndata['x_mask']
        atom91 = batch.ndata['atom91']
        atom91_mask = batch.ndata['atom91_mask']

        node_features = {
            0: node_s.unsqueeze(-1),
            1: node_v
        }
        edge_features = {
            0: edge_s.unsqueeze(-1),
            1: edge_v
        }

        denoised_density, seq_logits, pred_atom91 = zip(*diffuser.sample(node_features, edge_features, batch, steps=num_steps, return_trajectory=True))

        int_keys = [key for key in batch.ndata.keys() if isinstance(key, int)]
        density_dict = {
            l: batch.ndata[l]
            for l in int_keys
        }

        denoising_loss = [zernike_coeff_loss(density_dict, d, x_mask) for d in denoised_density[1:]]
        seq_loss = [seq_recov_loss(seq, s, x_mask) for s in seq_logits[1:]]
        per_seq_recov = [seq_recov(seq, s, x_mask) for s in seq_logits[1:]]
        print(denoised_density, seq_logits)
        for d, s, r in zip(denoising_loss, seq_loss, per_seq_recov):
            print(d, s, r)
        exit()
        atom91_rmsd = [atom91_rmsd_loss(atom91, pred_atom91, atom91_mask)]

        epoch_denoising_loss.append(denoising_loss.item())
        epoch_seq_loss.append(seq_loss.item())
        epoch_atom91_rmsd.append(atom91_rmsd.item())
        epoch_seq_recov.append(per_seq_recov.item())
        pbar.set_description((
            f"Denoising loss {denoising_loss.item():.4f}, seq loss {seq_loss.item():.4f}, atom91 rmsd {atom91_rmsd.item():.4f}, % seq recov {per_seq_recov.item():.4f}"))

    return np.mean(epoch_denoising_loss), np.mean(epoch_seq_loss), np.mean(epoch_atom91_rmsd), np.mean(epoch_seq_recov)
