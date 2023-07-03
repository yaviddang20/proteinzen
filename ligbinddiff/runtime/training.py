""" Training loops and related """
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


def atom91_rmsd_loss(ref_atom91, pred_atom91, atom91_mask):
    """ RMSD on relevant atom91 atoms """
    diff = ref_atom91[~atom91_mask] - pred_atom91[~atom91_mask]
    sd = torch.square(diff)
    return sd.mean().sqrt()


def cath_train_loop(diffuser, dataloader, optimizer, fabric, train=True):
    epoch_loss = []
    epoch_denoising_loss = []
    epoch_seq_loss = []
    epoch_atom91_rmsd = []

    for batch in (pbar := tqdm.tqdm(dataloader)):
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
        int_keys = [key for key in batch.ndata.keys() if isinstance(key, int)]
        density_dict = {
            l: batch.ndata[l]
            for l in int_keys
        }

        # for vecs in density_dict.values():
        #     if torch.isnan(vecs).any():
        #         print(density_dict)

        optimizer.zero_grad()
        noised_density_dict, ts = diffuser.forward_diffusion(density_dict)
        _, tau_ts = diffuser.snr_derivative(ts)

        # print({k: torch.isnan(v).any() for k, v in node_features.items()})
        # print({k: torch.isnan(v).any() for k, v in edge_features.items()})
        # print({k: torch.isnan(v).any() for k, v in noised_density_dict.items()})
        # print({k: v.shape for k, v in noised_density_dict.items()})
        # print(seq, batch.ndata['x_mask'])

        denoised_density, seq_logits, pred_atom91 = diffuser.denoising_step(node_features, edge_features, noised_density_dict, ts, batch)

        # print("Denoised", {k: torch.isnan(v).any() for k, v in denoised_density.items()})
        # print("Seq nan", torch.isnan(seq_logits).any())

        denoising_loss = zernike_coeff_loss(density_dict, denoised_density, x_mask)
        seq_loss = seq_recov_loss(seq, seq_logits, x_mask)
        atom91_rmsd = atom91_rmsd_loss(atom91, pred_atom91, atom91_mask)
        loss = -tau_ts * (denoising_loss + seq_loss + atom91_rmsd) #
        # loss = seq_loss

        if train:
            fabric.backward(loss)
            optimizer.step()

        epoch_loss.append(loss.item())
        epoch_denoising_loss.append(denoising_loss.item())
        epoch_seq_loss.append(seq_loss.item())
        epoch_atom91_rmsd.append(atom91_rmsd.item())
        pbar.set_description((
            f"Epoch loss {np.mean(epoch_loss):.4f}, point loss {loss.item():.4f}, "
            f"denoising loss {denoising_loss.item():.4f}, seq loss {seq_loss.item():.4f}, "
            f"atom91 rmsd {atom91_rmsd.item():.4f}, ts {ts.item():.4f} neg taus {-tau_ts.item():.4f}"))
        if torch.isnan(loss).any():
            exit()

    return np.mean(epoch_loss), np.mean(epoch_denoising_loss), np.mean(epoch_seq_loss), np.mean(epoch_atom91_rmsd)
