""" Training loops and related """
import tqdm

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from ligbinddiff.utils.type_l import type_l_sub, type_l_apply
from ligbinddiff.utils.fiber import compact_fiber_to_nl


def zernike_coeff_loss(ref_density, pred_density, mask):
    """ Invariant MSE loss on Zernike coeffs
    Average of norms of the difference between all type-l vectors """
    ref_density = compact_fiber_to_nl(ref_density)
    pred_density = compact_fiber_to_nl(pred_density)

    # only compute residues where all atoms are present
    ref_density = type_l_apply(lambda x: x[~mask], ref_density)
    pred_density = type_l_apply(lambda x: x[~mask], pred_density)

    # print({k: v.abs().max() for k, v in ref_density.items()})
    # print({k: v.abs().max() for k, v in pred_density.items()})

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


def seq_recov_loss(ref_seq, seq_logits, mask):
    """ CCE loss on logits """
    ref_seq = ref_seq[~mask]
    seq_logits = seq_logits[~mask]
    return F.cross_entropy(seq_logits, ref_seq)


def atom91_rmsd_loss(ref_atom91, pred_atom91, atom91_mask):
    """ RMSD on relevant atom91 atoms """
    diff = ref_atom91[..., 4:, :][~atom91_mask[..., 4:, :]] - pred_atom91[..., 4:, :][~atom91_mask[..., 4:, :]]
    sd = torch.square(diff)
    return sd.mean().sqrt()


def cath_train_loop(diffuser, dataloader, optimizer, fabric, train=True, warmup=False):
    epoch_loss = []
    epoch_ref_noise = []
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
        atom91_centered = batch.ndata['atom91_centered']
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

        optimizer.zero_grad()
        # ts = torch.tensor([[0.01]], device=batch.device)
        noised_density_dict, ts = diffuser.forward_diffusion(density_dict) #, ts)
        _, tau_ts = diffuser.snr_derivative(ts)

        denoised_density, seq_logits, pred_atom91 = diffuser.denoising_step(node_features, edge_features, noised_density_dict, ts, batch)

        denoising_loss = zernike_coeff_loss(density_dict, denoised_density, x_mask)
        ref_noise = zernike_coeff_loss(density_dict, noised_density_dict, x_mask)
        seq_loss = seq_recov_loss(seq, seq_logits, x_mask)
        atom91_rmsd = atom91_rmsd_loss(atom91_centered, pred_atom91, atom91_mask)
        unscaled_loss = denoising_loss + seq_loss + (not warmup) * atom91_rmsd
        loss = -0.5 * tau_ts * (
            unscaled_loss #+ torch.maximum(denoising_loss - ref_noise, torch.zeros_like(denoising_loss))
        )
        # loss = unscaled_loss

        if train:
            fabric.backward(loss)
            optimizer.step()
            grads = [
                param.grad.detach().flatten()
                for param in diffuser.parameters()
                if param.grad is not None
            ]
            norm = torch.cat(grads).norm()
        else:
            norm = 0

        epoch_loss.append(loss.item())
        epoch_denoising_loss.append(denoising_loss.item())
        epoch_ref_noise.append(ref_noise.item())
        epoch_seq_loss.append(seq_loss.item())
        epoch_atom91_rmsd.append(atom91_rmsd.item())
        pbar.set_description((
            f"Epoch loss {np.mean(epoch_loss):.4f}, point loss {loss.item():.4f}, norm {norm:.4f}, "
            f"denoise {denoising_loss.item():.4f}, ref noise {ref_noise.item():.4f}, "
            f"seq {seq_loss.item():.4f}, rmsd {atom91_rmsd.item():.4f}, ts {ts.item():.4f}"))
        if torch.isnan(loss).any():
            exit()

    return {
        "epoch_loss": np.mean(epoch_loss),
        "denoising_loss": np.mean(epoch_denoising_loss),
        "ref_noise": np.mean(epoch_ref_noise),
        "seq_loss": np.mean(epoch_seq_loss),
        "rmsd_loss": np.mean(epoch_atom91_rmsd)
    }
