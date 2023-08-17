""" Inference loops and related """
import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from ligbinddiff.utils.fiber import compact_fiber_to_nl
from ligbinddiff.runtime.loss import zernike_coeff_loss, seq_cce_loss, atom91_rmsd_loss, bond_length_loss, torsion_loss
from ligbinddiff.utils.atom_reps import num_to_letter, atom91_atom_masks
from ligbinddiff.data.io.pdb_utils import atom91_to_pdb


def seq_recov(ref_seq, seq_logits, mask):
    """ CCE loss on logits """
    ref_seq = ref_seq[~mask]
    seq_logits = seq_logits[~mask]

    pred_seq = seq_logits.argmax(dim=-1)
    return (pred_seq == ref_seq).type(torch.float).mean()


def cath_inference_loop(diffuser, dataloader, num_steps, loss_fn, device=None, save=False, truncate=None, use_channel_weights=True):
    epoch_denoising_loss = []
    epoch_seq_loss = []
    epoch_atom91_rmsd = []
    epoch_seq_recov = []

    for idx, batch in (pbar := tqdm.tqdm(enumerate(dataloader))):
        if truncate and idx > truncate:
            break

        if device:
            batch = batch.to(device)
        batch['loss_weight'] = 1

        denoised_batch, denoiser_outputs = diffuser.sample(batch, steps=num_steps)
        n_channels = diffuser.n_channels

        loss_dict = loss_fn(denoised_batch, denoiser_outputs, use_channel_weights=use_channel_weights)
        denoising_loss = loss_dict["denoising_loss"]
        seq_loss = loss_dict["seq_loss"]
        atom91_rmsd = loss_dict["atom91_rmsd"]
        bond_length_mse = loss_dict["bond_length_mse"]
        bond_angle_loss = loss_dict["bond_angle_loss"]
        chi_loss = loss_dict["chi_loss"]

        pred_seq = denoiser_outputs['seq_logits'].argmax(dim=-1).tolist()
        pred_seq = "".join([num_to_letter[i] for i in pred_seq])
        ref_seq = "".join([num_to_letter[i] for i in denoised_batch['seq'].tolist()])
        per_seq_recov = seq_recov(denoised_batch['seq'], denoiser_outputs['seq_logits'], mask=denoised_batch['x_mask'])
        print("pred seq:\t", pred_seq)
        print("ref seq:\t", ref_seq)


        epoch_denoising_loss.append(denoising_loss.item())
        epoch_seq_loss.append(seq_loss.item())
        epoch_atom91_rmsd.append(atom91_rmsd.item())
        epoch_seq_recov.append(per_seq_recov.item())
        pbar.set_description((
            f"Denoising loss {denoising_loss.item():.4f}, seq loss {seq_loss.item():.4f}, "
            f"atom91 rmsd {atom91_rmsd.item():.4f}, bond l mse {bond_length_mse.item():.4f}, "
            f"angle {bond_angle_loss.item():4f}, chi {chi_loss.item():.4f}, % seq recov {per_seq_recov.item():.4f}"))

        # if save:
        #     assert device is not None, "device cannot be None for saving densities"
        #     from ligbinddiff.data.io.density import density_to_mrc
        #     print(batch.name)
        #     zt = dataloader.dataset.zernike_transform
        #     print({k: v.shape for k,v in compact_fiber_to_nl(denoised_density[-1], n_channels=n_channels).items()})
        #     rho = zt.back_transform(compact_fiber_to_nl(denoised_density[-1], n_channels=n_channels), batch.ndata['x_cb'], device=device)

        #     atom91 = batch.ndata['atom91']
        #     min_coord = int(atom91[~atom91_mask].min() - 1)
        #     max_coord = int(atom91[~atom91_mask].max() + 1)

        #     density_to_mrc(rho, voxel_size=0.5, coord_bounds=(min_coord, max_coord), memory_mode=1, out_prefix=batch.name)

        #     atom91_uncentered = pred_atom91[-1] + X_cb.unsqueeze(-2)
        #     atom91_uncentered[..., :4, :] = atom91[..., :4, :]  # we don't train on the backbone atoms
        #     atom91_to_pdb(ref_seq, atom91_uncentered.numpy(force=True), batch.name)

    return {
        "denoising_loss": np.mean(epoch_denoising_loss),
        "seq_loss": np.mean(epoch_seq_loss),
        "seq_recov": np.mean(epoch_seq_recov),
        "rmsd_loss": np.mean(epoch_atom91_rmsd)
    }
