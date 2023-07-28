""" Inference loops and related """
import tqdm

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from ligbinddiff.utils.fiber import compact_fiber_to_nl
from ligbinddiff.runtime.loss import zernike_coeff_loss, seq_cce_loss, atom91_rmsd_loss, bond_length_loss
from ligbinddiff.utils.atom_reps import num_to_letter, atom91_atom_masks
from ligbinddiff.data.io.pdb_utils import atom91_to_pdb


def seq_recov(ref_seq, seq_logits, mask):
    """ CCE loss on logits """
    ref_seq = ref_seq[~mask]
    seq_logits = seq_logits[~mask]

    pred_seq = seq_logits.argmax(dim=-1)
    return (pred_seq == ref_seq).type(torch.float).mean()


def cath_inference_loop(diffuser, dataloader, num_steps, device=None, save=False, truncate=None, use_channel_weights=True):
    epoch_denoising_loss = []
    epoch_seq_loss = []
    epoch_atom91_rmsd = []
    epoch_seq_recov = []

    for idx, batch in (pbar := tqdm.tqdm(enumerate(dataloader))):
        if truncate and idx > truncate:
            break

        if device:
            batch = batch.to(device)
        node_s = batch.ndata['backbone_s']
        node_v = batch.ndata['backbone_v']
        edge_s = batch.edata['edge_s']
        edge_v = batch.edata['edge_v']
        seq = batch.ndata['seq']
        x_mask = batch.ndata['x_mask']
        X_cb = batch.ndata['x_cb']
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

        denoised_density, seq_logits, pred_atom91 = zip(*diffuser.sample(node_features, edge_features, batch, steps=num_steps, return_trajectory=True, show_progress=True))

        int_keys = [key for key in batch.ndata.keys() if isinstance(key, int)]
        density_dict = {
            l: batch.ndata[l]
            for l in int_keys
        }

        n_channels = diffuser.n_channels


        if use_channel_weights:
            channel_values = torch.as_tensor([
                atom91_atom_masks[atom][4:] for atom in ['C', 'N', 'O', 'S']
            ], dtype=torch.bool).T  # n_atom x 4
            atom_mask = ~atom91_mask.any(dim=-1, keepdim=True)  # n_res x n_atom x 1

            channel_atom_mask = channel_values.unsqueeze(0).to(atom_mask.device) * atom_mask[..., 4:, :]
            channel_atom_count = channel_atom_mask.float().sum(dim=-2)
            channel_atom_count[channel_atom_count == 0] = 0.5  # upweight pushing weights towards 0

            channel_weights = 1 / channel_atom_count
        else:
            channel_weights = None

        denoising_loss = [zernike_coeff_loss(density_dict, d, x_mask, n_channels=n_channels, channel_weights=channel_weights) for d in denoised_density[1:]]
        seq_loss = [seq_cce_loss(seq, s, x_mask) for s in seq_logits[1:]]
        per_seq_recov = [seq_recov(seq, s, x_mask) for s in seq_logits[1:]]
        atom91_rmsd = [atom91_rmsd_loss(atom91_centered, a, atom91_mask) for a in pred_atom91[1:]]

        for p_atom91 in pred_atom91[1:]:
            p_atom91[..., 4:, :] = atom91_centered[..., 4:, :]
        # bond_length_mse = [bond_length_loss(atom91_centered, a, atom91_mask.any(dim=-1)) for a in pred_atom91[1:]]
        bond_length_mse = bond_length_loss(atom91_centered, pred_atom91[-1], atom91_mask.any(dim=-1))
        # print(denoised_density, seq_logits)
        # for d, s, sr, r in zip(denoising_loss, seq_loss, per_seq_recov, atom91_rmsd):
        #     print(d, s, sr, r)

        pred_seq = seq_logits[-1].argmax(dim=-1).tolist()
        pred_seq = "".join([num_to_letter[i] for i in pred_seq])
        ref_seq = "".join([num_to_letter[i] for i in seq.tolist()])
        print("pred seq:\t", pred_seq)
        print("ref seq:\t", ref_seq)


        epoch_denoising_loss.append(denoising_loss[-1].item())
        epoch_seq_loss.append(seq_loss[-1].item())
        epoch_atom91_rmsd.append(atom91_rmsd[-1].item())
        epoch_seq_recov.append(per_seq_recov[-1].item())
        pbar.set_description((
            f"Denoising loss {denoising_loss[-1].item():.4f}, seq loss {seq_loss[-1].item():.4f}, "
            f"atom91 rmsd {atom91_rmsd[-1].item():.4f}, bond l mse {bond_length_mse.item():.4f}, % seq recov {per_seq_recov[-1].item():.4f}"))

        if save:
            assert device is not None, "device cannot be None for saving densities"
            from ligbinddiff.data.io.density import density_to_mrc
            print(batch.name)
            zt = dataloader.dataset.zernike_transform
            print({k: v.shape for k,v in compact_fiber_to_nl(denoised_density[-1], n_channels=n_channels).items()})
            rho = zt.back_transform(compact_fiber_to_nl(denoised_density[-1], n_channels=n_channels), batch.ndata['x_cb'], device=device)

            atom91 = batch.ndata['atom91']
            min_coord = int(atom91[~atom91_mask].min() - 1)
            max_coord = int(atom91[~atom91_mask].max() + 1)

            density_to_mrc(rho, voxel_size=0.5, coord_bounds=(min_coord, max_coord), memory_mode=1, out_prefix=batch.name)

            atom91_uncentered = pred_atom91[-1] + X_cb.unsqueeze(-2)
            atom91_uncentered[..., :4, :] = atom91[..., :4, :]  # we don't train on the backbone atoms
            atom91_to_pdb(ref_seq, atom91_uncentered.numpy(force=True), batch.name)

    return {
        "denoising_loss": np.mean(epoch_denoising_loss),
        "seq_loss": np.mean(epoch_seq_loss),
        "seq_recov": np.mean(epoch_seq_recov),
        "rmsd_loss": np.mean(epoch_atom91_rmsd)
    }
