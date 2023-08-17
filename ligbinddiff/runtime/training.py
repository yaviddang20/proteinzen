""" Training loops and related """
import tqdm

import numpy as np
import torch



def unbatch_density(density_dict, batch_num_nodes):
    densities = [{} for _ in batch_num_nodes]
    for l, batched_v in density_dict.items():
        vecs = torch.split(batched_v, batch_num_nodes, dim=-3)
        for density, vec in zip(densities, vecs):
            density[l] = vec

    return densities


def cath_train_loop(diffuser,
                    dataloader,
                    optimizer,
                    fabric,
                    loss_fn,
                    train=True,
                    use_channel_weights=True):
    epoch_loss = []
    epoch_ref_noise = []
    epoch_denoising_loss = []
    epoch_seq_loss = []
    epoch_atom91_rmsd = []

    for batch in (pbar := tqdm.tqdm(dataloader)):

        if train:
            optimizer.zero_grad()
        noised_batch = diffuser.forward_noising(batch)
        outputs = diffuser.reverse_noising(batch)

        loss_dict = loss_fn(noised_batch, outputs, use_channel_weights=use_channel_weights)
        loss = loss_dict["loss"]
        denoising_loss = loss_dict["denoising_loss"]
        ref_noise = loss_dict["ref_noise"]
        seq_loss = loss_dict["seq_loss"]
        atom91_rmsd = loss_dict["atom91_rmsd"]
        bond_length_mse = loss_dict["bond_length_mse"]
        bond_angle_loss = loss_dict["bond_angle_loss"]
        chi_loss = loss_dict["chi_loss"]
        t = noised_batch['t']

        if train:
            fabric.backward(loss)
            optimizer.step()

        epoch_loss.append(loss.item())
        epoch_denoising_loss.append(denoising_loss.item())
        epoch_ref_noise.append(ref_noise.item())
        epoch_seq_loss.append(seq_loss.item())
        epoch_atom91_rmsd.append(atom91_rmsd.item())
        pbar.set_description((
            f"Epoch loss {np.mean(epoch_loss):.4f}, point loss {loss.item():.4f}, "
            f"denoise {denoising_loss.item():.4f}, ref noise {ref_noise.item():.4f}, "
            f"seq {seq_loss.item():.4f}, rmsd {atom91_rmsd.item():.4f}, bond l mse {bond_length_mse.item():.4f}, "
            f"angle {bond_angle_loss.item():.4f}, chi {chi_loss.item():.4f}, t {t.item():.4f}"))

        if torch.isnan(loss).any():
            exit()

    return {
        "epoch_loss": np.mean(epoch_loss),
        "denoising_loss": np.mean(epoch_denoising_loss),
        "ref_noise": np.mean(epoch_ref_noise),
        "seq_loss": np.mean(epoch_seq_loss),
        "rmsd_loss": np.mean(epoch_atom91_rmsd)
    }
