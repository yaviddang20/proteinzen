""" Inference loops and related """
import tqdm

import numpy as np
import torch

from ligbinddiff.utils.atom_reps import num_to_letter, atom91_atom_masks
from ligbinddiff.data.io.pdb_utils import atom91_to_pdb

from .utils import format_list, gen_pbar_str, update_epoch_loss_dict


def seq_recov(ref_seq, seq_logits, num_nodes, mask):
    """ CCE loss on logits """
    masked_num_nodes = (~mask).split(num_nodes)
    num_nodes = [t.sum().item() for t in masked_num_nodes]

    ref_seq = ref_seq[~mask]
    seq_logits = seq_logits[~mask]

    pred_seq = seq_logits.argmax(dim=-1)
    same = (pred_seq == ref_seq).type(torch.float)

    batched_same = same.split(num_nodes)
    return torch.cat([t.mean().unsqueeze(0) for t in batched_same], dim=0)


def cath_inference_loop(diffuser,
                        dataloader,
                        num_steps,
                        loss_fn,
                        device=None,
                        save=False,
                        truncate=None,
                        use_channel_weights=True,
                        show_progress=False):
    epoch_denoising_loss = []
    epoch_seq_loss = []
    epoch_atom91_rmsd = []
    epoch_seq_recov = []
    epoch_bond_len_mse = []
    epoch_angle_loss = []
    epoch_chi_loss = []
    epoch_cl_denoising_loss = []
    epoch_cl_atom91_rmsd = []
    epoch_cl_bond_len_mse = []
    epoch_cl_angle_loss = []
    epoch_cl_chi_loss = []

    for idx, batch in (pbar := tqdm.tqdm(enumerate(dataloader))):
        if truncate and idx > truncate:
            break

        if device:
            batch = batch.to(device)

        if save:
            from torch_geometric.data import Batch
            batch = Batch.from_data_list([batch.to_data_list()[0]])
        batch['loss_weight'] = 1

        denoised_batch, denoiser_outputs = diffuser.sample(batch, steps=num_steps, show_progress=show_progress)

        loss_dict = loss_fn(denoised_batch, denoiser_outputs, use_channel_weights=use_channel_weights)
        print(batch.name)
        print(loss_dict)
        denoising_loss = loss_dict["denoising_loss"]
        seq_loss = loss_dict["seq_loss"]
        atom91_rmsd = loss_dict["atom91_rmsd"]
        bond_length_mse = loss_dict["bond_length_mse"]
        bond_angle_loss = loss_dict["bond_angle_loss"]
        chi_loss = loss_dict["chi_loss"]

        data_splits = batch._slice_dict['x']
        data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

        pred_seq = denoiser_outputs['seq_logits'].argmax(dim=-1).tolist()
        pred_seq = "".join([num_to_letter[i] for i in pred_seq])
        ref_seq = "".join([num_to_letter[i] for i in denoised_batch['seq'].tolist()])
        per_seq_recov = seq_recov(denoised_batch['seq'], denoiser_outputs['seq_logits'], num_nodes=data_lens, mask=denoised_batch['x_mask'])
        print()
        print("pred seq:\t", pred_seq)
        print("ref seq:\t", ref_seq)

        cl_denoising_loss = loss_dict["cl_denoising_loss"]
        cl_atom91_rmsd = loss_dict["cl_atom91_rmsd"]
        cl_bond_length_mse = loss_dict["cl_bond_length_mse"]
        cl_bond_angle_loss = loss_dict["cl_bond_angle_loss"]
        cl_chi_loss = loss_dict["cl_chi_loss"]

        epoch_denoising_loss += denoising_loss.tolist()
        epoch_seq_loss += seq_loss.tolist()
        epoch_atom91_rmsd += atom91_rmsd.tolist()
        epoch_seq_recov += per_seq_recov.tolist()
        epoch_bond_len_mse += bond_length_mse.tolist()
        epoch_angle_loss += bond_angle_loss.tolist()
        epoch_chi_loss += chi_loss.tolist()
        # for if seq_recov = 0
        epoch_cl_denoising_loss += cl_denoising_loss[cl_denoising_loss.isfinite()].tolist()
        epoch_cl_atom91_rmsd += cl_atom91_rmsd[cl_atom91_rmsd.isfinite()].tolist()
        epoch_cl_bond_len_mse += cl_bond_length_mse[cl_bond_length_mse.isfinite()].tolist()
        epoch_cl_angle_loss += cl_bond_angle_loss[cl_bond_angle_loss.isfinite()].tolist()
        epoch_cl_chi_loss += cl_chi_loss[cl_chi_loss.isfinite()].tolist()

        pbar.set_description((
            f"denoise {format_list(denoising_loss.tolist(), '{:.4f}')}, "
            f"seq {format_list(seq_loss.tolist(), '{:.4f}')}, seq_recov {format_list(per_seq_recov.tolist(), '{:.4f}')}, rmsd {format_list(atom91_rmsd.tolist(), '{:.4f}')}, bond l mse {format_list(bond_length_mse.tolist(), '{:.4f}')}, "
            f"angle {format_list(bond_angle_loss.tolist(), '{:.4f}')}, chi {format_list(chi_loss.tolist(), '{:.4f}')}"))

        if save:
            denoiser_outputs['decoded_latent'][..., :4, :] = batch['atom91_centered'][..., :4, :]  # we don't train on the backbone atoms
            atom91_uncentered = denoiser_outputs['decoded_latent'] + denoiser_outputs['x'].unsqueeze(-2)
            atom91_to_pdb(ref_seq, atom91_uncentered.numpy(force=True), batch.name[0] + "_native_seq")
            atom91_to_pdb(pred_seq, atom91_uncentered.numpy(force=True), batch.name[0] + "_designed_seq")

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
        "rmsd_loss": np.mean(epoch_atom91_rmsd),
        "bond_len_mse": np.mean(epoch_bond_len_mse),
        "angle_loss": np.mean(epoch_angle_loss),
        "chi_loss": np.mean(epoch_chi_loss),
        "cl_denoising_loss": np.mean(epoch_cl_denoising_loss),
        "cl_rmsd_loss": np.mean(epoch_cl_atom91_rmsd),
        "cl_bond_len_mse": np.mean(epoch_cl_bond_len_mse),
        "cl_angle_loss": np.mean(epoch_cl_angle_loss),
        "cl_chi_loss": np.mean(epoch_cl_chi_loss)
    }


def inpaint_inference_loop(diffuser,
                           dataloader,
                           num_steps,
                           loss_fn,
                           device=None,
                           save=False,
                           truncate=None,
                           use_channel_weights=True,
                           show_progress=False,
                           select_task=None):
    epoch_dict = {}

    for idx, batch in (pbar := tqdm.tqdm(enumerate(dataloader))):
        if truncate and idx > truncate:
            break

        if device:
            batch = batch.to(device)

        if save:
            from torch_geometric.data import Batch
            batch = Batch.from_data_list([batch.to_data_list()[0]])

        latent_outputs, decoder_outputs = diffuser.sample(batch, steps=num_steps, show_progress=show_progress, select_task=select_task)

        passthrough_outputs = {}
        passthrough_outputs.update(latent_outputs)
        passthrough_outputs.update(decoder_outputs)

        loss_dict = loss_fn(batch, latent_outputs, decoder_outputs, passthrough_outputs)
        print(batch.name)
        print(loss_dict)

        data_splits = batch._slice_dict['residue']['x']
        data_lens = (data_splits[1:] - data_splits[:-1]).tolist()

        if 'noising_mask' not in latent_outputs:
            latent_outputs['noising_mask'] = torch.ones_like(batch['residue']['seq']).bool()

        pred_seq = decoder_outputs['seq_logits'].argmax(dim=-1).tolist()
        pred_seq = "".join([num_to_letter[i] for i in pred_seq])
        ref_seq = "".join([num_to_letter[i] for i in batch['residue']['seq'].tolist()])
        total_seq_recov = seq_recov(batch['residue']['seq'], decoder_outputs['seq_logits'], num_nodes=data_lens, mask=batch['residue']['x_mask'])
        noised_seq_recov = seq_recov(batch['residue']['seq'], decoder_outputs['seq_logits'], num_nodes=data_lens, mask=batch['residue']['x_mask'] | ~latent_outputs['noising_mask'])
        unnoised_seq_recov = seq_recov(batch['residue']['seq'], decoder_outputs['seq_logits'], num_nodes=data_lens, mask=batch['residue']['x_mask'] | latent_outputs['noising_mask'])
        noised_pos = "".join(["X" if pos else " " for pos in latent_outputs['noising_mask'].tolist()])
        masked_pos = "".join(["_" if pos else " " for pos in batch['residue']['x_mask'].tolist()])
        print()
        print("pred seq:\t", pred_seq)
        print("ref seq: \t", ref_seq)
        print("noised:  \t", noised_pos)
        print("masked:  \t", masked_pos)

        loss_dict['total_seq_recov'] = total_seq_recov
        loss_dict['noised_seq_recov'] = noised_seq_recov
        loss_dict['unnoised_seq_recov'] = unnoised_seq_recov
        epoch_dict = update_epoch_loss_dict(epoch_dict, loss_dict)
        pbar_str = gen_pbar_str(loss_dict)
        pbar.set_description(pbar_str)

        if save:
            denoised_x_ca = latent_outputs['denoised_bb'][..., 1, :]
            atom91_uncentered = decoder_outputs['decoded_latent'] + denoised_x_ca.unsqueeze(-2)
            atom91_uncentered[..., :4, :] = latent_outputs['denoised_bb']  # set backbone atoms to those from denoiser
            atom91_to_pdb(ref_seq, atom91_uncentered.numpy(force=True), batch.name[0] + "_native_seq")
            atom91_to_pdb(pred_seq, atom91_uncentered.numpy(force=True), batch.name[0] + "_designed_seq")

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

    return {key: np.mean(losses) for key, losses in epoch_dict.items()}

def debug_inpaint_inference_loop(diffuser,
                           dataloader,
                           num_steps,
                           loss_fn,
                           device=None,
                           save=False,
                           truncate=None,
                           use_channel_weights=True,
                           show_progress=False):
    epoch_dict = {}

    for idx, batch in (pbar := tqdm.tqdm(enumerate(dataloader))):
        if truncate and idx > truncate:
            break

        if device:
            batch = batch.to(device)

        if save:
            from torch_geometric.data import Batch
            batch = Batch.from_data_list([batch.to_data_list()[0]])

        latent_outputs, denoiser_outputs = diffuser.sample(batch, steps=num_steps, show_progress=show_progress)

        loss_dict = loss_fn(batch, latent_outputs, denoiser_outputs)
        print(batch.name)
        print(loss_dict)

        epoch_dict = update_epoch_loss_dict(epoch_dict, loss_dict)
        pbar_str = gen_pbar_str(loss_dict)
        pbar.set_description(pbar_str)

        if save:
            denoised_bb = latent_outputs['denoised_bb']
            denoised_bb = torch.cat([denoised_bb, batch['bb'][:, 3:4]], dim=-2)  # add O
            atom91_to_pdb("".join(["G" for _ in range(batch.num_nodes)]), denoised_bb.numpy(force=True), batch.name[0] + "_bb")

    return {key: np.mean(losses) for key, losses in epoch_dict.items()}


def bb_inpaint_inference_loop(diffuser,
                           dataloader,
                           num_steps,
                           loss_fn,
                           device=None,
                           save=False,
                           truncate=None,
                           use_channel_weights=True,
                           show_progress=False):
    epoch_dict = {}

    for idx, batch in (pbar := tqdm.tqdm(enumerate(dataloader))):
        if truncate and idx > truncate:
            break

        if device:
            batch = batch.to(device)

        if save:
            from torch_geometric.data import Batch
            batch = Batch.from_data_list([batch.to_data_list()[0]])

        denoiser_outputs = diffuser.sample(batch, steps=num_steps, show_progress=show_progress)

        loss_dict = loss_fn(batch, denoiser_outputs)
        print(batch.name)
        print(loss_dict)

        epoch_dict = update_epoch_loss_dict(epoch_dict, loss_dict)
        pbar_str = gen_pbar_str(loss_dict)
        pbar.set_description(pbar_str)

        if save:
            denoised_bb = denoiser_outputs['denoised_bb']
            denoised_bb = torch.cat([denoised_bb, batch['bb'][:, 3:4]], dim=-2)  # add O
            atom91_to_pdb("".join(["G" for _ in range(batch['residue'].num_nodes)]), denoised_bb.numpy(force=True), batch.name[0] + "_bb")

    return {key: np.mean(losses) for key, losses in epoch_dict.items()}
