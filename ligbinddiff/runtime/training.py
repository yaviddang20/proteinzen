""" Training loops and related """
import tqdm

import numpy as np
import torch

from ligbinddiff.runtime.loss import zernike_coeff_loss, seq_cce_loss, atom91_rmsd_loss, density_consistency, bond_length_loss, torsion_loss
from ligbinddiff.utils.fiber import compact_fiber_to_nl
from ligbinddiff.utils.atom_reps import atom91_atom_masks


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
                    train=True,
                    warmup=False,
                    consistency_loss=False,
                    use_channel_weights=True):
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
        atom91 = batch.ndata['atom91']
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
        int_keys = [key for key in batch.ndata.keys() if isinstance(key, int)]
        density_dict = {
            l: batch.ndata[l]
            for l in int_keys
        }

        if train:
            optimizer.zero_grad()
        # ts = torch.tensor([[0.01]], device=batch.device)
        noised_density_dict, ts = diffuser.forward_diffusion(density_dict) #, ts)
        _, tau_ts = diffuser.snr_derivative(ts)

        denoised_density, seq_logits, pred_atom91 = diffuser.denoising_step(node_features, edge_features, noised_density_dict, ts, batch)

        n_channels = diffuser.n_channels

        if use_channel_weights:
            # compute channel weights for zernike loss
            channel_values = torch.as_tensor([
                atom91_atom_masks[atom][4:] for atom in ['C', 'N', 'O', 'S']
            ], dtype=torch.bool).T  # n_atom x 4
            atom_mask = ~atom91_mask.any(dim=-1, keepdim=True)  # n_res x n_atom x 1

            channel_atom_mask = channel_values.unsqueeze(0).to(atom_mask.device) * atom_mask[..., 4:, :]
            channel_atom_count = channel_atom_mask.float().sum(dim=-2)

            # normalize channel contribution across atom type
            channel_atom_count_per_channel = channel_atom_count.sum(dim=-2)
            total_atoms = channel_atom_count.sum()
            # to avoid divide-by-zero
            channel_atom_count_per_channel[channel_atom_count_per_channel == 0] = 1
            # normalize channel contribution across atom type
            channel_weights = channel_atom_count / channel_atom_count_per_channel * total_atoms
            channel_weights[channel_atom_count == 0] = 1  # retain pushing weights towards 0 for no density
        else:
            channel_weights = None

        denoising_loss = zernike_coeff_loss(density_dict, denoised_density, x_mask, n_channels=n_channels, channel_weights=channel_weights)
        ref_noise = zernike_coeff_loss(density_dict, noised_density_dict, x_mask, n_channels=n_channels, channel_weights=channel_weights)
        seq_loss = seq_cce_loss(seq, seq_logits, x_mask)
        atom91_rmsd = atom91_rmsd_loss(atom91_centered, pred_atom91, atom91_mask)
        bond_length_mse = bond_length_loss(atom91_centered, pred_atom91, atom91_mask.any(dim=-1))
        chi_loss = torsion_loss(atom91_centered, pred_atom91, atom91_mask.any(dim=-1))

        if consistency_loss:
            channel_values = torch.as_tensor([
                atom91_atom_masks[atom][4:] for atom in ['C', 'N', 'O', 'S']
            ], dtype=torch.bool).T
            batch_num_nodes = batch.batch_num_nodes().tolist()
            unbatched_densities = unbatch_density(denoised_density, batch_num_nodes)
            unbatched_densities = [compact_fiber_to_nl(density, n_channels=n_channels) for density in unbatched_densities]
            unbatched_sidechains = atom91_centered[..., 4:, :].split(batch_num_nodes)
            unbatched_sidechain_masks = atom91_mask[..., 4:, :].any(dim=-1).split(batch_num_nodes)

            denoising_consistency = [
                density_consistency(
                    density,
                    sidechains,
                    sidechain_masks,
                    channel_values,
                    dataloader.dataset.zernike_transform)
                for density, sidechains, sidechain_masks in zip(
                    unbatched_densities, unbatched_sidechains, unbatched_sidechain_masks
                )
            ]
            denoise_consistency_loss = -sum(denoising_consistency)
        else:
            denoise_consistency_loss = torch.zeros(1, device=denoising_loss.device)

        unscaled_loss = (
            denoising_loss + (not warmup) * (seq_loss + atom91_rmsd) + (consistency_loss) * denoise_consistency_loss +
            bond_length_mse + chi_loss
        )
        loss = -0.5 * tau_ts * (
            unscaled_loss #+ torch.maximum(denoising_loss - ref_noise, torch.zeros_like(denoising_loss))
        )
        # loss = unscaled_loss
        # loss = torch.tensor([torch.nan])

        if torch.isnan(loss).any():
            print(
                (f"Epoch loss {np.mean(epoch_loss):.4f}, point loss {loss.item():.4f}, "
                f"denoise {denoising_loss.item():.4f}, ref noise {ref_noise.item():.4f}, denoise consist {denoise_consistency_loss.item():.4f}, "
                f"seq {seq_loss.item():.4f}, rmsd {atom91_rmsd.item():.4f}, bond l mse {bond_length_mse.item():.4f}, "
                f"chi {chi_loss.item():.4f}, ts {ts.item():.4f}"))

            state_dict = {
                "diffuser": diffuser.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state_dict, "debug_last.pt")
            input_dict = {
                "density": density_dict,
                "node_features": node_features,
                "edge_features": edge_features,
                "noised_density": noised_density_dict,
                "ts": ts
            }
            torch.save(input_dict, "problematic_inputs.pt")
            import dgl

            int_keys = [key for key in batch.ndata.keys() if isinstance(key, int)]
            for i in int_keys:
                batch.ndata[str(i)] = batch.ndata[i]
                del batch.ndata[i]
            dgl.save_graphs("./input_graph.bin", [batch])

            print("F seems like you nan'd out in the forward pass. Retracing...")
            torch.autograd.set_detect_anomaly(True)
            if train:
                optimizer.zero_grad()

            def detect_naninf_hook(module, input, output):
                input_nans = []
                output_nans = []

                for i in input:
                    if isinstance(i, dict):
                        d_isnan = [torch.isfinite(v).all() for v in i.values()]
                        input_nans.append(torch.as_tensor(d_isnan).all())
                    elif isinstance(i, torch.Tensor):
                        input_nans.append(torch.isfinite(i).all())

                for o in output:
                    if isinstance(o, dict):
                        d_isnan = [torch.isfinite(v).all() for v in o.values()]
                        output_nans.append(torch.as_tensor(d_isnan).all())
                    elif isinstance(o, torch.Tensor):
                        output_nans.append(torch.isfinite(o).all())

                # print(input_nans, output_nans)

                if torch.as_tensor(input_nans, dtype=torch.bool).all() and (~torch.as_tensor(output_nans, dtype=torch.bool)).any():
                    print("module", module)
                    print("input", input)
                    print("output", output)

            print("Registering forward hooks")
            for module in diffuser.modules():
                if not isinstance(module, torch.jit.ScriptModule):
                    module.register_forward_hook(detect_naninf_hook)

            print("Denoising")
            denoised_density, seq_logits, pred_atom91 = diffuser.denoising_step(node_features, edge_features, noised_density_dict, ts, batch)

            n_channels = diffuser.n_channels

            print("Computing loss")
            denoising_loss = zernike_coeff_loss(density_dict, denoised_density, x_mask, n_channels=n_channels, channel_weights=channel_weights)
            ref_noise = zernike_coeff_loss(density_dict, noised_density_dict, x_mask, n_channels=n_channels, channel_weights=channel_weights)
            seq_loss = seq_cce_loss(seq, seq_logits, x_mask)
            atom91_rmsd = atom91_rmsd_loss(atom91_centered, pred_atom91, atom91_mask)

            if consistency_loss:
                unbatched_densities = unbatch_density(denoised_density, batch_num_nodes)
                unbatched_densities = [compact_fiber_to_nl(density, n_channels=n_channels) for density in unbatched_densities]
                unbatched_sidechains = atom91_centered[..., 4:, :].split(batch_num_nodes)
                unbatched_sidechain_masks = atom91_mask[..., 4:, :].any(dim=-1).split(batch_num_nodes)

                denoising_consistency = [
                    density_consistency(
                        density,
                        sidechains,
                        sidechain_masks,
                        channel_values,
                        dataloader.dataset.zernike_transform)
                    for density, sidechains, sidechain_masks in zip(
                        unbatched_densities, unbatched_sidechains, unbatched_sidechain_masks
                    )
                ]
                denoise_consistency_loss = -sum(denoising_consistency)
            else:
                denoise_consistency_loss = torch.zeros(1, device=denoising_loss.device)

            unscaled_loss = denoising_loss + (not warmup) * (seq_loss + atom91_rmsd) + (consistency_loss) * denoise_consistency_loss
            loss = -0.5 * tau_ts * (
                unscaled_loss #+ torch.maximum(denoising_loss - ref_noise, torch.zeros_like(denoising_loss))
            )

            if train:
                print("Tracing backward pass")
                fabric.backward(loss)
                grads = []
                for name, param in diffuser.named_parameters():
                    if param.grad is not None:
                        grads.append(param.grad.detach().flatten())
                        if not param.grad.isfinite().all():
                            print(name)
                            print(param)
                            print(param.grad)
                norm = torch.cat(grads).norm()
            else:
                norm = 0

            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(False)
            exit()

        if train:
            fabric.backward(loss)
            grads = []
            for name, param in diffuser.named_parameters():
                if param.grad is not None:
                    grads.append(param.grad.detach().flatten())
                    if not param.grad.isfinite().all():
                        print(name)
                        print(param)
                        print("param grad", param.grad)
            norm = torch.cat(grads).norm()

            if not torch.isfinite(norm):
                print("F seemes like you nan'd in the backwards pass. Rerunning .backward() with anomaly detect")
                optimizer.zero_grad()
                torch.autograd.set_detect_anomaly(True)

                print("Denoising")
                denoised_density, seq_logits, pred_atom91 = diffuser.denoising_step(node_features, edge_features, noised_density_dict, ts, batch)

                n_channels = diffuser.n_channels

                print("Computing loss")
                denoising_loss = zernike_coeff_loss(density_dict, denoised_density, x_mask, n_channels=n_channels, channel_weights=channel_weights)
                ref_noise = zernike_coeff_loss(density_dict, noised_density_dict, x_mask, n_channels=n_channels, channel_weights=channel_weights)
                seq_loss = seq_cce_loss(seq, seq_logits, x_mask)
                atom91_rmsd = atom91_rmsd_loss(atom91_centered, pred_atom91, atom91_mask)


                if consistency_loss:
                    unbatched_densities = unbatch_density(denoised_density, batch_num_nodes)
                    unbatched_densities = [compact_fiber_to_nl(density, n_channels=n_channels) for density in unbatched_densities]
                    unbatched_sidechains = atom91_centered[..., 4:, :].split(batch_num_nodes)
                    unbatched_sidechain_masks = atom91_mask[..., 4:, :].any(dim=-1).split(batch_num_nodes)

                    denoising_consistency = [
                        density_consistency(
                            density,
                            sidechains,
                            sidechain_masks,
                            channel_values,
                            dataloader.dataset.zernike_transform)
                        for density, sidechains, sidechain_masks in zip(
                            unbatched_densities, unbatched_sidechains, unbatched_sidechain_masks
                        )
                    ]
                    denoise_consistency_loss = -sum(denoising_consistency)
                else:
                    denoise_consistency_loss = torch.zeros(1, device=denoising_loss.device)

                unscaled_loss = denoising_loss + (not warmup) * (seq_loss + atom91_rmsd) + (consistency_loss) * denoise_consistency_loss
                loss = -0.5 * tau_ts * (
                    unscaled_loss #+ torch.maximum(denoising_loss - ref_noise, torch.zeros_like(denoising_loss))
                )

                fabric.backward(loss)
                # this should nan? its not guarenteed though
                torch.autograd.set_detect_anomaly(False)

            optimizer.step()
        else:
            norm = 0

        epoch_loss.append(loss.item())
        epoch_denoising_loss.append(denoising_loss.item())
        epoch_ref_noise.append(ref_noise.item())
        epoch_seq_loss.append(seq_loss.item())
        epoch_atom91_rmsd.append(atom91_rmsd.item())
        pbar.set_description((
            f"Epoch loss {np.mean(epoch_loss):.4f}, point loss {loss.item():.4f}, "
            f"denoise {denoising_loss.item():.4f}, ref noise {ref_noise.item():.4f}, " # denoise consist {denoise_consistency_loss.item():.4f}, "
            f"seq {seq_loss.item():.4f}, rmsd {atom91_rmsd.item():.4f}, bond l mse {bond_length_mse.item():.4f}, "
            f"chi {chi_loss.item():.4f}, ts {ts.item():.4f}"))

        if torch.isnan(loss).any():
            exit()

    return {
        "epoch_loss": np.mean(epoch_loss),
        "denoising_loss": np.mean(epoch_denoising_loss),
        "ref_noise": np.mean(epoch_ref_noise),
        "seq_loss": np.mean(epoch_seq_loss),
        "rmsd_loss": np.mean(epoch_atom91_rmsd)
    }
