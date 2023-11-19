""" Training loops and related """
import tqdm

import numpy as np
import torch

from ligbinddiff.runtime.utils import format_list, gen_pbar_str, update_epoch_loss_dict

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
                    use_channel_weights=True,
                    warmup=False,
                    debug=False,
                    debug_device='cpu',
                    discriminator=None,
                    discriminator_optimizer=None):
    epoch_loss = []
    epoch_ref_noise = []
    epoch_denoising_loss = []
    epoch_seq_loss = []
    epoch_atom91_rmsd = []
    epoch_bond_len_mse = []
    epoch_sidechain_dists_mse = []
    epoch_angle_loss = []
    epoch_chi_loss = []

    for batch in (pbar := tqdm.tqdm(dataloader)):

        try:
            # print(batch.name)
            if train:
                optimizer.zero_grad()
                if discriminator_optimizer is not None:
                    discriminator_optimizer.zero_grad()

            if debug:
                batch = batch.to(debug_device)
            if discriminator is not None:
                diffuser.zero_grad()
            noised_batch, outputs = diffuser.forward(batch, warmup=warmup)
            if discriminator is not None:
                outputs = discriminator(outputs)

            loss_dict = loss_fn(noised_batch, outputs, use_channel_weights=use_channel_weights)#, ae_loss_weight=diffuser.scheduler.weight(0))
            # print(loss_dict)
            if 'gen_loss' in loss_dict.keys():
                gen_loss = loss_dict['gen_loss']
            else:
                gen_loss = torch.zeros(1)
            # print("gen loss", loss_dict['gen_loss'])
            loss = loss_dict["loss"]
            denoising_loss = loss_dict["denoising_loss"]
            ref_noise = loss_dict["ref_noise"]
            seq_loss = loss_dict["seq_loss"]
            atom91_rmsd = loss_dict["atom91_rmsd"]
            bond_length_mse = loss_dict["bond_length_mse"]
            sidechain_dists_mse = loss_dict["sidechain_dists_mse"]
            bond_angle_loss = loss_dict["bond_angle_loss"]
            chi_loss = loss_dict["chi_loss"]
            t = noised_batch['t']

            if train:
                if debug:
                    loss.backward()
                else:
                    fabric.backward(loss)

                for name, param in diffuser.named_parameters():
                    if param.grad is not None and not param.grad.isfinite().all():
                        torch.set_printoptions(threshold=1000000)
                        print("nan in", name, "grad")
                        print("param", param)
                        print("param grad", param.grad)

                optimizer.step()

            if discriminator is not None:
                discriminator.zero_grad()
                outputs['decoded_latent'] = outputs['decoded_latent'].detach()
                outputs = discriminator(outputs)
                discrim_loss = discriminator_loss(outputs)
                d_loss = discrim_loss.mean()
                # print("discrim loss", discrim_loss)

                if train:
                    if discriminator_optimizer is not None:
                        if debug:
                            d_loss.backward()
                        else:
                            fabric.backward(d_loss)
                        discriminator_optimizer.step()

                        for name, param in discriminator.named_parameters():
                            if param.grad is not None and not param.grad.isfinite().all():
                                torch.set_printoptions(threshold=1000000)
                                print("nan in", name, "grad")
                                print("param", param)
                                print("param grad", param.grad)
            else:
                discrim_loss = torch.zeros(1)


            epoch_loss.append(loss.item())
            epoch_denoising_loss += denoising_loss.tolist()
            epoch_ref_noise += ref_noise.tolist()
            epoch_seq_loss += seq_loss.tolist()
            epoch_atom91_rmsd += atom91_rmsd.tolist()
            epoch_bond_len_mse += bond_length_mse.tolist()
            epoch_sidechain_dists_mse += sidechain_dists_mse.tolist()
            epoch_angle_loss += bond_angle_loss.tolist()
            epoch_chi_loss += chi_loss.tolist()
            pbar.set_description((
                f"Epoch loss {np.mean(epoch_loss):.4f}, point loss {loss.item():.4f}, "
                f"denoise {format_list(denoising_loss.tolist(), '{:.4f}')}, ref noise {format_list(ref_noise.tolist(), '{:.4f}')}, "
                f"seq {format_list(seq_loss.tolist(), '{:.4f}')}, rmsd {format_list(atom91_rmsd.tolist(), '{:.4f}')}, "
                f"bond l mse {format_list(bond_length_mse.tolist(), '{:.4f}')}, dists {format_list(sidechain_dists_mse.tolist(), '{:.4f}')}, "
                f"angle {format_list(bond_angle_loss.tolist(), '{:.4f}')}, chi {format_list(chi_loss.tolist(), '{:.4f}')}, t {t.tolist()}, "
                f"gen loss {format_list(gen_loss.tolist(), '{:.4f}')}, discrim loss {format_list(discrim_loss.tolist(), '{:.4f}')}"))

            if torch.isnan(loss).any():
                raise ValueError("loss is nan")
        except Exception as e:
            torch.save(batch, "problematic_inputs.pt")
            torch.save(outputs, "outputs.pt")
            raise e
        # print("Done")
        # exit()


    return {
        "epoch_loss": np.mean(epoch_loss),
        "denoising_loss": np.mean(epoch_denoising_loss),
        "ref_noise": np.mean(epoch_ref_noise),
        "seq_loss": np.mean(epoch_seq_loss),
        "rmsd_loss": np.mean(epoch_atom91_rmsd),
        "bond_len_mse": np.mean(epoch_bond_len_mse),
        "sidechain_dists_mse": np.mean(epoch_sidechain_dists_mse),
        "angle_loss": np.mean(epoch_angle_loss),
        "chi_loss": np.mean(epoch_chi_loss)
    }


def inpaint_train_loop(diffuser,
                       dataloader,
                       optimizer,
                       fabric,
                       loss_fn,
                       train=True,
                       warmup=False,
                       debug=False,
                       debug_device='cpu',
                       ):
    epoch_dict = {}

    for batch in (pbar := tqdm.tqdm(dataloader)):

        try:
            # print(batch.name)
            if train:
                optimizer.zero_grad()

            if debug:
                batch = batch.to(debug_device)
            latent_data, decoder_outputs, passthrough_outputs = diffuser.forward(batch, warmup=warmup)

            loss_dict = loss_fn(batch, latent_data, decoder_outputs, passthrough_outputs, warmup=warmup)
            # print(loss_dict)
            loss = loss_dict["loss"]

            if train:
                if debug:
                    loss.backward()
                else:
                    fabric.backward(loss)

                # check if there are any nans in grads
                # if so, rerun this step under anomaly detection
                nan_param_grad = False
                for name, param in diffuser.named_parameters():
                    if param.grad is not None and not param.grad.isfinite().all():
                        torch.set_printoptions(threshold=1000000)
                        print("nan in", name, "grad")
                        nan_param_grad = True

                if nan_param_grad:
                    optimizer.zero_grad()
                    with torch.autograd.detect_anomaly():
                        latent_data, decoder_outputs, passthrough_outputs = diffuser.forward(batch, warmup=warmup)

                        loss_dict = loss_fn(batch, latent_data, decoder_outputs, warmup=warmup)
                        loss = loss_dict["loss"]
                        if debug:
                            loss.backward()
                        else:
                            fabric.backward(loss)
                else:
                    optimizer.step()


            epoch_dict = update_epoch_loss_dict(epoch_dict, loss_dict)
            epoch_loss = np.mean(epoch_dict['loss'])
            pbar_str = f"epoch_loss: {np.mean(epoch_loss):.4f}, "
            pbar_str += gen_pbar_str(loss_dict)
            pbar_str += f"t: {format_list(latent_data['t'], '{:.4f}')}"
            pbar.set_description(pbar_str)

            if torch.isnan(loss).any():
                exit()
        except Exception as e:
            torch.save(batch, "problematic_inputs.pt")
            torch.save(latent_data, "latent_outputs.pt")
            torch.save(decoder_outputs, "decoder_outputs.pt")
            raise e
        # print("Done")
        # exit()

    return {key: np.mean(losses) for key, losses in epoch_dict.items()}

def debug_inpaint_train_loop(diffuser,
                       dataloader,
                       optimizer,
                       fabric,
                       loss_fn,
                       train=True,
                       warmup=False,
                       debug=False,
                       debug_device='cpu',
                       step_count=None,
                       checkpoint_steps=None,
                       early_checkpoint=100,
                       ):
    epoch_dict = {}

    for batch in (pbar := tqdm.tqdm(dataloader)):

        try:
            # print(batch.name)
            if train:
                optimizer.zero_grad()

            if debug:
                batch = batch.to(debug_device)
            latent_data, decoder_outputs = diffuser.forward(batch)

            loss_dict = loss_fn(batch, latent_data, decoder_outputs)
            loss = loss_dict["loss"]

            if train:
                # # TODO: get rid of this
                # if torch.isnan(loss).any():
                #     print("loss is nan")
                #     print(batch.name)
                #     continue

                if debug:
                    loss.backward()
                else:
                    fabric.backward(loss)

                # check if there are any nans in grads
                # if so, rerun this step under anomaly detection
                nan_param_grad = False
                for name, param in diffuser.named_parameters():
                    if param.grad is not None and not param.grad.isfinite().all():
                        torch.set_printoptions(threshold=1000000)
                        print("nan in", name, "grad")
                        nan_param_grad = True
                    # if param.grad is None:
                    #     print("no grads for", name)


                if nan_param_grad:
                    optimizer.zero_grad()
                    with torch.autograd.detect_anomaly():
                        latent_data, decoder_outputs = diffuser.forward(batch)

                        loss_dict = loss_fn(batch, latent_data, decoder_outputs)
                        loss = loss_dict["loss"]
                        if debug:
                            loss.backward()
                        else:
                            fabric.backward(loss)
                else:
                    optimizer.step()


            epoch_dict = update_epoch_loss_dict(epoch_dict, loss_dict)
            epoch_loss = np.mean(epoch_dict['loss'])
            pbar_str = f"epoch_loss: {np.mean(epoch_loss):.4f}, "
            pbar_str += gen_pbar_str(loss_dict)
            pbar_str += f", t: {format_list(latent_data['t'], '{:.4f}')}"
            pbar.set_description(pbar_str)
            # print(pbar_str)
            if step_count is not None:
                step_count += 1
                if checkpoint_steps is not None and step_count % checkpoint_steps == 0:
                    torch.save(diffuser.state_dict(), f"checkpoint_step_{step_count}.pt")
                if step_count == early_checkpoint:
                    torch.save(diffuser.state_dict(), f"checkpoint_step_{step_count}.pt")

            if torch.isnan(loss).any():
                raise ValueError("loss is nan")
        except Exception as e:
            print(batch.name)
            torch.save(batch, "problematic_inputs.pt")
            torch.save(latent_data, "latent_outputs.pt")
            torch.save(decoder_outputs, "decoder_outputs.pt")
            raise e
        # print("Done")
        # exit()

    return {key: np.mean(losses) for key, losses in epoch_dict.items()}, step_count


def bb_inpaint_train_loop(diffuser,
                       dataloader,
                       optimizer,
                       fabric,
                       loss_fn,
                       train=True,
                       warmup=False,
                       debug=False,
                       debug_device='cpu',
                       ):
    epoch_dict = {}

    for batch in (pbar := tqdm.tqdm(dataloader)):

        try:
            # print(batch.name)
            if train:
                optimizer.zero_grad()

            if debug:
                batch = batch.to(debug_device)
            denoiser_outputs = diffuser.forward(batch)

            loss_dict = loss_fn(batch, denoiser_outputs)
            # print(loss_dict)
            loss = loss_dict["loss"]

            if train:
                if debug:
                    loss.backward()
                else:
                    fabric.backward(loss)

                # check if there are any nans in grads
                # if so, rerun this step under anomaly detection
                nan_param_grad = False
                for name, param in diffuser.named_parameters():
                    if param.grad is not None and not param.grad.isfinite().all():
                        torch.set_printoptions(threshold=1000000)
                        print("nan in", name, "grad")
                        nan_param_grad = True

                if nan_param_grad:
                    optimizer.zero_grad()
                    with torch.autograd.detect_anomaly():
                        latent_data, decoder_outputs = diffuser.forward(batch, warmup=warmup)

                        loss_dict = loss_fn(batch, latent_data, decoder_outputs, warmup=warmup)
                        loss = loss_dict["loss"]
                        if debug:
                            loss.backward()
                        else:
                            fabric.backward(loss)
                else:
                    optimizer.step()


            epoch_dict = update_epoch_loss_dict(epoch_dict, loss_dict)
            epoch_loss = np.mean(epoch_dict['loss'])
            pbar_str = f"epoch_loss: {np.mean(epoch_loss):.4f}, "
            pbar_str += gen_pbar_str(loss_dict)
            pbar_str += f"t: {format_list(denoiser_outputs['t_per_graph'], '{:.4f}')}"
            pbar.set_description(pbar_str)

            if torch.isnan(loss).any():
                exit()
        except Exception as e:
            torch.save(batch, "problematic_inputs.pt")
            torch.save(denoiser_outputs, "decoder_outputs.pt")
            raise e
        # print("Done")
        # exit()

    return {key: np.mean(losses) for key, losses in epoch_dict.items()}
