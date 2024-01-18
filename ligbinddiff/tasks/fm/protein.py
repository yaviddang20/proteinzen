import copy
from typing import Sequence, Dict

import tqdm
import tree
import torch
import numpy as np

import torch_geometric.utils as pygu
from torch_geometric.data import HeteroData, Batch


from ligbinddiff.data.openfold import data_transforms
from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.tasks import Task
from ligbinddiff.model.utils.graph import batchwise_to_nodewise

from ligbinddiff.runtime.loss.frames import bb_frame_fm_loss
from ligbinddiff.runtime.loss.common import autoencoder_losses, latent_scalar_sidechain_fm_loss, _collect_from_seq
from ligbinddiff.stoch_interp.interpolate.se3 import _centered_gaussian, _uniform_so3
from ligbinddiff.stoch_interp.interpolate.latent import _centered_gaussian as _centered_rn_gaussian
from ligbinddiff.stoch_interp.interpolate.protein import ProteinInterpolant

import ligbinddiff.stoch_interp.interpolate.utils as du
from ligbinddiff.utils.framediff import all_atom


class ProteinInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'
    sidechain_x_1_key='latent_sidechain'
    sidechain_x_1_pred_key='pred_latent_sidechain'
    sidechain_x_t_key='noised_latent_sidechain'

    def __init__(self,
                 protein_noiser: ProteinInterpolant,
                 aux_loss_t_min=0.75,
                 compute_passthrough=False):
        super().__init__()
        self.se3_noiser = protein_noiser.se3_noiser
        self.sidechain_noiser = protein_noiser.sidechain_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.compute_passthrough = compute_passthrough

    def gen_diffuse_mask(self, data: HeteroData):
        return torch.ones_like(data['res_mask']).bool()

    def process_input(self, data: HeteroData):
        data = copy.deepcopy(data)
        self.se3_noiser.set_device(data['residue']['atom37'].device)
        self.sidechain_noiser.set_device(data['residue']['atom37'].device)
        res_data = data['residue']

        # compute base features
        chain_feats = {
            'aatype': torch.as_tensor(res_data['seq']).long(),
            'all_atom_positions': torch.as_tensor(res_data['atom37']).double(),
            'all_atom_mask': torch.as_tensor(res_data['atom37_mask']).double()
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles(prefix="")(chain_feats)  # TODO: uncurry this
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)

        rigids_1 = ru.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]

        # compute bb frame features
        diffuse_mask = self.gen_diffuse_mask(res_data)
        res_data['noising_mask'] = diffuse_mask
        res_data['x'] = rigids_1.get_trans()  # for HeteroData's sake
        res_data['rigids_1'] = rigids_1.to_tensor_7()
        res_data['noising_mask'] = diffuse_mask
        ##  noise data
        data = self.se3_noiser.corrupt_batch(data)

        # compute sidechain features
        ## generate data dict
        copy_keys = [
            "torsion_angles_sin_cos",
            "alt_torsion_angles_sin_cos",
            "torsion_angles_mask",
            "atom14_atom_exists",
            "atom14_gt_exists",
            "atom14_gt_positions",
            "atom14_alt_gt_exists",
            "atom14_alt_gt_positions",
        ]
        diff_feats_t = {k: chain_feats[k] for k in copy_keys}
        diff_feats_t['bb'] = diff_feats_t['atom14_gt_positions'][..., :4, :]
        diff_feats_t['atom37'] = chain_feats['all_atom_positions']
        diff_feats_t['atom37_mask'] = chain_feats['all_atom_mask']
        diff_feats_t = tree.map_structure(
            lambda x: torch.as_tensor(x),
            diff_feats_t)
        res_data.update(diff_feats_t)

        return data

    def process_sample_input(self, data: Dict, device='cpu'):
        self.se3_noiser.set_device(device)
        return data

    def _run_model(self, model, inputs, self_conditioning=None):
        # generate latent sidechains
        latent_data = model.encoder(inputs)
        ## sample only if we're training
        if model.training:
            latent_sigma = torch.exp(
                latent_data['latent_logvar'] * 0.5
            )
            latent_data[self.sidechain_x_1_key] = latent_data['latent_mu'] + latent_sigma * torch.randn_like(latent_sigma)

        else:
            latent_data[self.sidechain_x_1_key] = latent_data['latent_mu']

        ## center latent space as we can only fm/diffuse in centered Rn
        sampled_sidechain = latent_data[self.sidechain_x_1_key]
        sampled_center = pygu.scatter(
            sampled_sidechain,
            inputs['residue'].batch,
            dim_size=inputs.num_graphs
        )
        sampled_count = pygu.scatter(
            torch.ones_like(inputs['residue'].batch),
            inputs['residue'].batch,
            dim_size=inputs.num_graphs
        )
        sampled_center = sampled_center.sum(dim=-1) / (sampled_count * sampled_center.shape[-1])
        centered_sidechain = sampled_sidechain - batchwise_to_nodewise(
            sampled_center,
            inputs['residue'].batch
        )[..., None]
        latent_data[self.sidechain_x_1_key] = centered_sidechain


        # decoder
        decoder_outputs = model.decoder(inputs, latent_data)

        # fm
        noised_latent_data = self.sidechain_noiser.corrupt_batch(
            inputs,
            latent_data,
        )
        latent_outputs = model.denoiser(inputs, noised_latent_data, self_condition=self_conditioning)

        if self.compute_passthrough:
            # compute passthrough outputs
            passthrough_inputs = copy.copy(inputs)
            passthrough_inputs['residue']['rigids_1'] = latent_outputs['final_rigids'].to_tensor_7()
            passthrough_latent = {
                self.sidechain_x_1_key: latent_outputs[self.sidechain_x_1_pred_key]
            }
            passthrough_outputs = model.decoder(passthrough_inputs, passthrough_latent)
            passthrough_outputs.update(noised_latent_data)
            passthrough_outputs.update(latent_data)
        else:
            passthrough_outputs = None


        # update outputs for loss calculation
        latent_outputs.update(noised_latent_data)
        latent_outputs.update(latent_data)

        return latent_outputs, decoder_outputs, passthrough_outputs

    def run_eval(self, model, inputs):
        device = inputs['residue']['x'].device
        self.se3_noiser.set_device(device)
        self.sidechain_noiser.set_device(device)
        # TODO: should this be a separate flag?
        if model.self_conditioning and np.random.uniform() > 0.5:
            with torch.no_grad():
                self_conditioning, _, _ = self._run_model(model, inputs)
        else:
            self_conditioning = None
        denoiser_output, design_output, pt_outputs = self._run_model(model, inputs, self_conditioning)
        denoiser_output.update(design_output)
        denoiser_output["pt_outputs"] = pt_outputs
        return denoiser_output

    def run_predict(self,
                    model,
                    inputs,
                    device='cuda:0'):
        self.se3_noiser.set_device(device)
        self.sidechain_noiser.set_device(device)

        num_res = inputs['num_res']
        total_num_res = sum(num_res)
        data_list = []
        for n in num_res:
            data = HeteroData(
                residue={
                    "res_mask": torch.ones(n, device=device),
                    "noising_mask": torch.ones(n, device=device),
                    "num_nodes": n
                }
            )
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        res_data = batch['residue']
        # Set-up initial prior samples
        trans_0 = (
            _centered_gaussian(res_data.batch, device) * du.NM_TO_ANG_SCALE
        )
        rotmats_0 = _uniform_so3(total_num_res, device)
        sidechain_0 = _centered_rn_gaussian(
            res_data.batch,
            self.sidechain_noiser.dim_size,
            device
        )

        # Set-up time
        ts = torch.linspace(self.se3_noiser._cfg.min_t, 1.0, self.se3_noiser._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(
            trans_0,
            rotmats_0,
            torch.zeros((total_num_res, 2), device=device),
            sidechain_0
        )]
        clean_traj = []
        denoiser_out = None
        for t_2 in tqdm.tqdm(ts[1:]):
            # Run model.
            trans_t_1, rotmats_t_1, _, sidechain_t_1 = prot_traj[-1]
            res_data["trans_t"] = trans_t_1
            res_data["rotmats_t"] = rotmats_t_1
            res_data['rigids_t'] = ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_t_1),
                trans=trans_t_1
            ).to_tensor_7()
            t = torch.ones(batch.num_graphs, device=device) * t_1
            batch["t"] = t

            intermediates = {
                self.sidechain_x_t_key: sidechain_t_1
            }

            with torch.no_grad():
                denoiser_out = model.denoiser(batch, intermediates, self_condition=denoiser_out)

            # Process model output.
            pred_rigids = denoiser_out['final_rigids']
            pred_trans_1 = pred_rigids.get_trans()
            pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
            pred_psis = denoiser_out['psi'].detach().cpu()
            pred_latent_sidechain = denoiser_out[self.sidechain_x_1_pred_key]
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu(), pred_psis, pred_latent_sidechain.detach().cpu())
            )

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self.se3_noiser._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self.se3_noiser._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            sidechain_t_2 = self.sidechain_noiser._euler_step(d_t, t_1, pred_latent_sidechain, sidechain_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2, pred_psis, sidechain_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1, _, sidechain_t_1 = prot_traj[-1]
        res_data["trans_t"] = trans_t_1
        res_data["rotmats_t"] = rotmats_t_1
        res_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        t = torch.ones(batch.num_graphs, device=device) * t_1
        batch["t"] = t

        intermediates = {
            self.sidechain_x_t_key: sidechain_t_1
        }

        with torch.no_grad():
            denoiser_out = model.denoiser(batch, intermediates, self_condition=denoiser_out)

        # Process model output.
        pred_rigids = denoiser_out['final_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
        pred_psis = denoiser_out['psi'].detach().cpu()
        pred_latent_sidechain = denoiser_out[self.sidechain_x_1_pred_key].detach()
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu(), pred_psis, pred_latent_sidechain.cpu())
        )

        latent_output = {
            self.sidechain_x_1_key: pred_latent_sidechain
        }
        data_list = []
        for n in num_res:
            data = HeteroData(
                residue={
                    "res_mask": torch.ones(n, device=device).bool(),
                    "noising_mask": torch.ones(n, device=device).bool(),
                    "num_nodes": n
                }
            )
            data_list.append(data)

        decoder_inputs = Batch.from_data_list(data_list)
        decoder_inputs['residue'].update(
            {
                "rigids_1": pred_rigids.to_tensor_7(),
                "x": pred_rigids.get_trans(),
                "bb": denoiser_out["denoised_bb"][..., :4, :]
            }
        )
        decoder_output = model.decoder(decoder_inputs, latent_output)
        seq_logits = decoder_output['decoded_seq_logits']
        argmax_seq = seq_logits.argmax(dim=-1)
        all_atom14 = decoder_output['decoded_all_atom14']

        decoded_struct = _collect_from_seq(all_atom14, argmax_seq, torch.ones_like(argmax_seq).bool())

        # # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrotpsi_to_atom37(prot_traj, res_data.res_mask)
        # clean_atom37_traj = all_atom.transrotpsi_to_atom37(clean_traj, res_data.res_mask)

        return {
            "samples": decoded_struct.split(num_res),
            "seqs": argmax_seq.split(num_res),
            "inputs": inputs
        }


    def compute_loss(self, inputs, outputs: Dict):
        t_clip_max = 0.9
        bb_frame_diffusion_loss_dict = bb_frame_fm_loss(
            inputs, outputs)
        autoenc_loss_dict = autoencoder_losses(
            inputs, outputs
        )
        latent_loss_dict = latent_scalar_sidechain_fm_loss(inputs, outputs)

        bb_denoising_loss = (
            bb_frame_diffusion_loss_dict["trans_vf_loss"] * 2 +
            bb_frame_diffusion_loss_dict["rot_vf_loss"]
        )
        bb_denoising_finegrain_loss = (
            bb_frame_diffusion_loss_dict["scaled_pred_bb_mse"]
            + bb_frame_diffusion_loss_dict["scaled_dist_mat_loss"]
        ) * (inputs['t'] > self.aux_loss_t_min)

        vae_loss = (
            autoenc_loss_dict["atom14_mse"]
            + autoenc_loss_dict["seq_loss"]
            + autoenc_loss_dict["chi_loss"]
            + autoenc_loss_dict["kl_div"] * 1e-6
        )
        latent_denoising_loss = latent_loss_dict["latent_fm_loss"]

        if self.compute_passthrough:
            pt_outputs = outputs["pt_outputs"]
            assert pt_outputs is not None
            pt_loss_dict = autoencoder_losses(
                inputs, pt_outputs
            )
            pt_aux_loss = (
                pt_loss_dict["atom14_mse"]
                + pt_loss_dict["seq_loss"]
            ) * (inputs['t'] > self.aux_loss_t_min)
            pt_loss = (
                pt_loss_dict["chi_loss"]
            ) / torch.square(1 - torch.min(inputs['t'], torch.as_tensor(t_clip_max)))
            pt_loss = pt_loss + 0.25 * pt_aux_loss

            pt_loss_dict = {"pt_" + k: v for k,v in pt_loss_dict.items()}
        else:
            pt_loss = 0
            pt_loss_dict = {}

        loss = (
            bb_denoising_loss
            + latent_denoising_loss
            + 0.25 * bb_denoising_finegrain_loss
            + vae_loss
            + pt_loss
        ).mean()

        loss_dict = {"loss": loss}
        loss_dict.update(bb_frame_diffusion_loss_dict)
        loss_dict.update(autoenc_loss_dict)
        loss_dict.update(latent_loss_dict)
        loss_dict.update(pt_loss_dict)
        return loss_dict
