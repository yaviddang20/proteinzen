import copy
from typing import Sequence, Dict, Union
import os

import tqdm
import tree
import torch
import numpy as np
import torch.nn.functional as F

import torch_geometric.utils as pygu
from torch_geometric.data import HeteroData, Batch


from proteinzen.data.openfold import data_transforms
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.tasks import Task
from proteinzen.model.utils.graph import batchwise_to_nodewise
from proteinzen.runtime.loss.utils import _nodewise_to_graphwise

from proteinzen.runtime.loss.frames import bb_frame_fm_loss, all_atom_fape_loss
from proteinzen.runtime.loss.common import (
    autoencoder_losses, latent_scalar_sidechain_fm_loss, latent_encoder_consistency_loss,
    noisy_latent_losses, discrim_losses
)
from proteinzen.runtime.loss.atomic.holes import buried_cavity_loss
from proteinzen.runtime.loss.traj import traj_loss
from proteinzen.stoch_interp.interpolate.se3 import _centered_gaussian, _uniform_so3
from proteinzen.stoch_interp.interpolate.latent import _centered_gaussian as _centered_rn_gaussian
from proteinzen.stoch_interp.interpolate.protein import (
    ProteinInterpolant, ProteinFisherInterpolant, ProteinDirichletInterpolant,
    ProteinDirichletChiInterpolant, ProteinDirichletMultiChiInterpolant, ProteinCatFlowInterpolant,
    ProteinFisherMultiChiInterpolant
)

import proteinzen.stoch_interp.interpolate.utils as du
from proteinzen.utils.framediff import all_atom


def outputs_to_inputs(data, outputs, gt_seq=False):
    inputs = copy.copy(data)
    res_data = inputs['residue']
    res_data['rigids_1'] = outputs['final_rigids'].to_tensor_7()
    res_data['x'] = outputs['final_rigids'].get_trans()
    res_data['bb'] = outputs['denoised_bb'][:, :4]
    if gt_seq:
        res_data['atom14_gt_positions'] = outputs['decoded_atom14_gt_seq']
        res_data['seq_one_hot'] = outputs['gt_seq_one_hot']
    else:
        res_data['atom14_gt_positions'] = outputs['decoded_atom14']
        res_data['atom14_gt_exists'] = outputs['decoded_atom14_mask']
        res_data['atom14_noising_mask'] = outputs['decoded_atom14_mask']
        res_data['seq'] = outputs['decoded_seq_logits'].argmax(dim=-1)
        res_data['seq_one_hot'] = outputs['decoded_seq_one_hot']
    return inputs


def no_grad_for_model(model, *args, **kwargs):
    for param in model.parameters():
        param.requires_grad = False
    output = model(*args, **kwargs)
    for param in model.parameters():
        param.requires_grad = True
    return output

def no_grad_for_input(model, data):
    return model(data.detach())


class ProteinInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'
    sidechain_x_1_key='latent_sidechain'
    sidechain_x_1_pred_key='pred_latent_sidechain'
    sidechain_x_t_key='noised_latent_sidechain'

    def __init__(self,
                 protein_noiser: ProteinInterpolant,
                 aux_loss_t_min=0.25,
                 compute_passthrough=True,
                 pt_loss_t_min=0.0,
                 pt_clash_loss_t=1.1,
                 kl_strength=0,#1e-6,
                 mu_kl_strength=0,
                 rescale_kl_noise=False,
                 norm_latent_space=False,
                 use_smooth_lddt=False,
                 t_clip_se3=0.9,
                 t_clip_aa=0.9,
                 disable_bb_aux_loss=False,
                 square_bb_aux_loss_t_factor=False,
                 vae_seq_masking=False,
                 vae_loss=True,
                 vae_seq_loss=True,
                 train_vae_only=False,
                 no_grad_encoder=False,
                 latent_fm_loss_scale=1,
                 latent_pointwise_loss=False,
                 denoising_detach_gt_latent_grad=True,
                 denoising_twosided_detach_latent_grad=False,
                 beta_commit=0.25,
                 t_min=0.01,
                 t_max=0.99,
                 lognorm_sample_t=False,
                 lognorm_t_mu=0.0,
                 lognorm_t_std=1.0,
                 rigid_traj_loss=False,
                 seq_traj_loss=False,
                 aa_traj_loss=False,
                 dist_traj_loss=False,
                 use_fape_loss=False,
                 fape_length_scale=1.,
                 traj_decay_factor=0.99,
                 percent_all_mask=0.25,
                 percent_no_mask=0.25,
                 encoder_consistency_check=False,
                 supervise_noisy_latent=False,
                 use_buried_cavity_loss=False,
                 use_edge_dist_loss=False,
                 use_gan_losses=False
    ):
        super().__init__()
        assert not (denoising_detach_gt_latent_grad and denoising_twosided_detach_latent_grad)
        self.se3_noiser = protein_noiser.se3_noiser
        self.sidechain_noiser = protein_noiser.sidechain_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.compute_passthrough = compute_passthrough
        self.pt_loss_t_min = pt_loss_t_min
        self.pt_clash_loss_t = pt_clash_loss_t
        self.rng = np.random.default_rng()
        self.kl_strength = kl_strength
        self.mu_kl_strength = mu_kl_strength
        self.rescale_kl_noise = rescale_kl_noise
        self.use_smooth_lddt = use_smooth_lddt
        self.use_fape_loss = use_fape_loss
        self.fape_length_scale = fape_length_scale
        self.t_clip_se3 = t_clip_se3
        self.t_clip_aa = t_clip_aa
        self.disable_bb_aux_loss = disable_bb_aux_loss
        self.vae_seq_masking = vae_seq_masking
        self.vae_loss = vae_loss
        self.vae_seq_loss = vae_seq_loss
        self.no_grad_encoder = no_grad_encoder
        self.train_vae_only = train_vae_only
        self.latent_fm_loss_scale = latent_fm_loss_scale
        self.latent_pointwise_loss = latent_pointwise_loss
        self.traj_decay_factor = traj_decay_factor
        self.rigid_traj_loss = rigid_traj_loss
        self.seq_traj_loss = seq_traj_loss
        self.dist_traj_loss = dist_traj_loss
        self.square_bb_aux_loss_t_factor = square_bb_aux_loss_t_factor
        self.percent_all_mask = percent_all_mask
        self.percent_no_mask = percent_no_mask
        self.norm_latent_space = norm_latent_space
        self.encoder_consistency_check = encoder_consistency_check
        self.supervise_noisy_latent = supervise_noisy_latent
        self.use_buried_cavity_loss = use_buried_cavity_loss
        self.use_edge_dist_loss = use_edge_dist_loss
        self.use_gan_losses = use_gan_losses

        self.denoising_detach_gt_latent_grad = denoising_detach_gt_latent_grad
        self.denoising_twosided_detach_latent_grad = denoising_twosided_detach_latent_grad
        self.beta_commit = beta_commit

        self.t_min = t_min
        self.t_max = t_max
        self.lognorm_sample_t = lognorm_sample_t
        self.lognorm_t_mu = lognorm_t_mu
        self.lognorm_t_std = lognorm_t_std

    def _gen_diffuse_mask(self, data: HeteroData):
        return torch.ones_like(data['res_mask']).bool()

    def _gen_seq_noising_mask(self, data: HeteroData):
        # return torch.zeros_like(data['res_mask']).bool()
        select_noising_scheme = self.rng.random()
        if select_noising_scheme < self.percent_no_mask:
            return torch.ones_like(data['res_mask']).bool()
        elif select_noising_scheme > 1 - self.percent_all_mask:
            return torch.zeros_like(data['res_mask']).bool()
        else:
            percent = self.rng.random()
            return torch.rand(data['res_mask'].shape, device=data['res_mask'].device) > percent

    def _sample_t(self, num_batch, device):
        if self.lognorm_sample_t:
            t = torch.randn(num_batch, device=device).float()
            t = (t * self.lognorm_t_std) + self.lognorm_t_mu
            t = torch.sigmoid(t)
        else:
            t = torch.rand(num_batch, device=device).float()
        return t * (1 - 2 * self.t_min) + self.t_min

    def process_input(self, data: HeteroData):
        data = copy.deepcopy(data)
        self.se3_noiser.set_device(data['residue']['atom37'].device)
        self.sidechain_noiser.set_device(data['residue']['atom37'].device)
        res_data = data['residue']

        if "latent_mu" in res_data and "latent_logvar" in res_data:
            right_size = (
                res_data["latent_mu"].shape[0] == res_data.num_nodes
            ) and (
                res_data["latent_logvar"].shape[0] == res_data.num_nodes
            )
            if not right_size:
                del res_data["latent_mu"]
                del res_data["latent_logvar"]

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
        diffuse_mask = self._gen_diffuse_mask(res_data)
        res_data['noising_mask'] = diffuse_mask
        res_data['res_noising_mask'] = diffuse_mask
        if self.vae_seq_masking:
            res_data['seq_noising_mask'] = self._gen_seq_noising_mask(res_data)
        else:
            res_data['seq_noising_mask'] = diffuse_mask
        res_data['mlm_mask'] = ~diffuse_mask
        res_data['x'] = rigids_1.get_trans()  # for HeteroData's sake
        res_data['rigids_1'] = rigids_1.to_tensor_7()

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

        # redundant for convenience
        diff_feats_t['atom14'] = chain_feats['atom14_gt_positions']
        diff_feats_t['atom14_mask'] = chain_feats['atom14_gt_exists']
        res_data.update(diff_feats_t)

        # noising masks
        res_data['atom14_noising_mask'] = res_data['atom14_mask']

        data['t'] = self._sample_t(data.num_graphs, self.se3_noiser._device)
        data = self.se3_noiser.corrupt_batch(data)

        return data

    def process_sample_input(self, data: Dict, device='cpu'):
        self.se3_noiser.set_device(device)
        return data

    def _run_model(self, model, inputs, self_conditioning=None, pt_use_gt_seq=True):
        res_data = inputs['residue']

        discrim_outputs = {}

        if 'latent_mu' in res_data and 'latent_logvar' in res_data:
            # print("using cached latents")
            latent_data = {
                "latent_mu": res_data['latent_mu'],
                "latent_logvar": res_data['latent_logvar'],
            }
            if 'latent_norm_mu' in res_data and 'latent_norm_std' in res_data:
                latent_data.update({
                    "latent_norm_mu": res_data['latent_norm_mu'],
                    "latent_norm_std": res_data['latent_norm_std']
                })
        else:
            # print("generate latent")
            # generate latent sidechains
            if self.no_grad_encoder:
                with torch.no_grad():
                    latent_data = model.encoder(inputs, apply_noising_masks=self.vae_seq_masking)
            else:
                latent_data = model.encoder(inputs, apply_noising_masks=self.vae_seq_masking)

            if "tmpfile_path" in inputs:
                for i, tmpfile_path in enumerate(inputs['tmpfile_path']):
                    select = (res_data.batch == i)
                    save_dict = {
                        'latent_mu': latent_data['latent_mu'][select].detach().cpu(),
                        'latent_logvar': latent_data['latent_logvar'][select].detach().cpu(),
                    }
                    torch.save(save_dict, tmpfile_path)


        ## sample only if we're training
        if not isinstance(latent_data, dict):
            latent_data = {
                self.sidechain_x_1_key: latent_data,
                'latent_mu': latent_data,
                'latent_logvar': torch.zeros_like(latent_data)

            }
        elif model.training:
            latent_sigma = torch.exp(
                latent_data['latent_logvar'] * 0.5
            )
            latent_data[self.sidechain_x_1_key] = latent_data['latent_mu'] + latent_sigma * torch.randn_like(latent_sigma)
            if "latent_norm_mu" in latent_data and "latent_norm_std" in latent_data:
                latent_data[self.sidechain_x_1_key] = (
                    latent_data[self.sidechain_x_1_key] - latent_data['latent_norm_mu'][res_data.batch]
                ) / latent_data['latent_norm_std'][res_data.batch]
        else:
            latent_data[self.sidechain_x_1_key] = latent_data['latent_mu']

        if self.norm_latent_space:
            res_data = inputs['residue']
            latent_sidechains = latent_data[self.sidechain_x_1_key]
            sidechain_centers = pygu.scatter(
                latent_sidechains[res_data.res_mask],
                res_data.batch[res_data.res_mask],
                dim=0,
                reduce='mean'
            )
            sidechain_var = pygu.scatter(
                (latent_sidechains[res_data.res_mask] - sidechain_centers[res_data.batch[res_data.res_mask]])**2,
                res_data.batch[res_data.res_mask],
                dim=0,
            )
            sidechain_count = pygu.scatter(
                torch.ones_like(res_data.res_mask).float()[res_data.res_mask],
                res_data.batch[res_data.res_mask],
                dim=0,
            )
            sidechain_count[sidechain_count < 2] = 2
            sidechain_std = torch.sqrt(sidechain_var / (sidechain_count-1)[..., None])

            latent_sidechains = (latent_sidechains - sidechain_centers[res_data.batch]) / sidechain_std[res_data.batch]
            latent_data[self.sidechain_x_1_key] = latent_sidechains

            if self.rescale_kl_noise:
                latent_sigma = torch.exp(
                    latent_data['latent_logvar'] * 0.5
                )
                latent_sigma = latent_sigma / torch.maximum(sidechain_std[res_data.batch], torch.tensor(1e-8, device=latent_sigma.device))
                latent_data['latent_logvar'] = torch.log(2 * latent_sigma)

        # decoder
        decoder_outputs = model.decoder(inputs, latent_data)

        if self.use_gan_losses:
            copy_decoder_outputs = copy.copy(decoder_outputs)
            copy_decoder_outputs['final_rigids'] = ru.Rigid.from_tensor_7(inputs['residue']['rigids_1'])
            copy_decoder_outputs['denoised_bb'] = inputs['residue']['atom37'][:, (0, 1, 2, 4, 3)]
            repack_data = outputs_to_inputs(inputs, copy_decoder_outputs, gt_seq=True)
            discrim_outputs["gt_bb_gt_seq_repack_score_G_grad"] = no_grad_for_model(model.discriminator, repack_data)
            # discrim_outputs["gt_bb_gt_seq_repack_score_D_grad"] = no_grad_for_input(model.discriminator, repack_data)
            redesign_data = outputs_to_inputs(inputs, copy_decoder_outputs, gt_seq=False)
            discrim_outputs["gt_bb_pred_seq_repack_score_G_grad"] = no_grad_for_model(model.discriminator, redesign_data)
            # discrim_outputs["gt_bb_pred_seq_repack_score_D_grad"] = no_grad_for_input(model.discriminator, redesign_data)

        if self.train_vae_only:
            latent_outputs = latent_data
            passthrough_outputs = None
        else:
            # fm
            noised_latent_data = self.sidechain_noiser.corrupt_batch(
                inputs,
                latent_data,
            )
            latent_outputs = model.denoiser(inputs, noised_latent_data, self_condition=self_conditioning)
            if hasattr(model, "latent_supervisor"):
                noised_seq_logits, noised_torsion_logits = model.latent_supervisor(noised_latent_data[self.sidechain_x_t_key])
                latent_outputs['seq_logits_from_noisy_latent'] = noised_seq_logits
                latent_outputs['torsion_logits_from_noisy_latent'] = noised_torsion_logits

            if self.compute_passthrough:
                # compute passthrough outputs
                passthrough_inputs = copy.copy(inputs)
                passthrough_inputs['residue']['rigids_1'] = latent_outputs['final_rigids'].to_tensor_7()
                passthrough_inputs['residue']['x'] = latent_outputs['final_rigids'].get_trans()
                passthrough_inputs['residue']['bb'] = latent_outputs['denoised_bb'][:, :4]
                passthrough_latent = {
                    self.sidechain_x_1_key: latent_outputs[self.sidechain_x_1_pred_key]
                }
                passthrough_outputs = model.decoder(
                    passthrough_inputs,
                    passthrough_latent,
                    t=inputs['t'],
                    use_gt_seq=pt_use_gt_seq)
                passthrough_outputs.update(noised_latent_data)
                passthrough_outputs.update(latent_data)
                passthrough_outputs['final_rigids'] = latent_outputs['final_rigids']

                if self.encoder_consistency_check:
                    consistency_inputs = outputs_to_inputs(passthrough_inputs, passthrough_outputs)
                    consistency_data = no_grad_for_model(model.encoder, consistency_inputs, apply_noising_masks=False)
                    passthrough_outputs['consistency_data'] = consistency_data
                    passthrough_outputs.update(latent_outputs)

                if self.use_gan_losses:
                    passthrough_outputs['denoised_bb'] = latent_outputs['denoised_bb']
                    pt_gt_seq_data = outputs_to_inputs(passthrough_inputs, passthrough_outputs, gt_seq=True)
                    discrim_outputs["pred_bb_gt_seq_score_G_grad"] = no_grad_for_model(model.discriminator, pt_gt_seq_data)
                    # discrim_outputs["pred_bb_gt_seq_score_D_grad"] = no_grad_for_input(model.discriminator, pt_gt_seq_data)
                    pt_pred_seq_data = outputs_to_inputs(passthrough_inputs, passthrough_outputs, gt_seq=False)
                    discrim_outputs["pred_bb_pred_seq_score_G_grad"] = no_grad_for_model(model.discriminator, pt_pred_seq_data)
                    # discrim_outputs["pred_bb_pred_seq_score_D_grad"] = no_grad_for_input(model.discriminator, pt_pred_seq_data)
            else:
                passthrough_outputs = None

            # update outputs for loss calculation
            latent_outputs.update(noised_latent_data)
            latent_outputs.update(latent_data)

        latent_outputs['discrim_outputs'] = discrim_outputs
            # import json
            # print(json.dumps({k: v.mean().item() for k, v in discrim_outputs.items()}, indent=4))

        return latent_outputs, decoder_outputs, passthrough_outputs

    def run_eval(self, model, inputs):
        device = inputs['residue']['x'].device
        self.se3_noiser.set_device(device)
        self.sidechain_noiser.set_device(device)
        # TODO: should this be a separate flag?
        if not self.train_vae_only and model.self_conditioning and np.random.uniform() > 0.5:
            with torch.no_grad():
                self_conditioning, _, sc_pt_outputs = self._run_model(model, inputs, pt_use_gt_seq=True)
                if sc_pt_outputs is not None:
                    self_conditioning.update(sc_pt_outputs)
        else:
            self_conditioning = None

        denoiser_output, design_output, pt_outputs = self._run_model(model, inputs, self_conditioning)
        denoiser_output.update(design_output)
        denoiser_output["pt_outputs"] = pt_outputs
        return denoiser_output

    def run_discrim(self, model, inputs, model_outputs):
        passthrough_outputs = model_outputs['pt_outputs']

        discrim_outputs = {}
        inputs['residue']['seq_one_hot'] = F.one_hot(inputs['residue']['seq'], num_classes=21)
        discrim_outputs["gt_all_score"] = model.discriminator(inputs)

        copy_decoder_outputs = copy.copy(model_outputs)
        copy_decoder_outputs['final_rigids'] = ru.Rigid.from_tensor_7(inputs['residue']['rigids_1'])
        copy_decoder_outputs['denoised_bb'] = inputs['residue']['atom37'][:, (0, 1, 2, 4, 3)]
        repack_data = outputs_to_inputs(inputs, copy_decoder_outputs, gt_seq=True)
        # discrim_outputs["gt_bb_gt_seq_repack_score_G_grad"] = no_grad_for_model(model.discriminator, repack_data)
        discrim_outputs["gt_bb_gt_seq_repack_score_D_grad"] = no_grad_for_input(model.discriminator, repack_data)
        redesign_data = outputs_to_inputs(inputs, copy_decoder_outputs, gt_seq=False)
        # discrim_outputs["gt_bb_pred_seq_repack_score_G_grad"] = no_grad_for_model(model.discriminator, redesign_data)
        discrim_outputs["gt_bb_pred_seq_repack_score_D_grad"] = no_grad_for_input(model.discriminator, redesign_data)

        if not self.train_vae_only:
            passthrough_outputs['denoised_bb'] = model_outputs['denoised_bb']
            passthrough_inputs = copy.copy(inputs)
            passthrough_inputs['residue']['rigids_1'] = model_outputs['final_rigids'].to_tensor_7()
            passthrough_inputs['residue']['x'] = model_outputs['final_rigids'].get_trans()
            passthrough_inputs['residue']['bb'] = model_outputs['denoised_bb'][:, :4]
            pt_gt_seq_data = outputs_to_inputs(passthrough_inputs, passthrough_outputs, gt_seq=True)
            # discrim_outputs["pred_bb_gt_seq_score_G_grad"] = no_grad_for_model(model.discriminator, pt_gt_seq_data)
            discrim_outputs["pred_bb_gt_seq_score_D_grad"] = no_grad_for_input(model.discriminator, pt_gt_seq_data)
            pt_pred_seq_data = outputs_to_inputs(passthrough_inputs, passthrough_outputs, gt_seq=False)
            # discrim_outputs["pred_bb_pred_seq_score_G_grad"] = no_grad_for_model(model.discriminator, pt_pred_seq_data)
            discrim_outputs["pred_bb_pred_seq_score_D_grad"] = no_grad_for_input(model.discriminator, pt_pred_seq_data)

        discrim_loss_dict = discrim_losses(
            inputs, {"discrim_outputs": discrim_outputs}, losses_G=False, train_vae_only=self.train_vae_only
        )
        # import json
        # print(json.dumps({k: v.mean().item() for k, v in discrim_outputs.items()}, indent=4))
        if self.train_vae_only:
            loss = (
                discrim_loss_dict["discrim_gt_loss"] +
                discrim_loss_dict["discrim_fixed_bb_D_loss"] * 0.5
            )
        else:
            loss = (
                discrim_loss_dict["discrim_gt_loss"] +
                discrim_loss_dict["discrim_fixed_bb_D_loss"] * 0.25 +
                discrim_loss_dict["pt_discrim_pt_bb_D_loss"] * 0.25 * (inputs['t'] > self.pt_loss_t_min)
            )

        return loss.mean(), discrim_loss_dict["discrim_gt_loss"], discrim_loss_dict["discrim_gt_score"]


    def run_predict(self,
                    model,
                    inputs,
                    device='cuda:0'):
                    # device='cpu'):
        self.se3_noiser.set_device(device)
        self.sidechain_noiser.set_device(device)

        num_res = inputs['num_res']
        total_num_res = sum(num_res)
        data_list = []
        for n in num_res:
            data = HeteroData(
                residue={
                    "res_mask": torch.ones(n, device=device).bool(),
                    "seq_mask": torch.ones(n, device=device).bool(),
                    "noising_mask": torch.ones(n, device=device).bool(),
                    "res_noising_mask": torch.ones(n, device=device).bool(),
                    "seq_noising_mask": torch.ones(n, device=device).bool(),
                    "atom14_gt_positions": torch.zeros((n, 14, 3), device=device).float(),
                    "seq": torch.ones(n, device=device).long() * 20,  # should be X
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
        if getattr(model, "latent_conv_downsample_factor", 0) > 0:
            latent_prior = model.sample_prior(
                int(inputs['num_res'][0]),
                device,
                num_batch=len(inputs['num_res'])
            )
        else:
            latent_prior = model.sample_prior(
                int(res_data.batch.numel()),
                device
            )
        sidechain_0 = latent_prior['noised_latent_sidechain']

        # Set-up time
        ts = torch.linspace(self.se3_noiser._cfg.min_t, 1.0, self.se3_noiser._sample_cfg.num_timesteps)
        t_1 = ts[0]

        init_bb_psi = torch.zeros((total_num_res, 2), device=device)  # bb psi
        init_bb_psi[:, 0] = 1
        init_atom14 = all_atom.compute_backbone(
            ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_0),
                trans=trans_0
            ),
            init_bb_psi,
            impute_O=False,
        )[-1]

        prot_traj = [(
            trans_0,  # trans
            rotmats_0,  # rot
            init_bb_psi,
            sidechain_0,  # latent sidechain,
            torch.ones((total_num_res, 21), device=device).float(),  # seq logits
            init_atom14  # atom14 struct
        )]
        clean_traj = []
        denoiser_out = None

        for t_2 in tqdm.tqdm(ts[1:]):
            # Run model.
            trans_t_1, rotmats_t_1, _, sidechain_t_1, _, _ = prot_traj[-1]
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

                # denoiser_out[self.sidechain_x_1_pred_key] = torch.zeros_like(denoiser_out[self.sidechain_x_1_pred_key])

                pred_latent_sidechain = denoiser_out[self.sidechain_x_1_pred_key].detach()

                if 'latent_norm_mu' in inputs and 'latent_norm_std' in inputs:
                    print(pred_latent_sidechain.shape, inputs['latent_norm_mu'][None].shape)
                    norm_pred_latent_sidechain = (
                        pred_latent_sidechain + inputs['latent_norm_mu'][None]
                    ) * inputs['latent_norm_std'][None]
                    latent_output = {
                        self.sidechain_x_1_key: norm_pred_latent_sidechain
                    }
                else:
                    latent_output = {
                        self.sidechain_x_1_key: pred_latent_sidechain
                    }


                data_list = []
                for n in num_res:
                    data = HeteroData(
                        residue={
                            "res_mask": torch.ones(n, device=device).bool(),
                            "seq_mask": torch.ones(n, device=device).bool(),
                            "noising_mask": torch.ones(n, device=device).bool(),
                            "res_noising_mask": torch.ones(n, device=device).bool(),
                            "seq_noising_mask": torch.ones(n, device=device).bool(),
                            "atom14_gt_positions": torch.zeros((n, 14, 3), device=device).float(),
                            "seq": torch.ones(n, device=device).long() * 20,  # should be X
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
                decoder_output = model.decoder(
                    decoder_inputs,
                    latent_output,
                    t=t,
                    use_gt_seq=False
                )
                denoiser_out.update(decoder_output)

                clean_traj.append(
                    (
                        pred_trans_1.detach().cpu(),
                        pred_rotmats_1.detach().cpu(),
                        pred_psis,
                        pred_latent_sidechain.cpu(),
                        decoder_output['decoded_seq_logits'].detach().cpu().argmax(dim=-1),
                        decoder_output['decoded_atom14'].detach().cpu()
                    )
                )

            # # Process model output.
            # pred_rigids = denoiser_out['final_rigids']
            # pred_trans_1 = pred_rigids.get_trans()
            # pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
            # pred_psis = denoiser_out['psi'].detach().cpu()
            # pred_latent_sidechain = denoiser_out[self.sidechain_x_1_pred_key]
            # clean_traj.append(
            #     (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu(), pred_psis, pred_latent_sidechain.detach().cpu())
            # )

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self.se3_noiser._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self.se3_noiser._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            sidechain_t_2 = self.sidechain_noiser._euler_step(
                d_t=d_t,
                t=t_1,
                x_1=pred_latent_sidechain,
                x_t=sidechain_t_1)

            # sidechain_t_2 = torch.randn_like(sidechain_t_2) * 10
            # sidechain_t_2 = torch.zeros_like(sidechain_t_2)

            atom14_t_2 = all_atom.compute_backbone(
                ru.Rigid(
                    rots=ru.Rotation(rot_mats=rotmats_t_2),
                    trans=trans_t_2
                ),
                init_bb_psi,
                impute_O=False,
            )[-1]

            prot_traj.append(
                (trans_t_2,
                 rotmats_t_2,
                 pred_psis,
                 sidechain_t_2,
                 decoder_output["decoded_seq_logits"].argmax(dim=-1).detach().cpu(),
                 atom14_t_2.detach().cpu()
                ))
            t_1 = t_2

            if not model.self_conditioning:
                denoiser_out = None

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1, _, sidechain_t_1, _, _ = prot_traj[-1]
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

        if 'latent_norm_mu' in inputs and 'latent_norm_std' in inputs:
            norm_pred_latent_sidechain = (
                pred_latent_sidechain + inputs['latent_norm_mu'][None]
            ) * inputs['latent_norm_std'][None]
            latent_output = {
                self.sidechain_x_1_key: norm_pred_latent_sidechain
            }
        else:
            latent_output = {
                self.sidechain_x_1_key: pred_latent_sidechain
            }
        data_list = []
        for n in num_res:
            data = HeteroData(
                residue={
                    "res_mask": torch.ones(n, device=device).bool(),
                    "noising_mask": torch.ones(n, device=device).bool(),
                    "seq_noising_mask": torch.ones(n, device=device).bool(),
                    "atom14_gt_positions": torch.zeros((n, 14, 3), device=device).float(),
                    "seq": torch.ones(n, device=device).long() * 20,  # should be X
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
        decoder_output = model.decoder(decoder_inputs, latent_output, use_gt_seq=False)
        seq_logits = decoder_output['decoded_seq_logits']
        argmax_seq = seq_logits.argmax(dim=-1)
        decoded_struct = decoder_output['decoded_atom14']

        clean_traj.append(
            (
                pred_trans_1.detach().cpu(),
                pred_rotmats_1.detach().cpu(),
                pred_psis,
                pred_latent_sidechain.cpu(),
                decoder_output['decoded_seq_logits'].detach().cpu().argmax(dim=-1),
                decoder_output['decoded_atom14'].detach().cpu()
            )
        )

        encoder_inputs = batch#.copy()
        res_data = encoder_inputs['residue']
        res_data['atom14_gt_positions'] = decoder_output['decoded_atom14']
        res_data['atom14_gt_exists'] = decoder_output['decoded_atom14_mask']
        res_data['atom14_noising_mask'] = decoder_output['decoded_atom14_mask']
        res_data['rigids_1'] = pred_rigids.to_tensor_7()
        res_data['seq'] = argmax_seq
        encoder_inputs['name'] = "sample"
        latent_data = model.encoder(encoder_inputs, apply_noising_masks=self.vae_seq_masking)
        print(
            torch.mean(torch.linalg.vector_norm(pred_latent_sidechain - latent_data['latent_mu'], dim=-1)),
            torch.mean(torch.linalg.vector_norm((0.5 * latent_data['latent_logvar']).exp()))
        )
        print(
            torch.mean(
                torch.abs(
                    (pred_latent_sidechain - latent_data['latent_mu']) / (0.5 * latent_data['latent_logvar']).exp()
                )
            )
        )
        print(torch.mean(torch.linalg.vector_norm(pred_latent_sidechain, dim=-1)))
        print(torch.mean(torch.linalg.vector_norm(latent_data['latent_mu'], dim=-1)))

        # all_atom14 = decoder_output['decoded_all_atom14']

        # decoded_struct = _collect_from_seq(all_atom14, argmax_seq, torch.ones_like(argmax_seq).bool())

        # # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrotpsi_to_atom37(prot_traj, res_data.res_mask)
        # clean_atom37_traj = all_atom.transrotpsi_to_atom37(clean_traj, res_data.res_mask)
        clean_trajs = zip(*[traj[-1].split(num_res) for traj in clean_traj])
        clean_traj_seqs = zip(*[traj[-2].split(num_res) for traj in clean_traj])
        prot_trajs = zip(*[traj[-1].split(num_res) for traj in prot_traj])

        return {
            "samples": decoded_struct.split(num_res),
            "seqs": argmax_seq.split(num_res),
            "clean_trajs": clean_trajs,
            "clean_traj_seqs": clean_traj_seqs,
            "prot_trajs": prot_trajs,
            "inputs": inputs
        }


    def compute_loss(self, inputs, outputs: Dict):
        if not self.train_vae_only:
            bb_frame_diffusion_loss_dict = bb_frame_fm_loss(
                inputs, outputs, sep_rot_loss=True,
                t_norm_clip=self.t_clip_se3,
                square_aux_loss_time_factor=self.square_bb_aux_loss_t_factor)
            latent_loss_dict = latent_scalar_sidechain_fm_loss(
                inputs,
                outputs,
                scale=1,
                pointwise=self.latent_pointwise_loss,
                detach_gt_latent_grad=self.denoising_detach_gt_latent_grad,
                two_sided_loss=self.denoising_twosided_detach_latent_grad,
                beta_commit=self.beta_commit
            )
            bb_denoising_loss = (
                bb_frame_diffusion_loss_dict["trans_vf_loss"] +
                bb_frame_diffusion_loss_dict["rot_vf_loss"]
            )
            bb_denoising_finegrain_loss = (
                bb_frame_diffusion_loss_dict["scaled_pred_bb_mse"]
                + bb_frame_diffusion_loss_dict["scaled_dist_mat_loss"]
                + bb_frame_diffusion_loss_dict["scaled_edge_dist_loss"] * self.use_edge_dist_loss
            ) * (inputs['t'] > self.aux_loss_t_min)
            bb_denoising_finegrain_loss = bb_denoising_finegrain_loss * (not self.disable_bb_aux_loss)

            latent_denoising_loss = latent_loss_dict["latent_fm_loss"]

            with torch.no_grad():
                frameflow_bb_frame_diffusion_loss_dict = bb_frame_fm_loss(
                    inputs, outputs, sep_rot_loss=True,
                    t_norm_clip=0.9)
                frameflow_bb_denoising_loss = (
                    frameflow_bb_frame_diffusion_loss_dict["trans_vf_loss"] +
                    frameflow_bb_frame_diffusion_loss_dict["rot_vf_loss"]
                )
                frameflow_bb_denoising_finegrain_loss = (
                    frameflow_bb_frame_diffusion_loss_dict["scaled_pred_bb_mse"]
                    + frameflow_bb_frame_diffusion_loss_dict["scaled_dist_mat_loss"]
                ) * (inputs['t'] > 0.25)
                frameflow_loss = (frameflow_bb_denoising_loss + 0.25 * frameflow_bb_denoising_finegrain_loss).mean()

            if any([self.rigid_traj_loss, self.seq_traj_loss, self.dist_traj_loss]):
                traj_loss_dict = traj_loss(inputs, outputs, traj_decay_factor=self.traj_decay_factor)
            else:
                traj_loss_dict = {}


        else:
            bb_frame_diffusion_loss_dict = {}
            traj_loss_dict = {}
            latent_loss_dict = {}
            bb_denoising_loss = 0
            bb_denoising_finegrain_loss = 0
            latent_denoising_loss = 0
            frameflow_loss = 0


        autoenc_loss_dict = autoencoder_losses(
            inputs, outputs, use_smooth_lddt=self.use_smooth_lddt,
            t_norm_clip=self.t_clip_aa,
            apply_seq_noising_mask=self.vae_seq_masking,
            use_sidechain_dists_mse_loss=(not self.use_smooth_lddt),
            use_local_atomic_dist_loss=False,
            use_sidechain_clash_loss=False,
        )

        vae_loss = (
            autoenc_loss_dict["atom14_mse"]
            + autoenc_loss_dict["sidechain_dists_mse"] * (not self.use_smooth_lddt)
            # + autoenc_loss_dict["pred_sidechain_clash_loss"]
            + autoenc_loss_dict["seq_loss"] * self.vae_seq_loss
            + autoenc_loss_dict["chi_loss"]
            + autoenc_loss_dict["kl_div"] * self.kl_strength
            + autoenc_loss_dict["kl_div_mu"] * self.mu_kl_strength
            + autoenc_loss_dict["smooth_lddt"]
        )

        if self.use_buried_cavity_loss:
            buried_cavity_loss_dict = buried_cavity_loss(
                inputs, outputs
            )
            vae_loss = vae_loss + buried_cavity_loss_dict['cavity_volume_loss']
        else:
            buried_cavity_loss_dict = {}

        # latent_denoising_loss = latent_loss_dict["latent_denoising_loss"] * 0.01

        if not self.train_vae_only and self.compute_passthrough:
            pt_outputs = outputs["pt_outputs"]
            assert pt_outputs is not None
            pt_loss_dict = autoencoder_losses(
                inputs, pt_outputs,
                use_smooth_lddt=self.use_smooth_lddt,
                use_fape=self.use_fape_loss,
                fape_length_scale=self.fape_length_scale,
                t_norm_clip=self.t_clip_aa,
                apply_seq_noising_mask=False,
                use_sidechain_dists_mse_loss=(not self.use_smooth_lddt),
                use_local_atomic_dist_loss=False,
                use_sidechain_clash_loss=False,
            )
            norm = 1 - torch.min(inputs['t'], torch.as_tensor(self.t_clip_aa))
            pt_abs_pos_loss = (
                pt_loss_dict["scaled_atom14_mse"]
                + pt_loss_dict["sidechain_dists_mse"] * (not self.use_smooth_lddt)
                + pt_loss_dict["smooth_lddt"] # / (norm ** 2)
                + pt_loss_dict["fape"] # / (norm ** 2)
                # + pt_loss_dict["chi_loss"]
                # + pt_loss_dict["seq_loss"]
            ) * (inputs['t'] > self.aux_loss_t_min)
            pt_rel_pos_loss = (
                # pt_loss_dict["atom14_mse"]
                # + pt_loss_dict["sidechain_dists_mse"]
                + pt_loss_dict["chi_loss"]
                + pt_loss_dict["seq_loss"]
            )
            if self.encoder_consistency_check:
                consistency_loss_dict = latent_encoder_consistency_loss(
                    inputs,
                    pt_outputs
                )
                pt_abs_pos_loss += (
                    consistency_loss_dict["consistency_nll_gt_scaled"]
                    + consistency_loss_dict["consistency_nll_self_scaled"]
                ) * (inputs['t'] > self.aux_loss_t_min)
                pt_loss_dict.update(consistency_loss_dict)
            if self.use_buried_cavity_loss:
                pt_buried_cavity_loss_dict = buried_cavity_loss(
                    inputs, outputs
                )
                pt_abs_pos_loss = pt_abs_pos_loss + pt_buried_cavity_loss_dict['cavity_volume_loss']
                pt_loss_dict.update(pt_buried_cavity_loss_dict)

            pt_loss = (
                pt_abs_pos_loss
                + pt_rel_pos_loss
                + (inputs['t'] > self.pt_clash_loss_t) * pt_loss_dict['pred_sidechain_clash_loss']
            ) * (inputs['t'] > self.pt_loss_t_min)

            pt_loss_dict = {"pt_" + k: v for k,v in pt_loss_dict.items()}
        else:
            pt_loss = 0
            pt_loss_dict = {}

        loss = (
            bb_denoising_loss
            + self.latent_fm_loss_scale * latent_denoising_loss
            + 0.25 * bb_denoising_finegrain_loss
            + vae_loss * self.vae_loss
            + pt_loss
        )
        if not self.train_vae_only and any([self.rigid_traj_loss, self.seq_traj_loss, self.dist_traj_loss]):
            loss = loss + (
                traj_loss_dict['traj_bb_loss'] * self.rigid_traj_loss
                + traj_loss_dict['traj_pred_dist_loss'] * self.dist_traj_loss * 0.5
                + traj_loss_dict['traj_seq_loss'] * self.seq_traj_loss * 0.25
            )
        if not self.train_vae_only and self.supervise_noisy_latent:
            noisy_latent_loss_dict = noisy_latent_losses(inputs, outputs)
            loss = loss + (
                noisy_latent_loss_dict['noisy_latent_seq_loss']
                + noisy_latent_loss_dict['noisy_latent_torsion_loss']
            )
        else:
            noisy_latent_loss_dict = {}

        if self.use_gan_losses:
            discrim_loss_dict = discrim_losses(
                inputs, outputs, losses_G=True, train_vae_only=self.train_vae_only
            )
            if self.train_vae_only:
                loss = loss + (
                    discrim_loss_dict["discrim_fixed_bb_G_loss"] * 0.5
                )
            else:
                loss = loss + (
                    discrim_loss_dict["discrim_fixed_bb_G_loss"] * 0.25 +
                    discrim_loss_dict["pt_discrim_pt_bb_G_loss"] * 0.25 * (inputs['t'] > self.pt_loss_t_min)
                )
        else:
            discrim_loss_dict = {}

        loss = loss.mean()

        loss_dict = {"loss": loss, "frameflow_loss": frameflow_loss}
        loss_dict.update(bb_frame_diffusion_loss_dict)
        loss_dict.update(autoenc_loss_dict)
        loss_dict.update(latent_loss_dict)
        loss_dict.update(pt_loss_dict)
        loss_dict.update(traj_loss_dict)
        loss_dict.update(noisy_latent_loss_dict)
        loss_dict.update(discrim_loss_dict)
        return loss_dict


class ProteinSeqInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'
    sidechain_x_1_key='seq_probs_1'
    sidechain_x_1_pred_key='pred_seq_probs_1'
    sidechain_x_t_key='seq_probs_t'

    def __init__(self,
                 protein_noiser: Union[ProteinFisherInterpolant, ProteinDirichletInterpolant],
                 aux_loss_t_min=0.25,
                 use_clash_loss=False,
                 label_smoothing=0.0,
                 logit_norm_loss=0.0,
                 use_seq_vf_loss=True,
                 bb_aux_loss_scale=0.25,
                 aa_aux_loss_scale=1.0,
                 cond_atomic=False):
        super().__init__()
        self.se3_noiser = protein_noiser.se3_noiser
        self.sidechain_noiser = protein_noiser.sidechain_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.use_clash_loss = use_clash_loss

        if isinstance(protein_noiser, ProteinDirichletInterpolant):
            use_seq_vf_loss = False
        self.use_seq_vf_loss = use_seq_vf_loss

        self.label_smoothing = label_smoothing
        self.logit_norm_loss = logit_norm_loss
        self.cond_atomic = cond_atomic
        self.bb_aux_loss_scale = bb_aux_loss_scale
        self.aa_aux_loss_scale = aa_aux_loss_scale
        self.rng = np.random.default_rng()

    def _gen_diffuse_mask(self, data: HeteroData):
        return torch.ones_like(data['res_mask']).bool()

    def process_input(self, data: HeteroData):
        data = copy.deepcopy(data)
        self.se3_noiser.set_device(data['residue']['atom37'].device)
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
        diffuse_mask = self._gen_diffuse_mask(res_data)
        res_data['noising_mask'] = diffuse_mask
        res_data['mlm_mask'] = ~diffuse_mask
        res_data['x'] = rigids_1.get_trans().float()  # for HeteroData's sake
        res_data['rigids_1'] = rigids_1.to_tensor_7().float()

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

        # redundant for convenience
        diff_feats_t['atom14'] = chain_feats['atom14_gt_positions']
        diff_feats_t['atom14_mask'] = chain_feats['atom14_gt_exists']

        diff_feats_t = tree.map_structure(
            lambda x: torch.as_tensor(x),
            diff_feats_t)
        res_data.update(diff_feats_t)
        res_data['chi_mask'] = res_data['torsion_angles_mask'][..., 3:].contiguous().bool()

        # noising masks
        res_data['res_noising_mask'] = res_data['res_mask']
        res_data['seq_noising_mask'] = res_data['seq_mask']
        res_data['atom14_noising_mask'] = res_data['atom14_mask']
        ##  noise data
        data = self.se3_noiser.corrupt_batch(data)
        # data = self.sidechain_noiser.corrupt_batch(data)

        return data

    def process_sample_input(self, data: Dict, device='cpu'):
        self.se3_noiser.set_device(device)
        return data

    def run_eval(self, model, inputs):
        device = inputs['residue']['x'].device
        if isinstance(self.sidechain_noiser, torch.nn.Module):
            self.sidechain_noiser.to(device)
        inputs = self.sidechain_noiser.corrupt_batch(model, inputs)
        self.se3_noiser.set_device(device)
        if self.cond_atomic:
            gt_condition = (np.random.uniform() > 0.5)
            inputs['gt_conditioning'] = gt_condition
        else:
            gt_condition = False
            inputs['gt_conditioning'] = False

        # TODO: should this be a separate flag?
        if model.self_conditioning and np.random.uniform() > 0.5:
            with torch.no_grad():
                self_conditioning = model(inputs)
        else:
            self_conditioning = None
        denoiser_output = model(inputs, self_conditioning, gt_condition=gt_condition)
        return denoiser_output

    def run_predict(self,
                    model,
                    inputs,
                    device='cuda:0'):
                    # device='cpu'):
        self.se3_noiser.set_device(device)
        model = model.to(device)

        num_res = inputs['num_res']
        total_num_res = sum(num_res)
        data_list = []
        for n in num_res:
            data = HeteroData(
                residue={
                    "res_mask": torch.ones(n, device=device).bool(),
                    "noising_mask": torch.ones(n, device=device).bool(),
                    "res_noising_mask": torch.ones(n, device=device).bool(),
                    "seq": torch.zeros(n, device=device).long(),
                    "seq_mask": torch.ones(n, device=device).bool(),
                    "seq_noising_mask": torch.ones(n, device=device).bool(),
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
        # alphas = torch.ones((total_num_res, self.sidechain_noiser.D), device=device)
        # seq_probs_0 = torch.distributions.Dirichlet(alphas).sample()
        seq_probs_0 = self.sidechain_noiser.sample_prior(total_num_res, device=device)
        # seq_probs_0 = torch.full_like(seq_probs_0, 0.05)

        # Set-up time
        ts = torch.linspace(self.se3_noiser._cfg.min_t, 1.0, self.se3_noiser._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(
            trans_0,  # trans
            rotmats_0,  # rot
            seq_probs_0
        )]
        clean_traj = []
        denoiser_out = None
        for t_2 in tqdm.tqdm(ts[1:]):
            # Run model.
            trans_t_1, rotmats_t_1, seq_probs_t_1 = prot_traj[-1]

            res_data["trans_t"] = trans_t_1
            res_data["rotmats_t"] = rotmats_t_1
            res_data['rigids_t'] = ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_t_1),
                trans=trans_t_1
            ).to_tensor_7()
            res_data['seq_probs_t'] = seq_probs_t_1
            t = torch.ones(batch.num_graphs, device=device) * t_1
            batch["t"] = t

            with torch.no_grad():
                denoiser_out = model(batch, self_condition=denoiser_out)

                # Process model output.
                pred_rigids = denoiser_out['final_rigids']
                pred_trans_1 = pred_rigids.get_trans()
                pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
                pred_seq_logits = denoiser_out["decoded_seq_logits"]
                # pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)
                pred_seq_probs_1 = denoiser_out["seq_probs"]

                clean_traj.append(
                    (
                        pred_trans_1.detach().cpu(),
                        pred_rotmats_1.detach().cpu(),
                        pred_seq_probs_1.detach().cpu(),
                        denoiser_out['decoded_atom14'].detach().cpu()
                    )
                )

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self.se3_noiser._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self.se3_noiser._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            seq_probs_t_2 = self.sidechain_noiser.euler_step(
                d_t,
                t[res_data.batch],
                seq_probs_1=pred_seq_probs_1,
                seq_probs_t=seq_probs_t_1,
                batch=res_data.batch)

            prot_traj.append(
                (trans_t_2,
                 rotmats_t_2,
                 seq_probs_t_2,
                ))
            t_1 = t_2

            if not model.self_conditioning:
                denoiser_out = None

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        # Run model.
        trans_t_1, rotmats_t_1, seq_probs_t_1 = prot_traj[-1]
        res_data["trans_t"] = trans_t_1
        res_data["rotmats_t"] = rotmats_t_1
        res_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        res_data['seq_probs_t'] = seq_probs_t_1
        t = torch.ones(batch.num_graphs, device=device) * t_1
        batch["t"] = t

        with torch.no_grad():
            denoiser_out = model(batch, self_condition=denoiser_out)

            # Process model output.
            pred_rigids = denoiser_out['final_rigids']
            pred_trans_1 = pred_rigids.get_trans()
            pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
            pred_seq_logits = denoiser_out["decoded_seq_logits"]
            pred_seq_probs_1 = denoiser_out["seq_probs"]
            # pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)

        argmax_seq = pred_seq_logits.argmax(dim=-1)
        decoded_struct = denoiser_out['decoded_atom14']

        # decoded_struct = _collect_from_seq(all_atom14, argmax_seq, torch.ones_like(argmax_seq).bool())

        # # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrotpsi_to_atom37(prot_traj, res_data.res_mask)
        clean_trajs = list(zip(*[traj[-1].split(num_res) for traj in clean_traj]))
        clean_traj_seqs = list(zip(*[traj[-2].split(num_res) for traj in clean_traj]))
        prot_traj_seqs = list(zip(*[traj[-1].split(num_res) for traj in prot_traj]))
        prot_traj = all_atom.transrot_to_atom14(
            [data[:2] for data in prot_traj], res_data.res_mask
        )
        prot_trajs = list(zip(*[traj.split(num_res) for traj in prot_traj]))

        return {
            "samples": decoded_struct.split(num_res),
            "seqs": argmax_seq.split(num_res),
            "clean_trajs": clean_trajs,
            "clean_traj_seqs": clean_traj_seqs,
            "prot_trajs": prot_trajs,
            "prot_traj_seqs": prot_traj_seqs,
            "inputs": inputs
        }


    def compute_loss(self, inputs, outputs: Dict):
        bb_frame_diffusion_loss_dict = bb_frame_fm_loss(
            inputs, outputs, sep_rot_loss=True)
        autoenc_loss_dict = autoencoder_losses(
            inputs, outputs, label_smoothing=self.label_smoothing, logit_norm_loss=self.logit_norm_loss
        )

        bb_denoising_loss = (
            bb_frame_diffusion_loss_dict["trans_vf_loss"] +
            bb_frame_diffusion_loss_dict["rot_vf_loss"]
        )
        bb_denoising_finegrain_loss = (
            bb_frame_diffusion_loss_dict["scaled_pred_bb_mse"]
            + bb_frame_diffusion_loss_dict["scaled_dist_mat_loss"]
        ) * (inputs['t'] > self.aux_loss_t_min)

        sidechain_denoising_finegrain_loss = (
            autoenc_loss_dict["scaled_local_atomic_dist_loss"]
            + autoenc_loss_dict["scaled_atom14_mse"]
        ) * (inputs['t'] > self.aux_loss_t_min)

        # vae_loss = (
        #     ## smooth huber loss approx
        #     (torch.sqrt(autoenc_loss_dict["atom14_mse"] + 1) - 1)
        #     + (torch.sqrt(autoenc_loss_dict["sidechain_dists_mse"] + 1) - 1)
        #     # + autoenc_loss_dict["seq_loss"]
        #     + autoenc_loss_dict["chi_loss"]
        #     # + autoenc_loss_dict["kl_div"] * 1e-6
        # ) * (inputs['t'] > self.aux_loss_t_min)

        if self.use_clash_loss:
            clash_loss = autoenc_loss_dict["pred_sidechain_clash_loss"].clip(max=10)
        else:
            clash_loss = 0

        if self.use_seq_vf_loss:
            res_data = inputs['residue']
            if "seq_probs" in outputs and outputs['seq_probs'] is not None:
                pred_seq_probs = outputs['seq_probs']
            else:
                pred_seq_probs = F.softmax(outputs['decoded_seq_logits'], dim=-1)

            seq_probs_t = res_data['seq_probs_t']
            seq_probs_1 = res_data['seq_probs_1']
            nodewise_t = inputs['t'][res_data.batch]
            pred_hs_vf = self.sidechain_noiser.train_vf(nodewise_t, seq_probs_t, pred_seq_probs)
            gt_hs_vf = self.sidechain_noiser.train_vf(nodewise_t, seq_probs_t, seq_probs_1)
            seq_vf_loss = torch.square(pred_hs_vf - gt_hs_vf).sum(dim=-1)
            seq_vf_loss = _nodewise_to_graphwise(seq_vf_loss, res_data.batch, res_data.seq_mask & res_data.seq_noising_mask)
            if self.sidechain_noiser.train_sched == "learned":
                kappa = inputs['kappa_t']
                dkappa = inputs['dkappa_t']
                scale = 2 * kappa / ((1 - kappa) * 3) * dkappa
                kappa, dkappa = self.sidechain_noiser.kappa(
                    torch.linspace(0., 1., 11, device=kappa.device)[..., None]
                    )
                print(kappa, dkappa)
                # print(seq_vf_loss, scale, inputs['t'])
                # seq_vf_loss = seq_vf_loss * scale
        else:
            seq_vf_loss = autoenc_loss_dict["seq_loss"]

        # kappa, dkappa = self.sidechain_noiser.kappa(inputs['t'])
        # loss = kappa.sum()
        # loss.backward()
        # for name, param in self.sidechain_noiser.named_parameters():
        #     print(name, param.grad)

        loss = (
            bb_denoising_loss
            + self.bb_aux_loss_scale * bb_denoising_finegrain_loss
            + seq_vf_loss * (not inputs['gt_conditioning'])
            # + seq_vf_loss
            + autoenc_loss_dict['chi_loss']
            + autoenc_loss_dict['logit_norm_loss']
            + self.aa_aux_loss_scale * sidechain_denoising_finegrain_loss
            # + vae_loss
            + 0.1 * clash_loss
        ).mean()

        loss_dict = {
            "loss": loss,
            "frameflow_loss": (bb_denoising_loss + 0.25 * bb_denoising_finegrain_loss).mean(),
            "seq_vf_loss": seq_vf_loss,
            "corrected_seq_vf_loss": seq_vf_loss * (not inputs['gt_conditioning']),
        }
        loss_dict.update(bb_frame_diffusion_loss_dict)
        loss_dict.update(autoenc_loss_dict)
        return loss_dict


class ProteinSeqMultiChiInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'
    sidechain_x_1_key='seq_probs_1'
    sidechain_x_1_pred_key='pred_seq_probs_1'
    sidechain_x_t_key='seq_probs_t'

    def __init__(self,
                 protein_noiser: Union[ProteinFisherMultiChiInterpolant, ProteinDirichletMultiChiInterpolant],
                 aux_loss_t_min=0.25,
                 use_clash_loss=False,
                 use_fape_loss=False,
                 use_seq_vf_loss=True,
                 label_smoothing=0.0):
        super().__init__()
        self.se3_noiser = protein_noiser.se3_noiser
        self.sidechain_noiser = protein_noiser.sidechain_noiser
        self.chi_noiser = protein_noiser.chi_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.use_clash_loss = use_clash_loss
        self.use_fape_loss = use_fape_loss
        self.label_smoothing = label_smoothing
        if isinstance(protein_noiser, ProteinDirichletMultiChiInterpolant):
            use_seq_vf_loss = False
        self.use_seq_vf_loss = use_seq_vf_loss
        self.rng = np.random.default_rng()

    def _gen_diffuse_mask(self, data: HeteroData):
        return torch.ones_like(data['res_mask']).bool()

    def process_input(self, data: HeteroData):
        data = copy.deepcopy(data)
        self.se3_noiser.set_device(data['residue']['atom37'].device)
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
        diffuse_mask = self._gen_diffuse_mask(res_data)
        res_data['noising_mask'] = diffuse_mask
        res_data['mlm_mask'] = ~diffuse_mask
        res_data['x'] = rigids_1.get_trans()  # for HeteroData's sake
        res_data['rigids_1'] = rigids_1.to_tensor_7()

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
        # redundant for convenience
        diff_feats_t['atom14'] = chain_feats['atom14_gt_positions']
        diff_feats_t['atom14_mask'] = chain_feats['atom14_gt_exists']

        diff_feats_t = tree.map_structure(
            lambda x: torch.as_tensor(x),
            diff_feats_t)

        res_data.update(diff_feats_t)
        # noising masks
        res_data['res_noising_mask'] = res_data['res_mask']
        res_data['seq_noising_mask'] = res_data['seq_mask']
        res_data['atom14_noising_mask'] = res_data['atom14_mask']
        res_data['chis_1'] = res_data['torsion_angles_sin_cos'][..., 3:, :].contiguous().float()
        res_data['chi_mask'] = res_data['torsion_angles_mask'][..., 3:].contiguous().bool()
        res_data['chi_noising_mask'] = res_data['seq_mask']
        ##  noise data
        data = self.se3_noiser.corrupt_batch(data)
        data = self.chi_noiser.corrupt_batch(data)

        return data

    def process_sample_input(self, data: Dict, device='cpu'):
        self.se3_noiser.set_device(device)
        return data

    def run_eval(self, model, inputs):
        device = inputs['residue']['x'].device
        self.se3_noiser.set_device(device)
        inputs = self.sidechain_noiser.corrupt_batch(model, inputs)
        # TODO: should this be a separate flag?
        if model.self_conditioning and np.random.uniform() > 0.5:
            with torch.no_grad():
                self_conditioning = model(inputs)
        else:
            self_conditioning = None
        denoiser_output = model(inputs, self_conditioning)
        return denoiser_output

    def run_predict(self,
                    model,
                    inputs,
                    device='cuda:0'):
                    # device='cpu'):
        self.se3_noiser.set_device(device)
        model = model.to(device)

        num_res = inputs['num_res']
        total_num_res = sum(num_res)
        data_list = []
        for n in num_res:
            data = HeteroData(
                residue={
                    "res_mask": torch.ones(n, device=device).bool(),
                    "noising_mask": torch.ones(n, device=device).bool(),
                    "res_noising_mask": torch.ones(n, device=device).bool(),
                    "seq": torch.zeros(n, device=device).long(),
                    "seq_mask": torch.ones(n, device=device).bool(),
                    "seq_noising_mask": torch.ones(n, device=device).bool(),
                    "chi_noising_mask": torch.ones(n, device=device).bool(),
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
        seq_probs_0 = self.sidechain_noiser.sample_prior(total_num_res, device)
        angles_0 = torch.rand((total_num_res, 4), device=device) * 2 * torch.pi
        chis_0 = torch.stack(
            [torch.cos(angles_0), torch.sin(angles_0)],
            dim=-1
        )

        # Set-up time
        ts = torch.linspace(self.se3_noiser._cfg.min_t, 1.0, self.se3_noiser._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(
            trans_0,  # trans
            rotmats_0,  # rot
            seq_probs_0,
            chis_0
        )]
        clean_traj = []
        denoiser_out = None
        for t_2 in tqdm.tqdm(ts[1:]):
            # Run model.
            trans_t_1, rotmats_t_1, seq_probs_t_1, chis_t_1 = prot_traj[-1]

            res_data["trans_t"] = trans_t_1
            res_data["rotmats_t"] = rotmats_t_1
            res_data['rigids_t'] = ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_t_1),
                trans=trans_t_1
            ).to_tensor_7()
            res_data['seq_probs_t'] = seq_probs_t_1
            res_data['chis_t'] = chis_t_1
            t = torch.ones(batch.num_graphs, device=device) * t_1
            batch["t"] = t

            with torch.no_grad():
                denoiser_out = model(batch, self_condition=denoiser_out)

                # Process model output.
                pred_rigids = denoiser_out['final_rigids']
                pred_trans_1 = pred_rigids.get_trans()
                pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
                # pred_seq_logits = denoiser_out["decoded_seq_logits"]
                # pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)
                pred_seq_probs_1 = denoiser_out["seq_probs"]
                pred_chis_1 = denoiser_out["decoded_chis"]
                pred_atom14_1 = denoiser_out["decoded_atom14"]

                clean_traj.append(
                    (
                        pred_trans_1.detach().cpu(),
                        pred_rotmats_1.detach().cpu(),
                        pred_seq_probs_1.detach().cpu(),
                        pred_chis_1.detach().cpu(),
                        pred_atom14_1.detach().cpu(),
                    )
                )

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self.se3_noiser._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self.se3_noiser._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            seq_probs_t_2 = self.sidechain_noiser.euler_step(
                d_t,
                t[res_data.batch],
                pred_seq_probs_1,
                seq_probs_t_1,
                res_data.batch)
            chis_t_2 = self.chi_noiser.euler_step(
                d_t,
                t_1,
                denoiser_out["decoded_chis_all"],
                chis_t_1,
                pred_seq_probs_1
            )

            prot_traj.append(
                (trans_t_2,
                 rotmats_t_2,
                 seq_probs_t_2,
                 chis_t_2,
                ))
            t_1 = t_2

            if not model.self_conditioning:
                denoiser_out = None

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        # Run model.
        trans_t_1, rotmats_t_1, seq_probs_t_1, chis_t_1 = prot_traj[-1]
        res_data["trans_t"] = trans_t_1
        res_data["rotmats_t"] = rotmats_t_1
        res_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        res_data['seq_probs_t'] = seq_probs_t_1
        t = torch.ones(batch.num_graphs, device=device) * t_1
        batch["t"] = t

        with torch.no_grad():
            denoiser_out = model(batch, self_condition=denoiser_out)

            # Process model output.
            pred_rigids = denoiser_out['final_rigids']
            pred_trans_1 = pred_rigids.get_trans()
            pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
            pred_seq_logits = denoiser_out["decoded_seq_logits"]
            # pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)
            pred_seq_probs_1 = denoiser_out["seq_probs"]
            pred_chis_1 = denoiser_out["decoded_chis"]

        argmax_seq = pred_seq_logits.argmax(dim=-1)
        decoded_struct = denoiser_out['decoded_atom14']

        # decoded_struct = _collect_from_seq(all_atom14, argmax_seq, torch.ones_like(argmax_seq).bool())

        # # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrotpsi_to_atom37(prot_traj, res_data.res_mask)
        # clean_atom37_traj = all_atom.transrotpsi_to_atom37(clean_traj, res_data.res_mask)
        clean_trajs = zip(*[traj[-1].split(num_res) for traj in clean_traj])
        clean_traj_seqs = zip(*[traj[-3].split(num_res) for traj in clean_traj])
        prot_trajs = zip(*[traj[-1].split(num_res) for traj in prot_traj])

        return {
            "samples": decoded_struct.split(num_res),
            "seqs": argmax_seq.split(num_res),
            "clean_trajs": clean_trajs,
            "clean_traj_seqs": clean_traj_seqs,
            "prot_trajs": prot_trajs,
            "inputs": inputs
        }


    def compute_loss(self, inputs, outputs: Dict):
        bb_frame_diffusion_loss_dict = bb_frame_fm_loss(
            inputs, outputs, sep_rot_loss=True)
        autoenc_loss_dict = autoencoder_losses(
            inputs, outputs, label_smoothing=self.label_smoothing
        )

        bb_denoising_loss = (
            bb_frame_diffusion_loss_dict["trans_vf_loss"] * 2 +
            bb_frame_diffusion_loss_dict["rot_vf_loss"]
        )
        bb_denoising_finegrain_loss = (
            bb_frame_diffusion_loss_dict["scaled_pred_bb_mse"]
            + bb_frame_diffusion_loss_dict["scaled_dist_mat_loss"]
        ) * (inputs['t'] > self.aux_loss_t_min)
        sidechain_denoising_finegrain_loss = (
            autoenc_loss_dict["scaled_local_atomic_dist_loss"]
            + autoenc_loss_dict["scaled_atom14_mse"]
        ) * (inputs['t'] > self.aux_loss_t_min)

        # delta_squared = 1
        # vae_loss = (
        #     ## smooth huber loss approx
        #     delta_squared * (torch.sqrt(autoenc_loss_dict["atom14_mse"]/delta_squared + 1) - 1)
        #     + delta_squared * (torch.sqrt(autoenc_loss_dict["sidechain_dists_mse"]/delta_squared + 1) - 1)
        #     # + autoenc_loss_dict["seq_loss"]
        #     # + autoenc_loss_dict["chi_loss"]
        #     # + autoenc_loss_dict["kl_div"] * 1e-6
        # ) * (inputs['t'] > self.aux_loss_t_min)

        norm_scale = 1 - torch.min(inputs['t'], torch.as_tensor(0.9))
        if self.use_clash_loss:
            clash_loss = (
                0.01 * autoenc_loss_dict["pred_sidechain_clash_loss"].clip(max=10)
                / norm_scale
            ) * (inputs['t'] > self.aux_loss_t_min)
        else:
            clash_loss = 0

        if self.use_fape_loss:
            fape = all_atom_fape_loss(
                pred_atom14=outputs['decoded_atom14_gt_seq'],
                gt_atom14=inputs['residue']['atom14_gt_positions'],
                pred_rigids=outputs['final_rigids'],
                gt_rigids=ru.Rigid.from_tensor_7(inputs['residue']['rigids_1']),
                batch=inputs['residue'].batch,
                atom14_mask=inputs['residue']['atom14_gt_exists']
            )
        else:
            fape = 0

        if self.use_seq_vf_loss:
            res_data = inputs['residue']
            if "seq_probs" in outputs and outputs['seq_probs'] is not None:
                pred_seq_probs = outputs['seq_probs']
            else:
                pred_seq_probs = F.softmax(outputs['decoded_seq_logits'], dim=-1)

            seq_probs_t = res_data['seq_probs_t']
            seq_probs_1 = res_data['seq_probs_1']
            nodewise_t = inputs['t'][res_data.batch]
            pred_hs_vf = self.sidechain_noiser.train_vf(nodewise_t, seq_probs_t, pred_seq_probs)
            gt_hs_vf = self.sidechain_noiser.train_vf(nodewise_t, seq_probs_t, seq_probs_1)
            seq_vf_loss = torch.square(pred_hs_vf - gt_hs_vf).sum(dim=-1)
            seq_vf_loss = _nodewise_to_graphwise(seq_vf_loss, res_data.batch, res_data.seq_mask & res_data.seq_noising_mask)
        else:
            seq_vf_loss = autoenc_loss_dict["seq_loss"]

        loss = (
            bb_denoising_loss
            + 0.25 * bb_denoising_finegrain_loss
            + autoenc_loss_dict["seq_loss"]
            + autoenc_loss_dict["chi_loss"] / (norm_scale ** 2)
            + sidechain_denoising_finegrain_loss
            # + vae loss
            + clash_loss
            + fape
        ).mean()

        loss_dict = {
            "loss": loss,
            "frameflow_loss": (bb_denoising_loss + 0.25 * bb_denoising_finegrain_loss).mean(),
            "seq_vf_loss": seq_vf_loss
        }
        if self.use_fape_loss:
            loss_dict["fape"] = fape
        loss_dict.update(bb_frame_diffusion_loss_dict)
        loss_dict.update(autoenc_loss_dict)
        return loss_dict
