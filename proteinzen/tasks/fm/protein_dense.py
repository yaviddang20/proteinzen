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
    autoencoder_losses, latent_scalar_sidechain_fm_loss, latent_scalar_edge_fm_loss, _collect_from_seq, pt_autoencoder_losses,
    kl_losses,
    latent_scalar_dense_sidechain_fm_loss
)
from proteinzen.stoch_interp.interpolate.se3 import _centered_gaussian, _uniform_so3
from proteinzen.stoch_interp.interpolate.latent import _centered_gaussian as _centered_rn_gaussian
from proteinzen.stoch_interp.interpolate.protein import DenseProteinInterpolant

import proteinzen.stoch_interp.interpolate.utils as du
from proteinzen.utils.framediff import all_atom


class DenseProteinInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'
    sidechain_x_1_key='latent_sidechain'
    sidechain_x_1_pred_key='pred_latent_sidechain'
    sidechain_x_t_key='noised_latent_sidechain'
    edge_x_1_key='latent_edge'
    edge_x_1_pred_key='pred_latent_edge'
    edge_x_t_key='noised_latent_edge'

    def __init__(self,
                 protein_noiser: DenseProteinInterpolant,
                 aux_loss_t_min=0.25,
                 compute_passthrough=True,
                 pt_loss_t_min=0.0,
                 pt_clash_loss_t=1.1,
                 kl_strength=0,#1e-6,
                 use_kl_noise=True,
                 rescale_kl_noise=False,
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
                 t_min=0.01,
                 t_max=0.99,
                 lognorm_sample_t=False,
                 lognorm_t_mu=0.0,
                 lognorm_t_std=1.0,
                 rigid_traj_loss=False,
                 aa_traj_loss=False,
                 use_fape_loss=False,
                 fape_length_scale=1.,
                 percent_all_mask=0.25,
                 percent_no_mask=0.25,
                 norm_latent_space=False,
                 encoder_consistency_check=False
    ):
        super().__init__()
        self.se3_noiser = protein_noiser.se3_noiser
        self.latent_noiser = protein_noiser.latent_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.compute_passthrough = compute_passthrough
        self.pt_loss_t_min = pt_loss_t_min
        self.pt_clash_loss_t = pt_clash_loss_t
        self.rng = np.random.default_rng()
        self.kl_strength = kl_strength
        self.norm_latent_space = norm_latent_space
        self.use_kl_noise = use_kl_noise
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
        self.rigid_traj_loss = rigid_traj_loss
        self.aa_traj_loss = aa_traj_loss
        self.square_bb_aux_loss_t_factor = square_bb_aux_loss_t_factor
        self.percent_all_mask = percent_all_mask
        self.percent_no_mask = percent_no_mask
        self.encoder_consistency_check = encoder_consistency_check

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
        latent_data = model.encoder(inputs, apply_noising_masks=self.vae_seq_masking)

        ## sample only if we're training
        if self.use_kl_noise:
            latent_node_sigma = torch.exp(
                latent_data['latent_node_logvar'] * 0.5
            )
            latent_data[self.sidechain_x_1_key] = (
                latent_data['latent_node_mu']
                + latent_node_sigma * torch.randn_like(latent_node_sigma)
            )
            latent_edge_sigma = torch.exp(
                latent_data['latent_edge_logvar'] * 0.5
            )
            latent_data['latent_edge'] = (
                latent_data['latent_edge_mu']
                + latent_edge_sigma * torch.randn_like(latent_edge_sigma)
            )

        else:
            latent_data[self.sidechain_x_1_key] = latent_data['latent_node_mu']
            latent_data[self.edge_x_1_key] = latent_data['latent_edge_mu']

        # print(latent_data[self.sidechain_x_1_key].shape, latent_data[self.edge_x_1_key].shape)

        if self.norm_latent_space:
            latent_node = latent_data[self.sidechain_x_1_key]
            node_std, node_center = torch.std_mean(latent_node, dim=(0, 1))
            latent_edge = latent_data[self.edge_x_1_key]
            edge_std, edge_center = torch.std_mean(latent_edge, dim=(0, 1, 2))

            latent_node = (latent_node - node_center[None, None]) / node_std[None, None]
            latent_edge = (latent_edge - edge_center[None, None, None]) / edge_std[None, None, None]
            if self.rescale_kl_noise:
                latent_node_sigma = torch.exp(
                    latent_data['latent_node_logvar'] * 0.5
                )
                latent_node_sigma = latent_node_sigma / torch.maximum(latent_node_sigma, torch.tensor(1e-8, device=latent_node_sigma.device))
                latent_data['latent_node_logvar'] = torch.log(2 * latent_node_sigma)
                latent_edge_sigma = torch.exp(
                    latent_data['latent_edge_logvar'] * 0.5
                )
                latent_edge_sigma = latent_edge_sigma / torch.maximum(latent_edge_sigma, torch.tensor(1e-8, device=latent_edge_sigma.device))
                latent_data['latent_edge_logvar'] = torch.log(2 * latent_edge_sigma)

        # print(torch.linalg.vector_norm(latent_data[self.sidechain_x_1_key], dim=-1).mean())
        # # print(torch.std_mean(latent_data[self.sidechain_x_1_key], dim=(0, 1)))
        # print(torch.linalg.vector_norm(latent_data['latent_node_mu'], dim=-1).mean())
        # # print(torch.std_mean(latent_data['latent_node_mu'], dim=(0, 1)))
        # print(torch.linalg.vector_norm(latent_data[self.edge_x_1_key], dim=-1).mean())
        # # print(torch.std_mean(latent_data[self.edge_x_1_key], dim=(0, 1, 2)))
        # print(torch.linalg.vector_norm(latent_data['latent_edge_mu'], dim=-1).mean())
        # # print(torch.std_mean(latent_data['latent_edge_mu'], dim=(0, 1, 2)))

        # decoder
        decoder_outputs = model.decoder(inputs, latent_data)

        # fm
        noised_latent_data = self.latent_noiser.corrupt_batch(
            inputs,
            latent_data,
        )
        latent_outputs = model.denoiser(inputs, noised_latent_data, self_condition=self_conditioning)

        if self.compute_passthrough:
            # compute passthrough outputs
            passthrough_inputs = copy.copy(inputs)
            passthrough_inputs['residue']['rigids_1'] = latent_outputs['final_rigids'].to_tensor_7()
            passthrough_inputs['residue']['x'] = latent_outputs['final_rigids'].get_trans()
            passthrough_inputs['residue']['bb'] = latent_outputs['denoised_bb'][..., :4, :]
            passthrough_latent = {
                self.sidechain_x_1_key: latent_outputs[self.sidechain_x_1_pred_key],
                self.edge_x_1_key: latent_outputs[self.edge_x_1_pred_key]
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
                consistency_inputs = copy.copy(passthrough_inputs)
                c_res_data = consistency_inputs['res_data']
                c_res_data['atom14_gt_positions'] = passthrough_outputs['decoded_atom14']
                c_res_data['atom14_gt_exists'] = passthrough_outputs['decoded_atom14_mask']
                c_res_data['atom14_noising_mask'] = passthrough_outputs['decoded_atom14_mask']
                c_res_data['seq'] = passthrough_outputs['decoded_seq_logits'].argmax(dim=-1)
                for param in model.encoder.parameters():
                    param.requires_grad = False
                consistency_data = model.encoder(consistency_inputs, apply_noising_masks=self.vae_seq_masking)
                for param in model.encoder.parameters():
                    param.requires_grad = True
                passthrough_outputs['consistency_data'] = consistency_data

        else:
            passthrough_outputs = None

        # update outputs for loss calculation
        latent_outputs.update(noised_latent_data)
        latent_outputs.update(latent_data)

        return latent_outputs, decoder_outputs, passthrough_outputs

    def run_eval(self, model, inputs):
        device = inputs['residue']['x'].device
        self.se3_noiser.set_device(device)
        # TODO: should this be a separate flag?
        if not self.train_vae_only and model.self_conditioning and np.random.uniform() > 0.5:
            with torch.no_grad():
                self_conditioning, _, sc_pt_outputs = self._run_model(model, inputs, pt_use_gt_seq=False)
                if sc_pt_outputs is not None:
                    self_conditioning.update(sc_pt_outputs)
        else:
            self_conditioning = None

        denoiser_output, design_output, pt_outputs = self._run_model(model, inputs, self_conditioning)
        denoiser_output.update(design_output)
        denoiser_output["pt_outputs"] = pt_outputs
        return denoiser_output

    def run_predict(self,
                    model,
                    inputs,
                    keep_traj=False,
                    device='cuda:0'):
                    # device='cpu'):
        self.se3_noiser.set_device(device)

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
        latent_prior = model.sample_prior(
            batch.num_graphs,
            int(torch.sum(res_data.batch == 0)),
            device
        )
        sidechain_0 = latent_prior['noised_latent_sidechain']
        edge_0 = latent_prior['noised_latent_edge']

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
            edge_0,  # latent sidechain,
            torch.ones((total_num_res, 21), device=device).float(),  # seq logits
            init_atom14  # atom14 struct
        )]
        clean_traj = []
        denoiser_out = None

        for t_2 in tqdm.tqdm(ts[1:]):
            # Run model.
            trans_t_1, rotmats_t_1, _, sidechain_t_1, edge_t_1, _, _ = prot_traj[-1]
            res_data["trans_t"] = trans_t_1
            res_data["rotmats_t"] = rotmats_t_1
            res_data['rigids_t'] = ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_t_1),
                trans=trans_t_1
            ).to_tensor_7()
            t = torch.ones(batch.num_graphs, device=device) * t_1
            batch["t"] = t

            intermediates = {
                self.sidechain_x_t_key: sidechain_t_1,
                self.edge_x_t_key: edge_t_1
            }

            with torch.no_grad():
                denoiser_out = model.denoiser(batch, intermediates, self_condition=denoiser_out)

                # Process model output.
                pred_rigids = denoiser_out['final_rigids']
                pred_trans_1 = pred_rigids.get_trans()
                pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
                pred_psis = denoiser_out['psi'].detach().cpu()
                pred_latent_sidechain = denoiser_out[self.sidechain_x_1_pred_key].detach()
                pred_latent_edge = denoiser_out[self.edge_x_1_pred_key].detach()
                latent_output = {
                    self.sidechain_x_1_key: pred_latent_sidechain,
                    self.edge_x_1_key: pred_latent_edge
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
                        pred_latent_edge.cpu(),
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
            sidechain_t_2 = self.latent_noiser._euler_step(
                d_t=d_t,
                t=t_1,
                x_1=pred_latent_sidechain,
                x_t=sidechain_t_1)
            # TODO: this should probably be its own thing
            edge_t_2 = self.latent_noiser._euler_step(
                d_t=d_t,
                t=t_1,
                x_1=pred_latent_edge,
                x_t=edge_t_1)

            # sidechain_t_2 = torch.randn_like(sidechain_t_2) * 10

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
                 edge_t_2,
                 decoder_output["decoded_seq_logits"].argmax(dim=-1).detach().cpu(),
                 atom14_t_2.detach().cpu()
                ))
            t_1 = t_2

            if not model.self_conditioning:
                denoiser_out = None

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1, _, sidechain_t_1, edge_t_1, _, _ = prot_traj[-1]
        res_data["trans_t"] = trans_t_1
        res_data["rotmats_t"] = rotmats_t_1
        res_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        t = torch.ones(batch.num_graphs, device=device) * t_1
        batch["t"] = t

        intermediates = {
            self.sidechain_x_t_key: sidechain_t_1,
            self.edge_x_t_key: edge_t_1,
        }

        with torch.no_grad():
            denoiser_out = model.denoiser(batch, intermediates, self_condition=denoiser_out)

        # Process model output.
        pred_rigids = denoiser_out['final_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
        pred_psis = denoiser_out['psi'].detach().cpu()
        pred_latent_sidechain = denoiser_out[self.sidechain_x_1_pred_key].detach()
        pred_latent_edge = denoiser_out[self.edge_x_1_pred_key].detach()

        latent_output = {
            self.sidechain_x_1_key: pred_latent_sidechain,
            self.edge_x_1_key: pred_latent_edge
        }
        data_list = []
        for n in num_res:
            data = HeteroData(
                residue={
                    "res_mask": torch.ones(n, device=device).bool(),
                    "noising_mask": torch.ones(n, device=device).bool(),
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
                pred_latent_edge.cpu(),
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
        latent_data = model.encoder(encoder_inputs, apply_noising_masks=self.vae_seq_masking)
        print(
            torch.mean(torch.linalg.vector_norm(pred_latent_sidechain - latent_data['latent_node_mu'], dim=-1)),
            torch.mean(torch.linalg.vector_norm((0.5 * latent_data['latent_node_logvar']).exp()))
        )
        print(
            torch.mean(torch.linalg.vector_norm(pred_latent_edge - latent_data['latent_edge_mu'], dim=-1)),
            torch.mean(torch.linalg.vector_norm((0.5 * latent_data['latent_edge_logvar']).exp()))
        )
        print(torch.mean(torch.linalg.vector_norm(pred_latent_sidechain, dim=-1)))
        print(torch.mean(torch.linalg.vector_norm(pred_latent_edge, dim=-1)))


        # all_atom14 = decoder_output['decoded_all_atom14']

        # decoded_struct = _collect_from_seq(all_atom14, argmax_seq, torch.ones_like(argmax_seq).bool())

        # # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrotpsi_to_atom37(prot_traj, res_data.res_mask)
        # clean_atom37_traj = all_atom.transrotpsi_to_atom37(clean_traj, res_data.res_mask)
        clean_trajs = zip(*[traj[-1].split(num_res) for traj in clean_traj])
        clean_traj_seqs = zip(*[traj[-2].split(num_res) for traj in clean_traj])
        prot_trajs = zip(*[traj[-1].split(num_res) for traj in prot_traj])

        if keep_traj:
            return {
                "samples": decoded_struct.split(num_res),
                "seqs": argmax_seq.split(num_res),
                "clean_trajs": clean_trajs,
                "clean_traj_seqs": clean_traj_seqs,
                "prot_trajs": prot_trajs,
                "inputs": inputs
            }
        else:
            del clean_trajs
            del clean_traj_seqs
            del prot_trajs
            return {
                "samples": decoded_struct.split(num_res),
                "seqs": argmax_seq.split(num_res),
                "inputs": inputs
            }



    def compute_loss(self, inputs, outputs: Dict):
        bb_frame_diffusion_loss_dict = bb_frame_fm_loss(
            inputs, outputs, sep_rot_loss=True,
            t_norm_clip=self.t_clip_se3,
            square_aux_loss_time_factor=self.square_bb_aux_loss_t_factor)

        latent_loss_dict = latent_scalar_dense_sidechain_fm_loss(
            inputs,
            outputs,
            scale=1,
            pointwise=self.latent_pointwise_loss,
            detach_gt_latent_grad=True
        )
        latent_edge_loss_dict = latent_scalar_edge_fm_loss(
            inputs,
            outputs,
            scale=1,
            pointwise=self.latent_pointwise_loss,
            detach_gt_latent_grad=True
        )
        bb_denoising_loss = (
            bb_frame_diffusion_loss_dict["trans_vf_loss"] +
            bb_frame_diffusion_loss_dict["rot_vf_loss"]
        )
        bb_denoising_finegrain_loss = (
            bb_frame_diffusion_loss_dict["scaled_pred_bb_mse"]
            + bb_frame_diffusion_loss_dict["scaled_dist_mat_loss"]
        ) * (inputs['t'] > self.aux_loss_t_min)
        bb_denoising_finegrain_loss = bb_denoising_finegrain_loss * (not self.disable_bb_aux_loss)

        latent_denoising_loss = latent_loss_dict["latent_fm_loss"]
        latent_edge_denoising_loss = latent_edge_loss_dict["latent_fm_loss"]

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

        autoenc_loss_dict = autoencoder_losses(
            inputs, outputs, use_smooth_lddt=self.use_smooth_lddt,
            t_norm_clip=self.t_clip_aa,
            apply_seq_noising_mask=self.vae_seq_masking,
            use_sidechain_dists_mse_loss=(not self.use_smooth_lddt),
            use_local_atomic_dist_loss=False,
            use_sidechain_clash_loss=False,
            kl_loss=False
        )

        kl_loss_dict = kl_losses(
            inputs, outputs
        )

        vae_loss = (
            autoenc_loss_dict["atom14_mse"]
            + autoenc_loss_dict["sidechain_dists_mse"] * (not self.use_smooth_lddt)
            # + autoenc_loss_dict["pred_sidechain_clash_loss"]
            + autoenc_loss_dict["seq_loss"] * self.vae_seq_loss
            + autoenc_loss_dict["chi_loss"]
            + kl_loss_dict["node_kl_div"] * self.kl_strength
            + kl_loss_dict["edge_kl_div"] * self.kl_strength
            + autoenc_loss_dict["smooth_lddt"]
        )
        # latent_denoising_loss = latent_loss_dict["latent_denoising_loss"] * 0.01

        if self.compute_passthrough:
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
                kl_loss=False
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
            + self.latent_fm_loss_scale * latent_edge_denoising_loss
            + 0.25 * bb_denoising_finegrain_loss
            + vae_loss * self.vae_loss
            + pt_loss
        ).mean()

        loss_dict = {"loss": loss, "frameflow_loss": frameflow_loss}
        loss_dict.update(bb_frame_diffusion_loss_dict)
        loss_dict.update(autoenc_loss_dict)
        loss_dict.update(latent_loss_dict)
        loss_dict.update({
            "latent_edge_denoising_loss": latent_edge_loss_dict['latent_denoising_loss'],
            "latent_edge_fm_loss": latent_edge_loss_dict['latent_fm_loss'],
            "latent_edge_ref_noise": latent_edge_loss_dict['latent_ref_noise']
        })
        loss_dict.update(kl_loss_dict)
        loss_dict.update(pt_loss_dict)
        return loss_dict