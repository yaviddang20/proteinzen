import copy
from typing import Sequence, Dict

import tqdm
import tree
import torch
import numpy as np
import torch.nn.functional as F

import torch_geometric.utils as pygu
from torch_geometric.data import HeteroData, Batch


from ligbinddiff.data.openfold import data_transforms
from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.tasks import Task
from ligbinddiff.model.utils.graph import batchwise_to_nodewise
from ligbinddiff.runtime.loss.utils import _nodewise_to_graphwise

from ligbinddiff.runtime.loss.frames import bb_frame_fm_loss, all_atom_fape_loss
from ligbinddiff.runtime.loss.common import autoencoder_losses, latent_scalar_sidechain_fm_loss, _collect_from_seq, pt_autoencoder_losses
from ligbinddiff.stoch_interp.interpolate.se3 import _centered_gaussian, _uniform_so3
from ligbinddiff.stoch_interp.interpolate.latent import _centered_gaussian as _centered_rn_gaussian
from ligbinddiff.stoch_interp.interpolate.protein import ProteinInterpolant, ProteinFisherInterpolant, ProteinDirichletInterpolant, ProteinDirichletChiInterpolant, ProteinDirichletMultiChiInterpolant, ProteinCatFlowInterpolant, ProteinFisherMultiChiInterpolant

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
                 aux_loss_t_min=0.25,
                 compute_passthrough=True,
                 pt_clash_loss_t=1.1,
                 kl_strength=1e-6,
                 rescale_kl_noise=False):
        super().__init__()
        self.se3_noiser = protein_noiser.se3_noiser
        self.sidechain_noiser = protein_noiser.sidechain_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.compute_passthrough = compute_passthrough
        self.pt_clash_loss_t = pt_clash_loss_t
        self.rng = np.random.default_rng()
        self.kl_strength = kl_strength
        self.rescale_kl_noise = rescale_kl_noise

    def _gen_diffuse_mask(self, data: HeteroData):
        return torch.ones_like(data['res_mask']).bool()
        # if self.rng.random() > 0.25:
        #     return torch.ones_like(data['res_mask']).bool()
        # else:
        #     percent = self.rng.random()
        #     return torch.rand(data['res_mask'].shape, device=data['res_mask'].device) > percent

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
        diffuse_mask = self._gen_diffuse_mask(res_data)
        res_data['noising_mask'] = diffuse_mask
        res_data['mlm_mask'] = ~diffuse_mask
        res_data['x'] = rigids_1.get_trans()  # for HeteroData's sake
        res_data['rigids_1'] = rigids_1.to_tensor_7()
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

    def _run_model(self, model, inputs, self_conditioning=None, pt_use_gt_seq=True):
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
                self_conditioning, _, sc_pt_outputs = self._run_model(model, inputs, pt_use_gt_seq=False)
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
                    "noising_mask": torch.ones(n, device=device).bool(),
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
                pred_latent_sidechain = denoiser_out[self.sidechain_x_1_pred_key].detach()

                latent_output = {
                    self.sidechain_x_1_key: pred_latent_sidechain
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
            sidechain_t_2 = self.sidechain_noiser._euler_step(d_t, t_1, pred_latent_sidechain, sidechain_t_1)

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

        latent_output = {
            self.sidechain_x_1_key: pred_latent_sidechain
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
                decoder_output['decoded_seq_logits'].detach().cpu().argmax(dim=-1),
                decoder_output['decoded_atom14'].detach().cpu()
            )
        )

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
        t_clip_max = 0.9
        bb_frame_diffusion_loss_dict = bb_frame_fm_loss(
            inputs, outputs, sep_rot_loss=True)
        autoenc_loss_dict = autoencoder_losses(
            inputs, outputs
        )
        latent_loss_dict = latent_scalar_sidechain_fm_loss(inputs, outputs)

        # outputs["decoded_atom14_gt_seq"] = inputs["residue"]["atom14_gt_positions"]
        # autoenc_loss_dict = autoencoder_losses(
        #     inputs, outputs
        # )
        # print(autoenc_loss_dict)
        # exit()

        bb_denoising_loss = (
            bb_frame_diffusion_loss_dict["trans_vf_loss"] +
            bb_frame_diffusion_loss_dict["rot_vf_loss"]
        )
        bb_denoising_finegrain_loss = (
            bb_frame_diffusion_loss_dict["scaled_pred_bb_mse"]
            + bb_frame_diffusion_loss_dict["scaled_dist_mat_loss"]
        ) * (inputs['t'] > self.aux_loss_t_min)

        vae_loss = (
            autoenc_loss_dict["atom14_mse"]
            + autoenc_loss_dict["sidechain_dists_mse"]
            # + autoenc_loss_dict["pred_sidechain_clash_loss"]
            + autoenc_loss_dict["seq_loss"]
            + autoenc_loss_dict["chi_loss"]
            + autoenc_loss_dict["kl_div"] * self.kl_strength
        )
        latent_denoising_loss = latent_loss_dict["latent_fm_loss"]
        # latent_denoising_loss = latent_loss_dict["latent_denoising_loss"] * 0.01

        if self.compute_passthrough:
            pt_outputs = outputs["pt_outputs"]
            assert pt_outputs is not None
            pt_loss_dict = autoencoder_losses(
                inputs, pt_outputs
            )
            norm = 1 - torch.min(inputs['t'], torch.as_tensor(t_clip_max))
            pt_abs_pos_loss = (
                pt_loss_dict["atom14_mse"]
                + pt_loss_dict["sidechain_dists_mse"]
                # + pt_loss_dict["chi_loss"]
                # + pt_loss_dict["seq_loss"]
            ) * (inputs['t'] > self.aux_loss_t_min)
            pt_rel_pos_loss = (
                # pt_loss_dict["atom14_mse"]
                # + pt_loss_dict["sidechain_dists_mse"]
                + pt_loss_dict["chi_loss"]
                + pt_loss_dict["seq_loss"]
            ) * (inputs['t'] > self.aux_loss_t_min)
            pt_loss = (
                pt_abs_pos_loss * 0.01 / (norm ** 2)
                + pt_rel_pos_loss
                + (inputs['t'] > self.pt_clash_loss_t) * pt_loss_dict['pred_sidechain_clash_loss']
            )

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

        loss_dict = {"loss": loss, "frameflow_loss": (bb_denoising_loss + 0.25 * bb_denoising_finegrain_loss).mean()}
        loss_dict.update(bb_frame_diffusion_loss_dict)
        loss_dict.update(autoenc_loss_dict)
        loss_dict.update(latent_loss_dict)
        loss_dict.update(pt_loss_dict)
        return loss_dict


class ProteinDirichletInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'
    sidechain_x_1_key='seq_probs_1'
    sidechain_x_1_pred_key='pred_seq_probs_1'
    sidechain_x_t_key='seq_probs_t'

    def __init__(self,
                 protein_noiser: ProteinDirichletInterpolant,
                 aux_loss_t_min=0.25,
                 use_clash_loss=False,
                 cond_atomic=False):
        super().__init__()
        self.se3_noiser = protein_noiser.se3_noiser
        self.sidechain_noiser = protein_noiser.sidechain_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.use_clash_loss = use_clash_loss
        self.cond_atomic = cond_atomic
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
        res_data['res_noising_mask'] = diffuse_mask
        res_data['mlm_mask'] = ~diffuse_mask
        res_data['x'] = rigids_1.get_trans()  # for HeteroData's sake
        res_data['rigids_1'] = rigids_1.to_tensor_7()
        ##  noise data
        data = self.se3_noiser.corrupt_batch(data)
        data = self.sidechain_noiser.corrupt_batch(data)

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

    def run_eval(self, model, inputs):
        device = inputs['residue']['x'].device
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
                    "seq": torch.zeros(n, device=device).long(),
                    "seq_mask": torch.ones(n, device=device).bool(),
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
        alphas = torch.ones((total_num_res, self.sidechain_noiser.K), device=device)
        seq_probs_0 = torch.distributions.Dirichlet(alphas).sample()

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
                pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)

                clean_traj.append(
                    (
                        pred_trans_1.detach().cpu(),
                        pred_rotmats_1.detach().cpu(),
                        pred_seq_probs_1.detach().cpu(),
                    )
                )

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self.se3_noiser._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self.se3_noiser._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            seq_probs_t_2 = self.sidechain_noiser.euler_step(
                d_t * self.sidechain_noiser.t_max,
                t[res_data.batch] * self.sidechain_noiser.t_max,
                pred_seq_probs_1,
                seq_probs_t_1,
                res_data.batch)

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
            pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)

        argmax_seq = pred_seq_logits.argmax(dim=-1)
        decoded_struct = denoiser_out['decoded_atom14']

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
        bb_frame_diffusion_loss_dict = bb_frame_fm_loss(
            inputs, outputs, sep_rot_loss=True, square_aux_loss_time_factor=True)
        autoenc_loss_dict = autoencoder_losses(
            inputs, outputs
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

        loss = (
            bb_denoising_loss
            + 0.25 * bb_denoising_finegrain_loss
            + autoenc_loss_dict["seq_loss"] * (not inputs['gt_conditioning'])
            + autoenc_loss_dict['chi_loss']
            + sidechain_denoising_finegrain_loss
            # + vae_loss
            + 0.1 * clash_loss
        ).mean()

        loss_dict = {
            "loss": loss,
            "frameflow_loss": (bb_denoising_loss + 0.25 * bb_denoising_finegrain_loss).mean(),
        }
        if self.cond_atomic:
            loss_dict.update({
                "corrected_frameflow_loss": (bb_denoising_loss + 0.25 * bb_denoising_finegrain_loss).mean() * 2 * (not inputs['gt_conditioning']),
                "corrected_seq_loss": autoenc_loss_dict["seq_loss"] * 2 * (not inputs['gt_conditioning']),
                "corrected_chi_loss": autoenc_loss_dict['chi_loss'] * 2 * (not inputs['gt_conditioning']),
            })


        loss_dict.update(bb_frame_diffusion_loss_dict)
        loss_dict.update(autoenc_loss_dict)
        return loss_dict


class ProteinDirichletChiInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'
    sidechain_x_1_key='seq_probs_1'
    sidechain_x_1_pred_key='pred_seq_probs_1'
    sidechain_x_t_key='seq_probs_t'

    def __init__(self,
                 protein_noiser: ProteinDirichletChiInterpolant,
                 aux_loss_t_min=0.25,
                 use_clash_loss=False,
                 use_fape_loss=False):
        super().__init__()
        self.se3_noiser = protein_noiser.se3_noiser
        self.sidechain_noiser = protein_noiser.sidechain_noiser
        self.chi_noiser = protein_noiser.chi_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.use_clash_loss = use_clash_loss
        self.use_fape_loss = use_fape_loss
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
        ##  noise data
        data = self.se3_noiser.corrupt_batch(data)
        data = self.sidechain_noiser.corrupt_batch(data)

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

        res_data['chis_1'] = res_data['torsion_angles_sin_cos'][..., 3:, :].contiguous().float()
        res_data['chi_mask'] = res_data['torsion_angles_mask'][..., 3:].contiguous().bool()
        data = self.chi_noiser.corrupt_batch(data)

        return data

    def process_sample_input(self, data: Dict, device='cpu'):
        self.se3_noiser.set_device(device)
        return data

    def run_eval(self, model, inputs):
        device = inputs['residue']['x'].device
        self.se3_noiser.set_device(device)
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
                    "seq": torch.zeros(n, device=device).long(),
                    "seq_mask": torch.ones(n, device=device).bool(),
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
        alphas = torch.ones((total_num_res, self.sidechain_noiser.K), device=device)
        seq_probs_0 = torch.distributions.Dirichlet(alphas).sample()
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
                pred_seq_logits = denoiser_out["decoded_seq_logits"]
                pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)
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
                d_t * self.sidechain_noiser.t_max,
                t[res_data.batch] * self.sidechain_noiser.t_max,
                pred_seq_probs_1,
                seq_probs_t_1,
                res_data.batch)
            chis_t_2 = self.chi_noiser.euler_step(d_t, t_1, pred_chis_1, chis_t_1)

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
            pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)
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
            inputs, outputs)
        autoenc_loss_dict = autoencoder_losses(
            inputs, outputs
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

        loss = (
            bb_denoising_loss
            + 0.25 * bb_denoising_finegrain_loss
            + autoenc_loss_dict["seq_loss"]
            + autoenc_loss_dict["chi_loss"]
            + sidechain_denoising_finegrain_loss
            # + vae loss
            + clash_loss
            + fape
        ).mean()

        loss_dict = {"loss": loss, "frameflow_loss": (bb_denoising_loss + 0.25 * bb_denoising_finegrain_loss).mean()}
        if self.use_fape_loss:
            loss_dict["fape"] = fape
        loss_dict.update(bb_frame_diffusion_loss_dict)
        loss_dict.update(autoenc_loss_dict)
        return loss_dict


class ProteinDirichletMultiChiInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'
    sidechain_x_1_key='seq_probs_1'
    sidechain_x_1_pred_key='pred_seq_probs_1'
    sidechain_x_t_key='seq_probs_t'

    def __init__(self,
                 protein_noiser: ProteinDirichletMultiChiInterpolant,
                 aux_loss_t_min=0.25,
                 use_clash_loss=False,
                 use_fape_loss=False):
        super().__init__()
        self.se3_noiser = protein_noiser.se3_noiser
        self.sidechain_noiser = protein_noiser.sidechain_noiser
        self.chi_noiser = protein_noiser.chi_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.use_clash_loss = use_clash_loss
        self.use_fape_loss = use_fape_loss
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
        res_data['res_noising_mask'] = diffuse_mask
        res_data['mlm_mask'] = ~diffuse_mask
        res_data['x'] = rigids_1.get_trans()  # for HeteroData's sake
        res_data['rigids_1'] = rigids_1.to_tensor_7()
        ##  noise data
        data = self.se3_noiser.corrupt_batch(data)
        data = self.sidechain_noiser.corrupt_batch(data)

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

        res_data['chis_1'] = res_data['torsion_angles_sin_cos'][..., 3:, :].contiguous().float()
        res_data['chi_mask'] = res_data['torsion_angles_mask'][..., 3:].contiguous().bool()
        data = self.chi_noiser.corrupt_batch(data)

        return data

    def process_sample_input(self, data: Dict, device='cpu'):
        self.se3_noiser.set_device(device)
        return data

    def run_eval(self, model, inputs):
        device = inputs['residue']['x'].device
        self.se3_noiser.set_device(device)
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
                    "seq": torch.zeros(n, device=device).long(),
                    "seq_mask": torch.ones(n, device=device).bool(),
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
        alphas = torch.ones((total_num_res, self.sidechain_noiser.K), device=device)
        seq_probs_0 = torch.distributions.Dirichlet(alphas).sample()
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
                pred_seq_logits = denoiser_out["decoded_seq_logits"]
                pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)
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
                d_t * self.sidechain_noiser.t_max,
                t[res_data.batch] * self.sidechain_noiser.t_max,
                pred_seq_probs_1,
                seq_probs_t_1,
                res_data.batch)
            chis_t_2 = self.chi_noiser.euler_step(d_t, t_1, denoiser_out["decoded_chis_all"], chis_t_1, pred_seq_probs_1)

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
            pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)
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
            inputs, outputs
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

        loss_dict = {"loss": loss, "frameflow_loss": (bb_denoising_loss + 0.25 * bb_denoising_finegrain_loss).mean()}
        if self.use_fape_loss:
            loss_dict["fape"] = fape
        loss_dict.update(bb_frame_diffusion_loss_dict)
        loss_dict.update(autoenc_loss_dict)
        return loss_dict



class ProteinFisherInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'
    sidechain_x_1_key='seq_probs_1'
    sidechain_x_1_pred_key='pred_seq_probs_1'
    sidechain_x_t_key='seq_probs_t'

    def __init__(self,
                 protein_noiser: ProteinFisherInterpolant,
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
        res_data['res_noising_mask'] = diffuse_mask
        res_data['seq_noising_mask'] = diffuse_mask
        res_data['atom14_noising_mask'] = diffuse_mask
        res_data['mlm_mask'] = ~diffuse_mask
        res_data['x'] = rigids_1.get_trans().float()  # for HeteroData's sake
        res_data['rigids_1'] = rigids_1.to_tensor_7().float()
        ##  noise data
        data = self.se3_noiser.corrupt_batch(data)
        data = self.sidechain_noiser.corrupt_batch(data)

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

        return data

    def process_sample_input(self, data: Dict, device='cpu'):
        self.se3_noiser.set_device(device)
        return data

    def run_eval(self, model, inputs):
        device = inputs['residue']['x'].device
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
                    "seq": torch.zeros(n, device=device).long(),
                    "seq_mask": torch.ones(n, device=device).bool(),
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
        seq_probs_0 = self.sidechain_noiser.sample_prior(total_num_res).to(device)

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
                pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)

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
                pred_seq_probs_1,
                seq_probs_t_1,
                res_data.batch)

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
            pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)

        argmax_seq = pred_seq_logits.argmax(dim=-1)
        decoded_struct = denoiser_out['decoded_atom14']

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
            seq_vf_loss = _nodewise_to_graphwise(seq_vf_loss, res_data.batch, res_data.seq_mask)
            if self.sidechain_noiser.sample_sched == "linear":
                seq_vf_loss = seq_vf_loss * 0.01
        else:
            seq_vf_loss = autoenc_loss_dict["seq_loss"]

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


class ProteinFisherMultiChiInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'
    sidechain_x_1_key='seq_probs_1'
    sidechain_x_1_pred_key='pred_seq_probs_1'
    sidechain_x_t_key='seq_probs_t'

    def __init__(self,
                 protein_noiser: ProteinFisherMultiChiInterpolant,
                 aux_loss_t_min=0.25,
                 use_clash_loss=False,
                 use_fape_loss=False,
                 label_smoothing=0.0):
        super().__init__()
        self.se3_noiser = protein_noiser.se3_noiser
        self.sidechain_noiser = protein_noiser.sidechain_noiser
        self.chi_noiser = protein_noiser.chi_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.use_clash_loss = use_clash_loss
        self.use_fape_loss = use_fape_loss
        self.label_smoothing = label_smoothing
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
        ##  noise data
        data = self.se3_noiser.corrupt_batch(data)
        data = self.sidechain_noiser.corrupt_batch(data)

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

        res_data['chis_1'] = res_data['torsion_angles_sin_cos'][..., 3:, :].contiguous().float()
        res_data['chi_mask'] = res_data['torsion_angles_mask'][..., 3:].contiguous().bool()
        data = self.chi_noiser.corrupt_batch(data)

        return data

    def process_sample_input(self, data: Dict, device='cpu'):
        self.se3_noiser.set_device(device)
        return data

    def run_eval(self, model, inputs):
        device = inputs['residue']['x'].device
        self.se3_noiser.set_device(device)
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
                    "seq": torch.zeros(n, device=device).long(),
                    "seq_mask": torch.ones(n, device=device).bool(),
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
        alphas = torch.ones((total_num_res, self.sidechain_noiser.K), device=device)
        seq_probs_0 = torch.distributions.Dirichlet(alphas).sample()
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
                pred_seq_logits = denoiser_out["decoded_seq_logits"]
                pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)
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
                d_t * self.sidechain_noiser.t_max,
                t[res_data.batch] * self.sidechain_noiser.t_max,
                pred_seq_probs_1,
                seq_probs_t_1,
                res_data.batch)
            chis_t_2 = self.chi_noiser.euler_step(d_t, t_1, denoiser_out["decoded_chis_all"], chis_t_1, pred_seq_probs_1)

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
            pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)
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

        loss_dict = {"loss": loss, "frameflow_loss": (bb_denoising_loss + 0.25 * bb_denoising_finegrain_loss).mean()}
        if self.use_fape_loss:
            loss_dict["fape"] = fape
        loss_dict.update(bb_frame_diffusion_loss_dict)
        loss_dict.update(autoenc_loss_dict)
        return loss_dict


class ProteinCatFlowInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'
    sidechain_x_1_key='seq_probs_1'
    sidechain_x_1_pred_key='pred_seq_probs_1'
    sidechain_x_t_key='seq_probs_t'

    def __init__(self,
                 protein_noiser: ProteinCatFlowInterpolant,
                 aux_loss_t_min=0.25,
                 use_clash_loss=False,
                 label_smoothing=0.0):
        super().__init__()
        self.se3_noiser = protein_noiser.se3_noiser
        self.sidechain_noiser = protein_noiser.sidechain_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.use_clash_loss = use_clash_loss
        self.label_smoothing = label_smoothing
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
        ##  noise data
        data = self.se3_noiser.corrupt_batch(data)
        data = self.sidechain_noiser.corrupt_batch(data)

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

    def run_eval(self, model, inputs):
        device = inputs['residue']['x'].device
        self.se3_noiser.set_device(device)
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
                    "seq": torch.zeros(n, device=device).long(),
                    "seq_mask": torch.ones(n, device=device).bool(),
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
        seq_0 = self.sidechain_noiser.sample_prior(total_num_res).to(device)

        # Set-up time
        ts = torch.linspace(self.se3_noiser._cfg.min_t, 1.0, self.se3_noiser._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(
            trans_0,  # trans
            rotmats_0,  # rot
            seq_0
        )]
        clean_traj = []
        denoiser_out = None
        for t_2 in tqdm.tqdm(ts[1:]):
            # Run model.
            trans_t_1, rotmats_t_1, seq_t_1 = prot_traj[-1]

            res_data["trans_t"] = trans_t_1
            res_data["rotmats_t"] = rotmats_t_1
            res_data['rigids_t'] = ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_t_1),
                trans=trans_t_1
            ).to_tensor_7()
            # TODO: this is a hack
            res_data['seq_probs_t'] = seq_t_1
            t = torch.ones(batch.num_graphs, device=device) * t_1
            batch["t"] = t

            with torch.no_grad():
                denoiser_out = model(batch, self_condition=denoiser_out)

                # Process model output.
                pred_rigids = denoiser_out['final_rigids']
                pred_trans_1 = pred_rigids.get_trans()
                pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
                pred_seq_logits = denoiser_out["decoded_seq_logits"]
                pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)

                clean_traj.append(
                    (
                        pred_trans_1.detach().cpu(),
                        pred_rotmats_1.detach().cpu(),
                        pred_seq_probs_1.detach().cpu(),
                    )
                )

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self.se3_noiser._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self.se3_noiser._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            seq_t_2 = self.sidechain_noiser.euler_step(
                d_t,
                t[res_data.batch],
                pred_seq_logits,
                seq_t_1)

            prot_traj.append(
                (trans_t_2,
                 rotmats_t_2,
                 seq_t_2,
                ))
            t_1 = t_2

            if not model.self_conditioning:
                denoiser_out = None

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        # Run model.
        trans_t_1, rotmats_t_1, seq_t_1 = prot_traj[-1]
        res_data["trans_t"] = trans_t_1
        res_data["rotmats_t"] = rotmats_t_1
        res_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        res_data['seq_probs_t'] = seq_t_1
        t = torch.ones(batch.num_graphs, device=device) * t_1
        batch["t"] = t

        with torch.no_grad():
            denoiser_out = model(batch, self_condition=denoiser_out)

            # Process model output.
            pred_rigids = denoiser_out['final_rigids']
            pred_trans_1 = pred_rigids.get_trans()
            pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
            pred_seq_logits = denoiser_out["decoded_seq_logits"]
            pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)

        argmax_seq = pred_seq_logits.argmax(dim=-1)
        decoded_struct = denoiser_out['decoded_atom14']

        # decoded_struct = _collect_from_seq(all_atom14, argmax_seq, torch.ones_like(argmax_seq).bool())

        # # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrotpsi_to_atom37(prot_traj, res_data.res_mask)
        # clean_atom37_traj = all_atom.transrotpsi_to_atom37(clean_traj, res_data.res_mask)
        clean_trajs = zip(*[traj[-1].split(num_res) for traj in clean_traj])
        clean_traj_seqs = zip(*[traj[-1].split(num_res) for traj in clean_traj])
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

        loss = (
            bb_denoising_loss
            + 0.25 * bb_denoising_finegrain_loss
            + autoenc_loss_dict["seq_loss"]
            + autoenc_loss_dict['chi_loss']
            + sidechain_denoising_finegrain_loss
            # + vae_loss
            + 0.1 * clash_loss
        ).mean()

        loss_dict = {"loss": loss, "frameflow_loss": (bb_denoising_loss + 0.25 * bb_denoising_finegrain_loss).mean()}
        loss_dict.update(bb_frame_diffusion_loss_dict)
        loss_dict.update(autoenc_loss_dict)
        return loss_dict