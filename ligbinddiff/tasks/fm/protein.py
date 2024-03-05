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
from ligbinddiff.runtime.loss.common import autoencoder_losses, latent_scalar_sidechain_fm_loss, _collect_from_seq, pt_autoencoder_losses
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
                 aux_loss_t_min=0.25,
                 compute_passthrough=True,
                 pt_clash_loss_t=1.1):
        super().__init__()
        self.se3_noiser = protein_noiser.se3_noiser
        self.sidechain_noiser = protein_noiser.sidechain_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.compute_passthrough = compute_passthrough
        self.pt_clash_loss_t = pt_clash_loss_t
        self.rng = np.random.default_rng()

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
            inputs, outputs)
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
            bb_frame_diffusion_loss_dict["trans_vf_loss"] * 2 +
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
            + autoenc_loss_dict["kl_div"] * 1e-6
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
