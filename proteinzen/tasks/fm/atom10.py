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
    atom10_losses
)
from proteinzen.runtime.loss.atomic.holes import buried_cavity_loss
from proteinzen.runtime.loss.traj import traj_loss2
from proteinzen.stoch_interp.interpolate.se3 import _centered_gaussian, _uniform_so3
from proteinzen.stoch_interp.interpolate.protein import (
    ProteinAtom10Interpolant
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



class ProteinAtom10Interpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'
    sidechain_x_1_key='atom10_local_1'
    sidechain_x_1_pred_key='pred_atom10_local_1'
    sidechain_x_t_key='atom10_local_t'

    def __init__(self,
                 protein_noiser: ProteinAtom10Interpolant,
                 aux_loss_t_min=0.25,
                 use_clash_loss=False,
                 use_edge_dist_loss=False,
                 use_fape_loss=False,
                 atom10_preconditioning=False,
                 trans_preconditioning=False,
                 traj_decay_factor=0.99):
        super().__init__()
        self.se3_noiser = protein_noiser.se3_noiser
        self.sidechain_noiser = protein_noiser.sidechain_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.use_clash_loss = use_clash_loss
        self.use_edge_dist_loss = use_edge_dist_loss
        self.traj_decay_factor = traj_decay_factor
        self.use_fape_loss = use_fape_loss
        self.atom10_preconditioning = atom10_preconditioning
        self.trans_preconditioning = trans_preconditioning

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
        res_data['res_noising_mask'] = diffuse_mask
        res_data['x'] = rigids_1.get_trans().float()  # for HeteroData's sake
        res_data['rigids_1'] = rigids_1.to_tensor_7().float()

        # compute sidechain features
        ## generate data dict
        copy_keys = [
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

        ##  noise data
        data = self.se3_noiser.corrupt_batch(data)
        data = self.sidechain_noiser.corrupt_batch(data)

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
                    "res_noising_mask": torch.ones(n, device=device).bool(),
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
        atom10_local_0 = torch.randn(total_num_res, 10, 3, device=device)
        if self.sidechain_noiser.nonlocal_prior:
            atom10_local_0 = atom10_local_0 * self.sidechain_noiser.prior_std
            rigids_0 = ru.Rigid(
                trans=trans_0,
                rots=ru.Rotation(rot_mats=rotmats_0)
            )
            atom10_local_0 = rigids_0[..., None].invert_apply(atom10_local_0)


        seq_logits_0 = torch.randn(total_num_res, 20, device=device)

        # Set-up time
        ts = torch.linspace(self.se3_noiser._cfg.min_t, 1.0, self.se3_noiser._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(
            trans_0,  # trans
            rotmats_0,  # rot
            atom10_local_0,
            seq_logits_0
        )]
        clean_traj = []
        denoiser_out = None
        for t_2 in tqdm.tqdm(ts[1:]):
            # Run model.
            trans_t_1, rotmats_t_1, atom10_local_t_1, _ = prot_traj[-1]

            res_data["trans_t"] = trans_t_1
            res_data["rotmats_t"] = rotmats_t_1
            res_data['rigids_t'] = ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_t_1),
                trans=trans_t_1
            ).to_tensor_7()
            res_data['noised_atom10_local'] = atom10_local_t_1
            t = torch.ones(batch.num_graphs, device=device) * t_1
            batch["t"] = t
            if self.atom10_preconditioning:
                atom10_var_scaling_dict = self.sidechain_noiser.var_scaling_factors(t)
                batch['atom10_c_skip'] = atom10_var_scaling_dict['c_skip']
                batch['atom10_c_in'] = atom10_var_scaling_dict['c_in']
                batch['atom10_c_out'] = atom10_var_scaling_dict['c_out']
                batch['atom10_loss_weighting'] = atom10_var_scaling_dict['loss_weighting']
            batch['atom10_prior_offset'] = self.sidechain_noiser.get_prior_offset(device=trans_t_1.device)
            if self.trans_preconditioning:
                trans_var_scaling_dict = self.se3_noiser.var_scaling_factors(t)
                batch['trans_c_skip'] = trans_var_scaling_dict['c_skip']
                batch['trans_c_in'] = trans_var_scaling_dict['c_in']
                batch['trans_c_out'] = trans_var_scaling_dict['c_out']

            with torch.no_grad():
                denoiser_out = model(batch, self_condition=denoiser_out)

                # Process model output.
                pred_rigids = denoiser_out['final_rigids']
                pred_trans_1 = pred_rigids.get_trans()
                pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
                pred_seq_logits = denoiser_out["decoded_seq_logits"][..., :20]
                pred_atom10_local = denoiser_out["denoised_atom10_local"].view(-1, 10, 3)

                clean_traj.append(
                    (
                        pred_trans_1.detach().cpu(),
                        pred_rotmats_1.detach().cpu(),
                        denoiser_out['denoised_atom14'].detach().cpu(),
                        pred_seq_logits.detach().cpu()
                    )
                )

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self.se3_noiser._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self.se3_noiser._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            atom10_local_t_2 = self.sidechain_noiser._euler_step(
                d_t,
                t_1,
                x_1=pred_atom10_local,
                x_t=atom10_local_t_1
            )

            prot_traj.append(
                (trans_t_2,
                 rotmats_t_2,
                 atom10_local_t_2,
                 pred_seq_logits.detach().cpu()
                ))
            t_1 = t_2

            if not model.self_conditioning:
                denoiser_out = None

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        # Run model.
        trans_t_1, rotmats_t_1, atom10_local_t_1, _ = prot_traj[-1]
        res_data["trans_t"] = trans_t_1
        res_data["rotmats_t"] = rotmats_t_1
        res_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        res_data['noised_atom10_local'] = atom10_local_t_1
        t = torch.ones(batch.num_graphs, device=device) * t_1
        batch["t"] = t
        if self.atom10_preconditioning:
            atom10_var_scaling_dict = self.sidechain_noiser.var_scaling_factors(t)
            batch['atom10_c_skip'] = atom10_var_scaling_dict['c_skip']
            batch['atom10_c_in'] = atom10_var_scaling_dict['c_in']
            batch['atom10_c_out'] = atom10_var_scaling_dict['c_out']
            batch['atom10_loss_weighting'] = atom10_var_scaling_dict['loss_weighting']
        if self.trans_preconditioning:
            trans_var_scaling_dict = self.se3_noiser.var_scaling_factors(t)
            batch['trans_c_skip'] = trans_var_scaling_dict['c_skip']
            batch['trans_c_in'] = trans_var_scaling_dict['c_in']
            batch['trans_c_out'] = trans_var_scaling_dict['c_out']

        with torch.no_grad():
            denoiser_out = model(batch, self_condition=denoiser_out)

            # Process model output.
            pred_rigids = denoiser_out['final_rigids']
            pred_trans_1 = pred_rigids.get_trans()
            pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
            pred_seq_logits = denoiser_out["decoded_seq_logits"][..., :20]

        argmax_seq = pred_seq_logits.argmax(dim=-1)
        decoded_struct = denoiser_out['denoised_atom14']

        # decoded_struct = _collect_from_seq(all_atom14, argmax_seq, torch.ones_like(argmax_seq).bool())

        # # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrotpsi_to_atom37(prot_traj, res_data.res_mask)
        clean_trajs = list(zip(*[traj[-2].split(num_res) for traj in clean_traj]))
        clean_traj_seqs = list(zip(*[traj[-1].argmax(dim=-1).split(num_res) for traj in clean_traj]))
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
            inputs, outputs, sep_rot_loss=True,
            t_norm_clip=0.9,
            square_aux_loss_time_factor=True,
            trans_preconditioning=self.trans_preconditioning
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

        traj_loss_dict = traj_loss2(inputs, outputs, traj_decay_factor=self.traj_decay_factor)


        atom10_loss_dict = atom10_losses(
            inputs, outputs,
            t_norm_clip=0.9,
            use_sidechain_clash_loss=self.use_clash_loss,
            use_fape=self.use_fape_loss,
            preconditioning=self.atom10_preconditioning
        )

        atomic_loss = (
            atom10_loss_dict["atom10_fm_loss"] * (0.01 if self.sidechain_noiser.nonlocal_prior else 1)
            + atom10_loss_dict["scaled_atom14_mse"]
            + atom10_loss_dict["seq_loss"]
            + atom10_loss_dict["smooth_lddt"]
            + atom10_loss_dict['fape']
        )


        loss = (
            bb_denoising_loss
            + 0.25 * bb_denoising_finegrain_loss
            + atomic_loss
            + traj_loss_dict['traj_bb_loss']
            + traj_loss_dict['traj_pred_dist_loss'] * 0.5
            + traj_loss_dict['traj_seq_loss'] * 0.25
        )

        # if self.use_gan_losses:
        #     discrim_loss_dict = discrim_losses(
        #         inputs, outputs, losses_G=True, train_vae_only=self.train_vae_only
        #     )
        #     if self.train_vae_only:
        #         loss = loss + (
        #             discrim_loss_dict["discrim_fixed_bb_G_loss"] * 0.5
        #         )
        #     else:
        #         loss = loss + (
        #             discrim_loss_dict["discrim_fixed_bb_G_loss"] * 0.25 +
        #             discrim_loss_dict["pt_discrim_pt_bb_G_loss"] * 0.25 * (inputs['t'] > self.pt_loss_t_min)
        #         )
        # else:
        #     discrim_loss_dict = {}

        loss = loss.mean()

        loss_dict = {"loss": loss, "frameflow_loss": frameflow_loss}
        loss_dict.update(bb_frame_diffusion_loss_dict)
        loss_dict.update(atom10_loss_dict)
        loss_dict.update(traj_loss_dict)
        # loss_dict.update(discrim_loss_dict)
        return loss_dict

