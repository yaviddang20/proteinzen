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


from proteinzen.data.openfold import data_transforms, residue_constants
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.tasks import Task

from proteinzen.runtime.loss.common import atomic_losses
from proteinzen.runtime.loss.multiframe import multiframe_fm_loss
from proteinzen.runtime.loss.traj import multiframe_traj_loss
from proteinzen.stoch_interp.interpolate.multiframe import _centered_gaussian, _uniform_so3, MultiSE3Interpolant

import proteinzen.stoch_interp.interpolate.utils as du
from proteinzen.utils.framediff import all_atom
from proteinzen.utils.coarse_grain import compute_atom14_from_cg_frames



class MultiFrameInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'

    def __init__(self,
                 protein_noiser: MultiSE3Interpolant,
                 aux_loss_t_min=0.25,
                 use_fape=False,
                 use_fafe=False,
                 use_fafe_l2=False,
                 scale_frame_aligned_errors=False,
                 scale_fape=False,
                 scale_fafe=False,
                 fafe_weight=0.25,
                 polar_upweight=False,
                 sidechain_upweight=False,
                 trans_preconditioning=False,
                 rot_loss_exp_weighting=False,
                 rot_vf_angle_loss_weight=0.5,
                 rigids_traj_loss_scale=1
    ):
        super().__init__()
        self.frame_noiser = protein_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.use_fape = use_fape
        self.use_fafe = use_fafe
        self.use_fafe_l2 = use_fafe_l2
        self.scale_frame_aligned_errors = scale_frame_aligned_errors
        self.scale_fape = scale_fape
        self.scale_fafe = scale_fafe
        self.polar_upweight = polar_upweight
        self.sidechain_upweight = sidechain_upweight
        self.trans_preconditioning = trans_preconditioning
        self.rot_loss_exp_weighting = rot_loss_exp_weighting
        self.fafe_weight = fafe_weight
        self.rot_vf_angle_loss_weight = rot_vf_angle_loss_weight
        self.rigids_traj_loss_scale = rigids_traj_loss_scale

        self.rng = np.random.default_rng()

        self.polar_residues = torch.tensor([
            residue_constants.restype_order[i]
            for i in ['C', 'D', 'E', 'H', 'K', 'N', 'Q', 'R', 'S', 'T', 'Y']
        ])

    def _gen_diffuse_mask(self, data: HeteroData):
        return torch.ones_like(data['res_mask']).bool()

    def process_input(self, data: HeteroData):
        data = copy.deepcopy(data)
        self.frame_noiser.set_device(data['residue']['atom37'].device)
        res_data = data['residue']

        # compute base features
        chain_feats = {
            'aatype': torch.as_tensor(res_data['seq']).long(),
            'all_atom_positions': torch.as_tensor(res_data['atom37']).double(),
            'all_atom_mask': torch.as_tensor(res_data['atom37_mask']).double()
        }
        # chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.atom37_to_cg_frames(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles(prefix="")(chain_feats)  # TODO: uncurry this
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)

        # rigids_1 = ru.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, (0, 4, 5, 6, 7)]
        rigids_1 = ru.Rigid.from_tensor_4x4(chain_feats['cg_groups_gt_frames'])[:, (0, 2, 3)]

        # compute bb frame features
        diffuse_mask = self._gen_diffuse_mask(res_data)
        res_data['res_noising_mask'] = diffuse_mask
        res_data['x'] = rigids_1.get_trans()  # for HeteroData's sake
        rigids_1_tensor_7 = rigids_1.to_tensor_7()
        rigids_mask = chain_feats["cg_groups_gt_exists"][:, (0, 2, 3)]
        rigids_mask[..., 0] = res_data['res_mask']
        rigids_1_tensor_7 = (
            rigids_1_tensor_7 * rigids_mask[..., None] +
            rigids_1_tensor_7[..., 0:1, :] * (1 - rigids_mask[..., None].float())
        )
        res_data['rigids_1'] = rigids_1_tensor_7

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

        if self.polar_upweight:
            seq = res_data['seq']
            polar_mask = (seq[..., None] == self.polar_residues.to(seq.device)[None]).any(dim=-1)
            res_data['polar_mask'] = polar_mask

        data = self.frame_noiser.corrupt_batch(data)

        return data


    def run_eval(self, model, inputs):
        device = inputs['residue']['x'].device
        self.frame_noiser.set_device(device)
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
        self.frame_noiser.set_device(device)
        # model.self_conditioning = False

        num_res = inputs['num_res']
        total_num_res = sum(num_res)
        data_list = []
        for n in num_res:
            data = HeteroData(
                residue={
                    "res_mask": torch.ones(n, device=device).bool(),
                    "res_noising_mask": torch.ones(n, device=device).bool(),
                    "seq": torch.full((n,), residue_constants.restype_order['R'], device=device).long(),
                    "num_nodes": n
                }
            )
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        res_data = batch['residue']
        # Set-up initial prior samples
        if self.frame_noiser.trans_preconditioning:
            trans_0 = (
                _centered_gaussian(res_data.batch, self.frame_noiser.rigids_per_res, device)
                * self.frame_noiser.trans_preconditioning_std
            )
        else:
            trans_0 = (
                _centered_gaussian(res_data.batch, self.frame_noiser.rigids_per_res, device) * du.NM_TO_ANG_SCALE
            )
        rotmats_0 = _uniform_so3(total_num_res, self.frame_noiser.rigids_per_res, device)

        # Set-up time
        ts = torch.linspace(self.frame_noiser._cfg.min_t, 1.0, self.frame_noiser._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(
            trans_0,  # trans
            rotmats_0,  # rot
            torch.ones((total_num_res, 21), device=device).float(),  # seq logits
        )]
        clean_traj = []
        denoiser_out = None

        for t_2 in tqdm.tqdm(ts[1:]):
            # Run model.
            trans_t_1, rotmats_t_1,  _ = prot_traj[-1]
            res_data["trans_t"] = trans_t_1
            res_data["rotmats_t"] = rotmats_t_1
            res_data['rigids_t'] = ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_t_1),
                trans=trans_t_1
            ).to_tensor_7()
            t = torch.ones(batch.num_graphs, device=device) * t_1
            batch["t"] = t
            if self.trans_preconditioning:
                trans_var_scaling_dict = self.frame_noiser.var_scaling_factors(t)
                batch['trans_c_skip'] = trans_var_scaling_dict['c_skip']
                batch['trans_c_in'] = trans_var_scaling_dict['c_in']
                batch['trans_c_out'] = trans_var_scaling_dict['c_out']

            denoiser_out = model(batch, self_condition=denoiser_out)

            # Process model output.
            pred_rigids = denoiser_out['final_rigids']
            pred_trans_1 = pred_rigids.get_trans()
            pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()

            clean_traj.append(
                (
                    pred_trans_1.detach().cpu(),
                    pred_rotmats_1.detach().cpu(),
                    denoiser_out['decoded_seq_logits'].detach().cpu().argmax(dim=-1),
                    denoiser_out['denoised_atom14'].detach().cpu(),
                    denoiser_out['denoised_atom14_gt_seq'].detach().cpu()
                )
            )

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self.frame_noiser._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self.frame_noiser._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)

            prot_traj.append(
                (trans_t_2,
                 rotmats_t_2,
                 denoiser_out["decoded_seq_logits"].argmax(dim=-1).detach().cpu(),
                )
            )
            t_1 = t_2

            if not model.self_conditioning:
                denoiser_out = None

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1, _ = prot_traj[-1]
        res_data["trans_t"] = trans_t_1
        res_data["rotmats_t"] = rotmats_t_1
        res_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        t = torch.ones(batch.num_graphs, device=device) * t_1
        batch["t"] = t
        if self.trans_preconditioning:
            trans_var_scaling_dict = self.frame_noiser.var_scaling_factors(t)
            batch['trans_c_skip'] = trans_var_scaling_dict['c_skip']
            batch['trans_c_in'] = trans_var_scaling_dict['c_in']
            batch['trans_c_out'] = trans_var_scaling_dict['c_out']

        denoiser_out = model(batch, self_condition=denoiser_out)

        # Process model output.
        pred_rigids = denoiser_out['final_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
        decoded_struct = denoiser_out['denoised_atom14']
        argmax_seq = denoiser_out['decoded_seq_logits'].detach().cpu().argmax(dim=-1)

        clean_traj.append(
            (
                pred_trans_1.detach().cpu(),
                pred_rotmats_1.detach().cpu(),
                denoiser_out['decoded_seq_logits'].detach().cpu().argmax(dim=-1),
                denoiser_out['denoised_atom14'].detach().cpu()
            )
        )

        # # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrotpsi_to_atom37(prot_traj, res_data.res_mask)
        # clean_atom37_traj = all_atom.transrotpsi_to_atom37(clean_traj, res_data.res_mask)
        clean_trajs = zip(*[traj[-2].split(num_res) for traj in clean_traj])
        clean_traj_seqs = zip(*[traj[-3].split(num_res) for traj in clean_traj])
        all_R_clean_trajs = zip(*[traj[-1].split(num_res) for traj in clean_traj])
        prot_trajs = zip(*[traj[-1].split(num_res) for traj in prot_traj])

        return {
            "samples": decoded_struct.split(num_res),
            "seqs": argmax_seq.split(num_res),
            "clean_trajs": clean_trajs,
            "all_R_clean_trajs": all_R_clean_trajs,
            "clean_traj_seqs": clean_traj_seqs,
            "prot_trajs": prot_trajs,
            "inputs": inputs
        }


    def compute_loss(self, inputs, outputs: Dict):
        frame_fm_loss_dict = multiframe_fm_loss(
            inputs, outputs, sep_rot_loss=True,
            t_norm_clip=0.9,
            square_aux_loss_time_factor=True,
            use_fafe=self.use_fafe,
            use_fafe_l2=self.use_fafe_l2,
            polar_upweight=self.polar_upweight,
            sidechain_upweight=self.sidechain_upweight,
            trans_preconditioning=self.trans_preconditioning,
            rot_exp_weighting=self.rot_loss_exp_weighting,
            rot_vf_angle_loss_weight=self.rot_vf_angle_loss_weight
        )

        frame_vf_loss = (
            frame_fm_loss_dict["trans_vf_loss"] +
            frame_fm_loss_dict["rot_vf_loss"]
        )
        unscaled_frame_vf_loss = (
            frame_fm_loss_dict["unscaled_trans_vf_loss"] +
            frame_fm_loss_dict["unscaled_rot_vf_loss"]
        )
        bb_denoising_finegrain_loss = (
            frame_fm_loss_dict["scaled_pred_bb_mse"]
            + frame_fm_loss_dict["scaled_dist_mat_loss"]
        ) * (inputs['t'] > self.aux_loss_t_min)

        traj_loss_dict = multiframe_traj_loss(inputs, outputs, traj_decay_factor=0.99)

        atomic_loss_dict = atomic_losses(
            inputs,
            outputs,
            use_smooth_lddt=True,
            use_fape=self.use_fape,
            t_norm_clip=0.9,
            use_sidechain_dists_mse_loss=False,
            use_sidechain_clash_loss=False,
            polar_upweight=self.polar_upweight,
            sidechain_upweight=self.sidechain_upweight
        )

        atomic_loss = (
            atomic_loss_dict["scaled_atom14_mse"]
            + atomic_loss_dict["seq_loss"]
            + atomic_loss_dict["smooth_lddt"]
            + atomic_loss_dict[('scaled_fape' if self.scale_frame_aligned_errors or self.scale_fape else 'fape')]
        )


        loss = (
            frame_vf_loss
            + self.fafe_weight * frame_fm_loss_dict[('scaled_fafe' if self.scale_frame_aligned_errors or self.scale_fafe else 'fafe')]
            + 0.25 * bb_denoising_finegrain_loss
            + atomic_loss
            + traj_loss_dict['traj_bb_loss']
            + traj_loss_dict['traj_rigids_loss'] * self.rigids_traj_loss_scale
            + traj_loss_dict['traj_pred_dist_loss'] * 0.5
            + traj_loss_dict['traj_pred_framepair_dist_loss'] * 0.5
            + traj_loss_dict['traj_seq_loss'] * 0.25
            + traj_loss_dict['traj_seqpair_loss'] * 0.25
        )
        loss = loss.mean()

        loss_dict = {"loss": loss, "frame_vf_loss": frame_vf_loss, "unscaled_frame_vf_loss": unscaled_frame_vf_loss}
        loss_dict.update(frame_fm_loss_dict)
        loss_dict.update(atomic_loss_dict)
        loss_dict.update(traj_loss_dict)
        # loss_dict.update(discrim_loss_dict)
        return loss_dict


