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
from proteinzen.harness import TrainingHarness

from proteinzen.runtime.loss.common import atomic_losses_reduced
from proteinzen.runtime.loss.multiframe import multiframe_fm_loss_reduced
from proteinzen.runtime.loss.traj import multiframe_traj_loss
from proteinzen.runtime.training.task import TaskSampler
from proteinzen.stoch_interp.multiframe import _centered_gaussian, _uniform_so3, MultiSE3Interpolant


import proteinzen.stoch_interp.utils as du
from proteinzen.utils.framediff import all_atom
from proteinzen.utils.coarse_grain import compute_atom14_from_cg_frames



class MultiFrameInterpolation(TrainingHarness):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'

    def __init__(self,
                 protein_noiser: MultiSE3Interpolant,
                 task_sampler: TaskSampler,
                 aux_loss_t_min=0.25,
                 use_fape=False,
                 use_fafe=False,
                 use_fafe_l2=False,
                 scale_atom14_mse=True,
                 scale_frame_aligned_errors=False,
                 scale_fape=False,
                 scale_fafe=False,
                 square_fafe_l2=False,
                 square_fafe_scale=True,
                 fafe_l2_block_mask_size=1,
                 fafe_weight=0.25,
                 polar_upweight=False,
                 sidechain_upweight=False,
                 trans_preconditioning=False,
                 rot_preconditioning=False,
                 rot_loss_exp_weighting=False,
                 rot_vf_angle_loss_weight=0.5,
                 disable_rot_vf_time_scaling=False,
                 rigids_traj_loss_scale=1,
                 bb_traj_loss_scale=1,
                 ignore_rigid_2_vf_loss=False,
                 dummy_rigid_to_sidechain_rigid=False,
                 rot_cap_loss_weight=0.0,
                 cg_version=2,
                 resolve_sidechain_ambiguity=False,
                 self_condition_rate=0.5,
                 bb_aux_loss_weight=0.25,
                 sep_rot_loss=True,
                 t_norm_clip=0.9,
                 atom14_mse_weight=1.0,
                 smooth_lddt_weight=1.0,
                 seq_loss_weight=1.0,
                 use_traj_losses=True,
                 polar_residues_v2=False,
                 polar_residues_v3=False,
                 polar_residues_v4=False,
                 polar_residues_v5=False,
                 polar_residues_v6=False,
                 polar_residues_v7=False,
                 downweight_K=False
    ):
        super().__init__()
        self.frame_noiser = protein_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.use_fape = use_fape
        self.use_fafe = use_fafe
        self.use_fafe_l2 = use_fafe_l2
        self.fafe_l2_block_mask_size = fafe_l2_block_mask_size
        self.scale_frame_aligned_errors = scale_frame_aligned_errors
        self.scale_fape = scale_fape
        self.square_fafe_scale = square_fafe_scale
        self.square_fafe_l2 = square_fafe_l2
        self.scale_fafe = scale_fafe
        self.polar_upweight = polar_upweight
        self.sidechain_upweight = sidechain_upweight
        self.trans_preconditioning = trans_preconditioning
        self.rot_preconditioning = rot_preconditioning
        assert (not rot_loss_exp_weighting), "depreciated"
        self.fafe_weight = fafe_weight
        self.rot_vf_angle_loss_weight = rot_vf_angle_loss_weight
        self.rigids_traj_loss_scale = rigids_traj_loss_scale
        self.bb_traj_loss_scale = bb_traj_loss_scale
        self.ignore_rigid_2_vf_loss = ignore_rigid_2_vf_loss
        assert (not disable_rot_vf_time_scaling), "depreciated"
        self.rot_cap_loss_weight = rot_cap_loss_weight
        self.dummy_rigid_to_sidechain_rigid = dummy_rigid_to_sidechain_rigid
        self.cg_version = cg_version
        self.resolve_sidechain_ambiguity = resolve_sidechain_ambiguity
        self.self_condition_rate = self_condition_rate
        self.scale_atom14_mse = scale_atom14_mse
        self.bb_aux_loss_weight = bb_aux_loss_weight
        self.sep_rot_loss = sep_rot_loss
        self.t_norm_clip = t_norm_clip
        self.atom14_mse_weight = atom14_mse_weight
        self.smooth_lddt_weight = smooth_lddt_weight
        self.seq_loss_weight = seq_loss_weight
        self.use_traj_losses = use_traj_losses
        self.downweight_K = downweight_K

        self.rng = np.random.default_rng()

        if polar_residues_v2:
            self.polar_residues = torch.tensor([
                residue_constants.restype_order[i]
                for i in ['C', 'E', 'H', 'Q', 'R', 'S', 'T']
            ])
        elif polar_residues_v3:
            self.polar_residues = torch.tensor([
                residue_constants.restype_order[i]
                for i in ['C', 'D', 'E', 'H', 'Q', 'R', 'S', 'T']
            ])
        elif polar_residues_v4:
            self.polar_residues = torch.tensor([
                residue_constants.restype_order[i]
                for i in ['C', 'D', 'E', 'H', 'P', 'Q', 'R', 'S', 'T']
            ])
        elif polar_residues_v5:
            self.polar_residues = torch.tensor([
                residue_constants.restype_order[i]
                for i in ['C', 'D', 'E', 'H', 'P', 'Q', 'R', 'S', 'T', 'W']
            ])
        elif polar_residues_v6:
            self.polar_residues = torch.tensor([
                residue_constants.restype_order[i]
                for i in ['C', 'E', 'H', 'P', 'Q', 'R', 'S', 'T', 'W']
            ])
        elif polar_residues_v7:
            self.polar_residues = torch.tensor([
                residue_constants.restype_order[i]
                for i in ['C', 'E', 'H', 'P', 'Q', 'R', 'W']
            ])
        else:
            self.polar_residues = torch.tensor([
                residue_constants.restype_order[i]
                for i in ['C', 'D', 'E', 'H', 'K', 'N', 'Q', 'R', 'S', 'T', 'Y']
            ])

        self.task_sampler = task_sampler


    def _resolve_symmetry(self, data, chain_feats):
        res_data = data['residue']
        alt_atom37 = torch.zeros_like(res_data['atom37'])
        alt_atom37.scatter_add_(
            src=chain_feats['atom14_alt_gt_positions'] * chain_feats['atom14_alt_gt_exists'][..., None],
            index=chain_feats['residx_atom14_to_atom37'][..., None].expand(*chain_feats['residx_atom14_to_atom37'].shape, 3),
            dim=-2,
        )
        alt_chain_feats = {
            'aatype': torch.as_tensor(res_data['seq']).long(),
            'all_atom_positions': alt_atom37.double(),
            'all_atom_mask': torch.as_tensor(res_data['atom37_mask']).double()
        }
        alt_chain_feats = data_transforms.atom37_to_cg_frames(alt_chain_feats, cg_version=self.cg_version)

        # rigids_1 = ru.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, (0, 4, 5, 6, 7)]
        rigids_1 = ru.Rigid.from_tensor_4x4(chain_feats['cg_groups_gt_frames'])[:, (0, 2, 3)]
        alt_rigids_1 = ru.Rigid.from_tensor_4x4(alt_chain_feats['cg_groups_gt_frames'])[:, (0, 2, 3)]

        reconstructed_atom14 = compute_atom14_from_cg_frames(rigids_1, res_data['res_mask'], res_data['seq'], cg_version=self.cg_version)
        alt_reconstructed_atom14 = compute_atom14_from_cg_frames(alt_rigids_1, res_data['res_mask'], res_data['seq'], cg_version=self.cg_version)
        gt_diff = torch.sum(
            chain_feats['atom14_gt_exists'][..., None] *
            (reconstructed_atom14 - chain_feats['atom14_gt_positions']) ** 2, dim=(-1, -2)
        )
        alt_diff = torch.sum(
            chain_feats['atom14_gt_exists'][..., None] *
            (alt_reconstructed_atom14 - chain_feats['atom14_gt_positions']) ** 2, dim=(-1, -2)
        )
        select_alt = alt_diff < gt_diff
        print(select_alt.long().sum(), gt_diff.sum(), alt_diff.sum())
        tensor_4x4 = (
            chain_feats['cg_groups_gt_frames'] * (~select_alt[..., None, None, None])
            + alt_chain_feats['cg_groups_gt_frames'] * (select_alt[..., None, None, None])
        )
        return ru.Rigid.from_tensor_4x4(tensor_4x4)[:, (0, 2, 3)]


    def process_input(self, data: HeteroData, force_t0=False):
        data = copy.deepcopy(data)
        self.frame_noiser.set_device(data['residue']['atom37'].device)
        res_data = data['residue']

        # compute base features
        chain_feats = {
            'aatype': torch.as_tensor(res_data['seq']).long(),
            'all_atom_positions': torch.as_tensor(res_data['atom37']).double(),
            'all_atom_mask': torch.as_tensor(res_data['atom37_mask']).double(),
        }
        # chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.atom37_to_cg_frames(chain_feats, cg_version=self.cg_version)
        chain_feats = data_transforms.atom37_to_torsion_angles(prefix="")(chain_feats)  # TODO: uncurry this
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)

        if self.resolve_sidechain_ambiguity:
            rigids_1 = self._resolve_symmetry(data, chain_feats)
        else:
            rigids_1 = ru.Rigid.from_tensor_4x4(chain_feats['cg_groups_gt_frames'])[:, (0, 2, 3)]

        # compute bb frame features
        rigids_mask = chain_feats["cg_groups_gt_exists"][:, (0, 2, 3)]
        rigids_mask *= res_data['res_mask'][..., None]
        res_data['rigids_mask'] = rigids_mask
        # res_data['rigids_noising_mask'] = rigids_noising_mask
        res_data['x'] = rigids_1.get_trans()  # for HeteroData's sake
        rigids_1_tensor_7 = rigids_1.to_tensor_7()

        if self.dummy_rigid_to_sidechain_rigid:
            seq = res_data['seq']
            mask_AG = (seq == residue_constants.restype_order['G']) | (seq == residue_constants.restype_order['A'])
            dummy_rigid = rigids_1_tensor_7[..., 0, :] * mask_AG[..., None] + rigids_1_tensor_7[..., 1, :] * (~mask_AG[..., None])
            rigids_1_tensor_7 = (
                rigids_1_tensor_7 * rigids_mask[..., None] +
                dummy_rigid[..., None, :] * (1 - rigids_mask[..., None].float())
            )
        else:
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

        task = self.task_sampler.sample_task()
        data['task'] = task
        rigidwise_t, rigidwise_noising_mask, seq_noising_mask = task.sample_t_and_mask(data)
        if force_t0:
            rigidwise_t = torch.zeros_like(rigidwise_t)
        res_data['seq_noising_mask'] = seq_noising_mask
        data['rigidwise_t'] = rigidwise_t
        data['t'] = rigidwise_t.unflatten(0, (data.num_graphs, -1))[..., 0, 0]
        res_data['rigids_noising_mask'] = rigidwise_noising_mask

        data = self.frame_noiser.corrupt_batch(data)

        return data


    def run_eval(self, model, inputs):
        device = inputs['residue']['x'].device
        self.frame_noiser.set_device(device)
        # TODO: should this be a separate flag?
        if model.self_conditioning and np.random.uniform() < self.self_condition_rate:
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
        # if model.rot_preconditioning:
        #     self.frame_noiser._rots_cfg.sample_schedule = 'linear'
        #     self.frame_noiser._rots_cfg.sample_schedule = 'test'

        num_res = inputs['num_res']
        total_num_res = sum(num_res)
        data_list = []
        for n in num_res:
            data = HeteroData(
                residue={
                    "res_mask": torch.ones(n, device=device).bool(),
                    "rigids_mask": torch.ones((n, 3), device=device).bool(),
                    "rigids_noising_mask": torch.ones((n, 3), device=device).bool(),
                    "seq": torch.full((n,), residue_constants.restype_order_with_x['X'], device=device).long(),
                    "seq_mask": torch.ones(n, device=device).bool(),
                    "seq_noising_mask": torch.ones(n, device=device).bool(),
                    "chain_idx": torch.zeros(n, device=device),
                    # "chain_idx": torch.cat([torch.zeros(n//2, device=device), torch.ones(n//2, device=device)], dim=0),
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

        # ts = 1.0 - torch.logspace(-2, 0, self.frame_noiser._sample_cfg.num_timesteps + 1).flip(0)
        # ts = ts - torch.min(ts)
        # ts = ts / torch.max(ts)

        # shift_scale = np.sqrt(num_res[0]/100)
        # ts = ts / (ts * (1 - shift_scale) + shift_scale)

        t_1 = ts[0]

        prot_traj = [(
            trans_0,  # trans
            rotmats_0,  # rot
            torch.ones((total_num_res, 21), device=device).float(),  # seq logits
            compute_atom14_from_cg_frames(
                ru.Rigid(ru.Rotation(rot_mats=rotmats_0), trans_0),
                res_data.res_mask,
                res_data.seq,
                cg_version=self.cg_version
            )
        )]
        clean_traj = []
        denoiser_out = None

        for t_2 in tqdm.tqdm(ts[1:]):
            d_t = t_2 - t_1
            # Run model.
            trans_t_1, rotmats_t_1, _, _ = prot_traj[-1]
            t_hat, d_t_hat, trans_t_hat = self.frame_noiser._trans_churn(
                d_t,
                t_1,
                trans_t_1,
                noising_mask=res_data['rigids_noising_mask']
            )
            _, _, rotmats_t_hat = self.frame_noiser._rot_churn(
                d_t,
                t_1,
                rotmats_t_1,
                noising_mask=res_data['rigids_noising_mask']
            )

            res_data["trans_t"] = trans_t_hat
            res_data["rotmats_t"] = rotmats_t_hat
            res_data['rigids_t'] = ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_t_hat),
                trans=trans_t_hat
            ).to_tensor_7()
            t = torch.ones(batch.num_graphs, device=device) * t_hat
            batch["t"] = t
            batch["rigidwise_t"] = torch.ones((total_num_res, self.frame_noiser.rigids_per_res), device=device) * t_hat

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
            trans_t_2 = self.frame_noiser._trans_euler_step(
                d_t_hat,
                t_hat,
                pred_trans_1,
                trans_t_hat,
                noising_mask=res_data['rigids_noising_mask'],
                add_noise=False
            )
            rotmats_t_2 = self.frame_noiser._rots_euler_step(
                d_t_hat,
                t_hat,
                pred_rotmats_1,
                rotmats_t_hat,
                noising_mask=res_data['rigids_noising_mask'],
                add_noise=False
            )

            prot_traj.append(
                (trans_t_2,
                 rotmats_t_2,
                 denoiser_out["decoded_seq_logits"].argmax(dim=-1).detach().cpu(),
                 compute_atom14_from_cg_frames(
                     ru.Rigid(ru.Rotation(rot_mats=rotmats_t_2), trans_t_2),
                     res_data.res_mask,
                     res_data.seq,
                     cg_version=self.cg_version
                 )
                )
            )
            t_1 = t_2

            if not model.self_conditioning:
                denoiser_out = None

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1, _, _ = prot_traj[-1]
        res_data["trans_t"] = trans_t_1
        res_data["rotmats_t"] = rotmats_t_1
        res_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        t = torch.ones(batch.num_graphs, device=device) * t_1
        batch["t"] = t
        batch["rigidwise_t"] = torch.ones((total_num_res, self.frame_noiser.rigids_per_res), device=device) * t_1

        denoiser_out = model(batch, self_condition=denoiser_out)

        # Process model output.
        pred_rigids = denoiser_out['final_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
        decoded_struct = denoiser_out['denoised_atom14']
        argmax_seq = denoiser_out['decoded_seq_logits'].detach().cpu()[..., :-1].argmax(dim=-1)

        clean_traj.append(
            (
                pred_trans_1.detach().cpu(),
                pred_rotmats_1.detach().cpu(),
                denoiser_out['decoded_seq_logits'].detach().cpu().argmax(dim=-1),
                denoiser_out['denoised_atom14'].detach().cpu(),
                denoiser_out['denoised_atom14_gt_seq'].detach().cpu()
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
        frame_fm_loss_dict = multiframe_fm_loss_reduced(
            inputs, outputs, sep_rot_loss=self.sep_rot_loss,
            t_norm_clip=self.t_norm_clip,
            polar_upweight=self.polar_upweight,
            sidechain_upweight=self.sidechain_upweight,
            rot_vf_angle_loss_weight=self.rot_vf_angle_loss_weight,
            rot_cap_loss_weight=self.rot_cap_loss_weight,
            fafe_l2_block_mask_size=self.fafe_l2_block_mask_size,
            downweight_K=self.downweight_K
        )

        frame_vf_loss = (
            frame_fm_loss_dict["trans_vf_loss"] +
            frame_fm_loss_dict["rot_vf_loss"]
        )
        unscaled_frame_vf_loss = (
            frame_fm_loss_dict["unscaled_trans_vf_loss"] +
            frame_fm_loss_dict["unscaled_rot_vf_loss"]
        )

        atomic_loss_dict = atomic_losses_reduced(
            inputs,
            outputs,
            polar_upweight=self.polar_upweight,
            sidechain_upweight=self.sidechain_upweight
        )

        atomic_loss = (
            self.seq_loss_weight * atomic_loss_dict["seq_loss"]
            + self.smooth_lddt_weight * atomic_loss_dict["smooth_lddt"]
        )

        loss = (
            frame_vf_loss
            + self.fafe_weight * frame_fm_loss_dict[
                ('scaled_fafe' if self.scale_frame_aligned_errors or self.scale_fafe else 'fafe')
            ]
            + atomic_loss
        )
        loss = loss.mean()

        loss_dict = {"loss": loss, "frame_vf_loss": frame_vf_loss, "frame_vf_loss_unscaled": unscaled_frame_vf_loss}
        loss_dict[inputs['task'].name + "_loss"] = loss
        loss_dict[inputs['task'].name + "_seq_loss"] = atomic_loss_dict["seq_loss"]
        loss_dict[inputs['task'].name + "_frame_vf_loss"] = frame_vf_loss
        loss_dict[inputs['task'].name + "_frame_vf_loss_unscaled"] = unscaled_frame_vf_loss
        loss_dict.update(frame_fm_loss_dict)
        loss_dict.update(atomic_loss_dict)
        # loss_dict.update(discrim_loss_dict)
        return loss_dict


