import copy
from typing import Sequence, Dict

import tqdm
import tree
import torch
import numpy as np
from torch_scatter import scatter

from torch_geometric.data import HeteroData, Batch

from proteinzen.data.openfold import data_transforms
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.tasks import Task
from proteinzen.model.utils.graph import batchwise_to_nodewise

from proteinzen.runtime.loss.frames import bb_frame_fm_loss, rg_loss, frame_traj_loss, bb_frame_clash_loss, bb_plddt_loss
from proteinzen.runtime.loss.atomic.hbond import bb_hbond_loss
from proteinzen.stoch_interp.interpolate.se3 import SE3Interpolant, _centered_gaussian, _uniform_so3, HarmonicPrior
import proteinzen.stoch_interp.interpolate.utils as du
from proteinzen.utils.framediff import all_atom


class BackboneFrameInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'

    def __init__(self,
                 se3_noiser: SE3Interpolant,
                 aux_loss_t_min=0.25,
                 rigid_traj_loss=False,
                 sep_rot_loss=False,
                 local_atomic_dist_r=6,
                 trans_loss_rescale=1,
                 use_hbond_loss=False,
                 use_plddt_loss=False,
                 square_aux_loss_time_factor=False):
        super().__init__()
        self.se3_noiser = se3_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.rigid_traj_loss = rigid_traj_loss
        self.sep_rot_loss = sep_rot_loss
        self.local_atomic_dist_r = local_atomic_dist_r
        self.trans_loss_rescale = trans_loss_rescale
        self.use_plddt_loss = use_plddt_loss
        self.use_hbond_loss = use_hbond_loss
        self.square_aux_loss_time_factor = square_aux_loss_time_factor

    def gen_diffuse_mask(self, data: HeteroData):
        return torch.ones_like(data['res_mask']).bool()

    def process_input(self, data: HeteroData):
        data = copy.deepcopy(data)
        self.se3_noiser.set_device(data['residue']['atom37'].device)
        res_data = data['residue']

        # compute bb rigids
        chain_feats = {
            'aatype': torch.as_tensor(res_data['seq']).long(),
            'all_atom_positions': torch.as_tensor(res_data['atom37']).double(),
            'all_atom_mask': torch.as_tensor(res_data['atom37_mask']).double()
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        rigids_1 = ru.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]

        # compute noising mask
        diffuse_mask = self.gen_diffuse_mask(res_data)
        res_data['noising_mask'] = diffuse_mask
        res_data['res_noising_mask'] = diffuse_mask
        res_data['x'] = rigids_1.get_trans()  # for HeteroData's sake
        res_data['rigids_1'] = rigids_1.to_tensor_7()
        res_data['noising_mask'] = diffuse_mask

        # noise data
        data = self.se3_noiser.corrupt_batch(data)

        return data

    def process_sample_input(self, data: Dict, device='cpu'):
        self.se3_noiser.set_device(device)
        return data

    def run_eval(self, model, inputs):
        # TODO: should this be a separate flag?
        if model.self_conditioning and np.random.uniform() > 0.5:
            with torch.no_grad():
                self_conditioning = model(inputs)
        else:
            self_conditioning = None

        try:
            denoiser_output = model(inputs, self_conditioning)
        except torch.cuda.OutOfMemoryError:
            denoiser_output = torch.utils.checkpoint.checkpoint(
                model.forward,
                inputs,
                self_conditioning,
                use_reentrant=False)

        return denoiser_output

    def run_predict(self,
                    model,
                    inputs,
                    device='cuda:0'):
                    # device='cpu'):
        self.se3_noiser.set_device(device)

        num_res = inputs['num_res']
        total_num_res = sum(num_res)
        data_list = []
        for n in num_res:
            data = HeteroData(
                residue={
                    "res_mask": torch.ones(n, device=device),
                    "noising_mask": torch.ones(n, device=device),
                    "num_nodes": n,
                    "data_lens": torch.as_tensor([n], device=device)
                }
            )
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        res_data = batch['residue']
        # Set-up initial prior samples
        if self.se3_noiser.harmonic_trans_noise:
            noise = []
            for l in num_res:
                prior = HarmonicPrior(l)
                noise.append(prior.sample().to(device))
            noise = torch.cat(noise, dim=0)
            center = scatter(
                noise,
                index=res_data.batch,
                dim=0,
                reduce='mean'
            )
            trans_0 = noise - center[res_data.batch]
        else:
            trans_0 = (
                _centered_gaussian(res_data.batch, device) * du.NM_TO_ANG_SCALE
            )
        rotmats_0 = _uniform_so3(total_num_res, device)

        # Set-up time
        ts = torch.linspace(self.se3_noiser._cfg.min_t, 1.0, self.se3_noiser._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0, torch.zeros((total_num_res, 2), device=device))]
        clean_traj = []
        denoiser_out = None
        for t_2 in tqdm.tqdm(ts[1:]):
            # Run model.
            trans_t_1, rotmats_t_1, _ = prot_traj[-1]
            res_data["trans_t"] = trans_t_1
            res_data["rotmats_t"] = rotmats_t_1
            res_data['rigids_t'] = ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_t_1),
                trans=trans_t_1
            ).to_tensor_7()
            t = torch.ones(batch.num_graphs, device=device) * t_1
            batch["t"] = t
            with torch.no_grad():
                denoiser_out = model(batch, self_condition=denoiser_out)

            # Process model output.
            pred_rigids = denoiser_out['final_rigids']
            pred_trans_1 = pred_rigids.get_trans()
            pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
            pred_psis = denoiser_out['psi'].detach().cpu()
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu(), pred_psis)
            )

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self.se3_noiser._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self.se3_noiser._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)

            if self.se3_noiser.sfm:
                trans_sfm_noise = self.se3_noiser.trans_sfm_noise(trans_t_1, t[res_data.batch])
                rotmat_sfm_noise = self.se3_noiser.rot_sfm_noise(res_data.num_nodes, d_t, device)

                trans_t_2 = trans_t_2 + trans_sfm_noise
                rotmats_t_2 = rotmats_t_2 @ rotmat_sfm_noise

            prot_traj.append((trans_t_2, rotmats_t_2, pred_psis))
            t_1 = t_2

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
        with torch.no_grad():
            denoiser_out = model(batch, self_condition=denoiser_out)

        # Process model output.
        pred_rigids = denoiser_out['final_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
        pred_psis = denoiser_out['psi'].detach().cpu()
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu(), pred_psis)
        )

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrotpsi_to_atom37(prot_traj, res_data.res_mask)
        clean_atom37_traj = all_atom.transrotpsi_to_atom37(clean_traj, res_data.res_mask, impute_oxy=False)

        return {
            "samples": clean_atom37_traj[-1][..., (0, 1, 2, 4, 3), :].split(num_res),
            "clean_trajs": zip(*[clean[..., (0, 1, 2, 4, 3), :].split(num_res) for clean in clean_atom37_traj]),
            "prot_trajs": zip(*[prot[..., (0, 1, 2, 4, 3), :].split(num_res) for prot in atom37_traj]),
            "inputs": inputs
        }

    def compute_loss(self, inputs, outputs: Dict):
        bb_frame_diffusion_loss_dict = bb_frame_fm_loss(
            inputs, outputs,
            sep_rot_loss=self.sep_rot_loss,
            local_atomic_dist_r=self.local_atomic_dist_r,
            square_aux_loss_time_factor=self.square_aux_loss_time_factor)
        # bb_frame_clash_loss_dict = bb_frame_clash_loss(
        #     inputs, outputs
        # )
        # bb_hbond_loss_dict = bb_hbond_loss(
        #     inputs, outputs
        # )
        # rg_loss_dict = rg_loss(inputs, outputs)
        bb_denoising_loss = (
            bb_frame_diffusion_loss_dict["trans_vf_loss"] * 2 * self.trans_loss_rescale +
            bb_frame_diffusion_loss_dict["rot_vf_loss"]
        )

        bb_denoising_finegrain_loss = (
            bb_frame_diffusion_loss_dict["scaled_pred_bb_mse"]
            + bb_frame_diffusion_loss_dict["scaled_dist_mat_loss"]
            # + bb_frame_clash_loss_dict["scaled_bb_clash_loss"]
            # + torch.stack(list(bb_hbond_loss_dict.values())).sum(dim=0)
        ) * (inputs['t'] > self.aux_loss_t_min)

        if self.use_plddt_loss:
            plddt_loss_dict = bb_plddt_loss(
                inputs,
                outputs
            )
            plddt_loss = plddt_loss_dict['plddt_loss']
        else:
            plddt_loss = torch.zeros_like(bb_denoising_loss)


        loss = (
            bb_denoising_loss
            + 0.25 * bb_denoising_finegrain_loss
            + plddt_loss
        )

        if self.use_hbond_loss:
            bb_hbond_loss_dict = bb_hbond_loss(inputs, outputs)
            hbond_loss = (
                bb_hbond_loss_dict['delta_mse'] * 0.01 / (1 - inputs['t'].clip(max=0.9)) ** (2 if self.square_aux_loss_time_factor else 1)
                + bb_hbond_loss_dict['theta_mse']
                + bb_hbond_loss_dict['psi_mse']
                + bb_hbond_loss_dict['X_mse']
            )
            loss = loss + hbond_loss

        loss = loss.mean()

        ff_loss = (
            bb_denoising_loss
            + 0.25 * bb_denoising_finegrain_loss
        ).mean()

        loss_dict = {"loss": loss, "frameflow_loss": ff_loss}
        loss_dict.update(bb_frame_diffusion_loss_dict)
        if self.use_plddt_loss:
            loss_dict.update(plddt_loss_dict)
        if self.use_hbond_loss:
            loss_dict.update(bb_hbond_loss_dict)
        # loss_dict.update(bb_frame_clash_loss_dict)
        # loss_dict.update(bb_hbond_loss_dict)
        # loss_dict.update(rg_loss_dict)


        return loss_dict
