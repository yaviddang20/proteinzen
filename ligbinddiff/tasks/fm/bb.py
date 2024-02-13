import copy
from typing import Sequence, Dict

import tqdm
import tree
import torch
import numpy as np

from torch_geometric.data import HeteroData, Batch


from ligbinddiff.data.openfold import data_transforms
from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.tasks import Task
from ligbinddiff.model.utils.graph import batchwise_to_nodewise

from ligbinddiff.runtime.loss.frames import bb_frame_fm_loss, rg_loss, frame_traj_loss, bb_frame_clash_loss
from ligbinddiff.runtime.loss.atomic.hbond import bb_hbond_loss
from ligbinddiff.stoch_interp.interpolate.se3 import SE3Interpolant, _centered_gaussian, _uniform_so3
import ligbinddiff.stoch_interp.interpolate.utils as du
from ligbinddiff.utils.framediff import all_atom


class BackboneFrameInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'

    def __init__(self,
                 se3_noiser: SE3Interpolant,
                 aux_loss_t_min=0.25,
                 sep_rot_loss=False):
        super().__init__()
        self.se3_noiser = se3_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.sep_rot_loss = sep_rot_loss

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
        self.se3_noiser.set_device(device)

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
            inputs, outputs, sep_rot_loss=self.sep_rot_loss)
        # bb_frame_clash_loss_dict = bb_frame_clash_loss(
        #     inputs, outputs
        # )
        # bb_hbond_loss_dict = bb_hbond_loss(
        #     inputs, outputs
        # )
        # rg_loss_dict = rg_loss(inputs, outputs)
        bb_denoising_loss = (
            bb_frame_diffusion_loss_dict["trans_vf_loss"] * 2 +
            bb_frame_diffusion_loss_dict["rot_vf_loss"]
        )

        bb_denoising_finegrain_loss = (
            bb_frame_diffusion_loss_dict["scaled_pred_bb_mse"]
            + bb_frame_diffusion_loss_dict["scaled_dist_mat_loss"]
            # + bb_frame_clash_loss_dict["scaled_bb_clash_loss"]
            # + torch.stack(list(bb_hbond_loss_dict.values())).sum(dim=0)
        ) * (inputs['t'] > self.aux_loss_t_min)

        loss = (
            bb_denoising_loss
            + 0.25 * bb_denoising_finegrain_loss
        ).mean()

        ff_loss = (
            bb_denoising_loss
            + 0.25 * bb_denoising_finegrain_loss
        ).mean()

        loss_dict = {"loss": loss, "frameflow_loss": ff_loss}
        loss_dict.update(bb_frame_diffusion_loss_dict)
        # loss_dict.update(bb_frame_clash_loss_dict)
        # loss_dict.update(bb_hbond_loss_dict)
        # loss_dict.update(rg_loss_dict)
        return loss_dict
