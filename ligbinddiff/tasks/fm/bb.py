
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

from ligbinddiff.runtime.loss.frames import bb_frame_fm_loss
from ligbinddiff.stoch_interp.interpolate.interpolant import Interpolant


class BackboneFrameInterpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'

    def __init__(self,
                 se3_noiser: Interpolant,
                 aux_loss_t_min=0.75):
        super().__init__()
        self.se3_noiser = se3_noiser
        self.aux_loss_t_min = aux_loss_t_min

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
        denoiser_output = model(inputs, self_conditioning)
        return denoiser_output

    def run_predict(self,
                    model,
                    inputs,
                    device='cuda:0'):
        num_res = inputs['num_res']
        self.se3_noiser.set_device(device)
        _, clean_atom37_traj, _ = self.se3_noiser.sample(model, num_res)
        return {
            "samples": clean_atom37_traj[-1][..., (0, 1, 2, 4, 3), :].split(num_res),
            "inputs": inputs
        }

    def compute_loss(self, inputs, outputs: Dict):
        bb_frame_diffusion_loss_dict = bb_frame_fm_loss(
            inputs, outputs)
        bb_denoising_loss = (
            bb_frame_diffusion_loss_dict["trans_vf_loss"] * 2 +
            bb_frame_diffusion_loss_dict["rot_vf_loss"]
        )
        bb_denoising_finegrain_loss = (
            bb_frame_diffusion_loss_dict["scaled_pred_bb_mse"]
            + bb_frame_diffusion_loss_dict["scaled_dist_mat_loss"]
        ) * (inputs['t'] > self.aux_loss_t_min)
        loss = (bb_denoising_loss + 0.25 * bb_denoising_finegrain_loss).mean()

        loss_dict = {"loss": loss}
        loss_dict.update(bb_frame_diffusion_loss_dict)
        return loss_dict
