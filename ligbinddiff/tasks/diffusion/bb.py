
from typing import Sequence, Dict

import tree
import torch
import numpy as np

from torch_geometric.data import HeteroData


from ligbinddiff.data.openfold import data_transforms
from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.tasks import Task
from ligbinddiff.model.utils.graph import batchwise_to_nodewise, get_data_lens

from ligbinddiff.runtime.loss.frames import bb_frame_diffusion_loss


class BackboneFrameNoising(Task):

    bb_x_0_key='rigids_0'
    bb_x_0_pred_key='final_rigids'
    bb_x_t_key='rigids_t'

    def __init__(self,
                 se3_noiser,
                 sample_t_min=0.01,
                 aux_loss_t_max=0.25,
                 self_conditioning=False):
        super().__init__()
        self.se3_noiser = se3_noiser
        self.sample_t_min = sample_t_min
        self.aux_loss_t_max = aux_loss_t_max
        self.self_conditioning = self_conditioning

    def gen_diffuse_mask(self, data: HeteroData):
        return torch.ones_like(data['res_mask']).bool()

    def process_input(self, data: HeteroData):
        num_graphs = data.num_graphs
        res_data = data['residue']

        # compute bb rigids
        chain_feats = {
            'aatype': torch.as_tensor(res_data['seq']).long(),
            'all_atom_positions': torch.as_tensor(res_data['atom37']).double(),
            'all_atom_mask': torch.as_tensor(res_data['atom37_mask']).double()
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        rigids_0 = ru.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]

        # compute diffusion time
        t = torch.rand(num_graphs)
        nodewise_t = batchwise_to_nodewise(t, res_data.batch)

        # compute noising mask
        diffuse_mask = self.gen_diffuse_mask(res_data)

        diff_feats_t = self.se3_noiser.nodewise_forward_marginal(
            rigids_0=rigids_0,
            t=nodewise_t,
            diffuse_mask=diffuse_mask.float(),
        )
        diff_feats_t['x'] = rigids_0.get_trans()  # for HeteroData's sake
        diff_feats_t['rigids_0'] = rigids_0.to_tensor_7()
        diff_feats_t['t'] = t
        diff_feats_t['noising_mask'] = diffuse_mask
        diff_feats_t = tree.map_structure(
            lambda x: torch.as_tensor(x),
            diff_feats_t)

        data['residue'].update(diff_feats_t)
        return data

    def run_eval(self, model, inputs):
        if self.self_conditioning and np.random.uniform() > 0.5:
            with torch.no_grad():
                self_conditioning = model(inputs)
        else:
            self_conditioning = None
        denoiser_output = model(inputs, self_conditioning)
        pred_bb_score = self.bb_score_fn(inputs, denoiser_output)
        denoiser_output["pred_bb_score"] = pred_bb_score
        return denoiser_output

    def bb_score_fn(self, data, denoiser_output):
        res_data = data['residue']
        res_mask = res_data['res_mask']
        noising_mask = res_data['noising_mask']
        mask = res_mask & noising_mask

        bb_x_t = ru.Rigid.from_tensor_7(res_data[self.bb_x_t_key])
        t = res_data['t']
        bb_x_0_pred = denoiser_output[self.bb_x_0_pred_key]
        rots_x_t = bb_x_t.get_rots()
        rots_x_0_pred = bb_x_0_pred.get_rots()
        rot_score = self.se3_noiser.calc_rot_score2(
            rots_x_t,
            rots_x_0_pred,
            t,
            res_data.batch
        )
        rot_score = rot_score.squeeze(0)

        t_per_node = batchwise_to_nodewise(t, res_data.batch)
        trans_x_t = bb_x_t.get_trans()
        trans_x_0_pred = bb_x_0_pred.get_trans()
        trans_score = self.se3_noiser.calc_trans_score(
            trans_x_t,
            trans_x_0_pred,
            t_per_node[:, None],
            use_torch=True
        )

        rot_score = rot_score * mask[..., None]
        trans_score = trans_score * mask[..., None]
        return (rot_score, trans_score)

    def compute_loss(self, inputs, outputs: Dict):
        bb_frame_diffusion_loss_dict = bb_frame_diffusion_loss(
            inputs, outputs)
        bb_denoising_loss = (
            bb_frame_diffusion_loss_dict["pred_x_ca_mse"] / 100 +
            bb_frame_diffusion_loss_dict["rot_score_loss"] * 0.5
        )
        bb_denoising_finegrain_loss = (
            bb_frame_diffusion_loss_dict["pred_bb_mse"]
            + bb_frame_diffusion_loss_dict["dist_mat_loss"]
        )* (inputs['residue']['t'] < self.aux_loss_t_max)
        loss = (bb_denoising_loss + 0.25 * bb_denoising_finegrain_loss).mean()

        loss_dict = {"loss": loss}
        loss_dict.update(bb_frame_diffusion_loss_dict)
        return loss_dict
