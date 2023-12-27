""" Diffusion modules """
import torch
from torch import nn
import tqdm
import torch.distributions as dist
import numpy as np

from ligbinddiff.model.denoiser.bb.framediff import IpaScoreWrapper, KnnIpaScoreWrapper
from ligbinddiff.model.denoiser.bb.frames import GraphIpaFrameDenoiser
from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.model.utils.graph import batchwise_to_nodewise, get_data_lens

from torch_geometric.data import Batch, Data


class GraphFrameFlow(nn.Module):
    def __init__(self,
                 se3_interp,
                 c_s=128,
                 c_z=64,
                 c_hidden=128,
                 num_qk_pts=8,
                 num_v_pts=12,
                 num_heads=8,
                 num_layers=4,
                 bb_x_0_key='rigids_0',
                 bb_x_0_pred_key='final_rigids',
                 bb_x_t_key='rigids_t',
                 ):
        super().__init__()
        self.bb_x_0_key = bb_x_0_key
        self.bb_x_0_pred_key = bb_x_0_pred_key
        self.bb_x_t_key = bb_x_t_key

        self.denoiser = GraphIpaFrameDenoiser(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_pts=num_qk_pts,
            num_v_pts=num_v_pts,
            n_layers=num_layers
        )
        self.time_dist = dist.Uniform(0, 1)
        self.time_T = 1
        self.se3_interp = se3_interp

    def reverse_noising(self, data, t_clip=0.1):
        denoiser_output = self.denoiser(data)
        x_mask = data['x_mask']
        noising_mask = ~data['fixed_mask'].bool()
        mask = x_mask | ~noising_mask
        pred_bb_cond_v = self.bb_cond_v(data, denoiser_output, mask, t_clip=t_clip)
        denoiser_output["pred_bb_cond_v"] = pred_bb_cond_v
        return denoiser_output

    def bb_cond_v(self, data, denoiser_output, mask, t_clip=0):
        bb_x_t = ru.Rigid.from_tensor_7(data[self.bb_x_t_key])
        t = data['t']
        t_per_node = batchwise_to_nodewise(t, data.batch)
        bb_x_0_pred = denoiser_output[self.bb_x_0_pred_key]
        rots_x_t = bb_x_t.get_rots()
        rots_x_0_pred = bb_x_0_pred.get_rots()
        rot_cond_v = self.se3_interp.calc_rot_cond_v(
            rots_x_t,
            rots_x_0_pred,
            t_per_node,
            t_clip=t_clip
        )
        rot_cond_v = rot_cond_v.squeeze(0)

        trans_x_t = bb_x_t.get_trans()
        trans_x_0_pred = bb_x_0_pred.get_trans()
        trans_cond_v = self.se3_interp.calc_trans_cond_v(
            trans_x_t,
            trans_x_0_pred,
            t_per_node,
            data.batch,
            t_clip=t_clip
        )

        rot_cond_v = rot_cond_v * ~mask[..., None, None]
        trans_cond_v = trans_cond_v * ~mask[..., None]
        return (rot_cond_v, trans_cond_v)

    def forward(self, data):
        denoised_outputs = self.reverse_noising(data)
        return data, denoised_outputs

    def sample_prior(self, num_nodes, device):
        sampled_residx = torch.arange(num_nodes, device=device)
        noising_mask = torch.ones(num_nodes, device=device).bool()
        bb_prior = self.se3_interp.sample_ref(n_samples=num_nodes)
        out_dict = {
            'noised_residx': sampled_residx,
            'noising_mask': noising_mask
        }
        out_dict.update({
            'rigids_t': ru.Rigid(rots=ru.Rotation(rot_mats=bb_prior[0]), trans=bb_prior[1])
        })
        return out_dict

    def reverse_step(self, intermediates, delta_t, noise_scale=1.0):
        # assert delta_t < 0
        t = intermediates['t']
        noising_mask = intermediates['noising_mask']
        x_mask = torch.zeros_like(noising_mask).bool()
        mask = x_mask | ~noising_mask
        bb_x_t = intermediates[self.bb_x_t_key]

        data = Data(
            x=bb_x_t.get_trans(),
            x_mask=x_mask,
            noising_mask=noising_mask,
            fixed_mask=~noising_mask,
            rigids_t=bb_x_t.to_tensor_7(),
            t=t
        )
        data = Batch.from_data_list([data])
        data = data.to(t.device)

        denoiser_output = self.denoiser(data)
        rot_cond_v, trans_cond_v = self.bb_cond_v(
            data,
            denoiser_output,
            mask,
            t_clip=0
        )

        bb_x_tm1 = self.se3_interp.reverse(
            bb_x_t,
            rot_cond_v,
            trans_cond_v,
            t.item(),
            delta_t,
            noising_mask.float(),
        )
        # center sample
        bb_x_tm1 = bb_x_tm1.translate(-bb_x_tm1.get_trans().mean(dim=0)[None])

        tm1 = t - np.abs(delta_t) #+ delta_t

        return bb_x_tm1, tm1, denoiser_output

    def sample(self,
               num_nodes=None,
               steps=100,
               show_progress=False,
               device=None,
               noise_scale=1.0):

        prior = self.sample_prior(num_nodes, device=device)
        intermediates = prior

        delta_t = self.time_T / steps
        intermediates['t'] = torch.ones(1, device=device)
        bb_x_t = intermediates[self.bb_x_t_key]
        intermediates[self.bb_x_t_key] = bb_x_t.to(device)
        denoiser_output = None

        with torch.no_grad():
            if show_progress:
                pbar = tqdm.tqdm(total=steps)

            while (intermediates['t'] > np.abs(delta_t)).all():
                bb_x_tm1, tm1, denoiser_output = self.reverse_step(intermediates, delta_t, noise_scale=noise_scale)
                intermediates['t'] = tm1
                intermediates[self.bb_x_t_key] = bb_x_tm1

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        intermediates['t'] = torch.as_tensor(intermediates['t']).view(1, 1)
        diffusion_outputs = intermediates
        diffusion_outputs[self.bb_x_0_key] = diffusion_outputs[self.bb_x_t_key]
        diffusion_outputs[self.bb_x_t_key] = bb_x_t
        # add a bunch of keys so the loss fn is happy
        diffusion_outputs.update(denoiser_output)

        return diffusion_outputs, diffusion_outputs

    def sample_impute(self,
                      impute,
                      steps=100,
                      show_progress=False,
                      device=None,
                      noise_scale=1.0):

        num_nodes = impute.num_nodes
        prior = self.se3_interp.forward_marginal(
            ru.Rigid.from_tensor_7(impute['rigids_0']),
            t=torch.ones(num_nodes, device=device)
        )
        prior['noising_mask'] = torch.ones(num_nodes, device=device).bool()
        prior[self.bb_x_t_key] = ru.Rigid.from_tensor_7(prior[self.bb_x_t_key])

        intermediates = prior

        delta_t = self.time_T / steps
        intermediates['t'] = torch.ones(1, device=device)
        bb_x_t = intermediates[self.bb_x_t_key]
        denoiser_output = None

        with torch.no_grad():
            if show_progress:
                pbar = tqdm.tqdm(total=steps)

            while (intermediates['t'] > np.abs(delta_t)).all():
                bb_x_tm1, tm1, denoiser_output = self.reverse_step(intermediates, delta_t, noise_scale=noise_scale)
                intermediates['t'] = tm1
                intermediates[self.bb_x_t_key] = bb_x_tm1

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        intermediates['t'] = torch.as_tensor(intermediates['t']).view(1, 1)
        diffusion_outputs = intermediates
        diffusion_outputs[self.bb_x_0_key] = diffusion_outputs[self.bb_x_t_key]
        diffusion_outputs[self.bb_x_t_key] = bb_x_t
        # add a bunch of keys so the loss fn is happy
        diffusion_outputs.update(denoiser_output)

        return diffusion_outputs, diffusion_outputs
