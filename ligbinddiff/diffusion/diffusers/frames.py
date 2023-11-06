""" Diffusion modules """
import torch
from torch import nn
import tqdm
import torch.distributions as dist
import numpy as np

from ligbinddiff.model.denoiser.bb.framediff import IpaScoreWrapper, KnnIpaScoreWrapper
from ligbinddiff.model.denoiser.bb.frames import GraphIpaFrameDenoiser, PSAEBFrameDenoiser
from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.model.utils.graph import batchwise_to_nodewise, get_data_lens

from torch_geometric.data import Batch, Data


class FrameDiff(nn.Module):
    def __init__(self,
                 se3_noiser,
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

        self.denoiser = IpaScoreWrapper(
            se3_noiser,
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_points=num_qk_pts,
            num_v_points=num_v_pts,
            num_blocks=num_layers
        )
        self.time_dist = dist.Uniform(0, 1)
        self.time_T = 1
        self.se3_noiser = se3_noiser

    def reverse_noising(self, data):
        denoiser_output = self.denoiser(data)
        x_mask = data['x_mask']
        noising_mask = ~data['fixed_mask'].bool()
        mask = x_mask | ~noising_mask
        pred_bb_score = self.bb_score_fn(data, denoiser_output, mask)
        denoiser_output["pred_bb_score"] = pred_bb_score
        return denoiser_output

    def bb_score_fn(self, data, denoiser_output, mask):
        bb_x_t = ru.Rigid.from_tensor_7(data[self.bb_x_t_key])
        t = data['t']
        bb_x_0_pred = denoiser_output[self.bb_x_0_pred_key]
        num_nodes = bb_x_t.shape[0]
        bb_x_t = bb_x_t.view([data.num_graphs, -1])
        bb_x_0_pred = bb_x_0_pred.view([data.num_graphs, -1])
        rots_x_t = bb_x_t.get_rots()
        rots_x_0_pred = bb_x_0_pred.get_rots()
        rot_score = self.se3_noiser.calc_rot_score(
            rots_x_t,
            rots_x_0_pred,
            t
        )
        rot_score = rot_score.view(num_nodes, -1)

        trans_x_t = bb_x_t.get_trans()
        trans_x_0_pred = bb_x_0_pred.get_trans()
        trans_score = self.se3_noiser.calc_trans_score(
            trans_x_t,
            trans_x_0_pred,
            t[:, None, None],
            use_torch=True
        )
        trans_score = trans_score.view(num_nodes, -1)

        rot_score = rot_score * ~mask[..., None]
        trans_score = trans_score * ~mask[..., None]
        return (rot_score, trans_score)

    def forward(self, data):
        denoised_outputs = self.reverse_noising(data)
        return data, denoised_outputs

    def sample_prior(self, num_nodes, device):
        sampled_residx = torch.arange(num_nodes, device=device)
        noising_mask = torch.ones(num_nodes, device=device).bool()
        bb_prior = self.se3_noiser.sample_ref(n_samples=num_nodes)
        out_dict = {
            'noised_residx': sampled_residx,
            'noising_mask': noising_mask
        }
        out_dict.update(bb_prior)
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
        rot_score, trans_score = self.bb_score_fn(
            data,
            denoiser_output,
            mask
        )

        bb_x_tm1 = self.se3_noiser.reverse(
            bb_x_t,
            rot_score.numpy(force=True),
            trans_score.numpy(force=True),
            t.item(),
            delta_t,
            noising_mask.numpy(force=True),
            noise_scale=noise_scale
        )
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

class KnnFrameDiff(nn.Module):
    def __init__(self,
                 se3_noiser,
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

        self.denoiser = KnnIpaScoreWrapper(
            se3_noiser,
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_points=num_qk_pts,
            num_v_points=num_v_pts,
            num_blocks=num_layers
        )
        self.time_dist = dist.Uniform(0, 1)
        self.time_T = 1
        self.se3_noiser = se3_noiser

    def reverse_noising(self, data):
        denoiser_output = self.denoiser(data)
        x_mask = data['x_mask']
        noising_mask = ~data['fixed_mask'].bool()
        mask = x_mask | ~noising_mask
        pred_bb_score = self.bb_score_fn(data, denoiser_output, mask)
        denoiser_output["pred_bb_score"] = pred_bb_score
        return denoiser_output

    def bb_score_fn(self, data, denoiser_output, mask):
        bb_x_t = ru.Rigid.from_tensor_7(data[self.bb_x_t_key])
        t = data['t']
        bb_x_0_pred = denoiser_output[self.bb_x_0_pred_key]
        num_nodes = bb_x_t.shape[0]
        bb_x_t = bb_x_t.view([data.num_graphs, -1])
        bb_x_0_pred = bb_x_0_pred.view([data.num_graphs, -1])
        rots_x_t = bb_x_t.get_rots()
        rots_x_0_pred = bb_x_0_pred.get_rots()
        rot_score = self.se3_noiser.calc_rot_score(
            rots_x_t,
            rots_x_0_pred,
            t
        )
        rot_score = rot_score.view(num_nodes, -1)

        trans_x_t = bb_x_t.get_trans()
        trans_x_0_pred = bb_x_0_pred.get_trans()
        trans_score = self.se3_noiser.calc_trans_score(
            trans_x_t,
            trans_x_0_pred,
            t[:, None, None],
            use_torch=True
        )
        trans_score = trans_score.view(num_nodes, -1)

        rot_score = rot_score * ~mask[..., None]
        trans_score = trans_score * ~mask[..., None]
        return (rot_score, trans_score)

    def forward(self, data):
        denoised_outputs = self.reverse_noising(data)
        return data, denoised_outputs

    def sample_prior(self, num_nodes, device):
        sampled_residx = torch.arange(num_nodes, device=device)
        noising_mask = torch.ones(num_nodes, device=device).bool()
        bb_prior = self.se3_noiser.sample_ref(n_samples=num_nodes)
        out_dict = {
            'noised_residx': sampled_residx,
            'noising_mask': noising_mask
        }
        out_dict.update(bb_prior)
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
        rot_score, trans_score = self.bb_score_fn(
            data,
            denoiser_output,
            mask
        )

        bb_x_tm1 = self.se3_noiser.reverse(
            bb_x_t,
            rot_score.numpy(force=True),
            trans_score.numpy(force=True),
            t.item(),
            delta_t,
            noising_mask.numpy(force=True),
            noise_scale=noise_scale
        )
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


class GraphFrameDiff(nn.Module):
    def __init__(self,
                 se3_noiser,
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
                 self_conditioning=False,
                 graph_conditioning=False,
                 use_anchors=False,
                 ):
        super().__init__()
        self.bb_x_0_key = bb_x_0_key
        self.bb_x_0_pred_key = bb_x_0_pred_key
        self.bb_x_t_key = bb_x_t_key
        self.self_conditioning = self_conditioning

        self.denoiser = GraphIpaFrameDenoiser(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_pts=num_qk_pts,
            num_v_pts=num_v_pts,
            n_layers=num_layers,
            self_conditioning=self_conditioning,
            graph_conditioning=graph_conditioning,
            use_anchors=use_anchors
        )
        self.time_dist = dist.Uniform(0, 1)
        self.time_T = 1
        self.se3_noiser = se3_noiser

    def reverse_noising(self, data):
        if self.self_conditioning and np.random.uniform() > 0.5:
            with torch.no_grad():
                self_conditioning = self.denoiser(data)
        else:
            self_conditioning = None
        # print(self_conditioning is not None)
        denoiser_output = self.denoiser(data, self_conditioning)
        x_mask = data['x_mask']
        noising_mask = ~data['fixed_mask'].bool()
        mask = x_mask | ~noising_mask
        pred_bb_score = self.bb_score_fn(data, denoiser_output, mask)
        denoiser_output["pred_bb_score"] = pred_bb_score
        return denoiser_output

    def bb_score_fn(self, data, denoiser_output, mask):
        bb_x_t = ru.Rigid.from_tensor_7(data[self.bb_x_t_key])
        t = data['t']
        data_lens = get_data_lens(data, 'x')
        bb_x_0_pred = denoiser_output[self.bb_x_0_pred_key]
        rots_x_t = bb_x_t.get_rots()
        rots_x_0_pred = bb_x_0_pred.get_rots()
        rot_score = self.se3_noiser.calc_rot_score2(
            rots_x_t,
            rots_x_0_pred,
            t,
            data.batch
        )
        rot_score = rot_score.squeeze(0)

        t_per_node = batchwise_to_nodewise(t, data_lens)
        trans_x_t = bb_x_t.get_trans()
        trans_x_0_pred = bb_x_0_pred.get_trans()
        trans_score = self.se3_noiser.calc_trans_score(
            trans_x_t,
            trans_x_0_pred,
            t_per_node[:, None],
            use_torch=True
        )

        rot_score = rot_score * ~mask[..., None]
        trans_score = trans_score * ~mask[..., None]
        return (rot_score, trans_score)

    def forward(self, data):
        denoised_outputs = self.reverse_noising(data)
        return data, denoised_outputs

    def sample_prior(self, num_nodes, device):
        sampled_residx = torch.arange(num_nodes, device=device)
        noising_mask = torch.ones(num_nodes, device=device).bool()
        bb_prior = self.se3_noiser.sample_ref(n_samples=num_nodes)
        out_dict = {
            'noised_residx': sampled_residx,
            'noising_mask': noising_mask
        }
        out_dict.update(bb_prior)
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

        if self.self_conditioning and "self_condition" in intermediates.keys():
                denoiser_output = self.denoiser(data, self_condition=intermediates['self_condition'])
        else:
            denoiser_output = self.denoiser(data)

        rot_score, trans_score = self.bb_score_fn(
            data,
            denoiser_output,
            mask
        )

        bb_x_tm1 = self.se3_noiser.reverse(
            bb_x_t,
            rot_score.numpy(force=True),
            trans_score.numpy(force=True),
            t.item(),
            delta_t,
            noising_mask.numpy(force=True),
            noise_scale=noise_scale
        )
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
        denoiser_output = None

        with torch.no_grad():
            if show_progress:
                pbar = tqdm.tqdm(total=steps)

            while (intermediates['t'] > np.abs(delta_t)).all():
                bb_x_tm1, tm1, denoiser_output = self.reverse_step(intermediates, delta_t, noise_scale=noise_scale)
                intermediates['t'] = tm1
                intermediates[self.bb_x_t_key] = bb_x_tm1
                if self.self_conditioning:
                    intermediates['self_condition'] = denoiser_output

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


class PSAEBFrameDiff(nn.Module):
    def __init__(self,
                 se3_noiser,
                 c_s=128,
                 c_v=16,
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

        self.denoiser = PSAEBFrameDenoiser(
            c_s=c_s,
            c_v=c_v,
            c_z=c_z,
            c_hidden=c_hidden,
            num_heads=num_heads,
            num_qk_pts=num_qk_pts,
            num_v_pts=num_v_pts,
            n_layers=num_layers
        )
        self.time_dist = dist.Uniform(0, 1)
        self.time_T = 1
        self.se3_noiser = se3_noiser

    def reverse_noising(self, data):
        denoiser_output = self.denoiser(data)
        x_mask = data['x_mask']
        noising_mask = ~data['fixed_mask'].bool()
        mask = x_mask | ~noising_mask
        pred_bb_score = self.bb_score_fn(data, denoiser_output, mask)
        denoiser_output["pred_bb_score"] = pred_bb_score
        return denoiser_output

    def bb_score_fn(self, data, denoiser_output, mask):
        bb_x_t = ru.Rigid.from_tensor_7(data[self.bb_x_t_key])
        t = data['t']
        data_lens = get_data_lens(data, 'x')
        bb_x_0_pred = denoiser_output[self.bb_x_0_pred_key]
        rots_x_t = bb_x_t.get_rots()
        rots_x_0_pred = bb_x_0_pred.get_rots()
        rot_score = self.se3_noiser.calc_rot_score2(
            rots_x_t,
            rots_x_0_pred,
            t,
            data.batch
        )
        rot_score = rot_score.squeeze(0)

        t_per_node = batchwise_to_nodewise(t, data_lens)
        trans_x_t = bb_x_t.get_trans()
        trans_x_0_pred = bb_x_0_pred.get_trans()
        trans_score = self.se3_noiser.calc_trans_score(
            trans_x_t,
            trans_x_0_pred,
            t_per_node[:, None],
            use_torch=True
        )

        rot_score = rot_score * ~mask[..., None]
        trans_score = trans_score * ~mask[..., None]
        return (rot_score, trans_score)

    def forward(self, data):
        denoised_outputs = self.reverse_noising(data)
        return data, denoised_outputs

    def sample_prior(self, num_nodes, device):
        sampled_residx = torch.arange(num_nodes, device=device)
        noising_mask = torch.ones(num_nodes, device=device).bool()
        bb_prior = self.se3_noiser.sample_ref(n_samples=num_nodes)
        out_dict = {
            'noised_residx': sampled_residx,
            'noising_mask': noising_mask
        }
        out_dict.update(bb_prior)
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
        rot_score, trans_score = self.bb_score_fn(
            data,
            denoiser_output,
            mask
        )

        bb_x_tm1 = self.se3_noiser.reverse(
            bb_x_t,
            rot_score.numpy(force=True),
            trans_score.numpy(force=True),
            t.item(),
            delta_t,
            noising_mask.numpy(force=True),
            noise_scale=noise_scale
        )
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
