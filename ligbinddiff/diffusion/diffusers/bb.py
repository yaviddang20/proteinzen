""" Diffusion modules """

import torch
from torch import nn
import tqdm
import numpy as np

from ligbinddiff.model.denoiser.bb.r3 import BackboneR3Denoiser
from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2


class R3InpaintingDiffuser(nn.Module):
    def __init__(self,
                 bb_lmax_list,
                 bb_scheduler,
                 time_dist,
                 time_T,
                 time_discrete,
                 edge_channels_list,
                 h_time=64,
                 scalar_h_dim=128,
                 num_heads=8,
                 bb_channels=32,
                 h_channels=32,
                 num_layers=4,
                 k=30,
                 bb_x_0_key='bb_vecs',
                 bb_x_0_pred_key='denoised_bb_vecs',
                 bb_x_t_key='noised_bb_vecs'):
        super().__init__()
        # build these expensive coeff stores
        bb_SO3_rotation_list = nn.ModuleList()
        for lmax in bb_lmax_list:
            bb_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )

        bb_SO3_grid_list = nn.ModuleList()
        for l in range(max(bb_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(bb_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            bb_SO3_grid_list.append(SO3_m_grid)

        mappingReduced_bb = CoefficientMappingModule(bb_lmax_list, bb_lmax_list)

        denoiser = BackboneR3Denoiser(
            bb_lmax_list=bb_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_bb=mappingReduced_bb,
            bb_SO3_rotation=bb_SO3_rotation_list,
            bb_SO3_grid=bb_SO3_grid_list,
            num_heads=num_heads,
            h_channels=h_channels,
            bb_channels=bb_channels,
            h_time=h_time,
            scalar_h_dim=scalar_h_dim,
            n_layers=num_layers,
        )
        self.denoiser = denoiser
        self.time_dist = time_dist
        self.time_T = time_T
        self.time_discrete = time_discrete
        self.bb_scheduler = bb_scheduler

        self.bb_x_0_key = bb_x_0_key
        self.bb_x_0_pred_key = bb_x_0_pred_key
        self.bb_x_t_key = bb_x_t_key

        self.k = k

    def sample_noise_residx(self, data):
        device = data['residue']['x'].device
        x_mask = data['residue'].x_mask
        num_graphs = data.num_graphs

        node_splits = data._slice_dict['residue']['x']
        num_nodes = (node_splits[1:] - node_splits[:-1]).tolist()
        node_index_split = torch.arange(data['residue'].num_nodes, device=device).split(num_nodes, dim=-1)
        node_mask_split = x_mask.split(num_nodes, dim=-1)

        sampled_src_residx = []
        for node_index, node_mask in zip(node_index_split, node_mask_split):
            srcs = node_index[~node_mask]
            select_idx = torch.randint(low=0, high=srcs.shape[0], size=(1,), device=device)
            sampled_src_residx.append(srcs[select_idx])
        sampled_src_residx = torch.cat(sampled_src_residx, dim=0)

        # select the relevant edges
        edge_selector = data['residue', 'knn', 'residue'].edge_index[1][:, None] == sampled_src_residx[None, :]   # n_edge x n_graph
        edge_selector = edge_selector.any(dim=-1)
        # gather the dst of all src nodes
        sampled_dst_residx = data['residue', 'knn', 'residue'].edge_index[0][edge_selector]  # by construction these must be unique, n_edge
        # do a little reshaping so that the order is [(src, {dsts}), ...]
        sampled_dst_residx = sampled_dst_residx.view(num_graphs, self.k)
        sampled_residx = torch.cat([sampled_src_residx[:, None], sampled_dst_residx], dim=-1).long()
        sampled_residx = sampled_residx.flatten()

        num_nodes = data['residue'].num_nodes
        residxs = torch.arange(num_nodes, device=device)
        noising_mask = (residxs[:, None] == sampled_residx[None, :]).any(dim=-1)

        return sampled_residx, noising_mask

    def forward_noising(self, data, intermediates):
        out_dict = {}
        t = self.time_dist.sample([data.num_graphs])
        t = t.to(data['residue']['x'].device)
        bb_data_coeff, bb_noise_coeff = self.bb_scheduler.noising_coeffs(t)
        noised_data = self.noise_bb(data, intermediates['noised_residx'], bb_data_coeff, bb_noise_coeff)
        out_dict.update(noised_data)
        out_dict['t'] = t
        out_dict['bb_loss_weight'] = self.bb_scheduler.weight(t)
        return out_dict

    def noise_bb(self, data, sampled_residx, data_coeff, noise_coeff):
        bb_x_0 = data['residue']["bb"]
        X_ca = bb_x_0[:, 1]
        bb_x_0_rel = bb_x_0 - X_ca.unsqueeze(-2)
        bb_vecs_0 = bb_x_0_rel.clone()
        bb_vecs_0[:, 1] = X_ca

        selected_bb_vecs_0 = bb_vecs_0[sampled_residx]
        bb_noise = torch.randn_like(selected_bb_vecs_0)

        # print(data.name)
        data_coeff = data_coeff.repeat_interleave(self.k + 1)  # knn + self
        noise_coeff = noise_coeff.repeat_interleave(self.k + 1)
        selected_bb_vecs_t = selected_bb_vecs_0 * data_coeff[:, None, None] + bb_noise + noise_coeff[:, None, None]

        bb_vecs_t = bb_vecs_0.clone()
        bb_vecs_t[sampled_residx] = selected_bb_vecs_t

        bb_x_t = bb_vecs_t.clone()
        bb_x_t[:, (0, 2, 3)] += bb_x_t[:, 1].unsqueeze(-2)

        return {
            self.bb_x_t_key: bb_vecs_t,
            "noised_bb": bb_x_t
        }

    def reverse_noising(self, data, intermediates):
        return self.denoiser(data, intermediates)

    def forward(self, data, warmup=False, deterministic=False):
        sampled_residx, noising_select = self.sample_noise_residx(data)
        intermediates = {
            "noising_select": noising_select,
            "noised_residx": sampled_residx
        }
        noised_data = self.forward_noising(data, intermediates)
        noised_data.update(intermediates)
        denoised_outputs = self.reverse_noising(data, noised_data)
        denoised_outputs.update(noised_data)
        denoised_outputs.update(intermediates)

        denoised_bb_vecs = denoised_outputs[self.bb_x_0_pred_key]
        denoised_bb = denoised_bb_vecs.clone()
        denoised_bb[:, (0, 2, 3)] += denoised_bb[:, 1].unsqueeze(-2)
        denoised_outputs["denoised_bb"] = denoised_bb

        return denoised_outputs

    def sample_prior(self, data):
        sampled_residx, noising_select = self.sample_noise_residx(data)
        prior = {}
        one = torch.ones(data.num_graphs).unsqueeze(-1).to(data['residue']['x'].device)
        zero = torch.zeros(data.num_graphs).unsqueeze(-1).to(data['residue']['x'].device)
        noised_bb = self.noise_bb(data, sampled_residx, zero, one)

        prior.update(noised_bb)
        prior['noised_residx'] = sampled_residx
        prior['noising_select'] = noising_select
        return prior

    def score_fn(self, data, intermediates):
        denoiser_output = self.denoiser(data, intermediates)
        bb_score = self.bb_score_fn(denoiser_output)
        return bb_score, denoiser_output


    def bb_score_fn(self, denoiser_output):
        residx = denoiser_output['noised_residx']
        bb_x_t = denoiser_output[self.bb_x_t_key]
        t = denoiser_output['t']
        subset_t = t[residx]
        alphabar_t = self.bb_scheduler.alphabar(subset_t)

        bb_x_0_pred = denoiser_output[self.bb_x_0_pred_key]
        subset_bb_x_t = bb_x_t[residx]
        subset_bb_x_0_pred = bb_x_0_pred[residx]

        bb_score = torch.sqrt(alphabar_t)[:, None, None] * subset_bb_x_0_pred - subset_bb_x_t
        bb_score = 1 / (1 - alphabar_t[:, None, None]) * bb_score

        return bb_score

    def reverse_step(self, data, intermediates, delta_t):
        assert delta_t < 0
        t = intermediates['t']
        residx = intermediates['noised_residx']
        bb_x_t = intermediates[self.bb_x_t_key]

        subset_t = t[residx]
        bb_beta_t = self.bb_scheduler.beta(subset_t)
        bb_score, denoiser_output = self.score_fn(data, intermediates)

        bb_x_tm1 = self.bb_reverse_step(
            bb_x_t,
            bb_score,
            residx,
            bb_beta_t,
            delta_t)
        tm1 = t + delta_t

        return bb_x_tm1, tm1, denoiser_output

    def bb_reverse_step(self,
                        bb_x_t,
                        bb_score,
                        residx,
                        beta_t,
                        delta_t):
        f = -0.5 * beta_t
        g = torch.sqrt(beta_t)

        subset_bb_x_t = bb_x_t[residx]
        bb_noise = torch.randn_like(subset_bb_x_t)
        bb_drift = (f[:, None, None] * subset_bb_x_t - g[:, None, None]**2 * bb_score) * delta_t
        bb_diffusion = g[:, None, None] * np.sqrt(np.abs(delta_t)) * bb_noise
        bb_delta_x = bb_drift + bb_diffusion

        bb_x_tm1 = bb_x_t.clone()
        bb_x_tm1[residx] = bb_x_tm1[residx] + bb_delta_x
        return bb_x_tm1


    def sample(self,
               data,
               steps=None,
               show_progress=False,
               device=None):
        if device is None:
            device = data['residue']['x'].device

        # num_batch = len(data._slice_dict['x'])
        # num_nodes = num_batch * self.k
        num_nodes = data['residue'].num_nodes
        prior = self.sample_prior(data)
        intermediates = prior

        if steps is not None:
            if self.time_discrete:
                assert self.time_T % steps == 0
        else:
            steps = self.time_T

        intermediates['t'] = torch.ones([num_nodes], device=device).float() * (steps - 1)
        delta_t = - self.time_T // steps
        denoiser_output = None

        with torch.no_grad():
            if show_progress:
                pbar = tqdm.tqdm(total=steps)

            while (intermediates['t'] > 0).all():
                bb_x_tm1, tm1, denoiser_output = self.reverse_step(data, intermediates, delta_t)
                intermediates['t'] = tm1
                intermediates[self.bb_x_t_key] = bb_x_tm1

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        diffusion_outputs = denoiser_output
        diffusion_outputs[self.bb_x_0_key] = diffusion_outputs[self.bb_x_t_key]
        diffusion_outputs['bb_loss_weight'] = 1

        denoised_bb_vecs = diffusion_outputs[self.bb_x_t_key]
        denoised_bb = denoised_bb_vecs.clone()
        denoised_bb[:, (0, 2, 3)] += denoised_bb[:, 1].unsqueeze(-2)
        diffusion_outputs["denoised_bb"] = denoised_bb

        return diffusion_outputs
