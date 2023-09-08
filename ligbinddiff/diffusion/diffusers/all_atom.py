""" Diffusion modules """
import abc

import torch
from torch import nn
import tqdm
import numpy as np

from ligbinddiff.utils.so3_embedding import so3_add, so3_sub, so3_mult, so3_randn_like, so3_ones_like, gen_so3_unop

from ligbinddiff.model.all_atom.r3.denoiser import LatentDenoiser
from ligbinddiff.model.all_atom.r3.autoencoder import LatentEncoder, LatentDecoder
from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2


class LatentBaseDiffuser(nn.Module, abc.ABC):
    def __init__(self,
                 encoder,
                 decoder,
                 denoiser,
                 time_dist,
                 time_T,
                 time_discrete,
                 bb_scheduler,
                 sidechain_scheduler,
                 sidechain_x_0_key='latent_sidechain',
                 sidechain_x_0_pred_key='denoised_latent_sidechain',
                 sidechain_x_t_key='noised_latent_sidechain',
                 bb_x_0_key='x',
                 bb_x_0_pred_key='denoised_x',
                 bb_x_t_key='noised_x'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.denoiser = denoiser
        self.time_dist = time_dist
        self.time_T = time_T
        self.time_discrete = time_discrete
        self.sidechain_scheduler = sidechain_scheduler
        self.bb_scheduler = bb_scheduler

        self.sidechain_x_0_key = sidechain_x_0_key
        self.sidechain_x_0_pred_key = sidechain_x_0_pred_key
        self.sidechain_x_t_key = sidechain_x_t_key

        self.bb_x_0_key = bb_x_0_key
        self.bb_x_0_pred_key = bb_x_0_pred_key
        self.bb_x_t_key = bb_x_t_key

    def forward_noising(self, data, intermediates):
        out_dict = {}
        t = self.time_dist.sample([data.num_graphs])
        t = t.to(data['x'].device)
        bb_data_coeff, bb_noise_coeff = self.bb_scheduler.noising_coeffs(t)
        sidechain_data_coeff, sidechain_noise_coeff = self.sidechain_scheduler.noising_coeffs(t)
        noised_data = self.noise(data, intermediates, bb_data_coeff, bb_noise_coeff, sidechain_data_coeff, sidechain_noise_coeff)
        out_dict.update(noised_data)
        out_dict['t'] = t
        out_dict['bb_loss_weight'] = self.bb_scheduler.weight(t)
        out_dict['sidechain_loss_weight'] = self.sidechain_scheduler.weight(t)
        return out_dict

    def noise(self,
              data,
              intermediates,
              bb_data_coeff,
              bb_noise_coeff,
              sidechain_data_coeff,
              sidechain_noise_coeff):
        out_dict = {}
        sampled_residx = self.sample_noise_residx(data)
        out_dict['noised_residx'] = sampled_residx
        noised_bb_data = self.noise_bb(data, sampled_residx, bb_data_coeff, bb_noise_coeff)
        noised_sidechain_data = self.noise_sidechain(intermediates, sampled_residx, sidechain_data_coeff, sidechain_noise_coeff)
        out_dict.update(noised_bb_data)
        out_dict.update(noised_sidechain_data)
        return out_dict

    @abc.abstractmethod
    def sample_noise_residx(self, data):
        return NotImplemented

    @abc.abstractmethod
    def noise_bb(self, data, sampled_residx, data_coeff, noise_coeff):
        return NotImplemented

    @abc.abstractmethod
    def noise_sidechain(self, intermediates, sampled_residx, data_coeff, noise_coeff):
        return NotImplemented

    def reverse_noising(self, data, intermediates):
        return self.denoiser(data, intermediates)

    def score_fn(self, data, intermediates):
        denoiser_output = self.denoiser(data, intermediates)
        bb_score = self.bb_score_fn(denoiser_output)
        sidechain_score = self.sidechain_score_fn(denoiser_output)
        return bb_score, sidechain_score, denoiser_output

    @abc.abstractmethod
    def bb_score_fn(self, denoiser_output):
        return NotImplemented

    @abc.abstractmethod
    def sidechain_score_fn(self, denoiser_output):
        return NotImplemented

    def reverse_step(self, data, intermediates, delta_t):
        assert delta_t < 0
        t = intermediates['t']
        residx = intermediates['noised_residx']
        bb_x_t = intermediates[self.bb_x_t_key]
        sidechain_x_t = intermediates[self.sidechain_x_t_key]

        subset_t = t[residx]
        bb_beta_t = self.bb_scheduler.beta(subset_t)
        sidechain_beta_t = self.sidechain_scheduler.beta(subset_t)

        bb_score, sidechain_score, denoiser_output = self.score_fn(data, intermediates)

        bb_x_tm1 = self.bb_reverse_step(
            bb_x_t,
            bb_score,
            residx,
            bb_beta_t,
            delta_t)
        sidechain_x_tm1 = self.sidechain_reverse_step(
            sidechain_x_t,
            sidechain_score,
            residx,
            sidechain_beta_t,
            delta_t)
        tm1 = t + delta_t

        return bb_x_tm1, sidechain_x_tm1, tm1, denoiser_output

    @abc.abstractmethod
    def bb_reverse_step(self,
                        bb_x_t,
                        bb_score,
                        residx,
                        beta_t,
                        delta_t):
        return NotImplemented

    @abc.abstractmethod
    def sidechain_reverse_step(self,
                               sidechain_x_t,
                               sidechain_score,
                               residx,
                               beta_t,
                               delta_t):
        return NotImplemented

    @abc.abstractmethod
    def sample_prior(self, data):
        return NotImplemented

    def sample(self,
               data,
               steps=None,
               show_progress=False,
               device=None):
        if device is None:
            device = data['x'].device

        # num_batch = len(data._slice_dict['x'])
        # num_nodes = num_batch * self.k
        num_nodes = data['x'].shape[0]
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
                bb_x_tm1, sidechain_x_tm1, tm1, denoiser_output = self.reverse_step(data, intermediates, delta_t)
                intermediates['t'] = tm1
                intermediates[self.bb_x_t_key] = bb_x_tm1
                intermediates[self.sidechain_x_t_key] = sidechain_x_tm1

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        diffusion_outputs = intermediates
        diffusion_outputs[self.sidechain_x_0_key] = diffusion_outputs[self.sidechain_x_t_key]
        diffusion_outputs[self.bb_x_0_key] = diffusion_outputs[self.bb_x_t_key]
        decoded_outputs = self.decoder(data, diffusion_outputs)
        # we do this to recover the "ground truth" encoding
        latent_outputs = self.encoder(data)
        diffusion_outputs.update(latent_outputs)
        diffusion_outputs[self.sidechain_x_0_key] = diffusion_outputs['latent_mu']
        decoded_outputs['seq_logits'] = decoded_outputs['decoded_seq_logits']

        diffusion_outputs['sidechain_loss_weight'] = 1
        diffusion_outputs['bb_loss_weight'] = 1

        return diffusion_outputs, decoded_outputs


class NewLatentInpaintingDiffuser(LatentBaseDiffuser):
    def __init__(self,
                 bb_scheduler,
                 sidechain_scheduler,
                 time_dist,
                 time_T,
                 time_discrete,
                 node_lmax_list,
                 edge_channels_list,
                 h_time=64,
                 scalar_h_dim=128,
                 bb_lmax_list=[1],
                 bb_channels=6,
                 atom_lmax_list=[1],
                 atom_channels=91,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4,
                 k=30
                 ):
        # build these expensive coeff stores
        atom_super_lmax_list = [max(l1, l2) for l1, l2 in zip(atom_lmax_list, node_lmax_list)]
        bb_super_lmax_list = [max(l1, l2) for l1, l2 in zip(bb_lmax_list, node_lmax_list)]
        atom_super_SO3_rotation_list = nn.ModuleList()
        bb_super_SO3_rotation_list = nn.ModuleList()
        node_SO3_rotation_list = nn.ModuleList()
        for lmax in atom_super_lmax_list:
            atom_super_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in bb_super_lmax_list:
            bb_super_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in node_lmax_list:
            node_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )

        atom_super_SO3_grid_list = nn.ModuleList()
        bb_super_SO3_grid_list = nn.ModuleList()
        node_SO3_grid_list = nn.ModuleList()
        for l in range(max(atom_super_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(atom_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            atom_super_SO3_grid_list.append(SO3_m_grid)
        for l in range(max(bb_super_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(node_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            bb_super_SO3_grid_list.append(SO3_m_grid)
        bb_super_SO3_grid_list = nn.ModuleList()
        for l in range(max(node_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(atom_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            node_SO3_grid_list.append(SO3_m_grid)

        mappingReduced_super_atoms = CoefficientMappingModule(atom_super_lmax_list, atom_super_lmax_list)
        mappingReduced_super_bb = CoefficientMappingModule(bb_super_lmax_list, bb_super_lmax_list)
        mappingReduced_nodes = CoefficientMappingModule(node_lmax_list, node_lmax_list)

        denoiser = LatentDenoiser(
            node_lmax_list=node_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_nodes=mappingReduced_nodes,
            node_SO3_rotation=node_SO3_rotation_list,
            node_SO3_grid=node_SO3_grid_list,
            num_heads=num_heads,
            h_channels=h_channels,
            h_time=h_time,
            scalar_h_dim=scalar_h_dim,
            n_layers=num_layers,
        )
        encoder = LatentEncoder(
            node_lmax_list=node_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_nodes=mappingReduced_nodes,
            mappingReduced_super_bb=mappingReduced_super_bb,
            mappingReduced_super_atoms=mappingReduced_super_atoms,
            node_SO3_rotation=node_SO3_rotation_list,
            node_SO3_grid=node_SO3_grid_list,
            bb_super_SO3_rotation=bb_super_SO3_rotation_list,
            bb_super_SO3_grid=bb_super_SO3_grid_list,
            atom_super_SO3_rotation=atom_super_SO3_rotation_list,
            atom_super_SO3_grid=atom_super_SO3_grid_list,
            bb_lmax_list=bb_lmax_list,
            bb_channels=bb_channels,
            atom_lmax_list=atom_lmax_list,
            atom_channels=atom_channels,
            num_heads=num_heads,
            h_channels=h_channels,
            num_layers=num_layers
        )
        decoder = LatentDecoder(
            node_lmax_list=node_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_nodes=mappingReduced_nodes,
            mappingReduced_super_atoms=mappingReduced_super_atoms,
            node_SO3_rotation=node_SO3_rotation_list,
            node_SO3_grid=node_SO3_grid_list,
            bb_super_SO3_rotation=bb_super_SO3_rotation_list,
            atom_super_SO3_rotation=atom_super_SO3_rotation_list,
            atom_super_SO3_grid=atom_super_SO3_grid_list,
            atom_lmax_list=atom_lmax_list,
            atom_channels=atom_channels,
            num_heads=num_heads,
            h_channels=h_channels,
            num_layers=num_layers
        )
        super().__init__(
            encoder,
            decoder,
            denoiser,
            time_dist,
            time_T,
            time_discrete,
            bb_scheduler,
            sidechain_scheduler
            )
        self.k = k

    def sample_noise_residx(self, data):
        device = data['x'].device
        num_graphs = data.num_graphs
        # we noise by selecting a random residue, then noising its knn neighborhood
        edge_splits = data._slice_dict['edge_index']
        num_edges = (edge_splits[1:] - edge_splits[:-1]).tolist()
        edge_index_split = data.edge_index.split(num_edges, dim=-1)

        sampled_src_residx = []
        for edge_index in edge_index_split:
            # we sample an edge, then look at its source node
            # we can do this since we have the same number of edges per node
            # this also avoids the problem of sampling a masked node, since it won't have an edge to it
            srcs = edge_index[1]
            select_idx = torch.randint(low=0, high=srcs.shape[0], size=(1,), device=device)
            sampled_src_residx.append(srcs[select_idx])
        sampled_src_residx = torch.cat(sampled_src_residx, dim=0)

        # select the relevant edges
        edge_selector = data.edge_index[1][:, None] == sampled_src_residx[None, :]   # n_edge x n_graph
        edge_selector = edge_selector.any(dim=-1)
        # gather the dst of all src nodes
        sampled_dst_residx = data.edge_index[0][edge_selector]  # by construction these must be unique, n_edge
        # do a little reshaping so that the order is [(src, {dsts}), ...]
        sampled_dst_residx = sampled_dst_residx.view(num_graphs, self.k)
        sampled_residx = torch.cat([sampled_src_residx[:, None], sampled_dst_residx], dim=-1).long()
        sampled_residx = sampled_residx.flatten()
        return sampled_residx

    def noise_bb(self, data, sampled_residx, data_coeff, noise_coeff):
        bb_x_0 = data[self.bb_x_0_key]
        selected_bb_x_0 = bb_x_0[sampled_residx]
        bb_noise = torch.randn_like(selected_bb_x_0)

        # print(data.name)
        data_coeff = data_coeff.repeat_interleave(self.k + 1)  # knn + self
        noise_coeff = noise_coeff.repeat_interleave(self.k + 1)
        selected_bb_x_t = selected_bb_x_0 * data_coeff[:, None] + bb_noise + noise_coeff[:, None]

        bb_x_t = bb_x_0.clone()
        bb_x_t[sampled_residx] = selected_bb_x_t
        return {
            self.bb_x_t_key: bb_x_t
        }

    def noise_sidechain(self, intermediates, sampled_residx, data_coeff, noise_coeff):
        sidechain_x_0 = intermediates[self.sidechain_x_0_key]
        selected_sidechain_x_0 = sidechain_x_0.embedding[sampled_residx]
        sidechain_noise = torch.randn_like(selected_sidechain_x_0)

        # print(data.name)
        data_coeff = data_coeff.repeat_interleave(self.k + 1)  # knn + self
        noise_coeff = noise_coeff.repeat_interleave(self.k + 1)
        selected_sidechain_x_t = selected_sidechain_x_0 * data_coeff[:, None, None] + sidechain_noise * noise_coeff[:, None, None]

        sidechain_x_t = sidechain_x_0.clone()
        sidechain_x_t.embedding[sampled_residx] = selected_sidechain_x_t
        return {
            self.sidechain_x_t_key: sidechain_x_t
        }

    def reverse_noising(self, data, intermediates):
        return self.denoiser(data, intermediates)

    def bb_score_fn(self, denoiser_output):
        residx = denoiser_output['noised_residx']
        bb_x_t = denoiser_output[self.bb_x_t_key]
        t = denoiser_output['t']
        subset_t = t[residx]
        alphabar_t = self.bb_scheduler.alphabar(subset_t)

        bb_x_0_pred = denoiser_output[self.bb_x_0_pred_key]
        subset_bb_x_t = bb_x_t[residx]
        subset_bb_x_0_pred = bb_x_0_pred[residx]

        bb_score = torch.sqrt(alphabar_t)[:, None] * subset_bb_x_0_pred - subset_bb_x_t
        bb_score = 1 / (1 - alphabar_t[:, None]) * bb_score

        return bb_score

    def sidechain_score_fn(self, denoiser_output):
        residx = denoiser_output['noised_residx']
        sidechain_x_t = denoiser_output[self.sidechain_x_t_key]
        t = denoiser_output['t']
        subset_t = t[residx]

        sidechain_x_0_pred = denoiser_output[self.sidechain_x_0_pred_key]
        alphabar_t = self.sidechain_scheduler.alphabar(subset_t)

        subset_sidechain_x_t = sidechain_x_t.clone()
        subset_sidechain_x_t.set_embedding(sidechain_x_t.embedding[residx])

        subset_sidechain_x_0_pred = sidechain_x_0_pred.clone()
        subset_sidechain_x_0_pred.set_embedding(sidechain_x_0_pred.embedding[residx])

        sidechain_score = so3_sub(
            so3_mult(
                torch.sqrt(alphabar_t[:, None, None]),
                subset_sidechain_x_0_pred
            ),
            subset_sidechain_x_t
        )
        sidechain_score = so3_mult(1 / (1 - alphabar_t[:, None, None]), sidechain_score)
        return sidechain_score

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
        bb_drift = (f[:, None] * subset_bb_x_t - g[:, None]**2 * bb_score) * delta_t
        bb_diffusion = g[:, None] * np.sqrt(np.abs(delta_t)) * bb_noise
        bb_delta_x = bb_drift + bb_diffusion

        bb_x_tm1 = bb_x_t.clone()
        bb_x_tm1[residx] = bb_x_tm1[residx] + bb_delta_x
        return bb_x_tm1

    def sidechain_reverse_step(self,
                               sidechain_x_t,
                               sidechain_score,
                               residx,
                               beta_t,
                               delta_t):
        f = -0.5 * beta_t
        g = torch.sqrt(beta_t)

        subset_sidechain_x_t = sidechain_x_t.clone()
        subset_sidechain_x_t.set_embedding(sidechain_x_t.embedding[residx])

        sidechain_noise = so3_randn_like(subset_sidechain_x_t)
        sidechain_drift = so3_mult(
            so3_sub(so3_mult(f[:, None, None], subset_sidechain_x_t), so3_mult(g[:, None, None]**2, sidechain_score)),
            delta_t
        )
        sidechain_diffusion = so3_mult(g[:, None, None] * np.sqrt(np.abs(delta_t)), sidechain_noise)
        sidechain_delta_x = so3_add(sidechain_drift, sidechain_diffusion)

        sidechain_x_tm1 = sidechain_x_t.clone()
        sidechain_x_tm1.embedding[residx] = sidechain_x_tm1.embedding[residx] + sidechain_delta_x.embedding
        return sidechain_x_tm1

    def sample_prior(self, data):
        sampled_residx = self.sample_noise_residx(data)
        prior = {}
        one = torch.ones(data.num_graphs).unsqueeze(-1).to(data['x'].device)
        zero = torch.zeros(data.num_graphs).unsqueeze(-1).to(data['x'].device)
        noised_bb = self.noise_bb(data, sampled_residx, zero, one)
        # this is hacky
        # TODO: don't use the encoder
        intermediates = self.encoder(data)
        intermediates[self.sidechain_x_0_key] = intermediates['latent_mu']
        noised_sidechain = self.noise_sidechain(intermediates, sampled_residx, zero, one)

        prior.update(noised_bb)
        prior.update(noised_sidechain)
        prior['noised_residx'] = sampled_residx
        return prior

    def forward(self, data, warmup=False, deterministic=False):
        latent_data = self.encoder(data)

        if deterministic:
            latent_data['latent_sidechain'] = latent_data['latent_mu']
        else:
            latent_sigma = gen_so3_unop(torch.exp)(
                so3_mult(
                    latent_data['latent_logvar'],
                    0.5
                ),
            )
            latent_data['latent_sidechain'] = so3_add(
                latent_data['latent_mu'],
                so3_mult(
                    latent_sigma,
                    so3_randn_like(latent_sigma)
                )
            )
        decoded_outputs = self.decoder(data, latent_data)
        noised_latent = self.forward_noising(data, latent_data)
        latent_data.update(noised_latent)
        latent_outputs = self.reverse_noising(data, latent_data)
        latent_data.update(latent_outputs)

        return latent_data, decoded_outputs



class LatentInpaintingDiffuser(nn.Module):
    def __init__(self,
                 scheduler,
                 node_lmax_list,
                 edge_channels_list,
                 h_time=64,
                 scalar_h_dim=128,
                 bb_lmax_list=[1],
                 bb_channels=6,
                 atom_lmax_list=[1],
                 atom_channels=91,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4,
                 _add=so3_add,
                 _sub=so3_sub,
                 _mult=so3_mult,
                 _randn_like=so3_randn_like,
                 latent_x_0_key='latent',
                 latent_x_0_pred_key='denoised_latent',
                 latent_x_t_key='noised_latent',
                 bb_x_0_key='x',
                 bb_x_0_pred_key='denoised_x',
                 bb_x_t_key='noised_x',
                 k=30
                 ):
        super().__init__()
        # build these expensive coeff stores
        atom_super_lmax_list = [max(l1, l2) for l1, l2 in zip(atom_lmax_list, node_lmax_list)]
        bb_super_lmax_list = [max(l1, l2) for l1, l2 in zip(bb_lmax_list, node_lmax_list)]
        atom_super_SO3_rotation_list = nn.ModuleList()
        bb_super_SO3_rotation_list = nn.ModuleList()
        node_SO3_rotation_list = nn.ModuleList()
        for lmax in atom_lmax_list:
            atom_super_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in bb_super_lmax_list:
            bb_super_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in node_lmax_list:
            node_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )

        atom_super_SO3_grid_list = nn.ModuleList()
        bb_super_SO3_grid_list = nn.ModuleList()
        node_SO3_grid_list = nn.ModuleList()
        for l in range(max(atom_super_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(atom_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            atom_super_SO3_grid_list.append(SO3_m_grid)
        for l in range(max(bb_super_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(node_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            bb_super_SO3_grid_list.append(SO3_m_grid)
        bb_super_SO3_grid_list = nn.ModuleList()
        for l in range(max(node_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(atom_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            node_SO3_grid_list.append(SO3_m_grid)

        mappingReduced_super_atoms = CoefficientMappingModule(atom_super_lmax_list, atom_super_lmax_list)
        mappingReduced_super_bb = CoefficientMappingModule(bb_super_lmax_list, bb_super_lmax_list)
        mappingReduced_nodes = CoefficientMappingModule(node_lmax_list, node_lmax_list)

        self.denoiser = LatentDenoiser(
            node_lmax_list=node_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_nodes=mappingReduced_nodes,
            node_SO3_rotation=node_SO3_rotation_list,
            node_SO3_grid=node_SO3_grid_list,
            num_heads=num_heads,
            h_channels=h_channels,
            h_time=h_time,
            scalar_h_dim=scalar_h_dim,
            n_layers=num_layers,
        )
        self.encoder = LatentEncoder(
            node_lmax_list=node_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_nodes=mappingReduced_nodes,
            mappingReduced_super_bb=mappingReduced_super_bb,
            mappingReduced_super_atoms=mappingReduced_super_atoms,
            node_SO3_rotation=node_SO3_rotation_list,
            node_SO3_grid=node_SO3_grid_list,
            bb_super_SO3_rotation=bb_super_SO3_rotation_list,
            bb_super_SO3_grid=bb_super_SO3_grid_list,
            atom_super_SO3_rotation=atom_super_SO3_rotation_list,
            atom_super_SO3_grid=atom_super_SO3_grid_list,
            bb_lmax_list=bb_lmax_list,
            bb_channels=bb_channels,
            atom_lmax_list=atom_lmax_list,
            atom_channels=atom_channels,
            num_heads=num_heads,
            h_channels=h_channels,
            num_layers=num_layers
        )
        self.decoder = LatentDecoder(
            node_lmax_list=node_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_nodes=mappingReduced_nodes,
            mappingReduced_super_atoms=mappingReduced_super_atoms,
            node_SO3_rotation=node_SO3_rotation_list,
            node_SO3_grid=node_SO3_grid_list,
            bb_super_SO3_rotation=bb_super_SO3_rotation_list,
            atom_super_SO3_rotation=atom_super_SO3_rotation_list,
            atom_super_SO3_grid=atom_super_SO3_grid_list,
            atom_lmax_list=atom_lmax_list,
            atom_channels=atom_channels,
            num_heads=num_heads,
            h_channels=h_channels,
            num_layers=num_layers
        )
        self.scheduler = scheduler

        self._add = _add
        self._sub = _sub
        self._mult = _mult
        self._randn_like = _randn_like

        self.latent_x_0_key = latent_x_0_key
        self.latent_x_0_pred_key = latent_x_0_pred_key
        self.latent_x_t_key = latent_x_t_key

        self.bb_x_0_key = bb_x_0_key
        self.bb_x_0_pred_key = bb_x_0_pred_key
        self.bb_x_t_key = bb_x_t_key

        self.latent_lmax_list = node_lmax_list
        self.latent_n_channels = h_channels
        self.mappingReduced_super_atoms = mappingReduced_super_atoms
        self.mappingReduced_super_bb = mappingReduced_super_bb
        self.mappingReduced_nodes = mappingReduced_nodes
        self._apply_so3 = gen_so3_unop

        self.k = k

    def forward_noising(self, data):
        t = self.scheduler.sample_t([data.num_graphs])
        t = t.to(data['x'].device)
        data_coeff, noise_coeff = self.scheduler.noising_coeffs(t)
        noised_data = self.noise(data, data_coeff, noise_coeff)
        noised_data['t'] = t
        noised_data['loss_weight'] = self.scheduler.weight(t)
        return noised_data

    def noise(self, data, data_coeff, noise_coeff):
        sampled_residx = self.select_noise_residx(data)
        data['noised_residx'] = sampled_residx
        data = self.noise_bb(data, sampled_residx, data_coeff, noise_coeff)
        data = self.noise_sidechain(data, sampled_residx, data_coeff, noise_coeff)
        return data

    def select_noise_residx(self, data):
        device = data['x'].device
        num_graphs = data.num_graphs
        # we noise by selecting a random residue, then noising its knn neighborhood
        edge_splits = data._slice_dict['edge_index']
        num_edges = (edge_splits[1:] - edge_splits[:-1]).tolist()
        edge_index_split = data.edge_index.split(num_edges, dim=-1)

        sampled_src_residx = []
        for edge_index in edge_index_split:
            # we sample an edge, then look at its source node
            # we can do this since we have the same number of edges per node
            # this also avoids the problem of sampling a masked node, since it won't have an edge to it
            srcs = edge_index[1]
            select_idx = torch.randint(low=0, high=srcs.shape[0], size=(1,), device=device)
            sampled_src_residx.append(srcs[select_idx])
        sampled_src_residx = torch.cat(sampled_src_residx, dim=0)

        # select the relevant edges
        edge_selector = data.edge_index[1][:, None] == sampled_src_residx[None, :]   # n_edge x n_graph
        edge_selector = edge_selector.any(dim=-1)
        # gather the dst of all src nodes
        sampled_dst_residx = data.edge_index[0][edge_selector]  # by construction these must be unique, n_edge
        # do a little reshaping so that the order is [(src, {dsts}), ...]
        sampled_dst_residx = sampled_dst_residx.view(num_graphs, self.k)
        sampled_residx = torch.cat([sampled_src_residx[:, None], sampled_dst_residx], dim=-1).long()
        sampled_residx = sampled_residx.flatten()
        return sampled_residx

    def noise_bb(self, data, sampled_residx, data_coeff, noise_coeff):
        bb_x_0 = data[self.bb_x_0_key]
        selected_bb_x_0 = bb_x_0[sampled_residx]
        bb_noise = torch.randn_like(selected_bb_x_0)

        # print(data.name)
        data_coeff = data_coeff.repeat_interleave(self.k + 1)  # knn + self
        noise_coeff = noise_coeff.repeat_interleave(self.k + 1)
        selected_bb_x_t = selected_bb_x_0 * data_coeff[:, None] + bb_noise + noise_coeff[:, None]

        bb_x_t = bb_x_0.clone()
        bb_x_t[sampled_residx] = selected_bb_x_t
        data[self.bb_x_t_key] = bb_x_t
        return data

    def noise_sidechain(self, data, sampled_residx, data_coeff, noise_coeff):
        latent_x_0 = data[self.latent_x_0_key]
        selected_latent_x_0 = latent_x_0.embedding[sampled_residx]
        latent_noise = torch.randn_like(selected_latent_x_0)

        # print(data.name)
        data_coeff = data_coeff.repeat_interleave(self.k + 1)  # knn + self
        noise_coeff = noise_coeff.repeat_interleave(self.k + 1)
        selected_latent_x_t = selected_latent_x_0 * data_coeff[:, None, None] + latent_noise * noise_coeff[:, None, None]

        latent_x_t = latent_x_0.clone()
        latent_x_t.embedding[sampled_residx] = selected_latent_x_t
        data[self.latent_x_t_key] = latent_x_t
        return data

    def reverse_noising(self, data):
        return self.denoiser(data)

    def score_fn(self, data):
        residx = data['noised_residx']
        bb_x_t = data[self.bb_x_t_key]
        latent_x_t = data[self.latent_x_t_key]
        t = data['t']
        t = t[residx]

        denoiser_output = self.denoiser(data)
        bb_x_0_pred = denoiser_output[self.bb_x_0_pred_key]
        latent_x_0_pred = denoiser_output[self.latent_x_0_pred_key]
        alphabar_t = self.scheduler.alphabar(t)

        subset_bb_x_t = bb_x_t[residx]
        subset_latent_x_t = latent_x_t.clone()
        subset_latent_x_t.set_embedding(latent_x_t.embedding[residx])

        subset_bb_x_0_pred = bb_x_0_pred[residx]
        subset_latent_x_0_pred = latent_x_0_pred.clone()
        subset_latent_x_0_pred.set_embedding(latent_x_0_pred.embedding[residx])

        bb_score = torch.sqrt(alphabar_t).squeeze(-1) * subset_bb_x_0_pred - subset_bb_x_t
        bb_score = 1 / (1 - alphabar_t).squeeze(-1) * bb_score

        latent_score = self._sub(
            self._mult(
                torch.sqrt(alphabar_t),
                subset_latent_x_0_pred
            ),
            subset_latent_x_t
        )
        latent_score = self._mult(1 / (1 - alphabar_t), latent_score)

        return bb_score, latent_score, denoiser_output

    def reverse_step(self, data, delta_t):
        assert delta_t < 0
        t = data['t']
        residx = data['noised_residx']
        bb_x_t = data[self.bb_x_t_key]
        latent_x_t = data[self.latent_x_t_key]

        bb_score, latent_score, denoiser_output = self.score_fn(data)

        subset_t = t[residx]

        subset_bb_x_t = bb_x_t[residx]
        subset_latent_x_t = latent_x_t.clone()
        subset_latent_x_t.set_embedding(latent_x_t.embedding[residx])

        beta_t = self.scheduler.beta(subset_t)

        f = -0.5 * beta_t
        g = torch.sqrt(beta_t)

        bb_noise = torch.randn_like(subset_bb_x_t)
        bb_drift = (f.squeeze(-1) * subset_bb_x_t - g.squeeze(-1)**2 * bb_score) * delta_t
        bb_diffusion = g.squeeze(-1) * np.sqrt(np.abs(delta_t)) * bb_noise
        bb_delta_x = bb_drift + bb_diffusion

        bb_x_tm1 = bb_x_t.clone()
        bb_x_tm1[residx] = bb_x_tm1[residx] + bb_delta_x

        latent_noise = self._randn_like(subset_latent_x_t)
        latent_drift = self._mult(
            self._sub(self._mult(f, subset_latent_x_t), self._mult(g**2, latent_score)),
            delta_t
        )
        latent_diffusion = self._mult(g * np.sqrt(np.abs(delta_t)), latent_noise)
        latent_delta_x = self._add(latent_drift, latent_diffusion)

        latent_x_tm1 = latent_x_t.clone()
        latent_x_tm1.embedding[residx] = latent_x_tm1.embedding[residx] + latent_delta_x.embedding
        tm1 = t + delta_t
        return bb_x_tm1, latent_x_tm1, tm1, denoiser_output

    def sample_prior(self, data):
        # this is hacky
        latent_outputs = self.encoder(data)
        latent_outputs['latent'] = latent_outputs['latent_mu']
        one = torch.ones(data.num_graphs).unsqueeze(-1).to(data['x'].device)
        zero = torch.zeros(data.num_graphs).unsqueeze(-1).to(data['x'].device)
        data = self.noise(latent_outputs, zero, one)
        return data

    def sample(self,
               data,
               steps=None,
               show_progress=False,
               device=None):
        if device is None:
            device = data['x'].device

        # num_batch = len(data._slice_dict['x'])
        # num_nodes = num_batch * self.k
        num_nodes = data['x'].shape[0]
        data = self.sample_prior(data)

        if steps is not None:
            if self.scheduler.discrete:
                assert self.scheduler.T % steps == 0
        else:
            steps = self.scheduler.T

        data['t'] = torch.ones([num_nodes], device=device).view(
            -1, 1, 1).float() * (steps - 1)
        delta_t = - self.scheduler.T // steps
        denoiser_output = None

        with torch.no_grad():
            if show_progress:
                pbar = tqdm.tqdm(total=steps)

            while (data['t'] > 0).all():
                bb_x_tm1, latent_x_tm1, tm1, denoiser_output = self.reverse_step(data, delta_t)
                data['t'] = tm1
                data[self.bb_x_t_key] = bb_x_tm1
                data[self.latent_x_t_key] = latent_x_tm1

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        latent_outputs = data
        latent_outputs['latent'] = latent_outputs['noised_latent']
        latent_outputs['x_true'] = latent_outputs['x']  # save ground truth x_ca
        latent_outputs['x'] = latent_outputs['noised_x']
        decoded_outputs = self.decoder(latent_outputs)
        # we do this to recover the "ground truth" encoding
        decoded_outputs = self.encoder(decoded_outputs)
        decoded_outputs['latent'] = decoded_outputs['latent_mu']
        decoded_outputs['x'] = decoded_outputs['x_true']  # reset ground truth x_ca
        decoded_outputs['seq_logits'] = decoded_outputs['decoded_seq_logits']

        return decoded_outputs, denoiser_output


    def forward(self, data, warmup=False, deterministic=False):
        latent_data = self.encoder(data)

        if deterministic:
            latent_data['latent'] = latent_data['latent_mu']
        else:
            latent_sigma = self._apply_so3(torch.exp)(
                self._mult(
                    latent_data['latent_logvar'],
                    0.5
                ),
            )
            latent_data['latent'] = self._add(
                latent_data['latent_mu'],
                self._mult(
                    latent_sigma,
                    self._randn_like(latent_sigma)
                )
            )
        decoded_outputs = self.decoder(latent_data)
        noised_latent = self.forward_noising(latent_data)
        latent_outputs = self.reverse_noising(noised_latent)
        decoded_outputs.update(latent_outputs)

        return noised_latent, decoded_outputs
