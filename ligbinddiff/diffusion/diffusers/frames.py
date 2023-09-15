""" Diffusion modules """
import torch
from torch import nn
import torch.distributions as dist
import numpy as np

from ligbinddiff.utils.so3_embedding import so3_add, so3_mult, so3_randn_like, gen_so3_unop
from ligbinddiff.utils.frames import rigid_from_3_points
from ligbinddiff.model.all_atom.frames.denoiser import LatentDenoiser
from ligbinddiff.model.all_atom.frames.autoencoder import LatentEncoder, LatentDecoder
from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Rotation, SO3_Grid
from ligbinddiff.model.modules.openfold import rigid_utils as ru


class LatentFrameInpaintingDiffuser(nn.Module):
    def __init__(self,
                 se3_noiser,
                 sidechain_noiser,
                 node_lmax_list=[1],
                 h_channels=32,
                 edge_channels_list=[32, 32, 32],
                 h_time=64,
                 scalar_h_dim=128,
                 bb_lmax_list=[1],
                 bb_channels=6,
                 atom_lmax_list=[1],
                 atom_channels=91,
                 num_heads=8,
                 num_layers=4,
                 k=30,
                 sidechain_x_0_key='latent_sidechain',
                 sidechain_x_0_pred_key='denoised_latent_sidechain',
                 sidechain_x_t_key='noised_latent_sidechain',
                 bb_x_0_key='frames',
                 bb_x_0_pred_key='denoised_frames',
                 bb_x_t_key='noised_frames'):
        super().__init__()
        self.sidechain_x_0_key = sidechain_x_0_key
        self.sidechain_x_0_pred_key = sidechain_x_0_pred_key
        self.sidechain_x_t_key = sidechain_x_t_key

        self.bb_x_0_key = bb_x_0_key
        self.bb_x_0_pred_key = bb_x_0_pred_key
        self.bb_x_t_key = bb_x_t_key

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
            mappingReduced_super_bb=mappingReduced_super_bb,
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
        self.time_dist = dist.Uniform(0, 1)
        self.time_T = 1
        self.se3_noiser = se3_noiser
        self.sidechain_noiser = sidechain_noiser
        self.k = k

    def build_bb_frames(self, data):
        backbone_coords = data['bb']
        N, CA, C = backbone_coords[..., 0, :], backbone_coords[..., 1, :], backbone_coords[..., 2, :]
        frames = ru.Rigid.from_3_points(N, CA, C)
        return frames

    def forward_noising(self, data, intermediates):
        device = data['x'].device
        out_dict = {}

        # sample a t per graph, then expand to per node
        t_per_graph = self.time_dist.sample([data.num_graphs])
        t_per_graph = t_per_graph.to(device)
        num_nodes = data.num_nodes
        t_per_node = torch.empty(num_nodes, device=device)
        for i, batch_num in enumerate(range(data.batch.max().item() + 1)):
            select = (data.batch == batch_num)
            t_per_node[select] = t_per_graph[i]
        t = t_per_node
        out_dict['t'] = t

        sampled_residx, noising_mask = self.sample_noise_residx(data)
        out_dict['noised_residx'] = sampled_residx
        out_dict['noising_mask'] = noising_mask
        noised_bb_data = self.noise_bb(data, noising_mask, t)
        noised_sidechain_data = self.noise_sidechain(intermediates, noising_mask, t)
        out_dict.update(noised_bb_data)
        out_dict.update(noised_sidechain_data)
        return out_dict

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

        num_nodes = data.num_nodes
        residxs = torch.arange(num_nodes, device=device)
        noising_mask = (residxs[:, None] == sampled_residx[None, :]).any(dim=-1)

        return sampled_residx, noising_mask

    def noise_bb(self, data, noising_mask, t):
        bb_x_0 = data[self.bb_x_0_key]
        noised_bb = self.se3_noiser.forward_marginal(bb_x_0, t, noising_mask)
        return noised_bb

    def noise_sidechain(self, intermediates, noising_mask, t):
        sidechain_x_0 = intermediates[self.sidechain_x_0_key]
        noised_sidechain = self.sidechain_noiser.forward_marginal(sidechain_x_0, t, noising_mask)
        return noised_sidechain

    def reverse_noising(self, data, intermediates):
        denoiser_output = self.denoiser(data, intermediates)
        pred_bb_score = self.bb_score_fn(denoiser_output)
        pred_sidechain_score = self.bb_score_fn(denoiser_output)
        denoiser_output["pred_bb_score"] = pred_bb_score
        denoiser_output["pred_sidechain_score"] = pred_sidechain_score
        return denoiser_output

    def bb_score_fn(self, denoiser_output):
        bb_x_t = denoiser_output[self.bb_x_t_key]
        bb_x_0_pred = denoiser_output[self.bb_x_0_pred_key]
        t = denoiser_output['t']
        bb_score = self.se3_noiser.score(bb_x_t, bb_x_0_pred, t)
        return bb_score

    def sidechain_score_fn(self, denoiser_output):
        sidechain_x_t = denoiser_output[self.sidechain_x_t_key]
        sidechain_x_0_pred = denoiser_output[self.sidechain_x_0_pred_key]
        t = denoiser_output['t']
        sidechain_score = self.sidechain_noiser.score(sidechain_x_t, sidechain_x_0_pred, t)
        return sidechain_score

    def forward(self, data, warmup=False, deterministic=False):
        if self.bb_x_0_key not in data:
            data[self.bb_x_0_key] = self.build_bb_frames(data)
        latent_data = self.encoder(data)

        if deterministic:
            latent_data[self.sidechain_x_0_key] = latent_data['latent_mu']
        else:
            latent_sigma = gen_so3_unop(torch.exp)(
                so3_mult(
                    latent_data['latent_logvar'],
                    0.5
                ),
            )
            latent_data[self.sidechain_x_0_key] = so3_add(
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

    def sample_prior(self, data, intermediates):
        sampled_residx, noising_mask = self.sample_noise_residx(data)
        num_nodes = data.num_nodes
        bb_x_0 = data[self.bb_x_0_key]
        bb_prior = self.se3_noiser.sample_ref(num_nodes, bb_x_0, noising_mask)
        sidechain_x_0 = intermediates[self.sidechain_x_0_key]
        sidechain_prior = self.sidechain_noiser.sample_ref(num_nodes, sidechain_x_0, noising_mask)
        out_dict = {}
        out_dict.update(bb_prior)
        out_dict.update(sidechain_prior)
        return out_dict

    def reverse_step(self, data, intermediates, delta_t, noise_scale=1.0):
        assert delta_t < 0
        t = intermediates['t']
        noising_mask = intermediates['noising_mask']
        bb_x_t = intermediates[self.bb_x_t_key]
        sidechain_x_t = intermediates[self.sidechain_x_t_key]

        denoiser_output = self.denoiser(data, intermediates)
        rot_score, trans_score = self.bb_score_fn(denoiser_output)
        sidechain_score = self.sidechain_score_fn(denoiser_output)

        bb_x_tm1 = self.se3_noiser.reverse(
            bb_x_t,
            rot_score,
            trans_score,
            t,
            delta_t,
            noising_mask,
            noise_scale=noise_scale)
        sidechain_x_tm1 = self.sidechain_noiser.reverse(
            sidechain_x_t,
            sidechain_score,
            t,
            delta_t,
            noising_mask,
            noise_scale=noise_scale)
        tm1 = t - delta_t #+ delta_t

        return bb_x_tm1, sidechain_x_tm1, tm1, denoiser_output


    def sample(self,
               data,
               steps=None,
               show_progress=False,
               device=None):
        if device is None:
            device = data['x'].device

        if self.bb_x_0_key not in data:
            data[self.bb_x_0_key] = self.build_bb_frames(data)

        # num_batch = len(data._slice_dict['x'])
        # num_nodes = num_batch * self.k
        num_nodes = data.num_nodes
        latent_outputs = self.encoder(data)
        prior = self.sample_prior(data, latent_outputs)
        intermediates = prior

        if steps is None:
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
        diffusion_outputs.update(latent_outputs)
        diffusion_outputs[self.sidechain_x_0_key] = diffusion_outputs['latent_mu']
        decoded_outputs['seq_logits'] = decoded_outputs['decoded_seq_logits']

        return diffusion_outputs, decoded_outputs
