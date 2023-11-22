""" Diffusion modules """
import abc
import operator

import torch
from torch import nn
import tqdm
import numpy as np

from ligbinddiff.model.autoencoder.atom91 import Atom91Encoder, Atom91Decoder
# from ligbinddiff.model.autoencoder.torsion import Atom91Decoder
# from ligbinddiff.model.autoencoder.atomic import AtomicSidechainEncoder, MultiscaleSidechainEncoder
from ligbinddiff.model.autoencoder.hybrid import MultiscaleSidechainEncoder
from ligbinddiff.model.autoencoder.ipmp import IPMPEncoder, IPMPDecoder
from ligbinddiff.model.denoiser.sidechain.ipmp_latent import IPMPDenoiser
from ligbinddiff.model.autoencoder.two_track import Atom91SeqEncoder, Atom91SeqDecoder
from ligbinddiff.utils.so3_embedding import so3_add, so3_sub, so3_mult, so3_randn_like, so3_ones_like, gen_so3_unop
from ligbinddiff.model.modules.equiformer_v2.so3 import SO3_Embedding

from ligbinddiff.model.denoiser.sidechain.latent import LatentSidechainDenoiser
from ligbinddiff.model.denoiser.sidechain.two_track import LatentTwoTrackDenoiser

from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2


# TODO: brought this in to avoid importing dgl in the utils, do this in a less hacky way
def gen_compact_nmax_fiber(n_max):
    fiber_dict = {l: 0 for l in range(n_max+1)}
    n_levels = {l: [] for l in range(n_max+1)}
    for l in fiber_dict.keys():
        for n in range(n_max+1):
            if n < l: continue
            if (n-l) % 2 == 0:
                fiber_dict[l] = fiber_dict[l] + 1
                n_levels[l].append(n)
    return fiber_dict

def rand_compact_fiber_density(num_nodes, n_max, num_channels=1, device='cpu'):
    fiber = gen_compact_nmax_fiber(n_max)
    density = {}
    for l, num_vecs in fiber.items():
        m_tot = 2*l+1
        density[l] = torch.randn((num_nodes, num_vecs * num_channels, m_tot), device=device)
    return density


class Diffuser(nn.Module, abc.ABC):
    def __init__(self,
                 denoiser,
                 scheduler,
                 _add,
                 _sub,
                 _mult,
                 _randn_like,
                 x_0_key,
                 x_0_pred_key,
                 x_t_key,
        ):
        super().__init__()

        self.denoiser = denoiser
        self.scheduler = scheduler

        self._add = _add
        self._sub = _sub
        self._mult = _mult
        self._randn_like = _randn_like

        self.x_0_key = x_0_key
        self.x_0_pred_key = x_0_pred_key
        self.x_t_key = x_t_key

    def forward_noising(self, data, intermediates):
        t = self.scheduler.sample_t([data.num_graphs])
        t = t.to(data['residue']['x'].device)
        data_coeff, noise_coeff = self.scheduler.noising_coeffs(t)
        noised_data = self.noise(data, intermediates, data_coeff, noise_coeff)
        noised_data['t'] = t
        noised_data['loss_weight'] = self.scheduler.weight(t)
        return noised_data

    def noise(self, data, intermediates, data_coeff, noise_coeff):
        data_splits = data._slice_dict['residue']['x']
        data_lens = data_splits[1:] - data_splits[:-1]

        data_coeff = torch.cat([
            data_coeff[i].expand(l) for i, l in enumerate(data_lens)
        ]).view(-1, 1, 1)
        noise_coeff = torch.cat([
            noise_coeff[i].expand(l) for i, l in enumerate(data_lens)
        ]).view(-1, 1, 1)

        x_0 = intermediates[self.x_0_key]
        noise = self._randn_like(x_0)
        scaled_noise = self._mult(noise_coeff, noise)
        scaled_x_0 = self._mult(data_coeff, x_0)
        x_t = self._add(scaled_x_0, scaled_noise)
        intermediates[self.x_t_key] = x_t
        noising_mask = torch.ones_like(noise_coeff).view(-1).bool()
        intermediates["latent_sidechain_score_scaling"] = noising_mask.float()
        intermediates['noising_mask'] = noising_mask

        return intermediates

    def reverse_noising(self, data, intermediates):
        return self.denoiser(data, intermediates)

    def forward(self, data, warmup=None):
        noised_batch = self.forward_noising(data)
        outputs = self.reverse_noising(data, noised_batch)
        return noised_batch, outputs

    def score_fn(self, data, intermediates):
        x_t = intermediates[self.x_t_key]
        t = intermediates['t']
        denoiser_output = self.denoiser(data, intermediates)
        x_0_pred = denoiser_output[self.x_0_pred_key]
        alphabar_t = self.scheduler.alphabar(t)

        score = self._sub(
            self._mult(
                torch.sqrt(alphabar_t),
                x_0_pred
            ),
            x_t
        )
        score = self._mult(1 / (1 - alphabar_t), score)
        return score, denoiser_output

    def reverse_step(self, data, intermediates, delta_t):
        assert delta_t < 0
        t = intermediates['t']
        score, denoiser_output = self.score_fn(data, intermediates)
        x_t = intermediates[self.x_t_key]
        beta_t = self.scheduler.beta(t)

        f = -0.5 * beta_t
        g = torch.sqrt(beta_t)

        noise = self._randn_like(x_t)
        drift = self._mult(
            self._sub(self._mult(f, x_t), self._mult(g**2, score)),
            delta_t
        )
        diffusion = self._mult(g * np.sqrt(np.abs(delta_t)), noise)
        delta_x = self._add(drift, diffusion)

        x_tm1 = self._add(x_t, delta_x)
        tm1 = t + delta_t
        return x_tm1, tm1, denoiser_output

    @abc.abstractmethod
    def sample_prior(self, num_nodes, device):
        raise NotImplemented

    def sample(self,
               data,
               steps=None,
               show_progress=False,
               device=None):
        if device is None:
            device = data['residue']['x'].device
        num_nodes = data['residue'].num_nodes
        x_T = self.sample_prior(num_nodes, device)

        if steps is not None:
            if self.scheduler.discrete:
                assert self.scheduler.T % steps == 0
        else:
            steps = self.scheduler.T

        delta_t = - self.scheduler.T // steps
        intermediates = {}
        intermediates['t'] = torch.ones([num_nodes], device=device).view(
            -1, 1, 1).float() * (self.scheduler.T + delta_t)
        intermediates[self.x_t_key] = x_T

        with torch.no_grad():
            if show_progress:
                pbar = tqdm.tqdm(total=steps)

            while (intermediates['t'] > 0).all():
                x_tm1, tm1, denoiser_output = self.reverse_step(data, intermediates, delta_t)
                intermediates['t'] = tm1
                intermediates[self.x_t_key] = x_tm1

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()
        intermediates[self.x_0_pred_key] = denoiser_output[self.x_0_pred_key] #intermediates[self.x_t_key]
        intermediates[self.x_t_key] = x_T

        return intermediates, denoiser_output


## TODO: is there a way to not break the original structure?
class LatentDiffuser(Diffuser):
    def __init__(self,
                 scheduler,
                 node_lmax_list,
                 latent_lmax_list,
                 edge_channels_list,
                 h_time=64,
                 scalar_h_dim=128,
                 bb_lmax_list=[1],
                 bb_channels=7,
                 atom_lmax_list=[1],
                 atom_in_channels=18+1+5+1,
                 atom_h_channels=8,
                 atom_out_channels=91,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4,
                 ):
        # build these expensive coeff stores
        atom_super_lmax_list = [max(l1, l2) for l1, l2 in zip(atom_lmax_list, node_lmax_list)]
        bb_super_lmax_list = [max(l1, l2) for l1, l2 in zip(bb_lmax_list, node_lmax_list)]
        latent_super_lmax_list = [max(l1, l2) for l1, l2 in zip(latent_lmax_list, node_lmax_list)]

        atom_super_SO3_rotation_list = nn.ModuleList()
        bb_super_SO3_rotation_list = nn.ModuleList()
        latent_super_SO3_rotation_list = nn.ModuleList()
        atom_SO3_rotation_list = nn.ModuleList()
        node_SO3_rotation_list = nn.ModuleList()
        latent_SO3_rotation_list = nn.ModuleList()
        for lmax in atom_super_lmax_list:
            atom_super_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in bb_super_lmax_list:
            bb_super_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in latent_super_lmax_list:
            latent_super_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in atom_lmax_list:
            atom_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in node_lmax_list:
            node_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in latent_lmax_list:
            latent_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )

        atom_super_SO3_grid_list = nn.ModuleList()
        bb_super_SO3_grid_list = nn.ModuleList()
        latent_super_SO3_grid_list = nn.ModuleList()
        atom_SO3_grid_list = nn.ModuleList()
        node_SO3_grid_list = nn.ModuleList()
        latent_SO3_grid_list = nn.ModuleList()
        for l in range(max(atom_super_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            atom_super_SO3_grid_list.append(SO3_m_grid)
        for l in range(max(bb_super_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            bb_super_SO3_grid_list.append(SO3_m_grid)
        for l in range(max(latent_super_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            latent_super_SO3_grid_list.append(SO3_m_grid)
        for l in range(max(atom_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            atom_SO3_grid_list.append(SO3_m_grid)
        for l in range(max(node_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            node_SO3_grid_list.append(SO3_m_grid)
        for l in range(max(latent_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            latent_SO3_grid_list.append(SO3_m_grid)

        mappingReduced_super_atoms = CoefficientMappingModule(atom_super_lmax_list, atom_super_lmax_list)
        mappingReduced_super_latent = CoefficientMappingModule(latent_super_lmax_list, latent_super_lmax_list)
        mappingReduced_super_bb = CoefficientMappingModule(bb_super_lmax_list, bb_super_lmax_list)
        mappingReduced_atoms = CoefficientMappingModule(atom_lmax_list, atom_lmax_list)
        mappingReduced_nodes = CoefficientMappingModule(node_lmax_list, node_lmax_list)
        mappingReduced_latent = CoefficientMappingModule(latent_lmax_list, latent_lmax_list)

        denoiser = LatentSidechainDenoiser(
            node_lmax_list=node_lmax_list,
            latent_lmax_list=latent_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_nodes=mappingReduced_nodes,
            node_SO3_rotation=node_SO3_rotation_list,
            node_SO3_grid=node_SO3_grid_list,
            mappingReduced_super=mappingReduced_super_latent,
            super_SO3_rotation=latent_super_SO3_rotation_list,
            super_SO3_grid=latent_super_SO3_grid_list,
            num_heads=num_heads,
            h_channels=h_channels,
            h_time=h_time,
            scalar_h_dim=scalar_h_dim,
            n_layers=num_layers,
        )
        # init late so we can use common structure
        super().__init__(denoiser,
                         scheduler,
                         _add=so3_add,
                         _sub=so3_sub,
                         _mult=so3_mult,
                         _randn_like=so3_randn_like,
                         x_0_key='latent_sidechain',
                         x_0_pred_key='denoised_latent_sidechain',
                         x_t_key='noised_latent_sidechain',)
        # self.encoder = AtomicSidechainEncoder(
        #     node_lmax_list=node_lmax_list,
        #     edge_channels_list=edge_channels_list,
        #     mappingReduced_nodes=mappingReduced_nodes,
        #     mappingReduced_atoms=mappingReduced_atoms,
        #     mappingReduced_super_atoms=mappingReduced_super_atoms,
        #     node_SO3_rotation=node_SO3_rotation_list,
        #     node_SO3_grid=node_SO3_grid_list,
        #     atom_SO3_rotation=atom_super_SO3_rotation_list,
        #     atom_SO3_grid=atom_super_SO3_grid_list,
        #     atom_super_SO3_rotation=atom_super_SO3_rotation_list,
        #     atom_super_SO3_grid=atom_super_SO3_grid_list,
        #     atom_lmax_list=atom_lmax_list,
        #     atom_channels=atom_in_channels,
        #     num_heads=num_heads,
        #     h_channels=h_channels,
        #     num_layers=num_layers
        # )
        # self.encoder = MultiscaleSidechainEncoder(
        #     lmax_list=node_lmax_list,
        #     edge_channels_list=edge_channels_list,
        #     mappingReduced=mappingReduced_nodes,
        #     SO3_rotation=node_SO3_rotation_list,
        #     SO3_grid=node_SO3_grid_list,
        #     atom_channels=atom_in_channels,
        #     num_heads=num_heads,
        #     atom_h_channels=atom_h_channels,
        #     node_h_channels=h_channels,
        #     num_layers=num_layers-1
        # )
        self.encoder = Atom91Encoder(
            node_lmax_list=node_lmax_list,
            latent_lmax_list=latent_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_nodes=mappingReduced_nodes,
            mappingReduced_super_bb=mappingReduced_super_bb,
            mappingReduced_super_atoms=mappingReduced_super_atoms,
            mappingReduced_super_latent=mappingReduced_super_atoms,
            node_SO3_rotation=node_SO3_rotation_list,
            node_SO3_grid=node_SO3_grid_list,
            bb_super_SO3_rotation=bb_super_SO3_rotation_list,
            bb_super_SO3_grid=bb_super_SO3_grid_list,
            atom_super_SO3_rotation=atom_super_SO3_rotation_list,
            atom_super_SO3_grid=atom_super_SO3_grid_list,
            latent_super_SO3_rotation=latent_super_SO3_rotation_list,
            latent_super_SO3_grid=latent_super_SO3_grid_list,
            bb_lmax_list=bb_lmax_list,
            bb_channels=bb_channels,
            atom_lmax_list=atom_lmax_list,
            atom_channels=atom_out_channels,
            num_heads=num_heads,
            h_channels=h_channels,
            num_layers=num_layers
        )

        # self.decoder = Atom91Decoder(
        #     node_lmax_list=node_lmax_list,
        #     edge_channels_list=edge_channels_list,
        #     mappingReduced_nodes=mappingReduced_nodes,
        #     mappingReduced_super_bb=mappingReduced_super_bb,
        #     mappingReduced_super_atoms=mappingReduced_super_atoms,
        #     node_SO3_rotation=node_SO3_rotation_list,
        #     node_SO3_grid=node_SO3_grid_list,
        #     bb_super_SO3_rotation=bb_super_SO3_rotation_list,
        #     bb_super_SO3_grid=bb_super_SO3_grid_list,
        #     atom_super_SO3_rotation=atom_super_SO3_rotation_list,
        #     atom_super_SO3_grid=atom_super_SO3_grid_list,
        #     bb_lmax_list=bb_lmax_list,
        #     bb_channels=bb_channels,
        #     atom_lmax_list=atom_lmax_list,
        #     atom_channels=atom_out_channels,
        #     num_heads=num_heads,
        #     h_channels=h_channels,
        #     num_layers=num_layers
        # )
        self.decoder = Atom91Decoder(
            node_lmax_list=node_lmax_list,
            latent_lmax_list=latent_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_nodes=mappingReduced_nodes,
            mappingReduced_super_bb=mappingReduced_super_bb,
            mappingReduced_super_atoms=mappingReduced_super_atoms,
            mappingReduced_super_latent=mappingReduced_super_atoms,
            node_SO3_rotation=node_SO3_rotation_list,
            node_SO3_grid=node_SO3_grid_list,
            bb_super_SO3_rotation=bb_super_SO3_rotation_list,
            bb_super_SO3_grid=bb_super_SO3_grid_list,
            atom_super_SO3_rotation=atom_super_SO3_rotation_list,
            atom_super_SO3_grid=atom_super_SO3_grid_list,
            latent_super_SO3_rotation=latent_super_SO3_rotation_list,
            latent_super_SO3_grid=latent_super_SO3_grid_list,
            bb_lmax_list=bb_lmax_list,
            bb_channels=bb_channels,
            atom_lmax_list=atom_lmax_list,
            atom_channels=atom_out_channels,
            num_heads=num_heads,
            h_channels=h_channels,
            num_layers=num_layers
        )

        self.latent_lmax_list = latent_lmax_list
        self.latent_n_channels = h_channels
        self.mappingReduced_super_atoms = mappingReduced_super_atoms
        self.mappingReduced_super_bb = mappingReduced_super_bb
        self.mappingReduced_atoms = mappingReduced_super_atoms
        self.mappingReduced_nodes = mappingReduced_nodes
        self._apply_so3 = gen_so3_unop

    def sample_prior(self, num_nodes, device):
        latent = SO3_Embedding(
            num_nodes,
            lmax_list=self.latent_lmax_list,
            num_channels=self.latent_n_channels,
            device=device,
            dtype=torch.float
        )
        latent.embedding = torch.randn_like(latent.embedding)
        return latent

    def forward(self, data, warmup=False, deterministic=False):
        latent_data = self.encoder(data)
        if deterministic:
            latent_data[self.x_0_key] = latent_data['latent_mu']
        else:
            latent_sigma = self._apply_so3(torch.exp)(
                self._mult(
                    latent_data['latent_logvar'],
                    0.5
                ),
            )
            latent_data[self.x_0_key] = self._add(
                latent_data['latent_mu'],
                self._mult(
                    latent_sigma,
                    self._randn_like(latent_sigma)
                )
            )

        # decoded_outputs = self.decoder(data, latent_data)
        noised_latent = self.forward_noising(data, latent_data)
        latent_outputs = self.reverse_noising(data, noised_latent)
        latent_outputs.update(noised_latent)

        # x_0 = latent_data[self.x_0_key]
        # latent_data[self.x_0_key] = latent_outputs[self.x_0_pred_key]
        decoded_outputs = self.decoder(data, latent_data)
        # x_0 = latent_data[self.x_0_key]
        # latent_data[self.x_0_key] = latent_outputs[self.x_0_pred_key]
        # passthrough_outputs = self.decoder(data, latent_data)
        # latent_data[self.x_0_key] = x_0
        passthrough_outputs = None
        # decoded_outputs = passthrough_outputs


        return latent_outputs, decoded_outputs, passthrough_outputs  # dummy for passthrough outputs

    def sample(self,
               data,
               steps=None,
               show_progress=False,
               device=None,
               select_task=None):
        latent_outputs, denoiser_output = super().sample(data,
                                               steps,
                                               show_progress,
                                               device)
        latent_outputs[self.x_0_key] = denoiser_output[self.x_0_pred_key]
        decoded_outputs = self.decoder(data, latent_outputs)
        # we do this to recover the "ground truth" encoding
        encoder_outputs = self.encoder(data)
        latent_outputs[self.x_0_key] = encoder_outputs['latent_mu']

        latent_outputs.update(encoder_outputs)
        latent_outputs['loss_weight'] = 1
        latent_outputs['noising_mask'] = torch.ones_like(data['residue'].x_mask, device=device).bool()
        latent_outputs['latent_sidechain_score_scaling'] = torch.ones_like(data['residue'].x_mask, device=device).float()
        decoded_outputs['seq_logits'] = decoded_outputs['decoded_seq_logits']

        return latent_outputs, decoded_outputs


class LatentIPMPDiffuser(Diffuser):
    def __init__(self,
                 scheduler,
                 c_s=256,
                 c_z=128,
                 c_hidden=256,
                 num_layers=4,
                 ):
        denoiser = IPMPDenoiser(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            num_layers=num_layers
        )
        # init late so we can use common structure
        super().__init__(denoiser,
                         scheduler,
                         _add=operator.add,
                         _sub=operator.sub,
                         _mult=operator.mul,
                         _randn_like=torch.randn_like,
                         x_0_key='latent_sidechain',
                         x_0_pred_key='denoised_latent_sidechain',
                         x_t_key='noised_latent_sidechain',)
        self.encoder = IPMPEncoder(
            c_s,
            c_z,
            c_hidden=c_hidden,
            num_layers=num_layers
        )
        self.decoder = IPMPDecoder(
            c_s,
            c_z,
            c_hidden=c_hidden,
            num_layers=num_layers
        )
        self.c_s = c_s
        self._apply_so3 = lambda x: x

    def sample_prior(self, num_nodes, device):
        latent = torch.randn((num_nodes, self.c_s), device=device)
        return latent

    def forward(self, data, warmup=False, deterministic=False):
        latent_data = self.encoder(data)
        if deterministic:
            latent_data[self.x_0_key] = latent_data['latent_mu']
        else:
            latent_sigma = self._apply_so3(torch.exp)(
                self._mult(
                    latent_data['latent_logvar'],
                    0.5
                ),
            )
            latent_data[self.x_0_key] = self._add(
                latent_data['latent_mu'],
                self._mult(
                    latent_sigma,
                    self._randn_like(latent_sigma)
                )
            )

        # decoded_outputs = self.decoder(data, latent_data)
        noised_latent = self.forward_noising(data, latent_data)
        latent_outputs = self.reverse_noising(data, noised_latent)
        latent_outputs.update(noised_latent)

        # x_0 = latent_data[self.x_0_key]
        # latent_data[self.x_0_key] = latent_outputs[self.x_0_pred_key]
        decoded_outputs = self.decoder(data, latent_data)
        # x_0 = latent_data[self.x_0_key]
        # latent_data[self.x_0_key] = latent_outputs[self.x_0_pred_key]
        # passthrough_outputs = self.decoder(data, latent_data)
        # latent_data[self.x_0_key] = x_0
        passthrough_outputs = None
        # decoded_outputs = passthrough_outputs

        return latent_outputs, decoded_outputs, passthrough_outputs  # dummy for passthrough outputs


    def noise(self, data, intermediates, data_coeff, noise_coeff):
        data_splits = data._slice_dict['residue']['x']
        data_lens = data_splits[1:] - data_splits[:-1]

        data_coeff = torch.cat([
            data_coeff[i].expand(l) for i, l in enumerate(data_lens)
        ]).view(-1, 1)
        noise_coeff = torch.cat([
            noise_coeff[i].expand(l) for i, l in enumerate(data_lens)
        ]).view(-1, 1)

        x_0 = intermediates[self.x_0_key]
        noise = self._randn_like(x_0)
        scaled_noise = self._mult(noise_coeff, noise)
        scaled_x_0 = self._mult(data_coeff, x_0)
        x_t = self._add(scaled_x_0, scaled_noise)
        intermediates[self.x_t_key] = x_t
        noising_mask = torch.ones_like(noise_coeff).view(-1).bool()
        intermediates["latent_sidechain_score_scaling"] = noising_mask.float()
        intermediates['noising_mask'] = noising_mask

        return intermediates


    def sample(self,
               data,
               steps=None,
               show_progress=False,
               device=None,
               select_task=None):
        if device is None:
            device = data['residue']['x'].device
        num_nodes = data['residue'].num_nodes
        x_T = self.sample_prior(num_nodes, device)

        if steps is not None:
            if self.scheduler.discrete:
                assert self.scheduler.T % steps == 0
        else:
            steps = self.scheduler.T

        delta_t = - self.scheduler.T // steps
        intermediates = {}
        intermediates['t'] = torch.ones([num_nodes], device=device).view(
            -1, 1).float() * (self.scheduler.T + delta_t)
        intermediates[self.x_t_key] = x_T

        with torch.no_grad():
            if show_progress:
                pbar = tqdm.tqdm(total=steps)

            while (intermediates['t'] > 0).all():
                x_tm1, tm1, denoiser_output = self.reverse_step(data, intermediates, delta_t)
                intermediates['t'] = tm1
                intermediates[self.x_t_key] = x_tm1

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()
        intermediates[self.x_0_pred_key] = intermediates[self.x_t_key]
        intermediates[self.x_t_key] = x_T

        latent_outputs, denoiser_output = intermediates, denoiser_output

        latent_outputs[self.x_0_key] = denoiser_output[self.x_0_pred_key]
        decoded_outputs = self.decoder(data, latent_outputs)
        # we do this to recover the "ground truth" encoding
        encoder_outputs = self.encoder(data)
        latent_outputs[self.x_0_key] = encoder_outputs['latent_mu']
        latent_outputs.update(encoder_outputs)
        latent_outputs['loss_weight'] = 1
        latent_outputs['noising_mask'] = torch.ones_like(data['residue'].x_mask, device=device).bool()
        latent_outputs['latent_sidechain_score_scaling'] = torch.ones_like(data['residue'].x_mask, device=device).float()
        decoded_outputs['seq_logits'] = decoded_outputs['decoded_seq_logits']

        return latent_outputs, decoded_outputs

## TODO: is there a way to not break the original structure?
class LatentHybridDiffuser(Diffuser):
    def __init__(self,
                 scheduler,
                 node_lmax_list,
                 latent_lmax_list,
                 edge_channels_list,
                 h_time=64,
                 scalar_h_dim=128,
                 bb_lmax_list=[1],
                 bb_channels=7,
                 atom_lmax_list=[1],
                 atom_in_channels=18+1+5+1,
                 atom_h_channels=8,
                 atom_out_channels=91,
                 num_heads=8,
                 h_channels=32,
                 num_layers=4,
                 ):
        c_s = h_channels
        c_z = h_channels
        c_hidden = h_channels
        denoiser = IPMPDenoiser(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            num_layers=num_layers
        )
        # init late so we can use common structure
        super().__init__(denoiser,
                         scheduler,
                         _add=operator.add,
                         _sub=operator.sub,
                         _mult=operator.mul,
                         _randn_like=torch.randn_like,
                         x_0_key='latent_sidechain',
                         x_0_pred_key='denoised_latent_sidechain',
                         x_t_key='noised_latent_sidechain',)
        # build these expensive coeff stores
        atom_super_lmax_list = [max(l1, l2) for l1, l2 in zip(atom_lmax_list, node_lmax_list)]
        bb_super_lmax_list = [max(l1, l2) for l1, l2 in zip(bb_lmax_list, node_lmax_list)]
        latent_super_lmax_list = [max(l1, l2) for l1, l2 in zip(latent_lmax_list, node_lmax_list)]

        atom_super_SO3_rotation_list = nn.ModuleList()
        bb_super_SO3_rotation_list = nn.ModuleList()
        latent_super_SO3_rotation_list = nn.ModuleList()
        atom_SO3_rotation_list = nn.ModuleList()
        node_SO3_rotation_list = nn.ModuleList()
        latent_SO3_rotation_list = nn.ModuleList()
        for lmax in atom_super_lmax_list:
            atom_super_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in bb_super_lmax_list:
            bb_super_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in latent_super_lmax_list:
            latent_super_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in atom_lmax_list:
            atom_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in node_lmax_list:
            node_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )
        for lmax in latent_lmax_list:
            latent_SO3_rotation_list.append(
                SO3_Rotation(lmax)
            )

        atom_super_SO3_grid_list = nn.ModuleList()
        bb_super_SO3_grid_list = nn.ModuleList()
        latent_super_SO3_grid_list = nn.ModuleList()
        atom_SO3_grid_list = nn.ModuleList()
        node_SO3_grid_list = nn.ModuleList()
        latent_SO3_grid_list = nn.ModuleList()
        for l in range(max(atom_super_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            atom_super_SO3_grid_list.append(SO3_m_grid)
        for l in range(max(bb_super_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            bb_super_SO3_grid_list.append(SO3_m_grid)
        for l in range(max(latent_super_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            latent_super_SO3_grid_list.append(SO3_m_grid)
        for l in range(max(atom_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            atom_SO3_grid_list.append(SO3_m_grid)
        for l in range(max(node_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            node_SO3_grid_list.append(SO3_m_grid)
        for l in range(max(latent_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(l + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            latent_SO3_grid_list.append(SO3_m_grid)

        mappingReduced_super_atoms = CoefficientMappingModule(atom_super_lmax_list, atom_super_lmax_list)
        mappingReduced_super_latent = CoefficientMappingModule(latent_super_lmax_list, latent_super_lmax_list)
        mappingReduced_super_bb = CoefficientMappingModule(bb_super_lmax_list, bb_super_lmax_list)
        mappingReduced_atoms = CoefficientMappingModule(atom_lmax_list, atom_lmax_list)
        mappingReduced_nodes = CoefficientMappingModule(node_lmax_list, node_lmax_list)
        mappingReduced_latent = CoefficientMappingModule(latent_lmax_list, latent_lmax_list)

        self.encoder = MultiscaleSidechainEncoder(
            node_lmax_list=node_lmax_list,
            latent_lmax_list=latent_lmax_list,
            edge_channels_list=edge_channels_list,
            mappingReduced_node=mappingReduced_nodes,
            mappingReduced_super_latent=mappingReduced_super_atoms,
            node_SO3_rotation=node_SO3_rotation_list,
            node_SO3_grid=node_SO3_grid_list,
            latent_super_SO3_rotation=latent_super_SO3_rotation_list,
            latent_super_SO3_grid=latent_super_SO3_grid_list,
            atom_channels=atom_in_channels,
            num_heads=num_heads,
            atom_h_channels=atom_h_channels,
            node_h_channels=h_channels,
            num_layers=num_layers
        )
        self.decoder = IPMPDecoder(
            c_s=h_channels,
            c_z=edge_channels_list[0],
            c_hidden=h_channels,
            num_layers=num_layers
        )

        self.c_s = c_s
        self._apply_so3 = lambda x: x

    def sample_prior(self, num_nodes, device):
        latent = torch.randn((num_nodes, self.c_s), device=device)
        return latent

    def forward(self, data, warmup=False, deterministic=False):
        latent_data = self.encoder(data)
        if deterministic:
            latent_data[self.x_0_key] = latent_data['latent_mu']
        else:
            latent_sigma = self._apply_so3(torch.exp)(
                self._mult(
                    latent_data['latent_logvar'],
                    0.5
                ),
            )
            latent_data[self.x_0_key] = self._add(
                latent_data['latent_mu'],
                self._mult(
                    latent_sigma,
                    self._randn_like(latent_sigma)
                )
            )

        # decoded_outputs = self.decoder(data, latent_data)
        noised_latent = self.forward_noising(data, latent_data)
        latent_outputs = self.reverse_noising(data, noised_latent)
        latent_outputs.update(noised_latent)

        # x_0 = latent_data[self.x_0_key]
        # latent_data[self.x_0_key] = latent_outputs[self.x_0_pred_key]
        decoded_outputs = self.decoder(data, latent_data)
        # x_0 = latent_data[self.x_0_key]
        # latent_data[self.x_0_key] = latent_outputs[self.x_0_pred_key]
        # passthrough_outputs = self.decoder(data, latent_data)
        # latent_data[self.x_0_key] = x_0
        passthrough_outputs = None
        # decoded_outputs = passthrough_outputs

        return latent_outputs, decoded_outputs, passthrough_outputs  # dummy for passthrough outputs


    def noise(self, data, intermediates, data_coeff, noise_coeff):
        data_splits = data._slice_dict['residue']['x']
        data_lens = data_splits[1:] - data_splits[:-1]

        data_coeff = torch.cat([
            data_coeff[i].expand(l) for i, l in enumerate(data_lens)
        ]).view(-1, 1)
        noise_coeff = torch.cat([
            noise_coeff[i].expand(l) for i, l in enumerate(data_lens)
        ]).view(-1, 1)

        x_0 = intermediates[self.x_0_key]
        noise = self._randn_like(x_0)
        scaled_noise = self._mult(noise_coeff, noise)
        scaled_x_0 = self._mult(data_coeff, x_0)
        x_t = self._add(scaled_x_0, scaled_noise)
        intermediates[self.x_t_key] = x_t
        noising_mask = torch.ones_like(noise_coeff).view(-1).bool()
        intermediates["latent_sidechain_score_scaling"] = noising_mask.float()
        intermediates['noising_mask'] = noising_mask

        return intermediates


    def sample(self,
               data,
               steps=None,
               show_progress=False,
               device=None,
               select_task=None):
        if device is None:
            device = data['residue']['x'].device
        num_nodes = data['residue'].num_nodes
        x_T = self.sample_prior(num_nodes, device)

        if steps is not None:
            if self.scheduler.discrete:
                assert self.scheduler.T % steps == 0
        else:
            steps = self.scheduler.T

        delta_t = - self.scheduler.T // steps
        intermediates = {}
        intermediates['t'] = torch.ones([num_nodes], device=device).view(
            -1, 1).float() * (self.scheduler.T + delta_t)
        intermediates[self.x_t_key] = x_T

        with torch.no_grad():
            if show_progress:
                pbar = tqdm.tqdm(total=steps)

            while (intermediates['t'] > 0).all():
                x_tm1, tm1, denoiser_output = self.reverse_step(data, intermediates, delta_t)
                intermediates['t'] = tm1
                intermediates[self.x_t_key] = x_tm1

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()
        intermediates[self.x_0_pred_key] = intermediates[self.x_t_key]
        intermediates[self.x_t_key] = x_T

        latent_outputs, denoiser_output = intermediates, denoiser_output

        latent_outputs[self.x_0_key] = denoiser_output[self.x_0_pred_key]
        decoded_outputs = self.decoder(data, latent_outputs)
        # we do this to recover the "ground truth" encoding
        encoder_outputs = self.encoder(data)
        latent_outputs[self.x_0_key] = encoder_outputs['latent_mu']
        latent_outputs.update(encoder_outputs)
        latent_outputs['loss_weight'] = 1
        latent_outputs['noising_mask'] = torch.ones_like(data['residue'].x_mask, device=device).bool()
        latent_outputs['latent_sidechain_score_scaling'] = torch.ones_like(data['residue'].x_mask, device=device).float()
        decoded_outputs['seq_logits'] = decoded_outputs['decoded_seq_logits']

        return latent_outputs, decoded_outputs