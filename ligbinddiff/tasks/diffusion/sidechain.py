import logging
from typing import Sequence, Dict

import tqdm
import tree
import torch
import numpy as np

from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import scatter


from ligbinddiff.data.openfold import data_transforms
from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.tasks import Task
from ligbinddiff.model.utils.graph import batchwise_to_nodewise, get_data_lens
from ligbinddiff.utils.so3_embedding import gen_so3_unop
from ligbinddiff.diffusion.noisers.latent import SidechainDiffuser

from ligbinddiff.runtime.loss.common import autoencoder_losses, latent_scalar_sidechain_diffusion_loss


class DesignLatentSidechainNoising(Task):

    sidechain_x_0_key='latent_sidechain'
    sidechain_x_0_pred_key='pred_latent_sidechain'
    sidechain_x_t_key='noised_latent_sidechain'

    def __init__(self,
                 sidechain_noiser: SidechainDiffuser,
                 sample_t_min=0.01,
                 aux_loss_t_max=0.25,
                 self_conditioning=True):
        super().__init__()
        self.sidechain_noiser = sidechain_noiser
        self.sample_t_min = sample_t_min
        self.aux_loss_t_max = aux_loss_t_max
        self.self_conditioning = self_conditioning
        self._log = logging.getLogger(__name__)
        self.rng = np.random.default_rng()

    def _gen_diffuse_mask(self, data: HeteroData):
        if self.rng.random() > 0.25:
            return torch.ones_like(data['res_mask']).bool()
        else:
            percent = self.rng.random()
            return torch.rand(data['res_mask'].shape, device=data['res_mask'].device) > percent

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
        # for gt eval
        chain_feats = data_transforms.atom37_to_torsion_angles(prefix="")(chain_feats)  # TODO: uncurry this
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)

        rigids_0 = ru.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]

        # compute diffusion time
        if 't' in res_data.keys():
            t = res_data['t']
        else:
            t = torch.rand(num_graphs)

        # compute noising mask
        diffuse_mask = self._gen_diffuse_mask(res_data)

        # generate data dict
        copy_keys = [
            "torsion_angles_sin_cos",
            "alt_torsion_angles_sin_cos",
            "torsion_angles_mask",
            "atom14_atom_exists",
            "atom14_gt_exists",
            "atom14_gt_positions",
            "atom14_alt_gt_exists",
            "atom14_alt_gt_positions",
        ]
        diff_feats_t = {k: chain_feats[k] for k in copy_keys}

        diff_feats_t['x'] = rigids_0.get_trans()  # for HeteroData's sake
        diff_feats_t['bb'] = diff_feats_t['atom14_gt_positions'][..., :4, :]
        diff_feats_t['rigids_0'] = rigids_0.to_tensor_7()
        diff_feats_t['t'] = t
        diff_feats_t['noising_mask'] = diffuse_mask
        diff_feats_t['mlm_mask'] = ~diffuse_mask

        diff_feats_t['atom37'] = chain_feats['all_atom_positions']
        diff_feats_t['atom37_mask'] = chain_feats['all_atom_mask']

        diff_feats_t = tree.map_structure(
            lambda x: torch.as_tensor(x),
            diff_feats_t)

        data['residue'].update(diff_feats_t)
        return data

    def _run_model(self, model, inputs, self_conditioning=None):
        latent_data = model.encoder(inputs)

        if self.sidechain_noiser.mode == 'scalars':
            if model.training:
                latent_sigma = torch.exp(
                    latent_data['latent_logvar'] * 0.5
                )
                latent_data[self.sidechain_x_0_key] = latent_data['latent_mu'] + latent_sigma * torch.randn_like(latent_sigma)

            else:
                latent_data[self.sidechain_x_0_key] = latent_data['latent_mu']

        else:
            if model.training:
                latent_sigma = gen_so3_unop(torch.exp)(
                    latent_data['latent_logvar'] * 0.5
                )
                latent_data[self.sidechain_x_0_key] = (
                    latent_data['latent_mu'] + latent_sigma * gen_so3_unop(torch.randn_like)(latent_sigma)
                )
            else:
                latent_data[self.sidechain_x_0_key] = latent_data['latent_mu']

        t = inputs['residue']['t']
        nodewise_t = batchwise_to_nodewise(t, inputs['residue'].batch)

        decoder_outputs = model.decoder(inputs, latent_data)
        mask = inputs['residue']['res_mask'] & inputs['residue']['noising_mask']
        noised_latent = self.sidechain_noiser.forward_marginal(
            latent_data[self.sidechain_x_0_key],
            nodewise_t,
            torch.ones_like(mask).bool()
        )
        latent_outputs = model.denoiser(inputs, noised_latent, self_condition=self_conditioning)
        latent_outputs.update(noised_latent)
        latent_outputs.update(latent_data)

        return latent_outputs, decoder_outputs

    def sidechain_score_fn(self, data, denoiser_output):
        res_data = data['residue']
        res_mask = res_data['res_mask']
        noising_mask = res_data['noising_mask']
        mask = res_mask & noising_mask

        score = self.sidechain_noiser.score(
            denoiser_output[self.sidechain_x_t_key],
            denoiser_output[self.sidechain_x_0_pred_key],
            batchwise_to_nodewise(res_data['t'], res_data.batch)
        )
        return score * mask[..., None]

    def run_eval(self, model, inputs):
        if self.self_conditioning and np.random.uniform() > 0.5:
            with torch.no_grad():
                self_conditioning, _ = self._run_model(model, inputs)
        else:
            self_conditioning = None
        denoiser_output, design_output = self._run_model(model, inputs, self_conditioning)
        pred_bb_score = self.sidechain_score_fn(inputs, denoiser_output)
        denoiser_output["pred_sidechain_score"] = pred_bb_score
        denoiser_output.update(design_output)
        return denoiser_output

    def reverse_step(self, model, data, intermediates, delta_t, self_condition=None, noise_scale=1.0):
        if self.self_conditioning:
            denoiser_output = model.denoiser(data, intermediates, self_condition=self_condition)
        else:
            denoiser_output = model.denoiser(data, intermediates)

        score = self.sidechain_score_fn(
            data,
            denoiser_output
        )

        res_data = data['residue']
        sidechain_x_t = intermediates[self.sidechain_x_t_key]
        t = intermediates['t']
        nodewise_t = batchwise_to_nodewise(t, res_data.batch)
        noising_mask = res_data['noising_mask']
        res_mask = res_data['res_mask']
        mask = noising_mask & res_mask

        sidechain_x_tm1 = self.sidechain_noiser.reverse(
            sidechain_x_t,
            score,
            nodewise_t,
            delta_t,
            mask,
            noise_scale=noise_scale
        )
        tm1 = t - np.abs(delta_t)

        return sidechain_x_tm1, tm1, denoiser_output

    def run_predict(self,
                    model,
                    inputs,
                    steps=100,
                    show_progress=True,#False,
                    device=None,
                    noise_scale=1.0):
        res_data = inputs['residue']
        num_nodes = res_data.num_nodes
        if device is None:
            device = res_data['x'].device

        res_data['t'] = torch.ones(inputs.num_graphs, device=device)
        data = self.process_input(inputs)
        data['residue']['noising_mask'] = torch.ones_like(data['residue']['noising_mask'])
        data['residue']['mlm_mask'] = torch.zeros_like(data['residue']['noising_mask'])
        prior = model.sample_prior(num_nodes, device=device)
        intermediates = prior.copy()
        intermediates['t'] = torch.ones(inputs.num_graphs, device=device)

        delta_t = 1 / steps
        denoiser_output = None

        with torch.no_grad():
            if show_progress:
                pbar = tqdm.tqdm(total=steps)

            while (intermediates['t'] > delta_t).all():
                sidechain_x_tm1, tm1, denoiser_output = self.reverse_step(
                    model,
                    data,
                    intermediates,
                    delta_t,
                    self_condition=denoiser_output,
                    noise_scale=noise_scale
                )
                # TODO: remove this redundancy?
                intermediates['t'] = tm1
                res_data['t'] = tm1
                intermediates[self.sidechain_x_t_key] = sidechain_x_tm1

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        intermediates['t'] = torch.as_tensor(intermediates['t'])
        # TODO: this is useful, but figure out the underlying sampling issue above
        intermediates[self.sidechain_x_t_key] = denoiser_output[self.sidechain_x_0_pred_key]
        # diffusion_outputs = intermediates
        # # set this to see the starting point
        # diffusion_outputs[self.sidechain_x_t_key] = prior[self.sidechain_x_t_key]
        # # set this so we can feed this to the decoder
        # diffusion_outputs[self.sidechain_x_0_key] = diffusion_outputs[self.sidechain_x_t_key]

        decoder_outputs = model.decoder(
            data,
            {self.sidechain_x_0_key: intermediates[self.sidechain_x_t_key]}
        )

        # add encoder outputs as "ground truth" for latent diffusion
        encoder_outputs = model.encoder(data)
        decoder_outputs.update(encoder_outputs)
        decoder_outputs[self.sidechain_x_0_key] = decoder_outputs['latent_mu']
        decoder_outputs[self.sidechain_x_0_pred_key] = intermediates[self.sidechain_x_t_key]
        decoder_outputs[self.sidechain_x_t_key] = prior[self.sidechain_x_t_key]


        return decoder_outputs

    def compute_loss(self, inputs, outputs: Dict):
        # for design latent, kl 1e-5 and 0.01 on denoising loss seems good
        autoenc_loss_dict = autoencoder_losses(inputs, outputs)
        latent_loss_dict = latent_scalar_sidechain_diffusion_loss(inputs, outputs)

        vae_loss = (
            autoenc_loss_dict["atom14_mse"]
            + autoenc_loss_dict["seq_loss"]
            + autoenc_loss_dict["chi_loss"]
            + autoenc_loss_dict["kl_div"] *  1e-6  # 1e-2
        )

        loss = (
            vae_loss
            + latent_loss_dict["latent_denoising_loss"] * 0.01 # * 10
            # + latent_loss_dict["latent_denoising_nll"] #* 0.1
        )

        ret = {"loss": loss.mean()}
        for key, value in autoenc_loss_dict.items():
            ret[f"autoenc_{key}"] = value
        ret.update(latent_loss_dict)
        return ret
