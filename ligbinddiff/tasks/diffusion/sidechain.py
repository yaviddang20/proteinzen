import logging
from typing import Sequence, Dict

import tqdm
import tree
import torch
import numpy as np

from torch_geometric.data import HeteroData, Batch


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
                 self_conditioning=False):
        super().__init__()
        self.sidechain_noiser = sidechain_noiser
        self.sample_t_min = sample_t_min
        self.aux_loss_t_max = aux_loss_t_max
        self.self_conditioning = self_conditioning
        self._log = logging.getLogger(__name__)

    def _gen_diffuse_mask(self, data: HeteroData):
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
        # for gt eval
        chain_feats = data_transforms.atom37_to_torsion_angles(prefix="")(chain_feats)  # TODO: uncurry this
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)

        rigids_0 = ru.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]

        # compute diffusion time
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
        noised_latent = self.sidechain_noiser.forward_marginal(
            latent_data[self.sidechain_x_0_key],
            nodewise_t,
            inputs['residue']['noising_mask']
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

    def reverse_step(self, intermediates, delta_t, noise_scale=1.0):
        # assert delta_t < 0
        t = intermediates['t']
        noising_mask = intermediates['noising_mask']
        x_mask = torch.zeros_like(noising_mask).bool()
        mask = x_mask | ~noising_mask
        bb_x_t = intermediates[self.sidechain_x_t_key]

        data = HeteroData(
            residue=dict(
                x=bb_x_t.get_trans(),
                x_mask=x_mask,
                noising_mask=noising_mask,
                fixed_mask=~noising_mask,
                rigids_t=bb_x_t.to_tensor_7(),
                t=t
            )
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
               model,
               inputs,
               steps=100,
               show_progress=False,
               device=None,
               noise_scale=1.0):
        num_nodes = inputs['residue'].num_nodes
        if device is None:
            device = inputs['residue']['x'].device
        prior = model.sample_prior(num_nodes, device=device)
        intermediates = prior

        delta_t = self.time_T / steps
        intermediates['t'] = torch.ones(1, device=device)
        bb_x_t = intermediates[self.sidechain_x_t_key]
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

    def run_predict(self, model, inputs):
        raise NotImplementedError

    def compute_loss(self, inputs, outputs: Dict):
        autoenc_loss_dict = autoencoder_losses(inputs, outputs)
        latent_loss_dict = latent_scalar_sidechain_diffusion_loss(inputs, outputs)

        vae_loss = (
            autoenc_loss_dict["atom14_mse"]
            + autoenc_loss_dict["seq_loss"]
            + autoenc_loss_dict["chi_loss"]
            + autoenc_loss_dict["kl_div"] * 1e-6
        )

        loss = vae_loss # + latent_loss_dict["latent_denoising_loss"] * 0.01

        ret = {"loss": loss.mean()}
        for key, value in autoenc_loss_dict.items():
            ret[f"autoenc_{key}"] = value
        ret.update(latent_loss_dict)
        return ret
