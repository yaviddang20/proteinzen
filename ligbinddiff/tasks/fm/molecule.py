
import copy
import logging
from typing import Sequence, Dict

import tqdm
import tree
import torch
import numpy as np

from torch_geometric.data import HeteroData, Batch


from ligbinddiff.tasks import Task
from ligbinddiff.stoch_interp.interpolate.molecule import HarmonicPriorInterpolant, sample_harmonic_prior
from ligbinddiff.model.utils.graph import get_data_lens
from ligbinddiff.runtime.loss.molecular.harmonic import harmonic_fm_loss


class HarmonicFlowMatching(Task):

    atom_x_0_key='atom_pos'
    atom_x_0_pred_key='pred_atom_pos'
    atom_x_t_key='noised_atom_pos'

    def __init__(self,
                 harmonic_noiser: HarmonicPriorInterpolant,
                 num_timesteps=100):
        super().__init__()
        self.harmonic_noiser = harmonic_noiser
        self.num_timesteps = num_timesteps

    def gen_diffuse_mask(self, data: HeteroData):
        return torch.ones_like(data['res_mask']).bool()

    def process_input(self, data: HeteroData):
        data = copy.deepcopy(data)
        self.se3_noiser.set_device(data['ligand']['atom_pos'].device)
        # noise data
        noised_data = self.harmonic_noiser.corrupt_batch(data)
        data['ligand'].update(noised_data)

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
        self.harmonic_noiser.set_device(device)
        atom_data = inputs['ligand']
        # Set-up initial prior samples
        atoms_0 = sample_harmonic_prior(
            atom_data.num_nodes,
            inputs['ligand', 'bonds', 'ligand'].edge_index,
            ptr=atom_data.ptr
        )

        # Set-up time
        ts = torch.linspace(self.harmonic_noiser.min_t, 1.0, self.num_timesteps)
        t_1 = ts[0]

        mol_traj = [atoms_0]
        clean_traj = []
        denoiser_out = None
        for t_2 in tqdm.tqdm(ts[1:]):
            # Run model.
            atom_pos_t_1 = mol_traj[-1]
            atom_data[self.atom_x_t_key] = atom_pos_t_1
            t = torch.ones(atom_data.num_graphs, device=device) * t_1
            inputs["t"] = t
            with torch.no_grad():
                denoiser_out = model(inputs, self_condition=denoiser_out)

            # Process model output.
            pred_atom_pos = denoiser_out[self.atom_x_0_pred_key]
            clean_traj.append(
                pred_atom_pos.detach().cpu()
            )

            # Take reverse step
            d_t = t_2 - t_1
            atom_pos_t_2 = self.harmonic_noiser._euler_step(d_t, t_1, pred_atom_pos, atom_pos_t_1)
            mol_traj.append(atom_pos_t_2)
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        # Run model.
        atom_pos_t_1 = mol_traj[-1]
        atom_data[self.atom_x_t_key] = atom_pos_t_1
        t = torch.ones(atom_data.num_graphs, device=device) * t_1
        inputs["t"] = t
        with torch.no_grad():
            denoiser_out = model(inputs, self_condition=denoiser_out)

        # Process model output.
        pred_atom_pos = denoiser_out[self.atom_x_0_pred_key]
        clean_traj.append(
            pred_atom_pos.detach().cpu()
        )

        num_atoms_splits = get_data_lens(inputs, hetero_key='ligand')

        return {
            "samples": clean_traj[-1].split(num_atoms_splits),
            "inputs": inputs
        }

    def compute_loss(self, inputs, outputs: Dict):
        fm_loss_dict = harmonic_fm_loss(
            inputs, outputs)

        loss = (
            fm_loss_dict['atom_pos_mse'] / fm_loss_dict['norm_scale']
        ).mean()

        loss_dict = {"loss": loss}
        loss_dict.update(fm_loss_dict)
        return loss_dict