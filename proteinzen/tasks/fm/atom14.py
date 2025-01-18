import copy
from typing import Sequence, Dict, Union
import os

import tqdm
import tree
import torch
import numpy as np
import torch.nn.functional as F

import torch_geometric.utils as pygu
from torch_geometric.data import HeteroData, Batch


from proteinzen.data.openfold import data_transforms, residue_constants
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.tasks import Task

from proteinzen.runtime.loss.frames import bb_frame_fm_loss, all_atom_fape_loss
from proteinzen.runtime.loss.common import (
    atom14_fm_losses
)
from proteinzen.runtime.loss.traj import atomic_traj_loss
from proteinzen.stoch_interp.interpolate.se3 import _centered_gaussian, _uniform_so3
from proteinzen.stoch_interp.interpolate.atom14 import Atom14Interpolant


import proteinzen.stoch_interp.interpolate.utils as du
from proteinzen.utils.framediff import all_atom


def outputs_to_inputs(data, outputs, gt_seq=False):
    inputs = copy.copy(data)
    res_data = inputs['residue']
    res_data['rigids_1'] = outputs['final_rigids'].to_tensor_7()
    res_data['x'] = outputs['final_rigids'].get_trans()
    res_data['bb'] = outputs['denoised_bb'][:, :4]
    if gt_seq:
        res_data['atom14_gt_positions'] = outputs['decoded_atom14_gt_seq']
        res_data['seq_one_hot'] = outputs['gt_seq_one_hot']
    else:
        res_data['atom14_gt_positions'] = outputs['decoded_atom14']
        res_data['atom14_gt_exists'] = outputs['decoded_atom14_mask']
        res_data['atom14_noising_mask'] = outputs['decoded_atom14_mask']
        res_data['seq'] = outputs['decoded_seq_logits'].argmax(dim=-1)
        res_data['seq_one_hot'] = outputs['decoded_seq_one_hot']
    return inputs


def no_grad_for_model(model, *args, **kwargs):
    for param in model.parameters():
        param.requires_grad = False
    output = model(*args, **kwargs)
    for param in model.parameters():
        param.requires_grad = True
    return output

def no_grad_for_input(model, data):
    return model(data.detach())



class Atom14Interpolation(Task):

    bb_x_1_key='rigids_1'
    bb_x_1_pred_key='final_rigids'
    bb_x_t_key='rigids_t'
    sidechain_x_1_key='atom10_local_1'
    sidechain_x_1_pred_key='pred_atom10_local_1'
    sidechain_x_t_key='atom10_local_t'

    def __init__(self,
                 protein_noiser: Atom14Interpolant,
                 aux_loss_t_min=0.25,
                 traj_decay_factor=0.99,
                 preconditioning=False
    ):
        super().__init__()
        self.atom_noiser = protein_noiser
        self.aux_loss_t_min = aux_loss_t_min
        self.traj_decay_factor = traj_decay_factor
        self.preconditioning = preconditioning

        self.rng = np.random.default_rng()

        self.polar_residues = torch.tensor([
            residue_constants.restype_order[i]
            for i in ['C', 'D', 'E', 'H', 'K', 'N', 'Q', 'R', 'S', 'T', 'Y']
        ])

    def _gen_diffuse_mask(self, data: HeteroData):
        return torch.ones_like(data['res_mask']).bool()

    def process_input(self, data: HeteroData):
        data = copy.deepcopy(data)
        res_data = data['residue']

        # compute base features
        chain_feats = {
            'aatype': torch.as_tensor(res_data['seq']).long(),
            'all_atom_positions': torch.as_tensor(res_data['atom37']).double(),
            'all_atom_mask': torch.as_tensor(res_data['atom37_mask']).double()
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles(prefix="")(chain_feats)  # TODO: uncurry this
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)

        rigids_1 = ru.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]

        # compute bb frame features
        diffuse_mask = self._gen_diffuse_mask(res_data)
        res_data['res_noising_mask'] = diffuse_mask
        res_data['x'] = rigids_1.get_trans().float()  # for HeteroData's sake
        res_data['rigids_1'] = rigids_1.to_tensor_7().float()

        # compute sidechain features
        ## generate data dict
        copy_keys = [
            "atom14_atom_exists",
            "atom14_gt_exists",
            "atom14_gt_positions",
            "atom14_alt_gt_exists",
            "atom14_alt_gt_positions",
        ]
        diff_feats_t = {k: chain_feats[k] for k in copy_keys}
        diff_feats_t['bb'] = diff_feats_t['atom14_gt_positions'][..., :4, :]
        diff_feats_t['atom37'] = chain_feats['all_atom_positions'].float()
        diff_feats_t['atom37_mask'] = chain_feats['all_atom_mask']

        atom14 = chain_feats['atom14_gt_positions'].float()
        atom14_mask = chain_feats['atom14_gt_exists']
        atom14 = atom14 * atom14_mask[..., None] + atom14[..., [1], :] * (1 - atom14_mask[..., None])
        atom14 = atom14 * res_data['res_mask'][..., None, None]
        atom14_mask = atom14_mask * res_data['res_mask'][..., None]
        diff_feats_t['atom14'] = atom14
        diff_feats_t['atom14_mask'] = atom14_mask

        diff_feats_t = tree.map_structure(
            lambda x: torch.as_tensor(x),
            diff_feats_t)
        diff_feats_t = tree.map_structure(
            lambda x: x.float() if x.dtype == torch.float64 else x,
            diff_feats_t)
        res_data.update(diff_feats_t)

        ##  noise data
        data = self.atom_noiser.corrupt_batch(data)

        seq = res_data['seq']
        polar_mask = (seq[..., None] == self.polar_residues.to(seq.device)[None]).any(dim=-1)
        res_data['polar_mask'] = polar_mask

        return data

    def process_sample_input(self, data: Dict, device='cpu'):
        return data

    def run_eval(self, model, inputs):
        device = inputs['residue']['x'].device

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
                    # device='cpu'):
        model = model.to(device)

        num_res = inputs['num_res']
        total_num_res = sum(num_res)
        data_list = []
        for n in num_res:
            data = HeteroData(
                residue={
                    "res_mask": torch.ones(n, device=device).bool(),
                    "res_noising_mask": torch.ones(n, device=device).bool(),
                    "seq": torch.zeros(n, device=device).long(),
                    "seq_mask": torch.ones(n, device=device).bool(),
                    "atom14_mask": torch.ones((n, 14), device=device).bool(),
                    "num_nodes": n
                }
            )
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        res_data = batch['residue']
        # Set-up initial prior samples

        # TODO: this is jenk but i will fix it later
        atom14_0 = (
            _centered_gaussian(res_data.batch.repeat(14), device) * self.atom_noiser.prior_std
        ).reshape(-1, 14, 3)

        seq_logits_0 = torch.randn(total_num_res, 20, device=device)

        # Set-up time
        ts = torch.linspace(self.atom_noiser.min_t, 1.0, self.atom_noiser.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(
            atom14_0,
            seq_logits_0
        )]
        clean_traj = []
        denoiser_out = None
        for t_2 in tqdm.tqdm(ts[1:]):
            # Run model.
            atom14_t_1, _ = prot_traj[-1]

            res_data['noised_atom14'] = atom14_t_1
            t = torch.ones(batch.num_graphs, device=device) * t_1
            batch["t"] = t
            if model.preconditioning:
                atom14_var_scaling_dict = self.atom_noiser.var_scaling_factors(t)
                batch['c_skip'] = atom14_var_scaling_dict['c_skip']
                batch['c_in'] = atom14_var_scaling_dict['c_in']
                batch['c_out'] = atom14_var_scaling_dict['c_out']
                batch['c_data'] = atom14_var_scaling_dict['c_data']
                batch['loss_weighting'] = atom14_var_scaling_dict['loss_weighting']

            with torch.no_grad():
                denoiser_out = model(batch, self_condition=denoiser_out)

                # Process model output.
                pred_seq_logits = denoiser_out["decoded_seq_logits"][..., :20]
                pred_atom14 = denoiser_out["denoised_atom14"].view(-1, 14, 3)

                clean_traj.append(
                    (
                        pred_atom14.detach().cpu(),
                        pred_seq_logits.detach().cpu()
                    )
                )

            # Take reverse step
            d_t = t_2 - t_1
            atom14_t_2 = self.atom_noiser._euler_step(d_t, t_1, pred_atom14, atom14_t_1)

            prot_traj.append(
                (
                    atom14_t_2,
                    pred_seq_logits.detach().cpu()
                )
            )
            t_1 = t_2

            if not model.self_conditioning:
                denoiser_out = None

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        # Run model.
        atom14_t_1, _ = prot_traj[-1]

        res_data['noised_atom14'] = atom14_t_1
        t = torch.ones(batch.num_graphs, device=device) * t_1
        batch["t"] = t
        if model.preconditioning:
            atom14_var_scaling_dict = self.atom_noiser.var_scaling_factors(t)
            batch['c_skip'] = atom14_var_scaling_dict['c_skip']
            batch['c_in'] = atom14_var_scaling_dict['c_in']
            batch['c_out'] = atom14_var_scaling_dict['c_out']
            batch['loss_weighting'] = atom14_var_scaling_dict['loss_weighting']
            batch['c_data'] = atom14_var_scaling_dict['c_data']

        with torch.no_grad():
            denoiser_out = model(batch, self_condition=denoiser_out)

            # Process model output.
            pred_seq_logits = denoiser_out["decoded_seq_logits"][..., :20]
            pred_atom14 = denoiser_out["denoised_atom14"].view(-1, 14, 3)

        argmax_seq = pred_seq_logits.argmax(dim=-1)
        decoded_struct = pred_atom14

        # decoded_struct = _collect_from_seq(all_atom14, argmax_seq, torch.ones_like(argmax_seq).bool())

        # # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrotpsi_to_atom37(prot_traj, res_data.res_mask)
        clean_trajs = list(zip(*[traj[-2].split(num_res) for traj in clean_traj]))
        clean_traj_seqs = list(zip(*[traj[-1].argmax(dim=-1).split(num_res) for traj in clean_traj]))
        prot_traj_seqs = list(zip(*[traj[-1].split(num_res) for traj in prot_traj]))
        prot_trajs = list(zip(*[traj[-2].split(num_res) for traj in prot_traj]))

        return {
            "samples": decoded_struct.split(num_res),
            "seqs": argmax_seq.split(num_res),
            "clean_trajs": clean_trajs,
            "clean_traj_seqs": clean_traj_seqs,
            "prot_trajs": prot_trajs,
            "prot_traj_seqs": prot_traj_seqs,
            "inputs": inputs
        }


    def compute_loss(self, inputs, outputs: Dict):
        traj_loss_dict = atomic_traj_loss(inputs, outputs, traj_decay_factor=self.traj_decay_factor)

        atomic_loss_dict = atom14_fm_losses(
            inputs, outputs,
            t_norm_clip=0.9,
            use_smooth_lddt=True,
            polar_upweight=True,
            use_sidechain_dists_mse_loss=False,
            use_local_atomic_dist_loss=False,
            use_sidechain_clash_loss=False,
            preconditioning=self.preconditioning
        )

        atomic_loss = (
            atomic_loss_dict["scaled_atom14_mse"]
            + atomic_loss_dict["seq_loss"] * 0.25
            + atomic_loss_dict["smooth_lddt"]
        )

        loss = (
            + atomic_loss
            + traj_loss_dict['traj_ca_loss']
            + traj_loss_dict['traj_pred_dist_loss'] * 0.5
            + traj_loss_dict['traj_seq_loss'] * 0.25
        )

        loss = loss.mean()

        loss_dict = {"loss": loss}
        loss_dict.update(atomic_loss_dict)
        loss_dict.update(traj_loss_dict)
        return loss_dict

