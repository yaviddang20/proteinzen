import logging
from typing import Sequence, Dict

import tqdm
import tree
import torch
import numpy as np
import torch.nn.functional as F

from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import scatter


from proteinzen.data.openfold import data_transforms
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.tasks import Task
from proteinzen.model.utils.graph import batchwise_to_nodewise, get_data_lens
from proteinzen.runtime.loss.utils import _nodewise_to_graphwise
from proteinzen.stoch_interp.interpolate.dirichlet import DirichletConditionalFlow

from proteinzen.runtime.loss.common import seq_recov


class DirichletFlowMatching(Task):

    sidechain_x_1_key='seq_probs_1'
    sidechain_x_1_pred_key='pred_seq_probs_1'
    sidechain_x_t_key='seq_probs_t'

    def __init__(self,
                 sidechain_noiser: DirichletConditionalFlow,
                 sample_t_min=0.01,
                 aux_loss_t_max=0.25,
                 self_conditioning=True,
                 masking_on=False):
        super().__init__()
        self.sidechain_noiser = sidechain_noiser
        self.sample_t_min = sample_t_min
        self.aux_loss_t_max = aux_loss_t_max
        self.self_conditioning = self_conditioning
        self.masking_on = masking_on
        self._log = logging.getLogger(__name__)
        self.rng = np.random.default_rng()

    def _gen_diffuse_mask(self, data: HeteroData):
        if self.masking_on:
            if self.rng.random() > 0.25:
                return torch.ones_like(data['res_mask']).bool()
            else:
                percent = self.rng.random()
                return torch.rand(data['res_mask'].shape, device=data['res_mask'].device) > percent
        else:
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
        if 't' in res_data.keys():
            t = res_data['t']
        else:
            t = 1 - torch.rand(num_graphs) * (1 - self.sample_t_min)
            t = -torch.log(1-t)
            # t = t * 2
            data['t'] = t
            data['normalized_t'] = self.sidechain_noiser.get_normalized_t(t)


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
        data = self.sidechain_noiser.corrupt_batch(data)

        return data

    def run_eval(self, model, inputs):
        if not model.training:
            inputs['t'] = inputs['t'] * 0.
            inputs['normalized_t'] = inputs['normalized_t'] * 0.
            inputs = self.sidechain_noiser.corrupt_batch(inputs)

        if self.self_conditioning and np.random.uniform() > 0.5:
            with torch.no_grad():
                self_conditioning = model(inputs)
        else:
            self_conditioning = None
        output = model(inputs, self_condition=self_conditioning)
        return output

    def run_predict(self,
                    model,
                    inputs,
                    steps=100,
                    show_progress=True,#False,
                    device=None):
        res_data = inputs['residue']
        num_nodes = res_data.num_nodes
        if device is None:
            device = res_data['seq'].device


        data = self.process_input(inputs)
        data['t'] = torch.zeros(inputs.num_graphs, device=device)
        data['normalized_t'] = torch.zeros(inputs.num_graphs, device=device)
        data['residue']['noising_mask'] = torch.ones_like(data['residue']['noising_mask'])
        data['residue']['mlm_mask'] = torch.zeros_like(data['residue']['noising_mask'])
        prior = torch.ones((num_nodes, self.sidechain_noiser.K), device=device) / self.sidechain_noiser.K
        res_data['seq_probs_t'] = prior.float()
        batch = res_data.batch

        delta_t = self.sidechain_noiser.t_max / steps

        with torch.no_grad():
            if show_progress:
                pbar = tqdm.tqdm(total=steps)

            while (data['t'] < self.sidechain_noiser.t_max - delta_t).all():
                seq_probs_t = res_data['seq_probs_t']

                outputs = model(data)
                pred_seq_logits = outputs['pred_seq_logits']
                pred_seq_probs_1 = torch.softmax(pred_seq_logits, dim=-1)
                nodewise_t = res_data['t'][res_data.batch]
                seq_probs_tp1 = self.sidechain_noiser.euler_step(delta_t, nodewise_t, seq_probs_t, pred_seq_probs_1, batch)
                tp1 = data['t'] + delta_t
                data['t'] = tp1
                data['normalized_t'] = self.sidechain_noiser.get_normalized_t(tp1)
                res_data['seq_probs_t'] = seq_probs_tp1

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        outputs = model(data)

        return outputs

    def compute_loss(self, inputs, outputs: Dict):
        gt_seq = inputs['residue']['seq']
        seq_mask = inputs['residue']['seq_mask']
        res_mask = inputs['residue']['res_mask']
        mask = seq_mask & res_mask
        batch = inputs['residue']['batch']

        gt_seq = gt_seq * mask
        seq_logits = outputs['pred_seq_logits']

        cce = F.cross_entropy(seq_logits, gt_seq, reduction='none')
        seq_loss = _nodewise_to_graphwise(cce, batch, mask)
        per_seq_recov = seq_recov(gt_seq, seq_logits, batch, mask)

        ret = {
            "loss": seq_loss.mean(),
            "seq_loss": seq_loss,
            "per_seq_recov": per_seq_recov,
            "t": inputs['t']
        }
        return ret
