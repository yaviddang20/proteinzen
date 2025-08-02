import logging
import math
from functools import partial
import copy

import numpy as np
import tqdm
import torch
import tree
import lightning as L

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from proteinzen.boltz.data import const

from proteinzen.openfold.data import residue_constants
from proteinzen.openfold.utils import rigid_utils as ru

from .utils import gen_pbar_str
from .ema import EMAModel

from .loss.multiframe import multiframe_fm_loss_dense_batch
from .loss.common import seq_losses_dense_batch


DEFAULT_SEQ_WEIGHT = {
    c: 1.0
    for c in 'ACDEFGHIKLMNPQRSTVWY'
}
DEFAULT_SEQ_WEIGHT['X'] = 0.
for c in ['C', 'E', 'H', 'P', 'Q', 'R', 'W']:
    DEFAULT_SEQ_WEIGHT[c] = 2.0

DEFAULT_RESTYPE_WEIGHT = {
    c: 1.0
    for c in const.tokens
}
for c in ['CYS', 'GLU', 'HIS', 'PRO', 'GLN', 'ARG', 'TRP']:
    DEFAULT_RESTYPE_WEIGHT[c] = 2.0


def t_stratified_loss(batch_t, batch_loss, num_bins=4, loss_name=None):
    """Stratify loss by binning t."""
    batch_t = batch_t.float().numpy(force=True)
    batch_loss = batch_loss.float().numpy(force=True)
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins+1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = 'loss'
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin+1]
        t_range = f'{loss_name} t=[{bin_start:.2f},{bin_end:.2f})'
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


# from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return 0.0
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
def _get_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    else:
        return 1.0


def get_linear_warmup_schedule(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


RES_TO_AA = {}
for i, aa in enumerate(residue_constants.resnames):
    RES_TO_AA[const.token_ids[aa]] = i
AA_TO_RES = {j: i for i, j in RES_TO_AA.items()}

class BiomoleculeModule(L.LightningModule):
    def __init__(self,
                 model,
                 corrupter,
                 optim,
                 use_cosine_lr_sched=False,
                 cosine_warmup_steps=0,
                 cosine_total_steps=1e6,
                 use_linear_warmup=False,
                 linear_warmup_steps=0,
                 use_ema=True,
                 ema_decay=0.999,
                 use_posthoc_ema=False,
                 seq_weight=DEFAULT_RESTYPE_WEIGHT,
                 use_euclidean_for_rots=False,
                 learnable_noise_schedule=False,
                 direct_rot_vf_loss=False,
                 rot_angle_weight=0.5,
                 self_condition_rate=0.5,
                 atom_rigid_upweight=True,
                 compile_model=False,
                 apply_self_folding=False
    ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        if compile_model:
            self.model.compile()
        self.corrupter = corrupter
        self.optim = optim
        self.self_condition_rate = self_condition_rate
        self.use_cosine_lr_sched = use_cosine_lr_sched
        self.use_linear_warmup = use_linear_warmup
        self.linear_warmup_steps = linear_warmup_steps
        self.cosine_warmup_steps = cosine_warmup_steps
        self.cosine_total_steps = cosine_total_steps
        self.use_ema = use_ema
        self.use_posthoc_ema = use_posthoc_ema
        self.use_euclidean_for_rots = use_euclidean_for_rots
        self.learnable_noise_schedule = learnable_noise_schedule
        self.direct_rot_vf_loss = direct_rot_vf_loss
        self.rot_angle_weight = rot_angle_weight
        self.atom_rigid_upweight = atom_rigid_upweight
        self.apply_self_folding = apply_self_folding
        if learnable_noise_schedule:
            self.automatic_optimization = False

        seq_weight_tensor = torch.as_tensor([seq_weight[c] for c in const.tokens])
        self.seq_weight = seq_weight_tensor

        if use_ema:
            self.ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        else:
            self.ema = None

        if use_posthoc_ema:
            self.ema_long = EMAModel(model, gamma=6.94)
            self.ema_short = EMAModel(model, gamma=16.97)
        else:
            self.ema_long = None
            self.ema_short = None


        self.aatype_to_restype_tensor = torch.zeros(const.num_tokens)
        for aatype, restype in AA_TO_RES.items():
            self.aatype_to_restype_tensor[aatype] = restype

    def _generate_folding_batch(self, batch, pred_aatype):
        pred_seq_batch = copy.deepcopy(batch)

        # convert aatype to restype
        aa_to_res = self.aatype_to_restype_tensor.to(pred_aatype.device)
        pred_restype = aa_to_res[pred_aatype]
        restype = batch['token']['res_type'].clone()
        is_protein = batch['token']['mol_type'] == const.chain_type_ids["PROTEIN"]
        new_restype = restype * ~is_protein + pred_restype * is_protein

        # replace res_type (and seq which is here for legacy reasons)
        pred_seq_batch['token']['res_type'] = new_restype.long()
        pred_seq_batch['token']['seq'] = new_restype.long()
        # also don't noise sequence internally
        seq_noising_mask = pred_seq_batch['token']['seq_noising_mask'].clone()
        seq_noising_mask[is_protein] = False
        pred_seq_batch['token']['seq_noising_mask'] = seq_noising_mask

        # we'll also mask out any copy residues
        # since the idea here is to evaluate the denoiser output designability
        # without influence from conditioning
        token_mask = pred_seq_batch['token']['token_mask'].clone()
        token_is_copy_mask = pred_seq_batch['token']['token_is_copy_mask']
        token_mask[token_is_copy_mask] = False
        pred_seq_batch['token']['token_mask'] = token_mask
        # we also need to mask the corresponding rigids
        rigids_mask = pred_seq_batch['rigids']['rigids_mask'].clone()
        rigids_to_token = pred_seq_batch['rigids']['rigids_to_token']
        new_rigids_mask = torch.gather(
            token_mask,
            -1,
            rigids_to_token
        )
        pred_seq_batch['rigids']['rigids_mask'] = rigids_mask & new_rigids_mask

        return pred_seq_batch

    def training_step(self, batch, batch_idx):
        # update EMA
        if self.ema is not None and self.global_step > 0:
            self.ema.update_parameters(self.model)
        if self.ema_long is not None and self.global_step > 0:
            self.ema_long.update_parameters(self.model, self.global_step-1)
        if self.ema_short is not None and self.global_step > 0:
            self.ema_short.update_parameters(self.model, self.global_step-1)

        # corrupt data
        corrupter = self.corrupter
        batch['trans_t'] = batch['t']
        batch['rot_t'] = batch['t']
        batch = corrupter.corrupt_dense_batch(batch)

        # run denoiser, with self-conditioning as necessary
        model = self.model
        if model.self_conditioning and np.random.uniform() < self.self_condition_rate:
            with torch.no_grad():
                self_conditioning = model(batch)

                # run denoising with the predicted seq of the self conditioning denoising step
                if self.apply_self_folding:
                    pred_seq_batch = self._generate_folding_batch(batch, self_conditioning['pred_seq'])
                    # run "folding"
                    self_folding = model(pred_seq_batch)
                else:
                    self_folding = None

        else:
            self_conditioning = None
            self_folding = None
        outputs = model(batch, self_conditioning, self_folding)

        # compute loss
        loss_dict = self._loss_step(batch, outputs)

        # log loss
        log_dict = tree.map_structure(
            lambda x: torch.round(torch.mean(x), decimals=3) if torch.is_tensor(x) else x,
            loss_dict
        )

        loss_by_task = {}
        for i, task in enumerate(batch['task']):
            if task.name + "_loss" not in loss_by_task:
                loss_by_task[task.name + "_loss"] = []
            if task.name + "_seq_loss" not in loss_by_task:
                loss_by_task[task.name + "_seq_loss"] = []
            if task.name + "_frame_vf_loss" not in loss_by_task:
                loss_by_task[task.name + "_frame_vf_loss"] = []
            if task.name + "_frame_vf_loss_unscaled" not in loss_by_task:
                loss_by_task[task.name + "_frame_vf_loss_unscaled"] = []

            loss_by_task[task.name + "_loss"].append(loss_dict['loss_per_batch'][i])
            loss_by_task[task.name + "_seq_loss"].append(loss_dict["seq_loss"][i])
            loss_by_task[task.name + "_frame_vf_loss"].append(loss_dict['frame_vf_loss'][i])
            loss_by_task[task.name + "_frame_vf_loss_unscaled"].append(loss_dict['frame_vf_loss_unscaled'][i])

        loss_by_task = {
            key: torch.stack(values)
            for key, values in loss_by_task.items()
        }
        for key, value in loss_by_task.items():
            self.log(
                "task/" + key,
                value.mean(),
                prog_bar=False,
                logger=True,
                on_step=None,
                on_epoch=True,
                batch_size=value.shape[0],
                sync_dist=False)

        log_dict = {
            ("train/" + key): value
            for key, value in
            sorted(log_dict.items(), key = lambda x: x[0])
        }
        t = batch['t']
        for loss_name, loss_list in loss_dict.items():
            if loss_name in ['loss', 'frameflow_loss', "frame_vf_loss_unscaled", 'loss_per_batch']:
                continue
            if t.numel() != loss_list.numel():
                continue
            # if not loss_name.startswith("pt_") and not loss_name.startswith("latent_"):
            #     continue
            stratified_losses = t_stratified_loss(
                t, loss_list, loss_name=loss_name)
            stratified_losses = {
                f"train/{k}": torch.round(torch.as_tensor(v, device=log_dict['train/loss'].device), decimals=3)
                for k,v in stratified_losses.items()
            }
            self.log_dict(
                stratified_losses,
                prog_bar=False,
                logger=True,
                on_step=None,
                on_epoch=True,
                batch_size=t.shape[0],
                sync_dist=False)

        self.log_dict(
            log_dict,
            on_step=None,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=t.shape[0],
            sync_dist=True)

        return loss_dict

    def _loss_step(self, inputs, outputs):
        token_seq = inputs['token']['seq']
        seq_weight = self.seq_weight.to(token_seq.device)
        rigids_seq_idx = inputs['rigids']['rigids_seq_idx']
        rigids_seq = torch.gather(
            token_seq,
            -1,
            rigids_seq_idx,
        )
        rigidwise_weight = seq_weight[rigids_seq]

        frame_fm_loss_dict = multiframe_fm_loss_dense_batch(
            inputs, outputs, sep_rot_loss=True, use_euclidean_for_rots=self.use_euclidean_for_rots,
            t_norm_clip=0.9,
            rot_vf_angle_loss_weight=self.rot_angle_weight,
            fafe_l2_block_mask_size=1,
            rigidwise_weight=rigidwise_weight,
            direct_rot_vf_loss=self.direct_rot_vf_loss,
            upweight_atomic=self.atom_rigid_upweight
        )

        frame_vf_loss = (
            frame_fm_loss_dict["trans_vf_loss"] +
            frame_fm_loss_dict["rot_vf_loss"]
        )
        unscaled_frame_vf_loss = (
            frame_fm_loss_dict["unscaled_trans_vf_loss"] +
            frame_fm_loss_dict["unscaled_rot_vf_loss"]
        )

        atomic_seq_weight = seq_weight[token_seq]
        atomic_loss_dict = seq_losses_dense_batch(
            inputs,
            outputs,
            seqwise_weight=atomic_seq_weight
        )

        atomic_loss = (
            0.25 * atomic_loss_dict["seq_loss"]
        )

        if self.direct_rot_vf_loss:
            loss = (
                frame_vf_loss
                + 0.25 * atomic_loss_dict["seq_loss"]
                # + 0.5 * frame_fm_loss_dict['scaled_fafe']
            )
        else:
            loss = (
                frame_vf_loss
                + 0.5 * frame_fm_loss_dict['scaled_fafe']
                + atomic_loss
            )

        loss_dict = {"loss": loss.mean(), "frame_vf_loss": frame_vf_loss, "frame_vf_loss_unscaled": unscaled_frame_vf_loss}
        loss_dict['loss_per_batch'] = loss

        # TODO: for some reason this does not play well with nccl between different tasks
        # if 'motif_idx' in outputs:
        #     pred_motif_idx = outputs['motif_idx']
        #     gt_motif_idx = inputs['token']['token_seq_idx']
        #     is_motif_mask = ~inputs['token']['token_is_protein_output_mask'] & ~inputs['token']['token_is_ligand_mask']
        #     motif_idx_correct = (pred_motif_idx == gt_motif_idx) * is_motif_mask
        #     if is_motif_mask.sum() > 0:
        #         loss_dict['motif_idx_correct'] = motif_idx_correct.sum() / is_motif_mask.sum()

        # loss_dict[inputs['task'].name + "_loss"] = loss
        # loss_dict[inputs['task'].name + "_seq_loss"] = atomic_loss_dict["seq_loss"]
        # loss_dict[inputs['task'].name + "_frame_vf_loss"] = frame_vf_loss
        # loss_dict[inputs['task'].name + "_frame_vf_loss_unscaled"] = unscaled_frame_vf_loss


        loss_dict.update(frame_fm_loss_dict)
        loss_dict.update(atomic_loss_dict)
        # loss_dict.update(loss_by_task)

        return loss_dict


    def on_before_optimizer_step(self, optimizer):
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(name)

        with torch.no_grad():
            norms = []
            norm_dict = {}
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    n = torch.linalg.vector_norm(p.grad.view(-1), dim=-1)
                    norms.append(n)
                    norm_dict[name] = n.item()
            # import json
            # print(json.dumps(norm_dict, indent=4))
            total_norm = torch.linalg.vector_norm(
                torch.stack(norms, dim=0),
                dim=0
            )

            print("grad norm", total_norm)

        self.log(
            "grad_norm",
            total_norm,
            prog_bar=False,
            logger=True,
            on_step=None,
            on_epoch=True,
            batch_size=1,
            sync_dist=False
        )

    def predict_step(self, batch, batch_idx):
        if self.use_ema:
            model = self.ema.module
        else:
            model = self.model
        outputs = self._predict_step(model, batch)
        return outputs

    def _predict_step(
        self,
        model,
        batch
    ):
        corrupter = self.corrupter
        # Set-up time
        ts = torch.linspace(0.0, 1.0, corrupter.num_timesteps)

        rigids_data = batch['rigids']
        rigids_data['rigids_t'] = rigids_data['rigids_1']
        token_data = batch['token']
        rigids_0 = ru.Rigid.from_tensor_7(rigids_data['rigids_t'])
        trans_0 = rigids_0.get_trans()
        rotmats_0 = rigids_0.get_rots().get_rot_mats()
        rigids_noising_mask = rigids_data['rigids_noising_mask']
        seq_noising_mask = token_data['seq_noising_mask']

        t_1 = ts[0]

        num_batch, num_res = seq_noising_mask.shape
        device = self.device
        denoiser_out = None

        prot_traj = [(
            trans_0,
            rotmats_0,
            None
        )]

        clean_traj = []

        global_shift = 0
        for t_2 in tqdm.tqdm(ts[1:]):
            d_t = t_2 - t_1
            # Run model.
            trans_t_1, rotmats_t_1, _ = prot_traj[-1]
            # trans_t_1_center = (trans_t_1 * rigids_noising_mask[..., None]).sum(dim=-2) / rigids_noising_mask[..., None].float().sum(dim=-2)
            # # print(trans_t_1_center.shape, rigids_noising_mask.shape, trans_t_1.shape)
            # trans_t_1 = trans_t_1 - trans_t_1_center[..., None, :] * rigids_noising_mask[..., None]
            # # trans_t_1, center = frame_noiser._center_trans(trans_t_1, res_data.batch, rigids_noising_mask)
            # # global_shift += center

            t_hat, d_t_hat, trans_t_hat = corrupter._trans_churn(
                d_t,
                t_1,
                trans_t_1,
                noising_mask=rigids_noising_mask,
            )
            _, _, rotmats_t_hat = corrupter._rot_churn(
                d_t,
                t_1,
                rotmats_t_1,
                noising_mask=rigids_noising_mask,
            )

            rigids_data["trans_t"] = trans_t_hat
            rigids_data["rotmats_t"] = rotmats_t_hat
            rigids_data['rigids_t'] = ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_t_hat),
                trans=trans_t_hat
            ).to_tensor_7()
            t = torch.ones(num_batch, device=device)[..., None] * t_hat
            batch["t"] = t

            if self.apply_self_folding and denoiser_out is not None:
                pred_seq_batch = self._generate_folding_batch(batch, denoiser_out['pred_seq'])
                folding_out = model(pred_seq_batch, self_folding=denoiser_out)
            else:
                folding_out = None

            denoiser_out = model(batch, self_condition=denoiser_out, self_folding=folding_out)

            # Process model output.
            pred_rigids = denoiser_out['denoised_rigids']
            pred_trans_1 = pred_rigids.get_trans()
            pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()

            clean_traj.append(
                (pred_trans_1,
                 pred_rotmats_1,
                 denoiser_out["pred_seq"].detach().cpu(),
                )
            )

            trans_d_t_hat = d_t_hat
            rot_d_t_hat = d_t_hat
            trans_time = t_hat
            rot_time = t_hat
            trans_vf_scale = 1
            rot_vf_scale = 1

            trans_t_2 = corrupter._trans_euler_step(
                trans_d_t_hat,
                trans_time,
                pred_trans_1,
                trans_t_hat,
                noising_mask=rigids_noising_mask,
                vf_scale=trans_vf_scale,
            )
            rotmats_t_2 = corrupter._rots_euler_step(
                rot_d_t_hat,
                rot_time,
                pred_rotmats_1,
                rotmats_t_hat,
                noising_mask=rigids_noising_mask,
                vf_scale=rot_vf_scale,
                rot_vf=denoiser_out['pred_rot_vf']
            )

            prot_traj.append(
                (trans_t_2,
                 rotmats_t_2,
                 denoiser_out["pred_seq"].detach().cpu(),
                )
            )
            t_1 = t_2


            if not model.self_conditioning:
                denoiser_out = None

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1, _= prot_traj[-1]
        rigids_data["trans_t"] = trans_t_1
        rigids_data["rotmats_t"] = rotmats_t_1
        rigids_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        t = torch.ones(num_batch, device=device)[..., None] * t_1
        batch["t"] = t

        denoiser_out = model(batch, self_condition=denoiser_out)

        # Process model output.
        pred_rigids = denoiser_out['denoised_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()

        ret = []

        # data_list = batch.to_data_list()

        prot_traj = prot_traj[1:]

        for i, input_data in enumerate(batch['input_data']):
            # print(input_data)
            num_rigids = input_data['rigids']['tensor7'].shape[0]
            output_data = copy.deepcopy(input_data)
            tensor7 = pred_rigids.to_tensor_7().numpy(force=True)
            tensor7 = tensor7[i, :num_rigids]
            output_data['rigids']['tensor7'] = tensor7

            num_tokens = input_data['tokens']['token_idx'].shape[0]
            pred_seq = denoiser_out["pred_seq"].numpy(force=True)
            pred_seq = pred_seq[i, :num_tokens]
            output_data['tokens']['res_type'] = pred_seq

            # if we copy any tokens, figure out what generated residue corresponds to these fixed tokens
            # select masks
            token_data = output_data['tokens']
            token_is_copy_mask = token_data['is_copy']
            motif_idx = denoiser_out["motif_idx"][i, :num_tokens]
            motif_select_mask = (token_is_copy_mask & token_data['resolved_mask'])
            motif_seq_fixed = ~token_data['seq_noising_mask']
            # actual idxs
            fixed_bb_res_idx = motif_idx[motif_select_mask]
            fixed_seq_res_idx = motif_idx[motif_seq_fixed]
            fixed_bb_chain_idx = token_data['asym_id'][motif_select_mask]
            fixed_seq_chain_idx = token_data['asym_id'][motif_seq_fixed]

            prot_traj_i = [(_trans[i], _rot[i], _seq[i]) for _trans, _rot, _seq in prot_traj]
            ret_prot_traj = []
            for _trans, _rot, _seq in prot_traj_i:
                traj_data = copy.deepcopy(input_data)
                _quat = ru.rot_to_quat(_rot)
                _tensor7 = torch.cat([_quat, _trans], dim=-1)
                _tensor7 = _tensor7[:num_rigids].numpy(force=True)
                traj_data['rigids']['tensor7'] = _tensor7

                num_tokens = input_data['tokens']['token_idx'].shape[0]
                _seq = _seq[:num_tokens].numpy(force=True)
                traj_data['tokens']['res_type'] = _seq
                ret_prot_traj.append(traj_data)

            clean_traj_i = [(_trans[i], _rot[i], _seq[i]) for _trans, _rot, _seq in clean_traj]
            ret_clean_traj = []
            for _trans, _rot, _seq in clean_traj_i:
                traj_data = copy.deepcopy(input_data)
                _quat = ru.rot_to_quat(_rot)
                _tensor7 = torch.cat([_quat, _trans], dim=-1)
                _tensor7 = _tensor7[:num_rigids].numpy(force=True)
                traj_data['rigids']['tensor7'] = _tensor7

                num_tokens = input_data['tokens']['token_idx'].shape[0]
                _seq = _seq[:num_tokens].numpy(force=True)
                traj_data['tokens']['res_type'] = _seq
                ret_clean_traj.append(traj_data)

            ret.append({
                "input_data": input_data,
                "output_data": output_data,
                "prot_traj": ret_prot_traj,
                "clean_traj": ret_clean_traj,
                "fixed_bb_res_idx": fixed_bb_res_idx,
                "fixed_seq_res_idx": fixed_seq_res_idx,
                "fixed_bb_chain_idx": fixed_bb_chain_idx,
                "fixed_seq_chain_idx": fixed_seq_chain_idx,
                "name": batch["task"][i]
            })

        return ret


    def configure_optimizers(self):
        optimizer = self.optim(self.model.parameters())

        if self.use_cosine_lr_sched:
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cosine_warmup_steps,
                num_training_steps=int(self.cosine_total_steps),
                num_cycles=1
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        elif self.use_linear_warmup:
            scheduler = get_linear_warmup_schedule(
                optimizer,
                num_warmup_steps=self.linear_warmup_steps,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        else:
            return optimizer


