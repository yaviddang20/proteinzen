import logging
import math
from functools import partial
import copy
from typing import Any
import warnings
from dataclasses import replace
import os
import json

from xtb.interface import Calculator, Param

import numpy as np
import tqdm
import torch
import torch.distributed as dist
import tree
import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter
import sys

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from proteinzen.boltz.data import const

from proteinzen.openfold.data import residue_constants
from proteinzen.openfold.utils import rigid_utils as ru

from proteinzen.boltz.data.types import Structure
from proteinzen.data.featurize.sampling import construct_atoms, update_structure
from proteinzen.data.featurize.tokenize import Tokenized
# from proteinzen.data.write.mmcif import to_mmcif
from proteinzen.data.write.pdb import to_pdb

from .utils import gen_pbar_str
from .ema import EMAModel

from .loss.multiframe import multiframe_fm_loss_dense_batch
from .loss.common import seq_losses_dense_batch
from proteinzen.boltz.data.types import SamplingResidue


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


def _get_decay_lr_lambda(
    current_step: int, *, start_step: int, decay_step: int, decay_factor: float
):
    if current_step < start_step:
        return 1.0
    else:
        exponent = (current_step - start_step) // decay_step
        return decay_factor ** (exponent + 1)


def get_mult_decay_schedule(
    optimizer: torch.optim.Optimizer, start_step: int, decay_step: int, decay_factor: float
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
        _get_decay_lr_lambda,
        start_step=start_step,
        decay_step=decay_step,
        decay_factor=decay_factor
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



RES_TO_AA = {}
for i, aa in enumerate(residue_constants.resnames):
    RES_TO_AA[const.token_ids[aa]] = i
AA_TO_RES = {j: i for i, j in RES_TO_AA.items()}

ANGSTROM_TO_BOHR = 1.8897259886
HARTREE_TO_KCALMOL = 627.509474

# Module-level thread pool for parallel xTB single-points (GIL released in C ext).
# Sized to batch size; can be overridden before training via set_xtb_executor().
_xtb_executor = None


def set_xtb_executor(max_workers: int):
    global _xtb_executor
    _xtb_executor = __import__('concurrent.futures', fromlist=['ThreadPoolExecutor']).ThreadPoolExecutor(max_workers=max_workers)


def _xtb_single(nums, pos):
    try:
        calc = Calculator(Param.GFN2xTB, nums, pos)
        calc.set_verbosity(0)
        res = calc.singlepoint()
        return res.get_energy() * HARTREE_TO_KCALMOL
    except Exception:
        return float('nan')


def compute_xtb_energies(elements, positions, mask):
    """
    Compute GFN2-xTB single-point energies for a batch of molecules.

    Args:
        elements: [B, R] int tensor of atomic numbers
        positions: [B, R, 3] float tensor of atom positions in Angstroms
        mask: [B, R] bool tensor of valid atoms

    Returns:
        energies: [B] float tensor of energies in kcal/mol,
                  nan for samples where xTB fails
    """
    elements_np = elements.cpu().numpy()
    positions_np = positions.cpu().float().numpy()
    mask_np = mask.cpu().numpy()

    def _make_args(b):
        m = mask_np[b]
        nums = elements_np[b][m].astype(np.int32)
        pos = positions_np[b][m].astype(np.float64) * ANGSTROM_TO_BOHR
        return nums, pos

    B = elements_np.shape[0]
    args = [_make_args(b) for b in range(B)]

    executor = _xtb_executor
    if executor is not None:
        futures = [executor.submit(_xtb_single, n, p) for n, p in args]
        energies = [f.result() for f in futures]
    else:
        energies = [_xtb_single(n, p) for n, p in args]

    return torch.tensor(energies, dtype=torch.float32, device=elements.device)


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
                 use_lr_step_decay=False,
                 lr_step_decay_start=0,
                 lr_step_decay_step=1,
                 lr_step_decay_factor=0.95,
                 use_cosine_annealing=False,
                 cosine_annealing_T_max=100,
                 cosine_annealing_epoch_offset=None,
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
                 apply_self_folding=False,
                 strict_weight_loading=True,
                 bond_rotation_head_only=False,
                 scale_bond_length_loss=False,
                 scale_bond_angle_loss=False,
                 scale_ring_planarity_loss=False,
                 use_fafe_loss=True,
                 use_rot_vf_loss=True,
                 identity_rot_noise=False,
                 use_trans_mse_loss=False,
                 scale_trans_mse_loss=False,
                 use_min_conformer_head=False,
                 accumulate_grad_batches=1,
    ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        if compile_model:
            self.model.compile()
        self.corrupter = corrupter
        self.optim = optim
        self.self_condition_rate = self_condition_rate
        self.bond_rotation_head_only = bond_rotation_head_only
        self.scale_bond_length_loss = scale_bond_length_loss
        self.scale_bond_angle_loss = scale_bond_angle_loss
        self.scale_ring_planarity_loss = scale_ring_planarity_loss

        self.use_cosine_lr_sched = use_cosine_lr_sched
        self.use_linear_warmup = use_linear_warmup
        self.use_lr_step_decay = use_lr_step_decay
        self.linear_warmup_steps = linear_warmup_steps
        self.cosine_warmup_steps = cosine_warmup_steps
        self.cosine_total_steps = cosine_total_steps
        self.lr_step_decay_start = lr_step_decay_start
        self.lr_step_decay_step = lr_step_decay_step
        self.lr_step_decay_factor = lr_step_decay_factor
        self.use_cosine_annealing = use_cosine_annealing
        self.cosine_annealing_T_max = cosine_annealing_T_max
        self.cosine_annealing_epoch_offset = cosine_annealing_epoch_offset

        self.use_ema = use_ema
        self.use_posthoc_ema = use_posthoc_ema
        self.use_euclidean_for_rots = use_euclidean_for_rots
        self.learnable_noise_schedule = learnable_noise_schedule
        self.direct_rot_vf_loss = direct_rot_vf_loss
        self.rot_angle_weight = rot_angle_weight
        self.atom_rigid_upweight = atom_rigid_upweight
        self.apply_self_folding = apply_self_folding
        self.automatic_optimization = True

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

        
        self.use_fafe_loss = use_fafe_loss
        self.use_rot_vf_loss = use_rot_vf_loss
        self.identity_rot_noise = identity_rot_noise
        self.use_trans_mse_loss = use_trans_mse_loss
        self.scale_trans_mse_loss = scale_trans_mse_loss
        self.use_min_conformer_head = use_min_conformer_head
        self.accumulate_grad_batches = accumulate_grad_batches


        self.aatype_to_restype_tensor = torch.zeros(const.num_tokens)
        for aatype, restype in AA_TO_RES.items():
            self.aatype_to_restype_tensor[aatype] = restype

        if not strict_weight_loading or bond_rotation_head_only:
            warnings.warn("Model weights will be loaded with strict_loading=False, be sure you know what you're doing!")
            self.strict_loading = False

        if bond_rotation_head_only:
            assert hasattr(self.model, 'bond_rotation_head'), \
                "bond_rotation_head_only=True requires model.use_bond_rotation=True"
            for name, param in self.model.named_parameters():
                param.requires_grad = name.startswith('bond_rotation_head.')

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

    def _log_losses(self, loss_dict, batch, stage: str):
        # ---- global mean losses ----
        log_dict = tree.map_structure(
            lambda x: torch.round(torch.mean(x), decimals=3)
            if torch.is_tensor(x) else x,
            loss_dict
        )

        log_dict = {
            f"{stage}/{k}": v
            for k, v in sorted(log_dict.items(), key=lambda x: x[0])
        }

        # ---- per-task aggregation ----
        loss_by_task = {}
        for i, task in enumerate(batch["task"]):
            for key in [
                "loss_per_batch",
                "seq_loss",
                "frame_vf_loss",
                "frame_vf_loss_unscaled",
                "pred_trans_mse",
                "pred_heavy_atoms_trans_mse",
                "min_conformer_trans_mse",
                "min_conformer_heavy_trans_mse",
            ]:
                name = f"{task.name}_{key}"
                loss_by_task.setdefault(name, []).append(loss_dict[key][i])

        loss_by_task = {k: torch.stack(v) for k, v in loss_by_task.items()}

        for key, value in loss_by_task.items():
            self.log(
                f"task/{stage}/{key}",
                value.mean(),
                logger=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=value.shape[0],
                sync_dist=False,
            )

        # ---- t-stratified losses (TRAIN ONLY) ----
        if stage == "train":
            t = batch["t"]
            for loss_name, loss_list in loss_dict.items():
                if loss_name in [
                    "loss",
                    "frameflow_loss",
                    "frame_vf_loss_unscaled",
                    "loss_per_batch",
                ]:
                    continue
                if t.numel() != loss_list.numel():
                    continue

                stratified = t_stratified_loss(batch_t=t, batch_loss=loss_list, loss_name=loss_name)
                stratified = {
                    f"train/{k}": torch.round(
                        torch.as_tensor(v, device=t.device), decimals=3
                    )
                    for k, v in stratified.items()
                }

                self.log_dict(
                    stratified,
                    logger=True,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=t.shape[0],
                    sync_dist=False,
                )

        # ---- final logging ----
        self.log_dict(
            log_dict,
            logger=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["t"].shape[0],
            sync_dist=True,
        )

    def _shared_step(self, batch):
        # stochastic corruption for BOTH train and val
        batch["trans_t"] = batch["t"]
        batch["rot_t"] = batch["t"]
        batch = self.corrupter.corrupt_dense_batch(batch, self.identity_rot_noise)

        # self-conditioning (optional)
        self_conditioning = None
        self_folding = None
        if (
            self.model.self_conditioning
            and np.random.uniform() < self.self_condition_rate
        ):
            with torch.no_grad():
                self_conditioning = self.model(batch)
                if self.apply_self_folding:
                    pred_seq_batch = self._generate_folding_batch(
                        batch, self_conditioning["pred_seq"]
                    )
                    self_folding = self.model(pred_seq_batch)

        outputs = self.model(batch, self_conditioning, self_folding)
        return self._loss_step(batch, outputs)

    
    def training_step(self, batch, batch_idx):
        if self.global_step > 0:
            if self.ema is not None:
                self.ema.update_parameters(self.model)
            if self.ema_long is not None:
                self.ema_long.update_parameters(self.model, self.global_step - 1)
            if self.ema_short is not None:
                self.ema_short.update_parameters(self.model, self.global_step - 1)

        has_sequential = batch.get('rigids', {}).get('group1_rigid_mask', torch.tensor(False)).any()

        if not has_sequential:
            loss_dict = self._shared_step(batch)
        else:
            loss_dict = self._sequential_step(batch)

        self._log_losses(loss_dict, batch, stage="train")

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=1)
        return loss_dict["loss"].mean()

    def _corrupt_batch(self, batch):
        """Corrupt a batch in-place and return it."""
        batch["trans_t"] = batch["t"]
        batch["rot_t"] = batch["t"]
        return self.corrupter.corrupt_dense_batch(batch, self.identity_rot_noise)

    def _sequential_step(self, batch):
        """Two-pass training for MolSequentialScaffolding.

        Pass 1: both groups noised, loss on group 1.
        Pass 2: group 1 fixed at detached denoised positions from pass 1,
                group 2 still noised, loss on group 2.
        Both graphs held simultaneously; Lightning does a single backward on
        the combined loss.
        """
        group1_mask = batch['rigids']['group1_rigid_mask']  # [B, R]

        # --- Pass 1 ---
        batch1 = batch.copy()
        batch1['rigids'] = batch['rigids'].copy()
        batch1 = self._corrupt_batch(batch1)

        outputs1 = self.model(batch1)

        noising_mask_orig = batch1['rigids']['rigids_noising_mask'].clone()
        n_r = noising_mask_orig.shape[1]
        group1_mask_r = group1_mask[:, :n_r]  # clamp to actual rigid dim
        batch1['rigids']['rigids_noising_mask'] = noising_mask_orig & group1_mask_r
        loss_dict1 = self._loss_step(batch1, outputs1)

        denoised_trans1 = outputs1['denoised_rigids'].get_trans().detach()  # [B, R, 3]

        # --- Pass 2 ---
        batch2 = batch.copy()
        batch2['rigids'] = {k: v for k, v in batch['rigids'].items() if k != 'group1_rigid_mask'}
        batch2 = self._corrupt_batch(batch2)

        rigids_t = batch2['rigids']['rigids_t'].clone()
        n_r2 = rigids_t.shape[1]
        group1_mask_r2 = group1_mask[:, :n_r2]                # [B, n_r2]
        # Build src_trans [B, n_r2, 3] explicitly so mask application never sees
        # a shorter-than-n_r2 tensor (pass-1 may have fewer rigids than pass-2)
        src_trans = rigids_t[:, :, 4:].clone()                # [B, n_r2, 3] fallback
        n_copy = min(n_r2, denoised_trans1.shape[1])
        src_trans[:, :n_copy] = denoised_trans1[:, :n_copy]
        mask_exp = group1_mask_r2.unsqueeze(-1)               # [B, n_r2, 1]
        rigids_t[:, :, 4:] = torch.where(mask_exp, src_trans, rigids_t[:, :, 4:])
        batch2['rigids']['rigids_t'] = rigids_t
        batch2['rigids']['rigids_noising_mask'] = noising_mask_orig[:, :n_r2] & ~group1_mask_r2

        outputs2 = self.model(batch2)
        loss_dict2 = self._loss_step(batch2, outputs2)

        # Merge loss dicts for logging (average the scalar losses)
        loss_dict = {k: (loss_dict1[k] + loss_dict2[k]) / 2
                     if torch.is_tensor(loss_dict1.get(k)) and torch.is_tensor(loss_dict2.get(k))
                     else loss_dict1.get(k, loss_dict2.get(k))
                     for k in set(loss_dict1) | set(loss_dict2)}

        return loss_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        device = batch["t"].device
        B = batch["t"].shape[0]

        for t_val in (0.0, 0.5):
            batch_t = batch.copy()
            batch_t["t"] = torch.full((*batch_t["t"].shape,), t_val, device=device)

            loss_dict = self._shared_step(batch_t)

            # log under separate namespace
            self._log_losses(
                loss_dict,
                batch_t,
                stage=f"val/t_{t_val}",
            )

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        # Training epoch average: Lightning appends _epoch when on_step=True and on_epoch=True
        train_mse = metrics.get("train/pred_trans_mse_epoch", metrics.get("train/pred_trans_mse"))
        val_t05_mse = metrics.get("val/t_0.5/pred_trans_mse")
        val_t0_mse = metrics.get("val/t_0.0/pred_trans_mse")

        if train_mse is not None and val_t05_mse is not None and val_t0_mse is not None:
            composite = 0.4 * train_mse + 0.4 * val_t05_mse + 0.2 * val_t0_mse
            self.log(
                "val/composite_pred_trans_mse",
                composite,
                prog_bar=True,
                sync_dist=True,
            )

    # def training_step(self, batch, batch_idx):
    #     # update EMA
    #     if self.ema is not None and self.global_step > 0:
    #         self.ema.update_parameters(self.model)
    #     if self.ema_long is not None and self.global_step > 0:
    #         self.ema_long.update_parameters(self.model, self.global_step-1)
    #     if self.ema_short is not None and self.global_step > 0:
    #         self.ema_short.update_parameters(self.model, self.global_step-1)

    #     # corrupt data
    #     corrupter = self.corrupter
    #     batch['trans_t'] = batch['t']
    #     batch['rot_t'] = batch['t']
    #     batch = corrupter.corrupt_dense_batch(batch, self.identity_rot_noise)

    #     # print("rigids_data", batch['rigids'])
    #     # print("token_data", batch['token'])

    #     # torch.save(batch['rigids'], 'rigids.pt')
    #     # torch.save(batch['token'], 'token.pt')
    #     # torch.save(batch['input_data'], 'input_data.pt')

    #     # print("token_data", batch['token'])
    #     # print(batch)


    #     # run denoiser, with self-conditioning as necessary
    #     model = self.model
    #     if model.self_conditioning and np.random.uniform() < self.self_condition_rate:
    #         with torch.no_grad():
    #             self_conditioning = model(batch)

    #             # run denoising with the predicted seq of the self conditioning denoising step
    #             if self.apply_self_folding:
    #                 pred_seq_batch = self._generate_folding_batch(batch, self_conditioning['pred_seq'])
    #                 # run "folding"
    #                 self_folding = model(pred_seq_batch)
    #             else:
    #                 self_folding = None

    #     else:
    #         self_conditioning = None
    #         self_folding = None
    #     outputs = model(batch, self_conditioning, self_folding)

    #     # post_processed_outputs = self._post_process_outputs(batch, outputs)

    #     # for i, (post_processed_output, batch_item) in enumerate(zip(
    #     #     post_processed_outputs,
    #     #     batch['input_data']
    #     # )):
    #     #     input_structure = Structure(**batch_item['structure'])
    #     #     output_data = post_processed_output['output_data']
    #     #     output_struct = Structure(**output_data['structure'])
    #     #     tokenized_output = Tokenized(
    #     #         tokens=output_data['tokens'],
    #     #         rigids=output_data['rigids'],
    #     #         bonds=output_data['bonds'],
    #     #         structure=output_struct
    #     #     )
    #     #     new_residues = []
    #     #     for residue in output_struct.residues:
    #     #         new_residue = (
    #     #             residue['name'],
    #     #             residue['res_type'],
    #     #             residue['res_idx'],
    #     #             residue['atom_idx'],
    #     #             residue['atom_num'],
    #     #             residue['atom_center'],
    #     #             residue['atom_disto'],
    #     #             residue['is_standard'],
    #     #             residue['is_present'],
    #     #             False  # is_copy
    #     #         )
    #     #         new_residues.append(new_residue)
    #     #     new_residues = np.array(new_residues, dtype=SamplingResidue)
    #     #     output_struct = replace(output_struct, residues=new_residues)

    #     #     output_struct = construct_atoms(tokenized_output, output_struct)
    #     #     output_struct = update_structure(output_struct, tokenized_output.rigids['tensor7'])

    #     #     seen_asym_id = []
    #     #     chain_mask = []
    #     #     for chain in output_struct.chains:
    #     #         if chain['asym_id'] not in seen_asym_id:
    #     #             seen_asym_id.append(chain['asym_id'])
    #     #             chain_mask.append(True)
    #     #         else:
    #     #             chain_mask.append(False)
    #     #     output_struct = replace(output_struct, mask=np.array(chain_mask))

    #     #     output_pdb_str = to_pdb(output_struct)
    #     #     with open(f"output_pdb_{i}.pdb", "w") as f:
    #     #         f.write(output_pdb_str)
    #     #     input_pdb_str = to_pdb(input_structure)
    #     #     with open(f"input_pdb_{i}.pdb", "w") as f:
    #     #         f.write(input_pdb_str)
    #     #     print("written to", os.getcwd())
    #     # exit()

    #     # compute loss
    #     loss_dict = self._loss_step(batch, outputs)

    #     # log loss
    #     log_dict = tree.map_structure(
    #         lambda x: torch.round(torch.mean(x), decimals=3) if torch.is_tensor(x) else x,
    #         loss_dict
    #     )

    #     loss_by_task = {}
    #     for i, task in enumerate(batch['task']):
    #         if task.name + "_loss" not in loss_by_task:
    #             loss_by_task[task.name + "_loss"] = []
    #         if task.name + "_seq_loss" not in loss_by_task:
    #             loss_by_task[task.name + "_seq_loss"] = []
    #         if task.name + "_frame_vf_loss" not in loss_by_task:
    #             loss_by_task[task.name + "_frame_vf_loss"] = []
    #         if task.name + "_frame_vf_loss_unscaled" not in loss_by_task:
    #             loss_by_task[task.name + "_frame_vf_loss_unscaled"] = []
    #         if task.name + "_pred_trans_mse" not in loss_by_task:
    #             loss_by_task[task.name + "_pred_trans_mse"] = []

    #         loss_by_task[task.name + "_loss"].append(loss_dict['loss_per_batch'][i])
    #         loss_by_task[task.name + "_seq_loss"].append(loss_dict["seq_loss"][i])
    #         loss_by_task[task.name + "_frame_vf_loss"].append(loss_dict['frame_vf_loss'][i])
    #         loss_by_task[task.name + "_frame_vf_loss_unscaled"].append(loss_dict['frame_vf_loss_unscaled'][i])
    #         loss_by_task[task.name + "_pred_trans_mse"].append(loss_dict['pred_trans_mse'][i])

    #     loss_by_task = {
    #         key: torch.stack(values)
    #         for key, values in loss_by_task.items()
    #     }
    #     for key, value in loss_by_task.items():
    #         self.log(
    #             "task/" + key,
    #             value.mean(),
    #             prog_bar=False,
    #             logger=True,
    #             on_step=None,
    #             on_epoch=True,
    #             batch_size=value.shape[0],
    #             sync_dist=False)

    #     log_dict = {
    #         ("train/" + key): value
    #         for key, value in
    #         sorted(log_dict.items(), key = lambda x: x[0])
    #     }
    #     t = batch['t']
    #     for loss_name, loss_list in loss_dict.items():
    #         if loss_name in ['loss', 'frameflow_loss', "frame_vf_loss_unscaled", 'loss_per_batch']:
    #             continue
    #         if t.numel() != loss_list.numel():
    #             continue
    #         # if not loss_name.startswith("pt_") and not loss_name.startswith("latent_"):
    #         #     continue
    #         stratified_losses = t_stratified_loss(
    #             t, loss_list, loss_name=loss_name)
    #         stratified_losses = {
    #             f"train/{k}": torch.round(torch.as_tensor(v, device=log_dict['train/loss'].device), decimals=3)
    #             for k,v in stratified_losses.items()
    #         }
    #         self.log_dict(
    #             stratified_losses,
    #             prog_bar=False,
    #             logger=True,
    #             on_step=None,
    #             on_epoch=True,
    #             batch_size=t.shape[0],
    #             sync_dist=False)

    #     self.log_dict(
    #         log_dict,
    #         on_step=None,
    #         on_epoch=True,
    #         prog_bar=True,
    #         logger=True,
    #         batch_size=t.shape[0],
    #         sync_dist=True)

    #     return loss_dict

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
            upweight_atomic=self.atom_rigid_upweight,
            scale_bond_length_loss=self.scale_bond_length_loss,
            scale_bond_angle_loss=self.scale_bond_angle_loss,
            scale_ring_planarity_loss=self.scale_ring_planarity_loss,
            use_fafe_loss=self.use_fafe_loss,
            use_rot_vf_loss=self.use_rot_vf_loss,
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
        
        if self.bond_rotation_head_only:
            loss = frame_fm_loss_dict['bond_rot_mse']
        else:
            loss = loss + 1.0 * frame_fm_loss_dict['bond_angle_rmse'] + 1.0 * frame_fm_loss_dict['bond_length_rmse']
            loss = loss + frame_fm_loss_dict['ring_planarity_loss'] * 1.0 + frame_fm_loss_dict['bond_rot_mse'] * 0.01

        if self.use_trans_mse_loss:
            trans_mse = frame_fm_loss_dict['pred_trans_mse']
            if self.scale_trans_mse_loss:
                t_norm_clip = 0.9
                norm_scale = 1 - torch.min(inputs['t'], torch.full_like(inputs['t'], t_norm_clip))
                trans_mse = trans_mse / (norm_scale.squeeze(-1) ** 2)
            loss = loss + trans_mse.mean()

        if outputs.get('time_pred_val') is not None:
            loss = loss + 1.0 * (outputs['time_pred_val'] - inputs['t'].squeeze(-1)).abs().mean()

        if outputs.get('energy_pred_val') is not None:
            with torch.no_grad():
                e_gen = compute_xtb_energies(
                    inputs['rigids']['rigids_ref_element'],
                    outputs['denoised_rigids'].get_trans(),
                    inputs['rigids']['rigids_mask'],
                )
                delta_e = e_gen - inputs['e_min'].to(e_gen.device)
                valid = ~torch.isnan(delta_e)
            if valid.any():
                pred = outputs['energy_pred_val'][valid]
                target = delta_e[valid]
                loss = loss + 1.0 * (pred - target).abs().mean()

        # Min-conformer head: per-rigid translation prediction toward clean conformer
        zeros = torch.zeros_like(frame_fm_loss_dict['pred_trans_mse'])
        min_conformer_trans_mse = zeros
        min_conformer_heavy_trans_mse = zeros.clone()
        if outputs.get('min_conformer_pred_val') is not None:
            gt_trans = ru.Rigid.from_tensor_7(inputs['rigids']['rigids_1']).get_trans()
            pred_trans_mc = outputs['min_conformer_pred_val']  # [B, R, 3]
            # clamp to min rigid dim (input padding may differ from model output)
            n_mc = min(gt_trans.shape[1], pred_trans_mc.shape[1])
            gt_trans = gt_trans[:, :n_mc]
            pred_trans_mc = pred_trans_mc[:, :n_mc]
            rigids_mask = inputs['rigids']['rigids_mask'][:, :n_mc]
            rigids_noising_mask = inputs['rigids']['rigids_noising_mask'][:, :n_mc]
            total_mask = rigids_mask * rigids_noising_mask
            num_rigids = rigids_mask.long().sum(-1).clip(min=1)
            se_mc = torch.square(gt_trans - pred_trans_mc).sum(dim=-1)  # [B, n_mc]
            min_conformer_trans_mse = (se_mc * total_mask).sum(-1) / num_rigids
            is_heavy = (inputs['rigids']['rigids_ref_element'][:, :n_mc] != 1).float()
            heavy_mask = total_mask * is_heavy
            num_heavy = heavy_mask.sum(-1).clip(min=1)
            min_conformer_heavy_trans_mse = (se_mc * heavy_mask).sum(-1) / num_heavy
            loss = loss + min_conformer_trans_mse.mean()

        loss_dict = {"loss": loss.mean(), "frame_vf_loss": frame_vf_loss, "frame_vf_loss_unscaled": unscaled_frame_vf_loss}
        loss_dict['loss_per_batch'] = loss
        loss_dict['min_conformer_trans_mse'] = min_conformer_trans_mse
        loss_dict['min_conformer_heavy_trans_mse'] = min_conformer_heavy_trans_mse

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


        if 'bond_angles' in outputs:
            angles = outputs['bond_angles']  # (B, max_bonds, 2) — (sin θ, cos θ)
            num_rot_bonds = inputs['num_rot_bonds']  # (B,)
            valid = (torch.arange(angles.shape[1], device=angles.device)[None, :] < num_rot_bonds[:, None])
            theta_deg = torch.atan2(angles[..., 0], angles[..., 1]).abs() * (180.0 / torch.pi)
            mean_angle_deg = (theta_deg * valid).sum() / valid.sum().clamp(min=1)
            loss_dict['bond_rot_angle_deg'] = mean_angle_deg.detach()

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

            # print("grad norm", total_norm)

            if hasattr(self.model, 'bond_rotation_head'):
                head_norms = [
                    torch.linalg.vector_norm(p.grad.view(-1))
                    for p in self.model.bond_rotation_head.parameters()
                    if p.grad is not None
                ]
                if head_norms:
                    self.log("bond_rot_head_grad_norm", torch.stack(head_norms).norm(),
                             prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=1)
                else:
                    self.log("bond_rot_head_grad_norm", 0.0,
                             prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=1)

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


    def _post_process_outputs(
        self,
        batch,
        final_denoiser_out
    ):
        ret = []

        pred_rigids = final_denoiser_out['denoised_rigids']
        pred_tensor7 = pred_rigids.to_tensor_7().numpy(force=True)
        pred_seq = final_denoiser_out["pred_seq"].numpy(force=True)


        for i, input_data in enumerate(batch['input_data']):
            # chop off any padding for pred_rigids and pred_seq
            num_rigids = input_data['rigids']['tensor7'].shape[0]
            output_data = copy.deepcopy(input_data)
            # tensor7 = pred_rigids.to_tensor_7().numpy(force=True)
            _tensor7 = pred_tensor7[i, :num_rigids]
            output_data['rigids']['tensor7'] = _tensor7

            num_tokens = input_data['tokens']['token_idx'].shape[0]
            _seq = pred_seq[i, :num_tokens]
            output_data['tokens']['res_type'] = _seq

            # if we copy any tokens, figure out what generated residue corresponds to these fixed tokens
            # select masks
            token_data = output_data['tokens']
            token_is_copy_mask = token_data['is_copy']
            motif_idx = final_denoiser_out["motif_idx"][i, :num_tokens]
            motif_select_mask = (token_is_copy_mask & token_data['resolved_mask'])
            motif_seq_fixed = ~token_data['seq_noising_mask']
            # actual idxs
            fixed_bb_res_idx = motif_idx[motif_select_mask]
            fixed_seq_res_idx = motif_idx[motif_seq_fixed]
            fixed_bb_chain_idx = token_data['asym_id'][motif_select_mask]
            fixed_seq_chain_idx = token_data['asym_id'][motif_seq_fixed]

            # TODO: this is kinda jenk, we're doing this to allow us to have access to both
            # the "original unindexed index" (which is stored in res_idx)
            # and "new assigned index" (which is overwritten into token_idx)
            # we currently use both to impute the copy motif into the generated structure
            token_data['token_idx'][motif_select_mask] = motif_idx.numpy(force=True)[motif_select_mask]

            ret.append({
                "input_data": input_data,
                "output_data": output_data,
                "fixed_bb_res_idx": fixed_bb_res_idx,
                "fixed_seq_res_idx": fixed_seq_res_idx,
                "fixed_bb_chain_idx": fixed_bb_chain_idx,
                "fixed_seq_chain_idx": fixed_seq_chain_idx,
                "name": batch["task"][i]
            })

        return ret

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

        global_shift = torch.zeros_like(trans_0).mean(dim=-2)

        for t_2 in tqdm.tqdm(ts[1:]):
            d_t = t_2 - t_1
            # Run model.
            trans_t_1, rotmats_t_1, _ = prot_traj[-1]

            # trans_t_1_center = trans_t_1.mean(dim=-2)
            # trans_t_1 = trans_t_1 - trans_t_1_center[..., None, :]
            # global_shift += trans_t_1_center

            t_hat, d_t_hat, trans_t_hat = corrupter.trans_churn(
                d_t,
                t_1,
                trans_t_1,
                noising_mask=rigids_noising_mask,
            )
            _, _, rotmats_t_hat = corrupter.rot_churn(
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
                (pred_trans_1 + global_shift[..., None, :],
                 pred_rotmats_1,
                 denoiser_out["pred_seq"].detach().cpu(),
                )
            )

            trans_d_t_hat = d_t_hat
            rot_d_t_hat = d_t_hat
            trans_time = t_hat
            rot_time = t_hat
            trans_vf_scale = 0.5
            # trans_vf_scale = 1
            rot_vf_scale = 1

            trans_t_2 = corrupter.trans_euler_step(
                trans_d_t_hat,
                trans_time,
                pred_trans_1,
                trans_t_hat,
                noising_mask=rigids_noising_mask,
                vf_scale=trans_vf_scale,
            )
            rotmats_t_2 = corrupter.rots_euler_step(
                rot_d_t_hat,
                rot_time,
                pred_rotmats_1,
                rotmats_t_hat,
                noising_mask=rigids_noising_mask,
                vf_scale=rot_vf_scale,
                rot_vf=denoiser_out['pred_rot_vf']
            )

            prot_traj.append(
                (trans_t_2 + global_shift[..., None, :],
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

        denoiser_out = model(batch, self_condition=denoiser_out)#, sanitize_motif_idx=True)

        # Process model output.
        pred_rigids = denoiser_out['denoised_rigids']
        pred_trans_1 = pred_rigids.get_trans() + global_shift[..., None, :]
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

            # token_is_unindexed_mask = token_data['is_unindexed']
            # token_assign_index_mask = token_is_unindexed_mask & motif_select_mask
            # token_data['token_idx'][token_assign_index_mask] = motif_idx.numpy(force=True)[token_assign_index_mask]

            # TODO: this is kinda jenk, we're doing this to allow us to have access to both the "original unindexed index" and "new assigned index"
            # so that we can replace the motif into the structure
            token_data['token_idx'][motif_select_mask] = motif_idx.numpy(force=True)[motif_select_mask]
            # print(output_data['tokens'])

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
                "name": batch["task"][i],
                "smiles": batch["smiles"][i],
            })

        return ret


    def on_train_start(self):
        if self.use_cosine_annealing:
            last_epoch = self.cosine_annealing_epoch_offset if self.cosine_annealing_epoch_offset is not None else self.current_epoch
            scheduler = self.lr_schedulers()
            scheduler.T_max = self.cosine_annealing_T_max
            scheduler.last_epoch = last_epoch
            scheduler._step_count = last_epoch + 1
            print(f"Cosine annealing scheduler with T_max={scheduler.T_max} and last_epoch={scheduler.last_epoch}")

    def configure_optimizers(self):
        if self.bond_rotation_head_only:
            params = self.model.bond_rotation_head.parameters()
        else:
            params = self.model.parameters()
        optimizer = self.optim(params)

        if self.use_cosine_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cosine_annealing_T_max,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
            }

        elif self.use_lr_step_decay:
            scheduler = get_mult_decay_schedule(
                optimizer,
                start_step=self.lr_step_decay_start,
                decay_step=self.lr_step_decay_step,
                decay_factor=self.lr_step_decay_factor
            )
        elif self.use_cosine_lr_sched:
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


class BiomoleculeSamplingModule(L.LightningModule):
    def __init__(
        self,
        model,
        corrupter,
        run_cfg,
    ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        # the actual ema params don't matter here, we just wanna be able to load the weights
        self.ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
        self.corrupter = corrupter
        self.run_cfg = run_cfg

    def predict_step(self, batch, batch_idx):
        # structure_dict = batch['input_data'][0]['structure']
        # structure = Structure(**structure_dict)
        # pdb_str = to_pdb(structure)
        # with open(f"clean_data_mol_predict.pdb", "w") as f:
        #     f.write(pdb_str)
        # exit()
        # Set-up time

        # self.corrupter.num_timesteps = 20
        ts = torch.linspace(0.0, 1.0, self.corrupter.num_timesteps)
        # ts = torch.linspace(1.0, 0.0, self.corrupter.num_timesteps)

        rigids_data = batch['rigids']
        rigids_data['rigids_t'] = rigids_data['rigids_1']
        token_data = batch['token']
        # print("rigids_data before", rigids_data['rigids_ref_element'])

        rigids_data_center = rigids_data['rigids_t'][:, :, 4:].mean(dim=1)
        rigids_data['rigids_t'][:, :, 4:] -= rigids_data_center[..., None, :]

        # rigids_data['rigids_t'] = rigids_data['rigids_t'][:, :, :4]

        # test_rigids_data = torch.load('outputs/geom_identityRot_frame_384/train/rigids.pt')
        # test_token_data = torch.load('outputs/geom_identityRot_frame_384/train/token.pt')
        # batch['input_data'] = torch.load('outputs/geom_identityRot_frame_384/train/input_data.pt')
        # batch['rigids'] = test_rigids_data
        # batch['token'] = test_token_data
        # rigids_data = test_rigids_data
        # token_data = test_token_data

        # rigids_data['rigids_t'][:, :, :4] = test_rigids_data['rigids_t'][:, :32, :4]

        # print("rigids_data after", rigids_data['rigids_ref_element'])
        # print("token_data", token_data)
        # exit()


        for batch_item in batch['input_data']:
            residues = batch_item['structure']['residues']
            new_residues = []
            for residue in residues:
                new_residue = (
                    residue['name'],
                    residue['res_type'],
                    residue['res_idx'],
                    residue['atom_idx'],
                    residue['atom_num'],
                    residue['atom_center'],
                    residue['atom_disto'],
                    residue['is_standard'],
                    residue['is_present'],
                    False  # is_copy
                )
                new_residues.append(new_residue)
            batch_item['structure']['residues'] =  np.array(new_residues, dtype=SamplingResidue)

        rigids_0 = ru.Rigid.from_tensor_7(rigids_data['rigids_t'])
        trans_0 = rigids_0.get_trans()
        rotmats_0 = rigids_0.get_rots().get_rot_mats()
        rigids_noising_mask = rigids_data['rigids_noising_mask']
        seq_noising_mask = token_data['seq_noising_mask']

        prot_traj = [(
            trans_0,
            rotmats_0,
            None
        )]

        # print(trans_0)
        # exit()

        clean_traj = []
        

        # set up initial integration conditions
        t_1 = ts[0]
        prev_denoiser_out = None
        for t_2 in tqdm.tqdm(ts[1:]):
            trans_t, rotmats_t, _ = prot_traj[-1]
            prev_denoiser_out, prot_traj_point, clean_traj_point = self._integration_step(
                self.ema.module,
                batch,
                trans_t,
                rotmats_t,
                t_1,
                t_2,
                self_conditioning=prev_denoiser_out
            )

            

            prot_traj.append(prot_traj_point)
            clean_traj.append(clean_traj_point)
            t_1 = t_2

        trans_t, rotmats_t, _ = prot_traj[-1]
        # final_denoiser_out, prot_traj_point, _ = self._integration_step(
        #     self.ema.module,
        #     batch,
        #     trans_t,
        #     rotmats_t,
        #     ts[-1],
        #     ts[-1],
        #     self_conditioning=prev_denoiser_out,
        #     apply_step=False
        # )
        final_denoiser_out = self.ema.module(
            batch,
            self_condition=prev_denoiser_out,
            self_folding=None
        )
        pred_rigids = final_denoiser_out['denoised_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
        prot_traj_point = (pred_trans_1, pred_rotmats_1, final_denoiser_out["pred_seq"].detach().cpu())

        # Process model output.
        # prot_traj = prot_traj[1:]
        prot_traj[0] = (trans_0, rotmats_0, prot_traj_point[2])

        ret = self._post_process_outputs(
            batch,
            prot_traj,
            clean_traj,
            final_denoiser_out
        )

        return ret

    def chk(self, name, x):
        if not torch.isfinite(x).all():
            raise RuntimeError(f"NaN/Inf in {name}")

    def _integration_step(
        self,
        model,
        batch,
        trans_t_1,
        rotmats_t_1,
        t_1,
        t_2,
        self_conditioning=None,
        apply_step=True
    ):
        d_t = t_2 - t_1
        rigids_data = batch['rigids']
        token_data = batch['token']
        rigids_noising_mask = rigids_data['rigids_noising_mask']
        seq_noising_mask = token_data['seq_noising_mask']
        num_batch, num_res = seq_noising_mask.shape
        device = self.device
        denoiser_out = None

        t_hat, d_t_hat, trans_t_hat = self.corrupter.trans_churn(
            d_t,
            t_1,
            trans_t_1,
            noising_mask=rigids_noising_mask,
        )

        _, _, rotmats_t_hat = self.corrupter.rot_churn(
            d_t,
            t_1,
            rotmats_t_1,
            noising_mask=rigids_noising_mask,
        )

        # # Run model.
        rigids_data["trans_t"] = trans_t_hat
        rigids_data["rotmats_t"] = rotmats_t_hat
        rigids_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_hat),
            trans=trans_t_hat
        ).to_tensor_7()
        t = torch.ones(num_batch, device=device)[..., None] * t_hat
        batch["t"] = t


        denoiser_out = model(
            batch,
            self_condition=self_conditioning,
            self_folding=None
        )
        
        

        # total_mask = rigids_noising_mask & rigids_data['rigids_mask']
        # clean_rigids_1_trans = torch.stack([batch['task'][i]['clean_rigids_1'].get_trans() for i in range(num_batch)])
        # pred_frame_trans_se = torch.square(clean_rigids_1_trans - denoiser_out['denoised_rigids'].get_trans()).sum(dim=-1)
        # pred_frame_trans_mse = (pred_frame_trans_se * total_mask).sum(-1) / num_batch
        # print("pred_frame_trans_mse", pred_frame_trans_mse)
        # sys.stdout.flush()

        # Process model output.
        pred_rigids = denoiser_out['denoised_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()

        # print("1", pred_rotmats_1)

        clean_traj_point = (
            pred_trans_1,
            pred_rotmats_1,
            denoiser_out["pred_seq"].detach().cpu(),
        )

        # d_t_hat  = d_t
        # t_hat = t_1
        # trans_t_hat = trans_t_1
        # rotmats_t_hat = rotmats_t_1

        trans_d_t_hat = d_t_hat
        rot_d_t_hat = d_t_hat
        trans_time = t_hat
        rot_time = t_hat
        trans_vf_scale = 1
        rot_vf_scale = 1

        if apply_step:
            trans_t_2 = self.corrupter.trans_euler_step(
                trans_d_t_hat,
                trans_time,
                pred_trans_1,
                trans_t_hat,
                noising_mask=rigids_noising_mask,
                vf_scale=trans_vf_scale,
            )
            rotmats_t_2 = self.corrupter.rots_euler_step(
                rot_d_t_hat,
                rot_time,
                pred_rotmats_1,
                rotmats_t_hat,
                noising_mask=rigids_noising_mask,
                vf_scale=rot_vf_scale,
                rot_vf=denoiser_out['pred_rot_vf']
            )

            prot_traj_point = (
                trans_t_2,
                rotmats_t_2,
                denoiser_out["pred_seq"].detach().cpu(),
            )
        else:
            prot_traj_point = clean_traj_point

        # print("2", rotmats_t_2)
        return denoiser_out, prot_traj_point, clean_traj_point

    def _post_process_outputs(
        self,
        batch,
        prot_traj,
        clean_traj,
        final_denoiser_out
    ):
        ret = []

        pred_rigids = final_denoiser_out['denoised_rigids']
        pred_tensor7 = pred_rigids.to_tensor_7().numpy(force=True)
        pred_seq = final_denoiser_out["pred_seq"].numpy(force=True)


        for i, input_data in enumerate(batch['input_data']):
            # chop off any padding for pred_rigids and pred_seq
            num_rigids = input_data['rigids']['tensor7'].shape[0]
            output_data = copy.deepcopy(input_data)
            # tensor7 = pred_rigids.to_tensor_7().numpy(force=True)
            _tensor7 = pred_tensor7[i, :num_rigids]
            output_data['rigids']['tensor7'] = _tensor7

            num_tokens = input_data['tokens']['token_idx'].shape[0]
            _seq = pred_seq[i, :num_tokens]
            output_data['tokens']['res_type'] = _seq

            # if we copy any tokens, figure out what generated residue corresponds to these fixed tokens
            # select masks
            token_data = output_data['tokens']
            token_is_copy_mask = token_data['is_copy']
            motif_idx = final_denoiser_out["motif_idx"][i, :num_tokens]
            motif_select_mask = (token_is_copy_mask & token_data['resolved_mask'])
            motif_seq_fixed = ~token_data['seq_noising_mask']
            # actual idxs
            fixed_bb_res_idx = motif_idx[motif_select_mask]
            fixed_seq_res_idx = motif_idx[motif_seq_fixed]
            fixed_bb_chain_idx = token_data['asym_id'][motif_select_mask]
            fixed_seq_chain_idx = token_data['asym_id'][motif_seq_fixed]

            # TODO: this is kinda jenk, we're doing this to allow us to have access to both
            # the "original unindexed index" (which is stored in res_idx)
            # and "new assigned index" (which is overwritten into token_idx)
            # we currently use both to impute the copy motif into the generated structure
            token_data['token_idx'][motif_select_mask] = motif_idx.numpy(force=True)[motif_select_mask]

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
                "name": batch["task"][i],
                "smiles": batch["smiles"][i],
            })

        return ret


class PDBWriter(BasePredictionWriter):
    def __init__(self, output_dir, run_cfg):
        super().__init__(write_interval="batch_and_epoch")
        self.output_dir = os.path.abspath(output_dir)
        self.samples_dir = os.path.abspath(os.path.join(output_dir, "samples"))
        self.metadata_dir = os.path.abspath(os.path.join(output_dir, "metadata"))
        self.traj_dir = os.path.abspath(os.path.join(output_dir, "traj"))
        self.run_cfg = run_cfg

        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.traj_dir, exist_ok=True)


        self.samples_metadata = {}
        self.task_name_counters = {}  # Track counters per task name across function calls

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        pwd = os.getcwd()
        os.chdir(self.samples_dir)

        samples_metadata = {}

        curr_sample_id = 0
        rank = trainer.global_rank
        for sample_data in prediction:
            # sample_coords = sample_data['sample_coord']
            sample_output = sample_data['output_data']
            # TODO: idek why i have to do this...
            sample_output['structure']['mask'] = np.ones_like(sample_output['structure']['mask'].astype(bool))
            struct = Structure(**sample_output['structure'])
            sample_output = Tokenized(
                tokens=sample_output['tokens'],
                rigids=sample_output['rigids'],
                bonds=sample_output['bonds'],
                structure=struct,
            )
            struct = construct_atoms(sample_output, struct)
            sample_len = sample_output.tokens.shape[0]
            # Use task name from YAML if available, otherwise fall back to default naming
            task_name = sample_data.get('name', None)
            
            if task_name is not None:
                # Initialize counter for this task name if it doesn't exist
                if task_name not in self.task_name_counters:
                    self.task_name_counters[task_name] = 0
                # Use current counter value and increment for next time
                sample_name = f"{task_name}_{self.task_name_counters[task_name]}"
                self.task_name_counters[task_name] += 1
            else:
                # Fall back to original naming scheme
                sample_name = f"len_{sample_len}_protein_id{rank}_{batch_idx}_{curr_sample_id}" #.pdb"
            struct = update_structure(struct, sample_output.rigids['tensor7'])

            if self.run_cfg['output_motif_chains']:
                # we rename the motif chain to something different
                # so that it will be separated when outputted
                # it doesn't particularly matter what this letter is since it'll be
                # overwritten by to_pdb
                num_chains = len(struct.chains)
                struct.chains['asym_id'] = np.arange(num_chains)

                def get_next_free_chain_name(seen_names):
                    CHAIN_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                    for c in CHAIN_ALPHABET:
                        if c not in seen_names:
                            return c
                    raise ValueError("output has too many chains to be represented in .pdb format")

                seen_names = []
                new_chain_names = []
                for chain in struct.chains:
                    if chain['name'] not in seen_names:
                        seen_names.append(chain['name'])
                        new_chain_names.append(chain['name'])
                    else:
                        chain_rename = get_next_free_chain_name(seen_names)
                        seen_names.append(chain_rename)
                        new_chain_names.append(chain_rename)
                struct.chains['name'] = np.array(new_chain_names)

                for chain in struct.chains:
                    res_start = chain["res_idx"]
                    res_end = chain["res_idx"] + chain["res_num"]
                    residues = struct.residues[res_start:res_end]
                    residues['res_idx'] = np.arange(chain["res_num"])
            else:
                # we basically detect which chain is the motif by any duplicate chains
                # we rely on the fact that the motif is appended to the generated residues
                # so that it will always be second
                seen_asym_id = []
                chain_mask = []
                for chain in struct.chains:
                    if chain['asym_id'] not in seen_asym_id:
                        seen_asym_id.append(chain['asym_id'])
                        chain_mask.append(True)
                    else:
                        chain_mask.append(False)
                struct = replace(struct, mask=np.array(chain_mask))
                # print(struct, chain_mask)

            pdb_str = to_pdb(struct, smiles=sample_data.get('smiles'))
            with open(sample_name + ".pdb", 'w') as fp:
                fp.write(pdb_str)

            if self.run_cfg['save_traj']:
                clean_traj = sample_data['clean_traj']
                prot_traj = sample_data['prot_traj']

                if task_name is not None:
                    traj_base_name = f"{task_name}_{self.task_name_counters[task_name] - 1}"
                else:
                    traj_base_name = f"len_{sample_len}_protein_id{rank}_{batch_idx}_{curr_sample_id}"
                
                clean_traj_name = os.path.join(
                    self.traj_dir,
                    f"{traj_base_name}_clean_traj.pdb"
                )
                prot_traj_name = os.path.join(
                    self.traj_dir,
                    f"{traj_base_name}_prot_traj.pdb"
                )
                clean_model_strs = []
                prot_model_strs = []

                for i, traj_data in enumerate(clean_traj):
                    traj_struct = Structure(**traj_data['structure'])
                    traj_output = Tokenized(
                        tokens=traj_data['tokens'],
                        rigids=traj_data['rigids'],
                        bonds=traj_data['bonds'],
                        structure=traj_struct,

                    )
                    traj_struct = construct_atoms(traj_output, traj_struct)
                    traj_struct = update_structure(traj_struct, traj_output.rigids['tensor7'])
                    pdb_str = to_pdb(traj_struct)
                    clean_model_strs.append(f"MODEL        {i}\n")
                    clean_model_strs.append(pdb_str.split("END")[0])
                    clean_model_strs.append(f"ENDMDL       \n")
                clean_model_strs.append("END\n")

                with open(clean_traj_name, 'w') as fp:
                    for pdb_str in clean_model_strs:
                        fp.write(pdb_str)

                for i, traj_data in enumerate(prot_traj):
                    traj_struct = Structure(**traj_data['structure'])
                    traj_output = Tokenized(
                        tokens=traj_data['tokens'],
                        rigids=traj_data['rigids'],
                        bonds=traj_data['bonds'],
                        structure=traj_struct,
                    )
                    traj_struct = construct_atoms(traj_output, traj_struct)
                    traj_struct = update_structure(traj_struct, traj_output.rigids['tensor7'])
                    pdb_str = to_pdb(traj_struct)
                    prot_model_strs.append(f"MODEL        {i}\n")
                    prot_model_strs.append(pdb_str.split("END")[0])
                    prot_model_strs.append(f"ENDMDL       \n")
                prot_model_strs.append("END\n")
                with open(prot_traj_name, 'w') as fp:
                    for pdb_str in prot_model_strs:
                        fp.write(pdb_str)

            sample_path = os.path.abspath(sample_name + ".pdb")

            chain_data = struct.chains
            chain_mapping = {
                c['asym_id']: c['name']
                for c in chain_data
            }

            samples_metadata[sample_name] = {
                "path": sample_path,
                "name": sample_data['name'] if 'name' in sample_data else None,
                "length": sample_len,
                "fixed_bb_res_idx": [i+1 for i in sample_data['fixed_bb_res_idx'].tolist()],  # 1-indexed chain for pyrosetta
                "fixed_bb_chain": [chain_mapping[int(i)] for i in sample_data['fixed_bb_chain_idx']],
                "fixed_seq_res_idx": [i+1 for i in sample_data['fixed_seq_res_idx'].tolist()],  # 1-indexed chain for pyrosetta
                "fixed_seq_chain": [chain_mapping[int(i)] for i in sample_data['fixed_seq_chain_idx']],
            }
            curr_sample_id += 1
        
        

        with open(os.path.join(self.metadata_dir, f"samples_metadata_rank{rank}_batch{batch_idx}.json"), 'w') as fp:
            json.dump(samples_metadata, fp)
        self.samples_metadata.update(samples_metadata)

        os.chdir(pwd)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # collect metadata across all processes
        gathered = [None for _ in range(trainer.world_size)]
        # Call the collective on *every* rank or it will hang
        if dist.is_available() and dist.is_initialized() and trainer.world_size > 1:
            dist.all_gather_object(gathered, self.samples_metadata)   # blocks until all ranks participate
        else:
            gathered = [self.samples_metadata]

        samples_metadata = {}
        for d in gathered:
            samples_metadata.update(d)

        # Now only rank 0 writes/merges
        if trainer.global_rank == 0:
            # merged = ...  # flatten/concatenate/serialize as you like
            # write merged to disk
            with open(os.path.join(self.output_dir, "samples_metadata.json"), 'w') as fp:
                json.dump(samples_metadata, fp)

            pmpnn_fixed_pos_dict = {}
            for name, metadata in samples_metadata.items():
                entry = {
                    chain: []
                    for chain in set(metadata['fixed_seq_chain'])
                }
                for pos, pos_chain in zip(metadata['fixed_seq_res_idx'], metadata['fixed_seq_chain']):
                    entry[pos_chain].append(pos)
                pmpnn_fixed_pos_dict[name] = entry

            with open(os.path.join(self.output_dir, "pmpnn_fixed_pos_dict.jsonl"), 'w') as fp:
                json.dump(pmpnn_fixed_pos_dict, fp)

        # (optional) keep ranks in lockstep before exiting
        trainer.strategy.barrier()