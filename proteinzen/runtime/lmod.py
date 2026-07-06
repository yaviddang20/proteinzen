import logging
import math
from functools import partial
import copy
from typing import Any
import warnings
from dataclasses import replace
import os
import shutil
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
import p_tqdm

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from proteinzen.boltz.data import const

from proteinzen.openfold.data import residue_constants
from proteinzen.openfold.utils import rigid_utils as ru

from proteinzen.boltz.data.types import Structure
from proteinzen.data.featurize.sampling import construct_atoms, update_structure
from proteinzen.data.featurize.tokenize import Tokenized
from proteinzen.data.featurize.sampling import construct_atoms
# from proteinzen.data.write.mmcif import to_mmcif
from proteinzen.data.write.pdb import to_pdb

from proteinzen.model.utils import gather_helper
from proteinzen.model.denoiser_v2 import MonotonicIncreasingFn
from proteinzen.stoch_interp.integration import Integrator, EulerIntegrator
from proteinzen.stoch_interp.diffeq import BaseEulerODEStep
from proteinzen.stoch_interp.model_wrapper import BaseModelForward
from proteinzen.stoch_interp import so3_utils
from proteinzen.stoch_interp.multiframe import align_structures

from .utils import gen_pbar_str
from .ema import EMAModel

from .loss.multiframe import multiframe_fm_loss_dense_batch, sym_permute_gt_rigids
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


def t_stratified_loss(batch_t, batch_loss, num_bins=10, loss_name=None):
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
    from concurrent.futures import ThreadPoolExecutor
    _xtb_executor = ThreadPoolExecutor(max_workers=max_workers, initializer=_xtb_worker_init)


def _xtb_worker_init():
    """Pin each executor thread to 1 OMP thread so parallelism is across molecules,
    not within each xTB call (avoids 20 × OMP_NUM_THREADS thread explosion)."""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'


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


_ELEMENT_SYMBOLS = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
_CHAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _build_all_atom_records(rigid_tensor7, rigids_mask, ref_elements, is_atom_mask,
                             rigids_sc_idx, rigids_to_token, rigids_seq_idx,
                             res_types_tok, asym_ids_tok, token_residue_idx=None):
    """Return a list of atom dicts for one sample, with full sidechains for protein."""
    from collections import defaultdict
    from proteinzen.data.featurize.tokenize import RES_TO_AA
    from proteinzen.utils import coarse_grain as cg_utils
    from proteinzen.openfold.data import residue_constants as rc

    protein_id = const.chain_type_ids["PROTEIN"]
    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]

    records = []

    # --- Protein: group 3 backbone rigids per token, reconstruct all atoms ---
    protein_bb_mask = rigids_mask & (ref_elements == -1)
    token_to_sc = defaultdict(dict)
    for r_idx in np.where(protein_bb_mask)[0]:
        tok = int(rigids_to_token[r_idx])
        sc = int(rigids_sc_idx[r_idx])
        token_to_sc[tok][sc] = r_idx

    for tok in sorted(token_to_sc.keys()):
        sc_map = token_to_sc[tok]
        if len(sc_map) != 3:
            continue
        r_indices = [sc_map[sc] for sc in (0, 1, 2)]
        tensor7 = torch.as_tensor(rigid_tensor7[r_indices])
        rigids_obj = ru.Rigid.from_tensor_7(tensor7)

        res_type = int(res_types_tok[tok])
        aa_type = RES_TO_AA[res_type]
        seq_t = torch.tensor([aa_type], device='cpu')
        dummy_mask = torch.ones(1, dtype=torch.bool, device='cpu')
        atom14, atom14_mask = cg_utils.compute_atom14_from_cg_frames(
            rigids_obj, dummy_mask, seq_t, return_atom_mask=True
        )
        atom14 = atom14.squeeze(0).numpy()
        atom14_mask = atom14_mask.squeeze(0).bool().numpy()

        one_letter = rc.restypes_with_x[res_type] if res_type < len(rc.restypes_with_x) else 'X'
        res_name = rc.restype_1to3.get(one_letter, 'UNK')
        atom14_names = rc.restype_name_to_atom14_names.get(res_name, [])
        asym_id = int(asym_ids_tok[tok])
        seq_idx = int(token_residue_idx[int(rigids_seq_idx[r_indices[1]])]) if token_residue_idx is not None else int(rigids_seq_idx[r_indices[1]])

        for aname, present, xyz in zip(atom14_names, atom14_mask, atom14):
            if not present or not aname:
                continue
            records.append({
                'asym_id': asym_id, 'seq_idx': seq_idx, 'res_name': res_name,
                'atom_name': aname, 'element': aname[0], 'xyz': xyz,
                'mol_type': protein_id,
            })

    # --- Ligand: use rigid translations directly ---
    ligand_mask = rigids_mask & is_atom_mask & (ref_elements != 1)
    for r_idx in np.where(ligand_mask)[0]:
        tok = int(rigids_to_token[r_idx])
        asym_id = int(asym_ids_tok[tok])
        seq_idx = int(token_residue_idx[int(rigids_seq_idx[r_idx])]) if token_residue_idx is not None else int(rigids_seq_idx[r_idx])
        el = int(ref_elements[r_idx])
        elem_sym = _ELEMENT_SYMBOLS.get(el, 'X')
        xyz = rigid_tensor7[r_idx, 4:]
        records.append({
            'asym_id': asym_id, 'seq_idx': seq_idx, 'res_name': 'LIG',
            'atom_name': elem_sym, 'element': elem_sym, 'xyz': xyz,
            'mol_type': nonpolymer_id,
        })

    records.sort(key=lambda r: (r['asym_id'], r['seq_idx']))
    return records


def _write_model_block(f, atom_records, model_num):
    protein_id = const.chain_type_ids["PROTEIN"]
    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]

    f.write(f"MODEL        {model_num}\n")
    prev_asym_id = None
    for i, rec in enumerate(atom_records):
        asym_id = rec['asym_id']
        if prev_asym_id is not None and asym_id != prev_asym_id:
            f.write(f"TER   {i:>5}\n")
        prev_asym_id = asym_id

        chain_letter = _CHAIN_ALPHABET[asym_id % len(_CHAIN_ALPHABET)]
        x, y, z = float(rec['xyz'][0]), float(rec['xyz'][1]), float(rec['xyz'][2])
        res_serial = rec['seq_idx'] + 1
        elem = rec['element'].upper()
        aname = rec['atom_name']
        pdb_aname = f" {aname:<3}" if len(aname) < 4 else aname

        if rec['mol_type'] == nonpolymer_id:
            record = "HETATM"
        else:
            record = "ATOM  "

        f.write(
            f"{record}{i+1:>5} {pdb_aname} {rec['res_name']:>3} {chain_letter}{res_serial:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {elem:>2}\n"
        )
    f.write("ENDMDL\n")


def write_val_pdb(gt_rigid7, pred_rigid7, rigids_mask, ref_elements, is_atom_mask,
                  rigids_sc_idx, rigids_to_token, rigids_seq_idx,
                  res_types_tok, asym_ids_tok, path, token_residue_idx=None):
    """Write a two-MODEL PDB: MODEL 1 = GT, MODEL 2 = predicted, with full sidechains."""
    gt_records = _build_all_atom_records(
        gt_rigid7, rigids_mask, ref_elements, is_atom_mask,
        rigids_sc_idx, rigids_to_token, rigids_seq_idx, res_types_tok, asym_ids_tok,
        token_residue_idx=token_residue_idx,
    )
    pred_records = _build_all_atom_records(
        pred_rigid7, rigids_mask, ref_elements, is_atom_mask,
        rigids_sc_idx, rigids_to_token, rigids_seq_idx, res_types_tok, asym_ids_tok,
        token_residue_idx=token_residue_idx,
    )
    with open(path, "w") as f:
        _write_model_block(f, gt_records, 1)
        _write_model_block(f, pred_records, 2)
        f.write("END\n")


def _detach_cpu_batch(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: _detach_cpu_batch(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_detach_cpu_batch(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_detach_cpu_batch(v) for v in obj)
    return obj


def _move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    return obj


def _slice_batch(obj, i):
    if isinstance(obj, torch.Tensor):
        return obj[i:i+1]
    elif isinstance(obj, dict):
        return {k: _slice_batch(v, i) for k, v in obj.items()}
    elif isinstance(obj, list):
        return obj[i:i+1]
    elif isinstance(obj, tuple):
        return tuple(_slice_batch(v, i) for v in obj)
    return obj


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
                 no_seq_loss=False,
                 distogram_loss=False,
                 pred_local_fafe_loss=False,
                 pae_loss=False,
                 use_interface_fafe_loss=False,
                 use_interchain_fafe_loss=False,
                 use_brownian_rot_path_loss=False,
                 postalign_noise=False,
                 epoch_sample_every_n_epochs=5,
                 epoch_sample_num_steps=100,
                 # use_stabilized_high_t_loss=False
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
        self.no_seq_loss = no_seq_loss
        self.direct_rot_vf_loss = direct_rot_vf_loss
        self.rot_angle_weight = rot_angle_weight
        self.atom_rigid_upweight = atom_rigid_upweight
        self.apply_self_folding = apply_self_folding
        self.automatic_optimization = True
        self.distogram_loss = distogram_loss
        self.pred_local_fafe_loss = pred_local_fafe_loss
        self.pae_loss = pae_loss
        self.use_interface_fafe_loss = use_interface_fafe_loss
        self.use_interchain_fafe_loss = use_interchain_fafe_loss
        self.use_brownian_rot_path_loss = use_brownian_rot_path_loss
        self.postalign_noise = postalign_noise
        # self.use_stabilized_high_t_loss = use_stabilized_high_t_loss

        if learnable_noise_schedule:
            self.automatic_optimization = False
            self.t_sched = MonotonicIncreasingFn(n_res_ident=const.num_tokens)
        else:
            self.t_sched = None

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
        self.epoch_sample_every_n_epochs = epoch_sample_every_n_epochs
        self.epoch_sample_num_steps = epoch_sample_num_steps
        self._epoch_sample_train_batch = None
        self._epoch_sample_val_batch = None
        self._pending_traj_writes = []

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
            per_task_keys = [
                "loss_per_batch",
                "seq_loss",
                "frame_vf_loss",
                "frame_vf_loss_unscaled",
                "pred_trans_mse",
                "pred_heavy_atoms_trans_mse",
            ]
            if "min_conformer_trans_mse" in loss_dict:
                per_task_keys += ["min_conformer_trans_mse", "min_conformer_heavy_trans_mse"]
            for key in per_task_keys:
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
            _skip = {"loss", "frameflow_loss", "frame_vf_loss_unscaled", "loss_per_batch"}
            for loss_name, loss_list in loss_dict.items():
                if loss_name in _skip:
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

            # t-stratified per-task losses
            for task_key, task_loss in loss_by_task.items():
                t_per_sample = t[:task_loss.shape[0]]
                if t_per_sample.numel() != task_loss.numel():
                    continue
                stratified = t_stratified_loss(batch_t=t_per_sample, batch_loss=task_loss, loss_name=task_key)
                stratified = {
                    f"task/train/{k}": torch.round(
                        torch.as_tensor(v, device=t.device), decimals=3
                    )
                    for k, v in stratified.items()
                }
                self.log_dict(
                    stratified,
                    logger=True,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=t_per_sample.shape[0],
                    sync_dist=False,
                )

        # ---- val: per-task losses (val uses fixed t values via stage name) ----
        if stage.startswith("val"):
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

        # ---- final logging ----
        self.log_dict(
            log_dict,
            logger=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["t"].shape[0],
            sync_dist=False,
        )

    def _shared_step(self, batch, return_outputs=False):

        corrupter = self.corrupter
        if self.t_sched is not None:
            t = batch['t']
            res_type = batch['token']['res_type']
            rigid_parent_res = batch['rigids']['rigids_seq_idx']
            with torch.no_grad():
                t_per_aa_ident = self.t_sched(t)
            _, dlogk_per_aa_ident = torch.func.jvp(
                lambda x: self.t_sched(1-x),
                (t,),
                (torch.ones_like(t),)
            )
            velocity_factor_per_aa_ident = dlogk_per_aa_ident / (1 - t_per_aa_ident.clip(max=0.99))
            t_per_aa_ident.requires_grad = True
            t_per_res = gather_helper(t_per_aa_ident, res_type)
            velocity_factor_per_res = gather_helper(velocity_factor_per_aa_ident, res_type)
            t_per_rigid = gather_helper(t_per_res, rigid_parent_res)
            velocity_factor_per_rigid = gather_helper(velocity_factor_per_res, rigid_parent_res)
            batch['trans_t'] = t_per_rigid[..., 0]
            batch['rot_t'] = t_per_rigid[..., 1]
            batch['velocity_factor_per_rigid'] = velocity_factor_per_rigid
        else:
            t_per_aa_ident = None
            batch['trans_t'] = batch['t']
            batch['rot_t'] = batch['t']

        batch = corrupter.corrupt_dense_batch(batch, self.identity_rot_noise)

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
        loss_dict = self._loss_step(batch, outputs)
        if return_outputs:
            return loss_dict, batch, outputs
        return loss_dict

    
    def training_step(self, batch, batch_idx):
        if self.global_step > 0:
            if self.ema is not None:
                self.ema.update_parameters(self.model)
            if self.ema_long is not None:
                self.ema_long.update_parameters(self.model, self.global_step - 1)
            if self.ema_short is not None:
                self.ema_short.update_parameters(self.model, self.global_step - 1)

        if batch_idx == 0:
            self._epoch_sample_train_batch = _detach_cpu_batch(batch)

        has_sequential = any(t.name == 'mol_sequential_scaffolding' for t in batch['task'])

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

        num_gpus = max(1, self.trainer.world_size)
        n_samples = max(5, 2 * num_gpus) // num_gpus
        write_pdbs = (
            self.trainer.current_epoch % 5 == 0
            and batch_idx == 0  # only first val batch
        )
        run_epoch_sample = (
            self.trainer.current_epoch % self.epoch_sample_every_n_epochs == 0
            and batch_idx == 0
        )
        if batch_idx == 0:
            self._epoch_sample_val_batch = _detach_cpu_batch(batch)

        for t_val in (0.0, 0.5):
            batch_t = batch.copy()
            batch_t["t"] = torch.full((*batch_t["t"].shape,), t_val, device=device)

            if write_pdbs:
                loss_dict, batch_out, outputs = self._shared_step(batch_t, return_outputs=True)
                pdb_data = self._collect_val_pdb_data(batch_out, outputs, t_val, n_samples=n_samples)
                del batch_out, outputs
                torch.cuda.empty_cache()
                self._write_val_pdbs(pdb_data)
            else:
                loss_dict = self._shared_step(batch_t)

            # log under separate namespace
            self._log_losses(
                loss_dict,
                batch_t,
                stage=f"val/t_{t_val}",
            )

        if run_epoch_sample:
            if self._epoch_sample_train_batch is not None:
                self._run_epoch_sample(self._epoch_sample_train_batch, "train")
            self._run_epoch_sample(_detach_cpu_batch(batch), "val")

    def _collect_val_pdb_data(self, batch, outputs, t_val: float, n_samples: int = 5):
        """Extract all GPU tensors to CPU numpy."""
        epoch = self.trainer.current_epoch
        rank = self.trainer.global_rank
        out_dir = os.path.join(self.trainer.log_dir, f"val_pdbs/epoch_{epoch:04d}/t_{t_val}")
        os.makedirs(out_dir, exist_ok=True)

        B = batch["t"].shape[0]
        rigids = batch['rigids']
        # All .cpu().numpy() calls happen here in the main thread — no CUDA ops in the writer thread
        rigids_mask          = rigids['rigids_mask'].cpu().numpy().astype(bool)
        rigids_noising_mask  = rigids['rigids_noising_mask'].cpu().numpy().astype(bool)
        gt_trans             = ru.Rigid.from_tensor_7(rigids['rigids_1']).get_trans().cpu().numpy()
        pred_trans           = outputs['denoised_rigids'].get_trans().cpu().numpy()
        ref_elements         = rigids['rigids_ref_element'].cpu().numpy()
        rigids_seq_idx       = rigids['rigids_seq_idx'].cpu().numpy()
        rigids_to_token      = rigids['rigids_to_token'].cpu().numpy()
        rigids_sc_idx        = rigids['rigids_sidechain_idx'].cpu().numpy()
        is_atom_mask         = rigids['rigids_is_atom_mask'].cpu().numpy().astype(bool)
        gt_rigid7            = rigids['rigids_1'].cpu().numpy()
        pred_rigid7          = outputs['denoised_rigids'].to_tensor_7().cpu().numpy()
        res_types_tok        = batch['token']['res_type'].cpu().numpy()
        asym_ids_tok         = batch['token']['asym_id'].cpu().numpy()
        token_residue_idx    = batch['token']['residue_idx'].cpu().numpy()

        pred_rigid7_display = np.where(rigids_noising_mask[:, :, None], pred_rigid7, gt_rigid7)

        noised_heavy_mask = rigids_mask & rigids_noising_mask & (ref_elements != 1)
        n_noised = noised_heavy_mask.sum(axis=-1).clip(min=1)
        se = np.square(pred_trans - gt_trans).sum(axis=-1)
        per_sample_mse = (se * noised_heavy_mask).sum(axis=-1) / n_noised

        return dict(
            out_dir=out_dir, B=B, n_samples=n_samples, rank=rank,
            rigids_mask=rigids_mask, rigids_noising_mask=rigids_noising_mask,
            ref_elements=ref_elements, is_atom_mask=is_atom_mask,
            rigids_sc_idx=rigids_sc_idx, rigids_to_token=rigids_to_token,
            rigids_seq_idx=rigids_seq_idx, res_types_tok=res_types_tok,
            asym_ids_tok=asym_ids_tok, token_residue_idx=token_residue_idx,
            gt_rigid7=gt_rigid7,
            pred_rigid7_display=pred_rigid7_display, per_sample_mse=per_sample_mse,
            record_ids=batch.get('record_id', [None] * B),
            task_name=batch['task'][0].name if batch.get('task') else "unknown",
        )

    def _write_val_pdbs(self, data: dict):
        out_dir   = data['out_dir']
        B         = data['B']
        n_samples = data['n_samples']
        rank      = data['rank']
        rigids_mask         = data['rigids_mask']
        ref_elements        = data['ref_elements']
        is_atom_mask        = data['is_atom_mask']
        gt_rigid7           = data['gt_rigid7']
        pred_rigid7_display = data['pred_rigid7_display']
        per_sample_mse      = data['per_sample_mse']
        record_ids          = data['record_ids']

        for i in range(min(n_samples, B)):
            has_protein = ((ref_elements[i] == -1) & rigids_mask[i]).any()
            has_ligand  = (is_atom_mask[i] & rigids_mask[i] & (ref_elements[i] != 1)).any()
            if not has_protein and not has_ligand:
                continue

            mse_val = float(per_sample_mse[i])
            rid = record_ids[i] if record_ids[i] is not None else f"sample_{i:02d}"
            rid = rid.replace("/", "_").replace(" ", "_")
            task_name = data.get('task_name', 'unknown')
            path = os.path.join(out_dir, f"{task_name}_{rid}_rank{rank}_mse={mse_val:.3f}.pdb")
            write_val_pdb(
                gt_rigid7[i],
                pred_rigid7_display[i],
                rigids_mask[i],
                ref_elements[i],
                is_atom_mask[i],
                data['rigids_sc_idx'][i],
                data['rigids_to_token'][i],
                data['rigids_seq_idx'][i],
                data['res_types_tok'][i],
                data['asym_ids_tok'][i],
                path,
                token_residue_idx=data['token_residue_idx'][i],
            )

    def on_validation_epoch_end(self):
        for entry in self._pending_traj_writes:
            i = entry['i']
            for traj_np, traj_path in [
                (entry['prot_traj_np'], entry['noise_path']),
                (entry['clean_traj_np'], entry['clean_path']),
            ]:
                with open(traj_path, 'w') as f:
                    for step_idx, step_rigid7 in enumerate(traj_np):
                        step_records = _build_all_atom_records(
                            step_rigid7[i], entry['rigids_mask_np'][i], entry['ref_elements'][i], entry['is_atom_mask'][i],
                            entry['sc_idx_np'], entry['to_tok_np'], entry['seq_idx_np'],
                            entry['res_type_np'], entry['asym_id_np'],
                            token_residue_idx=entry['res_idx_np'],
                        )
                        _write_model_block(f, step_records, step_idx + 1)
                    f.write("END\n")
        self._pending_traj_writes.clear()

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
                sync_dist=False,
            )

    def on_train_epoch_end(self):
        pass

    @torch.no_grad()
    def _run_epoch_sample(self, batch_cpu, split):
        """Run full ODE integration from t=0 on a stashed batch and write a PDB."""
        epoch = self.trainer.current_epoch
        rank = self.trainer.global_rank
        device = self.device

        model = self.ema.module if (self.use_ema and self.ema is not None) else self.model
        model.eval()

        batch = _move_to_device(batch_cpu, device)

        rigids_data = batch['rigids']
        rigids_mask = rigids_data['rigids_mask']
        rigids_noising_mask = rigids_data['rigids_noising_mask']

        # Generate noise independently of x_1 (same as sampling datamodule)
        gt_trans = ru.Rigid.from_tensor_7(rigids_data['rigids_1']).get_trans()
        per_sample_std = batch.get('trans_prior_std', None)
        if per_sample_std is not None:
            std = torch.where(
                torch.isnan(per_sample_std),
                torch.full_like(per_sample_std, self.corrupter.trans_prior_std),
                per_sample_std,
            )
            trans_t = torch.randn_like(gt_trans) * std[:, None, None]
        else:
            trans_t = torch.randn_like(gt_trans) * self.corrupter.trans_prior_std
        trans_t = trans_t - trans_t.mean(dim=1, keepdim=True)
        trans_t = torch.where(rigids_noising_mask[..., None], trans_t, gt_trans)

        eye = torch.eye(3, device=device, dtype=torch.float32)
        rotmats_t = eye[None, None].expand_as(
            ru.Rigid.from_tensor_7(rigids_data['rigids_1']).get_rots().get_rot_mats()
        ).clone()

        # Pre-fill rigids_1 with noise so EulerIntegrator uses it as the start
        gt_rigid7 = rigids_data['rigids_1'].clone()
        rigids_data['rigids_1'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t), trans=trans_t
        ).to_tensor_7()

        integrator = EulerIntegrator(
            wrapped_model=BaseModelForward(model),
            diffeq=BaseEulerODEStep(),
            no_rot_sampling=not self.use_rot_vf_loss,
        )
        ts = torch.linspace(0.0, 1.0, self.epoch_sample_num_steps + 1)
        prot_traj, clean_traj, final_denoiser_out = integrator.sample(batch, ts)
        final_rigids = final_denoiser_out['denoised_rigids']

        # Pre-convert trajectories to CPU numpy now, before any file IO,
        # so CUDA syncs happen here (inside validation_step) rather than later.
        prot_traj_np = [
            ru.Rigid(rots=ru.Rotation(rot_mats=r), trans=t).to_tensor_7().cpu().numpy()
            for t, r, _ in prot_traj
        ]
        clean_traj_np = [
            ru.Rigid(rots=ru.Rotation(rot_mats=r), trans=t).to_tensor_7().cpu().numpy()
            for t, r, _ in clean_traj
        ]

        # Restore GT for MSE computation
        rigids_data['rigids_1'] = gt_rigid7

        out_dir = os.path.join(self.trainer.log_dir, f"epoch_samples/epoch_{epoch:04d}/{split}")
        os.makedirs(out_dir, exist_ok=True)

        gt_rigids_t = ru.Rigid.from_tensor_7(rigids_data['rigids_1'])
        gt_rigids_t = sym_permute_gt_rigids(final_rigids.get_trans(), gt_rigids_t, batch)
        gt_trans_t = gt_rigids_t.get_trans()
        pred_trans_t = final_rigids.get_trans()

        ref_elements_t = rigids_data['rigids_ref_element']
        noised_heavy_mask_t = (rigids_mask & rigids_noising_mask & (ref_elements_t != 1)).bool()

        gt_rigid7 = rigids_data['rigids_1'].cpu().numpy()
        pred_rigid7 = final_rigids.to_tensor_7().cpu().numpy()
        rigids_mask_np = rigids_mask.cpu().numpy().astype(bool)
        noising_mask_np = rigids_noising_mask.cpu().numpy().astype(bool)
        ref_elements = rigids_data['rigids_ref_element'].cpu().numpy()
        is_atom_mask = rigids_data['rigids_is_atom_mask'].cpu().numpy().astype(bool)
        pred_rigid7_display = np.where(noising_mask_np[:, :, None], pred_rigid7, gt_rigid7)

        task = batch.get('task', [None])[0]
        task_name = task.name if task is not None else "unknown"

        record_ids = batch.get('record_id', [None] * gt_rigid7.shape[0])

        n_batch = gt_rigid7.shape[0]
        for i in range(n_batch):
            n_noised = noised_heavy_mask_t[i].long().sum().clamp(min=1)

            align_mask_i = noised_heavy_mask_t[i]
            align_batch_i = torch.zeros(align_mask_i.sum(), dtype=torch.long, device=align_mask_i.device)
            _, _, R_i = align_structures(pred_trans_t[i][align_mask_i], align_batch_i, gt_trans_t[i][align_mask_i])
            pred_mean_i = pred_trans_t[i][align_mask_i].mean(0)
            gt_mean_i = gt_trans_t[i][align_mask_i].mean(0)
            pred_aligned_i = (pred_trans_t[i] - pred_mean_i) @ R_i[0] + gt_mean_i

            se_i = torch.square(pred_trans_t[i] - gt_trans_t[i]).sum(dim=-1)
            integration_mse = float((se_i * noised_heavy_mask_t[i]).sum() / n_noised)
            se_kabsch_i = torch.square(pred_aligned_i - gt_trans_t[i]).sum(dim=-1)
            integration_mse_kabsch = float((se_kabsch_i * noised_heavy_mask_t[i]).sum() / n_noised)

            rid = record_ids[i] if record_ids[i] is not None else f"sample_{i}"
            rid = rid.replace("/", "_").replace(" ", "_")
            path = os.path.join(out_dir, f"{task_name}_{rid}_mse={integration_mse:.3f}_kabsch={integration_mse_kabsch:.3f}.pdb")

            write_val_pdb(
                gt_rigid7[i],
                pred_rigid7_display[i],
                rigids_mask_np[i],
                ref_elements[i],
                is_atom_mask[i],
                rigids_data['rigids_sidechain_idx'].cpu().numpy()[i],
                rigids_data['rigids_to_token'].cpu().numpy()[i],
                rigids_data['rigids_seq_idx'].cpu().numpy()[i],
                batch['token']['res_type'].cpu().numpy()[i],
                batch['token']['asym_id'].cpu().numpy()[i],
                path,
                token_residue_idx=batch['token']['residue_idx'].cpu().numpy()[i],
            )

            pred_rigid7_aligned = pred_rigid7_display[i].copy()
            pred_rigid7_aligned[noising_mask_np[i], 4:] = pred_aligned_i.cpu().numpy()[noising_mask_np[i]]
            write_val_pdb(
                gt_rigid7[i],
                pred_rigid7_aligned,
                rigids_mask_np[i],
                ref_elements[i],
                is_atom_mask[i],
                rigids_data['rigids_sidechain_idx'].cpu().numpy()[i],
                rigids_data['rigids_to_token'].cpu().numpy()[i],
                rigids_data['rigids_seq_idx'].cpu().numpy()[i],
                batch['token']['res_type'].cpu().numpy()[i],
                batch['token']['asym_id'].cpu().numpy()[i],
                path.replace('.pdb', '_kabsch.pdb'),
                token_residue_idx=batch['token']['residue_idx'].cpu().numpy()[i],
            )

            sc_idx_np = rigids_data['rigids_sidechain_idx'].cpu().numpy()[i]
            to_tok_np = rigids_data['rigids_to_token'].cpu().numpy()[i]
            seq_idx_np = rigids_data['rigids_seq_idx'].cpu().numpy()[i]
            res_type_np = batch['token']['res_type'].cpu().numpy()[i]
            asym_id_np = batch['token']['asym_id'].cpu().numpy()[i]
            res_idx_np = batch['token']['residue_idx'].cpu().numpy()[i]

            self._pending_traj_writes.append({
                'prot_traj_np': prot_traj_np,
                'clean_traj_np': clean_traj_np,
                'noise_path': path.replace('.pdb', '_traj_noise.pdb'),
                'clean_path': path.replace('.pdb', '_traj_clean.pdb'),
                'i': i,
                'rigids_mask_np': rigids_mask_np,
                'ref_elements': ref_elements,
                'is_atom_mask': is_atom_mask,
                'sc_idx_np': sc_idx_np,
                'to_tok_np': to_tok_np,
                'seq_idx_np': seq_idx_np,
                'res_type_np': res_type_np,
                'asym_id_np': asym_id_np,
                'res_idx_np': res_idx_np,
            })

            self.log(f"epoch_sample/{split}/{task_name}/integration_mse", integration_mse, prog_bar=False, sync_dist=False)
            self.log(f"epoch_sample/{split}/{task_name}/integration_mse_kabsch", integration_mse_kabsch, prog_bar=False, sync_dist=False)
            self._log.info(f"Epoch {epoch} integration sample ({split}) written: {path} (mse={integration_mse:.3f}, kabsch={integration_mse_kabsch:.3f})")

        model.train()

    #     return loss_dict
    # def training_step(self, batch, batch_idx):
    #     # update EMA
    #     if self.ema is not None and self.global_step > 0:
    #         self.ema.update_parameters(self.model)
    #     if self.ema_long is not None and self.global_step > 0:
    #         self.ema_long.update_parameters(self.model, self.global_step-1)
    #     if self.ema_short is not None and self.global_step > 0:
    #         self.ema_short.update_parameters(self.model, self.global_step-1)

    #     has_sequential = any(t.name == 'mol_sequential_scaffolding' for t in batch['task'])

    #     # corrupt data
    #     corrupter = self.corrupter
    #     if self.t_sched is not None:
    #         t = batch['t']
    #         res_type = batch['token']['res_type']
    #         rigid_parent_res = batch['rigids']['rigids_seq_idx']
    #         with torch.no_grad():
    #             t_per_aa_ident = self.t_sched(t)
    #         _, dlogk_per_aa_ident = torch.func.jvp(
    #             lambda x: self.t_sched(1-x),
    #             (t,),
    #             (torch.ones_like(t),)
    #         )
    #         velocity_factor_per_aa_ident = dlogk_per_aa_ident / (1 - t_per_aa_ident.clip(max=0.99))
    #         t_per_aa_ident.requires_grad = True
    #         t_per_res = gather_helper(t_per_aa_ident, res_type)
    #         velocity_factor_per_res = gather_helper(velocity_factor_per_aa_ident, res_type)
    #         t_per_rigid = gather_helper(t_per_res, rigid_parent_res)
    #         velocity_factor_per_rigid = gather_helper(velocity_factor_per_res, rigid_parent_res)
    #         batch['trans_t'] = t_per_rigid[..., 0]
    #         batch['rot_t'] = t_per_rigid[..., 1]
    #         batch['velocity_factor_per_rigid'] = velocity_factor_per_rigid
    #     else:
    #         t_per_aa_ident = None
    #         batch['trans_t'] = batch['t']
    #         batch['rot_t'] = batch['t']
    #     batch = corrupter.corrupt_dense_batch(batch)

    #     self_conditioning = None
    #     self_folding = None
    #     if (
    #         self.model.self_conditioning
    #         and np.random.uniform() < self.self_condition_rate
    #     ):
    #         with torch.no_grad():
    #             self_conditioning = self.model(batch)
    #             if self.apply_self_folding:
    #                 pred_seq_batch = self._generate_folding_batch(
    #                     batch, self_conditioning["pred_seq"]
    #                 )
    #                 self_folding = self.model(pred_seq_batch)

    #     outputs = self.model(batch, self_conditioning, self_folding)

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

    #         loss_by_task[task.name + "_loss"].append(loss_dict['loss_per_batch'][i])
    #         loss_by_task[task.name + "_seq_loss"].append(loss_dict["seq_loss"][i])
    #         loss_by_task[task.name + "_frame_vf_loss"].append(loss_dict['frame_vf_loss'][i])
    #         loss_by_task[task.name + "_frame_vf_loss_unscaled"].append(loss_dict['frame_vf_loss_unscaled'][i])

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
    #             sync_dist=True)

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
    #             sync_dist=True)

    #     self.log_dict(
    #         log_dict,
    #         on_step=None,
    #         on_epoch=True,
    #         prog_bar=True,
    #         logger=True,
    #         batch_size=t.shape[0],
    #         sync_dist=True)

    #     if self.t_sched is None:
    #         return loss_dict
    #     else:
    #         opt_ret = self.optimizers()
    #         if isinstance(opt_ret, dict):
    #             opt = opt_ret['optimizer']
    #         else:
    #             opt = opt_ret
    #         opt.zero_grad()
    #         loss = loss_dict['loss']
    #         self.manual_backward(loss)
    #         per_res_trans_vf_loss = loss_dict['raw_trans_vf_loss']
    #         per_res_rot_vf_loss = loss_dict['raw_rot_vf_loss']
    #         stack_vf_loss = torch.stack([per_res_trans_vf_loss, per_res_rot_vf_loss], dim=-1).detach()

    #         t = batch['t']
    #         res_type = batch['token']['res_type']
    #         rigid_parent_res = batch['rigids']['rigids_seq_idx']
    #         t_per_aa_ident_redo = self.t_sched(t)
    #         t_per_res_redo = gather_helper(t_per_aa_ident_redo, res_type)
    #         t_per_rigid_redo = gather_helper(t_per_res_redo, rigid_parent_res)
    #         assert t_per_aa_ident is not None and t_per_aa_ident.grad is not None
    #         t_per_res_redo_grad = gather_helper(t_per_aa_ident.grad, res_type)
    #         t_per_rigid_redo_grad = gather_helper(t_per_res_redo_grad, rigid_parent_res)
    #         t_sched_grad = 2 * t_per_rigid_redo_grad * stack_vf_loss
    #         self.manual_backward(t_per_rigid_redo, gradient=t_sched_grad)
    #         # with torch.no_grad():
    #         #     torch.set_printoptions(threshold=10000000)
    #         #     for name, parameter in self.t_sched.named_parameters():
    #         #         print(name, parameter.grad)
    #         with torch.no_grad():
    #             torch.set_printoptions(threshold=10000000)
    #             print(self.t_sched(torch.arange(10, device=t.device)[..., None] / 10))
    #             exit()
    #         opt.step()


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

        if self.postalign_noise:
            rigids_data = inputs['rigids']
            rigids_mask = rigids_data['rigids_mask']
            rigids_noising_mask = rigids_data['rigids_noising_mask']
            align_mask = (rigids_mask * rigids_noising_mask).bool()
            num_batch = rigids_mask.shape[0]
            align_batch = torch.arange(num_batch, device=align_mask.device)[:, None].expand_as(rigids_mask)
            align_batch = align_batch[align_mask]

            pred_trans = outputs['denoised_rigids'].get_trans()
            gt_trans = ru.Rigid.from_tensor_7(rigids_data['rigids_1']).get_trans()

            with torch.no_grad():
                _, _, align_rot_mats = align_structures(
                    pred_trans[align_mask],
                    align_batch,
                    gt_trans[align_mask],
                )
                if align_rot_mats.shape[0] != num_batch:
                    num_pad = num_batch - align_rot_mats.shape[0]
                    eye = torch.eye(3, device=align_rot_mats.device, dtype=align_rot_mats.dtype)
                    align_rot_mats = torch.cat([align_rot_mats, eye[None].expand(num_pad, -1, -1)], dim=0)

            aligned_trans = torch.einsum("bni,bij->bnj", pred_trans, align_rot_mats)
            outputs = dict(outputs)
            outputs['denoised_rigids'] = ru.Rigid(
                rots=outputs['denoised_rigids'].get_rots(),
                trans=aligned_trans,
            )

        frame_fm_loss_dict = multiframe_fm_loss_dense_batch(
            inputs, outputs, sep_rot_loss=not self.learnable_noise_schedule, # use_euclidean_for_rots=self.use_euclidean_for_rots,
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
            t_sched_weight=inputs['velocity_factor_per_rigid'] if 'velocity_factor_per_rigid' in inputs else None,
            compute_interface_fafe=self.use_interface_fafe_loss,
            compute_interchain_fafe=self.use_interchain_fafe_loss,
            brownian_rot_path=self.use_brownian_rot_path_loss,
            # stabilize_high_t_loss=self.use_stabilized_t_loss
        )

        frame_vf_loss = (
            frame_fm_loss_dict["raw_trans_vf_loss"] +
            frame_fm_loss_dict["raw_rot_vf_loss"]
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
        ) if not self.no_seq_loss else 0

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
        min_conformer_active = outputs.get('min_conformer_pred_val') is not None
        if min_conformer_active:
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

        if self.use_interface_fafe_loss:
            loss = loss + 0.5 * frame_fm_loss_dict['scaled_interface_fafe']

        if self.use_interchain_fafe_loss:
            loss = loss + 0.5 * frame_fm_loss_dict['scaled_interchain_fafe']

        if self.distogram_loss:
            loss = loss + 0.1 * frame_fm_loss_dict['distogram_cross_entropy']

        if self.pred_local_fafe_loss:
            loss = loss + 0.1 * frame_fm_loss_dict['pred_local_trans_fafe_loss'] + 0.1 * frame_fm_loss_dict['pred_local_rot_fafe_loss']

        if self.pae_loss:
            loss = loss + 0.1 * frame_fm_loss_dict['pae_cross_entropy']

        loss_dict = {"loss": loss.mean(), "frame_vf_loss": frame_vf_loss, "frame_vf_loss_unscaled": unscaled_frame_vf_loss}
        loss_dict['loss_per_batch'] = loss
        if min_conformer_active:
            loss_dict['min_conformer_trans_mse'] = min_conformer_trans_mse
            loss_dict['min_conformer_heavy_trans_mse'] = min_conformer_heavy_trans_mse

        # if self.t_sched is not None:
        #     self.

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
        task_name = batch['task'][0]
        non_copy_task_names = {"protein_conditioned_generate_ligand", "ligand_conditioned_generate_protein"}
        
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
            
            # if task_name in non_copy_task_names:
            #     motif_select_mask = (~token_data['seq_noising_mask'] & token_data['resolved_mask'])
            # else:
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
        if self.trainer.is_global_zero:
            # Copy Hydra config to version dir for per-run documentation
            hydra_cfg_path = '.hydra/config.yaml'
            if os.path.exists(hydra_cfg_path) and self.trainer.log_dir is not None:
                shutil.copy(hydra_cfg_path, os.path.join(self.trainer.log_dir, 'config.yaml'))
            # Log effective batch size
            dm = self.trainer.datamodule
            if dm is not None and hasattr(dm, 'batch_size'):
                self.log('batch_size', float(dm.batch_size), rank_zero_only=True)

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

        if self.t_sched is not None:
            optimizer = self.optim(
                list(params)
                + list(self.t_sched.parameters())
            )
        else:
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
        integrator_init,
        run_cfg,
        strict_weight_loading=True,
    ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        # the actual ema params don't matter here, we just wanna be able to load the weights
        self.ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
        self.integrator: Integrator = integrator_init(self.ema.module)
        self.run_cfg = run_cfg
        if not strict_weight_loading:
            self.strict_loading = False

    def predict_step(self, batch, batch_idx):
        # Set-up time
        assert (batch['t'] == batch['t'][0]).all(), "batch t must all be the same"
        ts = torch.linspace(
            float(batch['t'][0]),
            1.0,
            self.run_cfg['num_timesteps']
        )

        if self.run_cfg.get('identity_rot_noise', False):
            B, N = batch['rigids']['rigids_1'].shape[:2]
            eye = torch.eye(3, device=batch['rigids']['rigids_1'].device)
            identity_quat = ru.Rotation(rot_mats=eye.expand(B, N, 3, 3)).get_quats()
            batch['rigids']['rigids_1'][..., :4] = identity_quat

        clean_traj, prot_traj, final_denoiser_out = self.integrator.sample(
            batch,
            ts
        )

        ret = self._post_process_outputs(
            batch,
            prot_traj,
            clean_traj,
            final_denoiser_out
        )

        return ret

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

        task_name = batch['task'][0]
        non_copy_task_names = {"protein_conditioned_generate_ligand", "ligand_conditioned_generate_protein"}

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

            # if task_name in non_copy_task_names:
            #     motif_select_mask = (~token_data['seq_noising_mask'] & token_data['resolved_mask'])
            # else:
            motif_select_mask = (token_is_copy_mask & token_data['resolved_mask'])

            motif_seq_fixed = (motif_select_mask & ~token_data['seq_noising_mask'])
            # actual residue idxs and chain asym ids
            motif_res_idx = token_data["res_idx"][motif_idx.numpy(force=True)]
            fixed_bb_res_idx = motif_res_idx[motif_select_mask]
            fixed_seq_res_idx = motif_res_idx[motif_seq_fixed]

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


def _write_traj(entry):
    path, traj = entry
    model_strs = []
    for i, traj_data in enumerate(traj):
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
        model_strs.append(f"MODEL        {i}\n")
        model_strs.append(pdb_str.split("END")[0])
        model_strs.append(f"ENDMDL       \n")
    model_strs.append("END\n")

    model_pdb_str = "".join(model_strs)
    with open(path, "w") as fp:
        fp.write(model_pdb_str)


class PDBWriter(BasePredictionWriter):
    def __init__(self, output_dir, run_cfg):
        super().__init__(write_interval="batch_and_epoch")
        self.output_dir = output_dir
        self.samples_dir = os.path.join(output_dir, "samples")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        self.traj_dir = os.path.join(output_dir, "traj")
        self.run_cfg = run_cfg

        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.traj_dir, exist_ok=True)

        self.use_task_name_labels = run_cfg['use_task_name_labels']
        self.samples_metadata = {}

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        pwd = os.getcwd()
        os.chdir(self.samples_dir)

        samples_metadata = {}

        traj_list = []

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

            if self.use_task_name_labels:
                sample_name = sample_data['name'] + f"_gpu{rank}_batch{batch_idx}_idx{curr_sample_id}" #.pdb"
            else:
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

                clean_traj_name = os.path.join(
                    self.traj_dir,
                    sample_name + f"_clean_traj.pdb"
                )
                prot_traj_name = os.path.join(
                    self.traj_dir,
                    sample_name + f"_prot_traj.pdb"
                )

                traj_list.append(
                    (
                        clean_traj_name,
                        clean_traj
                    )
                )
                traj_list.append(
                    (
                        prot_traj_name,
                        prot_traj
                    )
                )
                # _write_traj(clean_traj_name, clean_traj)
                # _write_traj(prot_traj_name, prot_traj)

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

        for traj_entry in traj_list:
            _write_traj(traj_entry)
        # results = p_tqdm.p_map(_write_traj, traj_list, num_cpus=8)
        # print(results)

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
            assert d is not None
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

    def teardown(self, trainer, pl_module, stage):
        super().teardown(trainer, pl_module, stage)
