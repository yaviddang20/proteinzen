import logging
import math
from functools import partial

import numpy as np
import tqdm
import torch
import tree
import lightning as L

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from proteinzen.harness import TrainingHarness
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.utils.coarse_grain import compute_atom14_from_cg_frames

from .utils import gen_pbar_str
from .ema import EMAModel


def t_stratified_loss(batch_t, batch_loss, num_bins=4, loss_name=None):
    """Stratify loss by binning t."""
    batch_t = batch_t.numpy(force=True)
    batch_loss = batch_loss.numpy(force=True)
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


class ProteinModule(L.LightningModule):
    def __init__(self,
                 model,
                 optim,
                 training_harness: TrainingHarness,
                 val_dir="validation",
                 use_amp=False,
                 use_cosine_lr_sched=False,
                 cosine_warmup_steps=0,
                 cosine_total_steps=1e6,
                 use_ema=False,
                 ema_decay=0.999,
                 use_posthoc_ema=False,
    ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        self.optim = optim
        self.training_harness = training_harness
        self.val_dir = val_dir
        self.use_amp = use_amp
        self.use_cosine_lr_sched = use_cosine_lr_sched
        self.cosine_warmup_steps = cosine_warmup_steps
        self.cosine_total_steps = cosine_total_steps
        self.use_ema = use_ema
        self.use_posthoc_ema = use_posthoc_ema
        if self.use_amp:
            torch.set_float32_matmul_precision("medium")

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


    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if hasattr(self.trainer.train_dataloader.batch_sampler, "epoch"):
            self.trainer.train_dataloader.batch_sampler.set_epoch(self.trainer.current_epoch)


    def training_step(self, batch):
        training_harness = self.training_harness

        if self.ema is not None and self.global_step > 0:
            self.ema.update_parameters(self.model)
        if self.ema_long is not None and self.global_step > 0:
            self.ema_long.update_parameters(self.model, self.global_step-1)
        if self.ema_short is not None and self.global_step > 0:
            self.ema_short.update_parameters(self.model, self.global_step-1)

        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = training_harness.run_eval(self.model, batch)
            loss_dict = training_harness.compute_loss(batch, outputs)
        else:
            outputs = training_harness.run_eval(self.model, batch)
            loss_dict = training_harness.compute_loss(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.round(torch.mean(x), decimals=3) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict = {
            ("train/" + key): value
            for key, value in
            sorted(log_dict.items(), key = lambda x: x[0])
        }
        t = batch['t']
        for loss_name, loss_list in loss_dict.items():
            if loss_name in ['loss', 'frameflow_loss']:
                continue
            if not loss_name.startswith("pt_") and not loss_name.startswith("latent_"):
                continue
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
                batch_size=batch.num_graphs,
                sync_dist=False)

        self.log_dict(
            log_dict,
            on_step=None,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)

        # self._log.info(gen_pbar_str(loss_dict))
        return loss_dict

    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            norms = []
            norm_dict = {}
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    n = torch.linalg.vector_norm(p.grad.view(-1), dim=-1)
                    norms.append(n)
                    norm_dict[name] = n.item()
            # print(json.dumps(norm_dict, indent=4))
            total_norm = torch.linalg.vector_norm(
                torch.stack(norms, dim=0),
                dim=0
            )

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

    def validation_step(self, batch, batch_idx):
        training_harness = self.training_harness
        outputs = training_harness.run_eval(self.model, batch)
        loss_dict = training_harness.compute_loss(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.round(torch.mean(x), decimals=3) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict = {
            ("val/" + key): value
            for key, value in
            sorted(log_dict.items(), key = lambda x: x[0])
        }
        t = batch['t']
        for loss_name, loss_list in loss_dict.items():
            if loss_name in ['loss', 'frameflow_loss']:
                continue
            if not loss_name.startswith("pt_") and not loss_name.startswith("latent_"):
                continue
            stratified_losses = t_stratified_loss(
                t, loss_list, loss_name=loss_name)
            stratified_losses = {
                f"val/{k}": torch.round(torch.as_tensor(v, device=log_dict['val/loss'].device), decimals=3)
                for k,v in stratified_losses.items()
            }
            self.log_dict(
                stratified_losses,
                prog_bar=False,
                logger=True,
                on_step=None,
                on_epoch=True,
                batch_size=batch.num_graphs,
                sync_dist=False)

        self.log_dict(
            log_dict,
            on_step=None,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)

        # self._log.info(gen_pbar_str(loss_dict))
        return loss_dict


    def predict_step(self, batch, batch_idx):
        training_harness = self.training_harness
        if self.use_ema:
            # TODO: remove compatibility code
            if isinstance(batch, dict):
                outputs = training_harness.run_predict(self.ema.module, batch, device=self.device)
            else:
                outputs = self._predict_step(self.ema.module, batch)
        else:
            outputs = training_harness.run_predict(self.model, batch, device=self.device)
            # outputs = self._predict_step(self.model, batch)
        return outputs

    def _predict_step(
        self,
        model,
        batch
    ):
        frame_noiser = self.training_harness.frame_noiser
        # Set-up time
        ts = torch.linspace(frame_noiser._cfg.min_t, 1.0, frame_noiser._sample_cfg.num_timesteps)

        res_data = batch['residue']
        rigids_0 = ru.Rigid.from_tensor_7(res_data['rigids_t'])
        trans_0 = rigids_0.get_trans()
        rotmats_0 = rigids_0.get_rots().get_rot_mats()
        rigids_noising_mask = res_data['rigids_noising_mask']

        t_1 = ts[0]

        total_num_res = res_data.num_nodes
        device = self.device
        prot_traj = [(
            trans_0,  # trans
            rotmats_0,  # rot
            torch.ones((total_num_res, 21), device=device).float(),  # seq logits
            compute_atom14_from_cg_frames(
                ru.Rigid(ru.Rotation(rot_mats=rotmats_0), trans_0),
                res_data.res_mask,
                torch.zeros_like(res_data.seq),
                cg_version=model.cg_version
            )
        )]
        clean_traj = []
        denoiser_out = None

        for t_2 in tqdm.tqdm(ts[1:]):
            d_t = t_2 - t_1
            # Run model.
            trans_t_1, rotmats_t_1, _, _ = prot_traj[-1]
            t_hat, d_t_hat, trans_t_hat = frame_noiser._trans_churn(
                d_t,
                t_1,
                trans_t_1,
                noising_mask=rigids_noising_mask,
            )
            _, _, rotmats_t_hat = frame_noiser._rot_churn(
                d_t,
                t_1,
                rotmats_t_1,
                noising_mask=rigids_noising_mask,
            )

            res_data["trans_t"] = trans_t_hat
            res_data["rotmats_t"] = rotmats_t_hat
            res_data['rigids_t'] = ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_t_hat),
                trans=trans_t_hat
            ).to_tensor_7()
            t = torch.ones(batch.num_graphs, device=device) * t_hat
            batch["t"] = t
            batch["rigidwise_t"] = torch.ones((total_num_res, frame_noiser.rigids_per_res), device=device) * t_hat

            denoiser_out = model(batch, self_condition=denoiser_out)

            # Process model output.
            pred_rigids = denoiser_out['final_rigids']
            pred_trans_1 = pred_rigids.get_trans()
            pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
            # if torch.isnan(pred_trans_1).any() or torch.isnan(pred_rotmats_1).any():
            #     print("pred", t_1)
            #     exit()

            clean_traj.append(
                (
                    pred_trans_1.detach().cpu(),
                    pred_rotmats_1.detach().cpu(),
                    denoiser_out['pred_seq'].detach().cpu(),
                    denoiser_out['denoised_atom14'].detach().cpu(),
                    denoiser_out['denoised_atom14_gt_seq'].detach().cpu()
                )
            )

            # Take reverse step
            trans_t_2 = frame_noiser._trans_euler_step(
                d_t_hat,
                t_hat,
                pred_trans_1,
                trans_t_hat,
                noising_mask=rigids_noising_mask,
                add_noise=False
            )
            rotmats_t_2 = frame_noiser._rots_euler_step(
                d_t_hat,
                t_hat,
                pred_rotmats_1,
                rotmats_t_hat,
                noising_mask=rigids_noising_mask,
                add_noise=False
            )
            # if torch.isnan(trans_t_2).any():
            #     print("trans step", t_1)
            #     exit()

            # if torch.isnan(rotmats_t_2).any():
            #     print("rot step", t_1)
            #     exit()

            prot_traj.append(
                (trans_t_2,
                 rotmats_t_2,
                 denoiser_out["pred_seq"].detach().cpu(),
                 compute_atom14_from_cg_frames(
                     ru.Rigid(ru.Rotation(rot_mats=rotmats_t_2), trans_t_2),
                     res_data.res_mask,
                     denoiser_out["pred_seq"],
                     cg_version=model.cg_version
                 )
                )
            )
            t_1 = t_2

            if not model.self_conditioning:
                denoiser_out = None

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1, _, _ = prot_traj[-1]
        res_data["trans_t"] = trans_t_1
        res_data["rotmats_t"] = rotmats_t_1
        res_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        t = torch.ones(batch.num_graphs, device=device) * t_1
        batch["t"] = t
        batch["rigidwise_t"] = torch.ones((total_num_res, frame_noiser.rigids_per_res), device=device) * t_1

        denoiser_out = model(batch, self_condition=denoiser_out)

        # Process model output.
        pred_rigids = denoiser_out['final_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()
        decoded_struct = denoiser_out['denoised_atom14']
        argmax_seq = denoiser_out['pred_seq'].detach().cpu()

        clean_traj.append(
            (
                pred_trans_1.detach().cpu(),
                pred_rotmats_1.detach().cpu(),
                denoiser_out['pred_seq'].detach().cpu(),
                denoiser_out['denoised_atom14'].detach().cpu(),
                denoiser_out['denoised_atom14_gt_seq'].detach().cpu()
            )
        )

        # # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrotpsi_to_atom37(prot_traj, res_data.res_mask)
        # clean_atom37_traj = all_atom.transrotpsi_to_atom37(clean_traj, res_data.res_mask)
        num_res = res_data['num_res'].cpu()
        res_mask = res_data['res_mask'].cpu()
        def _package_traj(traj, idx):
            # we take the particular index of data at that timestep
            # and select only real residues
            traj_component = [t[idx][res_mask.to(t[idx].device)] for t in traj]
            # we then need to split the data into their corresponding batches
            unbatched_traj = [t.split(num_res.tolist(), dim=0) for t in traj_component]
            # and we zip everything together
            return list(zip(*unbatched_traj))

        clean_trajs = _package_traj(clean_traj, -2)
        clean_traj_seqs = _package_traj(clean_traj, -3)
        prot_trajs = _package_traj(prot_traj, -1)

        fixed_residues = (~rigids_noising_mask).all(dim=-1)[res_mask].split(num_res.tolist(), dim=0)
        # print([t.shape for t in fixed_residues])
        fixed_res_idx = [torch.arange(t.numel())[t.cpu()].long() for t in fixed_residues]
        fixed_res_chain_idx = res_data['chain_idx'][res_mask].split(num_res.tolist(), dim=0)
        fixed_res_chain_idx = [
            chain[mask]
            for chain, mask in zip(fixed_res_chain_idx, fixed_residues)
        ]

        samples = decoded_struct[
            res_mask.to(device=decoded_struct.device)
        ].split(num_res.tolist())
        seqs = argmax_seq[
            res_mask.to(device=argmax_seq.device)
        ].split(num_res.tolist())

        ret = []

        data_list = batch.to_data_list()

        for i, sample_input in enumerate(data_list):
            ret.append({
                "sample_coord": samples[i],
                "seq": seqs[i],
                "clean_traj": clean_trajs[i],
                "clean_traj_seq": clean_traj_seqs[i],
                "prot_traj": prot_trajs[i],
                "input": sample_input,
                "fixed_res_idx": fixed_res_idx[i],
                "fixed_res_chain_idx": fixed_res_chain_idx[i]
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
        else:
            return optimizer


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