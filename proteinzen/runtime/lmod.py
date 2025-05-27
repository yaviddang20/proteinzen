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
from proteinzen.data.datasets.featurize.rigid_assembler import rigids_to_atom14
from proteinzen.data.openfold import residue_constants
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.utils.coarse_grain import compute_atom14_from_cg_frames

from .utils import gen_pbar_str
from .ema import EMAModel

from .loss.multiframe import multiframe_fm_loss_dense_batch
from .loss.common import atomic_losses_dense_batch


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


DEFAULT_SEQ_WEIGHT = {
    c: 1.0
    for c in 'ACDEFGHIKLMNPQRSTVWY'
}
DEFAULT_SEQ_WEIGHT['X'] = 0.
for c in ['C', 'E', 'H', 'P', 'Q', 'R', 'W']:
    DEFAULT_SEQ_WEIGHT[c] = 2.0

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
                 use_dense_batch_loss=False,
                 use_predict_v2=False,
                 seq_weight=DEFAULT_SEQ_WEIGHT,
                 use_euclidean_for_rots=False,
                 learnable_noise_schedule=False
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
        self.use_dense_batch_loss = use_dense_batch_loss
        self.use_euclidean_for_rots = use_euclidean_for_rots
        self.learnable_noise_schedule = learnable_noise_schedule
        if learnable_noise_schedule:
            self.automatic_optimization = False

        seq_weight_tensor = torch.as_tensor([seq_weight[c] for c in residue_constants.restypes_with_x])
        self.seq_weight = seq_weight_tensor

        if use_dense_batch_loss:
            use_predict_v2 = True

        self.use_predict_v2 = use_predict_v2

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

        # def detect_nan_hook(module, input, output):
        #     if isinstance(output, torch.Tensor):
        #         if torch.isnan(output).any():
        #             print(f"NaN detected in: {module}")
        #     elif isinstance(output, (tuple, list)):
        #         for i, out in enumerate(output):
        #             if torch.is_tensor(out) and torch.isnan(out).any():
        #                 print(f"NaN detected in output {i} of: {module}")

        # for name, module in self.model.named_modules():
        #     module.register_forward_hook(detect_nan_hook)


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

        corrupter = self.training_harness.frame_noiser
        if self.learnable_noise_schedule:
            with torch.no_grad():
                l = batch['token']['token_is_protein_output_mask'].float().sum(dim=-1) / 100
                trans_t = self.model.trans_gamma_t(batch['t'], l[..., None])
                rot_t = self.model.rot_gamma_t(batch['t'], l[..., None])
            trans_t.requires_grad = True
            rot_t.requires_grad = True
            print(trans_t.view(-1).tolist(), rot_t.view(-1).tolist(), batch['t'].view(-1).tolist())
            batch['trans_t'] = trans_t
            batch['rot_t'] = rot_t
        else:
            batch['trans_t'] = batch['t']
            batch['rot_t'] = batch['t']

        if self.use_dense_batch_loss:
            with torch.autograd.detect_anomaly():
                batch = corrupter.corrupt_dense_batch(batch)
        # else:
        #     batch = corrupter.corrupt_batch(batch)

        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = training_harness.run_eval(self.model, batch)

        else:
            outputs = training_harness.run_eval(self.model, batch)

        if self.use_dense_batch_loss:
            loss_dict = self._loss_step(batch, outputs)
        else:
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
            if self.use_dense_batch_loss:
                self.log_dict(
                    stratified_losses,
                    prog_bar=False,
                    logger=True,
                    on_step=None,
                    on_epoch=True,
                    batch_size=t.shape[0],
                    sync_dist=False)
            else:
                self.log_dict(
                    stratified_losses,
                    prog_bar=False,
                    logger=True,
                    on_step=None,
                    on_epoch=True,
                    batch_size=batch.num_graphs,
                    sync_dist=False)

        if self.use_dense_batch_loss:
            self.log_dict(
                log_dict,
                on_step=None,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=t.shape[0],
                sync_dist=True)
        else:
            self.log_dict(
                log_dict,
                on_step=None,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch.num_graphs,
                sync_dist=True)

        # self._log.info(gen_pbar_str(loss_dict))
        if not self.learnable_noise_schedule:
            return loss_dict
        else:
            opt = self.optimizers()
            opt.zero_grad()
            loss = loss_dict['loss']
            with torch.autograd.detect_anomaly():
                self.manual_backward(loss)
            loss_per_batch = loss_dict['frame_vf_loss'] # loss_dict['loss_per_batch'].detach()
            trans_t = batch['trans_t']
            rot_t = batch['rot_t']
            print("trans_t grad is_nan", trans_t.grad)
            print("rot_t grad is_nan", rot_t.grad)

            t = batch['t']
            l = batch['token']['token_is_protein_output_mask'].float().sum(dim=-1) / 100
            with torch.autograd.detect_anomaly():
                trans_t_copy = self.model.trans_gamma_t(t, l[..., None])
                rot_t_copy = self.model.rot_gamma_t(t, l[..., None])
                trans_t_copy.backward(trans_t.grad * loss_per_batch[..., None] * 2)
                rot_t_copy.backward(rot_t.grad * loss_per_batch[..., None] * 2)
            # for name, param in self.model.trans_gamma_t.named_parameters():
            #     if param.grad is not None:
            #         print("trans", name, param.grad)
            # for name, param in self.model.rot_gamma_t.named_parameters():
            #     if param.grad is not None:
            #         print("rot", name, param.grad)
            opt.step()


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

        if self.learnable_noise_schedule:
            t = inputs['t']
            l = inputs['token']['token_is_protein_output_mask'].float().sum(dim=-1) / 100
            log_trans_time_fn = lambda x: torch.log(1 - self.model.trans_gamma_t(x, l[..., None]))
            log_rot_time_fn = lambda x: torch.log(1 - self.model.rot_gamma_t(x, l[..., None]))
            log_trans_t_grad = torch.func.jvp(log_trans_time_fn, (t,), (torch.ones_like(t),))[1]
            log_rot_t_grad = torch.func.jvp(log_rot_time_fn, (t,), (torch.ones_like(t),))[1]
            print(log_trans_t_grad, log_rot_t_grad, 1 / (1 - t.clip(max=0.9)))
            log_trans_t_grad = (
                log_trans_t_grad.clip(min=-10, max=10).detach() / (log_trans_t_grad.detach() + 1e-4)
            ) * log_trans_t_grad
            log_rot_t_grad = (
                log_rot_t_grad.clip(min=-10, max=10).detach() / (log_rot_t_grad.detach() + 1e-4)
            ) * log_rot_t_grad
            # log_rot_t_grad = log_rot_t_grad.clip(min=-10, max=10)
            log_trans_t_grad = log_trans_t_grad * (1 - t.clip(max=0.9))
            log_rot_t_grad = log_rot_t_grad * (1 - t.clip(max=0.9))
        else:
            log_trans_t_grad = 1
            log_rot_t_grad = 1

        frame_fm_loss_dict = multiframe_fm_loss_dense_batch(
            inputs, outputs, sep_rot_loss=True, use_euclidean_for_rots=self.use_euclidean_for_rots,
            t_norm_clip=0.9,
            rot_vf_angle_loss_weight=0.5,
            fafe_l2_block_mask_size=1,
            rigidwise_weight=rigidwise_weight,
            trans_rigidwise_weight=log_trans_t_grad**2,
            rot_rigidwise_weight=log_rot_t_grad**2,
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
        atomic_loss_dict = atomic_losses_dense_batch(
            inputs,
            outputs,
            seqwise_weight=atomic_seq_weight
        )

        atomic_loss = (
            0.25 * atomic_loss_dict["seq_loss"]
            + 1 * atomic_loss_dict["smooth_lddt"]
        )

        if self.learnable_noise_schedule:
            loss = (
                frame_vf_loss
                + 0.25 * atomic_loss_dict["seq_loss"]
            )
        else:
            loss = (
                frame_vf_loss
                + 0.5 * frame_fm_loss_dict['scaled_fafe']
                + atomic_loss
            )

        loss_dict = {"loss": loss.mean(), "frame_vf_loss": frame_vf_loss, "frame_vf_loss_unscaled": unscaled_frame_vf_loss}
        loss_dict['loss_per_batch'] = loss

        if 'motif_idx' in outputs:
            pred_motif_idx = outputs['motif_idx']
            gt_motif_idx = inputs['token']['token_seq_idx']
            is_motif_mask = ~inputs['token']['token_is_protein_output_mask'] & ~inputs['token']['token_is_ligand_mask']
            motif_idx_correct = (pred_motif_idx == gt_motif_idx) * is_motif_mask
            if is_motif_mask.sum() > 0:
                loss_dict['motif_idx_correct'] = motif_idx_correct.sum() / is_motif_mask.sum()
        loss_dict[inputs['task'].name + "_loss"] = loss
        loss_dict[inputs['task'].name + "_seq_loss"] = atomic_loss_dict["seq_loss"]
        loss_dict[inputs['task'].name + "_frame_vf_loss"] = frame_vf_loss
        loss_dict[inputs['task'].name + "_frame_vf_loss_unscaled"] = unscaled_frame_vf_loss
        loss_dict.update(frame_fm_loss_dict)
        loss_dict.update(atomic_loss_dict)


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

            print(total_norm)

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
        if self.use_ema:
            if self.use_predict_v2:
                outputs = self._predict_step_v2(self.ema.module, batch)
            else:
                outputs = self._predict_step(self.ema.module, batch)
        else:
            outputs = self._predict_step(self.model, batch)
        return outputs

    def _predict_step(
        self,
        model,
        batch
    ):
        frame_noiser = self.training_harness.frame_noiser
        # Set-up time
        ts = torch.linspace(0.0, 1.0, frame_noiser.num_timesteps)

        # frame_noiser.churn = 0
        print(batch.num_graphs)

        res_data = batch['residue']
        rigids_0 = ru.Rigid.from_tensor_7(res_data['rigids_t'])
        trans_0 = rigids_0.get_trans()
        rotmats_0 = rigids_0.get_rots().get_rot_mats()
        rigids_noising_mask = res_data['rigids_noising_mask']
        seq_noising_mask = res_data['seq_noising_mask']

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

        global_shift = 0
        for t_2 in tqdm.tqdm(ts[1:]):
            d_t = t_2 - t_1
            # Run model.
            trans_t_1, rotmats_t_1, _, _ = prot_traj[-1]
            trans_t_1, center = frame_noiser._center_trans(trans_t_1, res_data.batch, rigids_noising_mask)
            global_shift += center

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
                # add_noise=False,
                # use_score=False
            )
            rotmats_t_2 = frame_noiser._rots_euler_step(
                d_t_hat,
                t_hat,
                pred_rotmats_1,
                rotmats_t_hat,
                noising_mask=rigids_noising_mask,
                # add_noise=False,
                # use_score=False
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

        # TODO: there's smth funky going on here
        # where seq is fixed and bb is fixed but rigid is not fixed
        fixed_residues = (~rigids_noising_mask).all(dim=-1)[res_mask].split(num_res.tolist(), dim=0)
        fixed_bb_residues = (~rigids_noising_mask[..., 0])[res_mask].split(num_res.tolist(), dim=0)
        fixed_seq_residues = (~seq_noising_mask)[res_mask].split(num_res.tolist(), dim=0)
        # print([t.shape for t in fixed_residues])
        fixed_res_idx = [torch.arange(t.numel())[t.cpu()].long() for t in fixed_residues]
        fixed_bb_res_idx = [torch.arange(t.numel())[t.cpu()].long() for t in fixed_bb_residues]
        fixed_seq_res_idx = [torch.arange(t.numel())[t.cpu()].long() for t in fixed_seq_residues]

        res_chain_idx = res_data['chain_idx'][res_mask].split(num_res.tolist(), dim=0)
        fixed_res_chain_idx = [
            chain[mask]
            for chain, mask in zip(res_chain_idx, fixed_residues)
        ]
        fixed_bb_chain_idx = [
            chain[mask]
            for chain, mask in zip(res_chain_idx, fixed_bb_residues)
        ]
        fixed_seq_chain_idx = [
            chain[mask]
            for chain, mask in zip(res_chain_idx, fixed_seq_residues)
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
                "fixed_bb_res_idx": fixed_bb_res_idx[i],
                "fixed_seq_res_idx": fixed_seq_res_idx[i],
                "fixed_res_chain_idx": fixed_res_chain_idx[i],
                "fixed_bb_chain_idx": fixed_bb_chain_idx[i],
                "fixed_seq_chain_idx": fixed_seq_chain_idx[i],
            })

        return ret

    def _predict_step_v2(
        self,
        model,
        batch
    ):
        frame_noiser = self.training_harness.frame_noiser
        # Set-up time
        ts = torch.linspace(0.0, 1.0, frame_noiser.num_timesteps)

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
        token_mask = token_data['token_mask']
        device = self.device
        prot_traj = [(
            trans_0,  # trans
            rotmats_0,  # rot
            torch.ones((num_batch, num_res, 21), device=device).float(),  # seq logits
            rigids_to_atom14(
                rigids_0,
                rigids_data['rigids_mask'],
                rigids_data['rigids_is_protein_output_mask'],
                rigids_data['rigids_is_atomized_mask'],
                token_data['token_is_atomized_mask'],
                token_data['token_is_protein_output_mask'],
                token_data['seq'],
                cg_version=model.cg_version
            )
        )]
        clean_traj = []
        denoiser_out = None

        global_shift = 0
        for t_2 in tqdm.tqdm(ts[1:]):
            d_t = t_2 - t_1
            # Run model.
            trans_t_1, rotmats_t_1, _, _ = prot_traj[-1]
            # trans_t_1, center = frame_noiser._center_trans(trans_t_1, res_data.batch, rigids_noising_mask)
            # global_shift += center

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

            rigids_data["trans_t"] = trans_t_hat
            rigids_data["rotmats_t"] = rotmats_t_hat
            rigids_data['rigids_t'] = ru.Rigid(
                rots=ru.Rotation(rot_mats=rotmats_t_hat),
                trans=trans_t_hat
            ).to_tensor_7()
            t = torch.ones(num_batch, device=device)[..., None] * t_hat
            batch["t"] = t

            denoiser_out = model(batch, self_condition=denoiser_out)

            # Process model output.
            pred_rigids = denoiser_out['denoised_rigids']
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

            # if self.learnable_noise_schedule:
            #     trans_vf_scale =

            # Take reverse step
            trans_t_2 = frame_noiser._trans_euler_step(
                d_t_hat,
                t_hat,
                pred_trans_1,
                trans_t_hat,
                noising_mask=rigids_noising_mask,
                # add_noise=False,
                # use_score=False
            )
            rotmats_t_2 = frame_noiser._rots_euler_step(
                d_t_hat,
                t_hat,
                pred_rotmats_1,
                rotmats_t_hat,
                noising_mask=rigids_noising_mask,
                # add_noise=False,
                # use_score=False
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
                 rigids_to_atom14(
                     ru.Rigid(ru.Rotation(rot_mats=rotmats_t_2), trans_t_2),
                     rigids_data['rigids_mask'],
                     rigids_data['rigids_is_protein_output_mask'],
                     rigids_data['rigids_is_atomized_mask'],
                     token_data['token_is_atomized_mask'],
                     token_data['token_is_protein_output_mask'],
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
        output_mask = token_data['token_is_protein_output_mask'].cpu()
        def _package_traj(traj, idx):
            traj_component = []
            for t in traj:
                unbatched_traj_frame = []
                for batch_idx in range(output_mask.shape[0]):
                    unbatched_traj_frame.append(
                        t[idx][batch_idx][output_mask.to(t[idx].device)[batch_idx]]
                        # t[idx][batch_idx]
                    )
                traj_component.append(unbatched_traj_frame)
            return list(zip(*traj_component))

        # clean_trajs = _package_traj(clean_traj, -2)
        # clean_traj_seqs = _package_traj(clean_traj, -3)
        # prot_trajs = _package_traj(prot_traj, -1)
        clean_trajs = [[] for _ in range(output_mask.shape[0])]
        clean_traj_seqs = [[] for _ in range(output_mask.shape[0])]
        prot_trajs = [[] for _ in range(output_mask.shape[0])]

        # fixed_residues = (~rigids_noising_mask).all(dim=-1).split(num_res.tolist(), dim=0)
        # fixed_bb_residues = (~rigids_noising_mask[..., 0]).split(num_res.tolist(), dim=0)
        # fixed_seq_residues = (~seq_noising_mask).split(num_res.tolist(), dim=0)
        # print([t.shape for t in fixed_residues])

        fixed_res_idx = [torch.as_tensor([]) for _ in range(output_mask.shape[0])]

        motif_idx = denoiser_out['motif_idx']
        motif_bb_mask = (~token_data['token_is_protein_output_mask'] & ~token_data['token_is_ligand_mask'] & token_data['token_mask'])
        # print(motif_idx, token_data['token_is_protein_output_mask'], token_data['token_is_ligand_mask'], token_data['token_mask'])
        fixed_bb_res_idx = []
        fixed_seq_res_idx = []
        for i in range(motif_idx.shape[0]):
            motif_idx_ = motif_idx[i, motif_bb_mask[i]]
            motif_seq_fixed = ~token_data['seq_noising_mask'][i, motif_bb_mask[i]]
            fixed_bb_res_idx.append(motif_idx_)
            fixed_seq_res_idx.append(motif_idx_[motif_seq_fixed])


        res_chain_idx = []
        fixed_bb_chain_idx = []
        fixed_seq_chain_idx = []
        for i in range(motif_idx.shape[0]):
            _chain_idx = token_data['chain_idx'][i, motif_bb_mask[i]]
            fixed_bb_chain_idx.append(_chain_idx)
            motif_seq_fixed = ~token_data['seq_noising_mask'][i, motif_bb_mask[i]]
            fixed_seq_chain_idx.append(_chain_idx[motif_seq_fixed])

        fixed_res_chain_idx = [
            torch.as_tensor([]) for _ in range(output_mask.shape[0])
        ]

        samples = []
        seqs = []
        max_mask_len = output_mask.sum(dim=-1).max().item()
        for i in range(output_mask.shape[0]):
            seqs.append(
                argmax_seq[i][output_mask.to(device=argmax_seq.device)[i]]
            )
            samples.append(
                decoded_struct[i][output_mask.to(device=decoded_struct.device)[i, :max_mask_len]]
            )


        ret = []

        # data_list = batch.to_data_list()

        for i, _ in enumerate(samples):
            ret.append({
                "sample_coord": samples[i],
                "seq": seqs[i],
                "clean_traj": clean_trajs[i],
                "clean_traj_seq": clean_traj_seqs[i],
                "prot_traj": prot_trajs[i],
                "input": {'task': batch['task'][i]},
                "fixed_res_idx": fixed_res_idx[i],
                "fixed_bb_res_idx": fixed_bb_res_idx[i],
                "fixed_seq_res_idx": fixed_seq_res_idx[i],
                "fixed_res_chain_idx": fixed_res_chain_idx[i],
                "fixed_bb_chain_idx": fixed_bb_chain_idx[i],
                "fixed_seq_chain_idx": fixed_seq_chain_idx[i],
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