import logging

import numpy as np
import torch
import tree
import lightning as L

from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


from ligbinddiff.tasks import TaskList
from .utils import gen_pbar_str

class MedianMetric(Metric):
    is_differentiable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("seq_recov", default=[], dist_reduce_fx="cat")

    def update(self, seq_recov: torch.Tensor) -> None:
        self.seq_recov.append(seq_recov)

    def compute(self) -> torch.Tensor:
        return torch.median(dim_zero_cat(self.seq_recov))


class LightningWrapper(L.LightningModule):
    def __init__(self, model, optim):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        self.optim = optim

        self.median_metric = MedianMetric()

    def training_step(self, batch):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        loss_dict = task.compile_task_losses(batch, outputs)

        stratified_loss = {}
        for key, loss_value in loss_dict.items():
            if loss_value.numel() < 2:
                continue
            stratified_loss.update(t_stratified_loss(batch['t'], loss_value, loss_name=key))


        log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict.update(stratified_loss)
        log_dict = dict(sorted(log_dict.items(), key = lambda x: x[0]))

        self.log_dict(
            log_dict,
            on_step=None,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)
        self._log.info(gen_pbar_str(loss_dict) + ", " + gen_pbar_str(stratified_loss))
        return loss_dict

    def validation_step(self, batch, batch_idx):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        loss_dict = task.compile_task_losses(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict = {"val_" + k: v for k,v in log_dict.items()}

        self.log_dict(
            log_dict,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)
        self._log.info(gen_pbar_str(loss_dict))

        sample_outputs = task.run_predicts(self.model, batch)
        sample_loss_dict = task.compile_task_losses(batch, sample_outputs)
        sample_log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            sample_loss_dict
        )
        sample_log_dict = {"sample_" + k: v for k,v in sample_log_dict.items()}
        self.log_dict(
            sample_log_dict,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)
        self._log.info(gen_pbar_str(sample_loss_dict))

        return loss_dict

    def test_step(self, batch, batch_idx):
        task: TaskList = batch.task
        sample_outputs = task.run_predicts(self.model, batch)
        sample_loss_dict = task.compile_task_losses(batch, sample_outputs)
        sample_log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            sample_loss_dict
        )
        sample_log_dict = {"sample_" + k: v for k,v in sample_log_dict.items()}

        self.log_dict(
            sample_log_dict,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)

        self.median_metric.update(sample_loss_dict['autoenc_per_seq_recov'])
        self.log("median_seq_recov", self.median_metric, on_step=False, on_epoch=True)
        self._log.info(gen_pbar_str(sample_loss_dict))
        return sample_loss_dict

    def predict_step(self, batch, batch_idx):
        task: TaskList = batch['task']
        outputs = task.run_predicts(self.model, batch)
        return outputs


    def forward(self, batch):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        return outputs

    def configure_optimizers(self):
        return self.optim(self.model.parameters())


class ProteinModule(L.LightningModule):
    def __init__(self, model, optim_ae, optim_denoiser):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        self.optim_ae = optim_ae
        self.optim_denoiser = optim_denoiser

        self.median_metric = MedianMetric()

    def training_step(self, batch):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        loss_dict = task.compile_task_losses(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict.update(stratified_loss)

        self.log_dict(
            log_dict,
            on_step=None,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)
        self._log.info(gen_pbar_str(loss_dict))
        return loss_dict

    def validation_step(self, batch, batch_idx):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        loss_dict = task.compile_task_losses(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict = {"val_" + k: v for k,v in log_dict.items()}

        self.log_dict(
            log_dict,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)
        self._log.info(gen_pbar_str(loss_dict))

        sample_outputs = task.run_predicts(self.model, batch)
        sample_loss_dict = task.compile_task_losses(batch, sample_outputs)
        sample_log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            sample_loss_dict
        )
        sample_log_dict = {"sample_" + k: v for k,v in sample_log_dict.items()}
        self.log_dict(
            sample_log_dict,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)
        self._log.info(gen_pbar_str(sample_loss_dict))

        return loss_dict

    def test_step(self, batch, batch_idx):
        task: TaskList = batch.task
        sample_outputs = task.run_predicts(self.model, batch)
        sample_loss_dict = task.compile_task_losses(batch, sample_outputs)
        sample_log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            sample_loss_dict
        )
        sample_log_dict = {"sample_" + k: v for k,v in sample_log_dict.items()}

        self.log_dict(
            sample_log_dict,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)

        self.median_metric.update(sample_loss_dict['autoenc_per_seq_recov'])
        self.log("median_seq_recov", self.median_metric, on_step=False, on_epoch=True)
        self._log.info(gen_pbar_str(sample_loss_dict))
        return sample_loss_dict

    def predict_step(self, batch, batch_idx):
        task: TaskList = batch['task']
        outputs = task.run_predicts(self.model, batch)
        return outputs


    def forward(self, batch):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        return outputs

    def configure_optimizers(self):
        optim_ae = self.optim_ae(
            self.model.encoder.parameters() +
            self.model.decoder.parameters()
        )
        optim_denoiser = self.optim_denoiser(
            self.model.denoiser.parameters()
        )
        return optim_ae, optim_denoiser


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