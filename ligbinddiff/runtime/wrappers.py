import logging

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

        log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            loss_dict
        )

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
