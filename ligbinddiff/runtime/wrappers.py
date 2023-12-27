import logging

import torch
import tree
import lightning as L


from ligbinddiff.tasks import TaskList
from .utils import gen_pbar_str

class LightningWrapper(L.LightningModule):
    def __init__(self, model, optim):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        self.optim = optim

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
        return loss_dict

    def test_step(self, batch, batch_idx):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        loss_dict = task.compile_task_losses(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict = {"test_" + k: v for k,v in log_dict.items()}

        self.log_dict(
            log_dict,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)
        self._log.info(gen_pbar_str(loss_dict))
        return loss_dict

    def forward(self, batch):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        return outputs

    def configure_optimizers(self):
        return self.optim(self.model.parameters())
