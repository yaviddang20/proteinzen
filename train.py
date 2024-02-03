""" Train a model """

import logging
import os

import hydra
from hydra_zen import zen
import omegaconf
import torch

from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

import wandb

from ligbinddiff.runtime.config import config_hydra_store, remove_zen_keys
from ligbinddiff.tasks.task import single_task_sampler


# A logger for this file
log = logging.getLogger(__name__)


class Experiment:
    def __init__(self,
                 model,
                 datamodule,
                 cfg):
        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._datamodule: LightningDataModule = datamodule
        self._model: LightningModule = model

    def train(self):
        callbacks = []
        if self._cfg.debug:
            log.info("Debug mode.")
            logger = None
        else:
            logger = WandbLogger(
                **remove_zen_keys(self._exp_cfg.wandb),
            )

        # Checkpoint directory.
        ckpt_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(ckpt_dir, exist_ok=True)
        log.info(f"Checkpoints saved to {ckpt_dir}")

        # Model checkpoints
        callbacks.append(
            ModelCheckpoint(
                **remove_zen_keys(self._exp_cfg.checkpointer)
        ))

        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = 1

        log.info(f"Using devices: {devices}")

        # some kinda jenk code to override the defaults
        trainer_cfg = remove_zen_keys(self._exp_cfg.lightning)
        overrides = dict(
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
            reload_dataloaders_every_n_epochs=1,
            # strategy='ddp_find_unused_parameters_true',
            # detect_anomaly=True
        )
        trainer_cfg = omegaconf.OmegaConf.to_container(trainer_cfg)
        trainer_cfg.update(overrides)

        trainer = Trainer(
            **trainer_cfg,
        )
        trainer.fit(
            model=self._model,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )


def main(model,
         corrupter,
         lmodule,
         datamodule,
         experiment,
         tasks,
         zen_cfg):
    # change into the output directory
    os.chdir(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Experiment started in folder: {os.getcwd()}")

    # datamodule and optim are all partial'd __init__s
    # so we instantiate instances of each
    model = lmodule(model, experiment['optim'])
    os.makedirs(model.val_dir, exist_ok=True)
    task_sampler = single_task_sampler(tasks(corrupter))
    datamodule_inst = datamodule(task_sampler=task_sampler)
    exp = Experiment(
        model=model,
        datamodule=datamodule_inst,
        cfg=zen_cfg)
    exp.train()


if __name__ == '__main__':
    config_hydra_store()

    # we use 1.2 so we can use .cache at the root dir
    # and change into the output directory at main()
    zen(main).hydra_main(
        config_name="main",
        version_base="1.2",
        config_path="."
    )
