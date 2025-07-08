""" Train a model """

import logging
import os
import shutil

import hydra
from hydra_zen import zen
import omegaconf
import torch
import numpy as np

from lightning import LightningDataModule, LightningModule, Trainer
# from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

# import wandb

from proteinzen.runtime.config_boltz import config_hydra_store, remove_zen_keys
from proteinzen.harness.fm.multiframe import MultiFrameInterpolation
from proteinzen.runtime.training.task import TaskSampler


# A logger for this file
log = logging.getLogger(__name__)


class Experiment:
    def __init__(self,
                 model,
                 datamodule,
                 checkpointer,
                 cfg):
        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._datamodule: LightningDataModule = datamodule
        self._model: LightningModule = model
        self._checkpointer: ModelCheckpoint = checkpointer

    def train(self):
        callbacks = []
        if self._cfg.debug:
            log.info("Debug mode.")
            logger = None
        else:
            raise NotImplementedError()
            # logger = WandbLogger(
            #     **remove_zen_keys(self._exp_cfg.wandb),
            # )

        # # Checkpoint directory.
        # ckpt_dir = self._exp_cfg.checkpointer.dirpath
        # os.makedirs(ckpt_dir, exist_ok=True)
        # log.info(f"Checkpoints saved to {ckpt_dir}")

        # Model checkpoints
        callbacks.append(self._checkpointer)

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
            # reload_dataloaders_every_n_epochs=1,
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
            ckpt_path=self._exp_cfg.warm_start,
        )


def main(model,
         corrupter,
         lmodule,
         dataset,
         datamodule,
         experiment,
         zen_cfg):
    # change into the output directory
    os.chdir(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Experiment started in folder: {os.getcwd()}")

    # assemble task sampler
    print(dataset)
    dataset_config = omegaconf.OmegaConf.load(dataset['config'])
    train_dataset = hydra.utils.instantiate(dataset_config)
    shutil.copy(dataset['config'], os.path.join(os.getcwd(), "dataset_config.yaml"))
    datamodule_inst = datamodule(train_dataset=train_dataset)

    # datamodule and optim are all partial'd __init__s
    # so we instantiate instances of each
    model = lmodule(model, corrupter, experiment['optim'])
    checkpointer = experiment['checkpointer']

    exp = Experiment(
        model=model,
        datamodule=datamodule_inst,
        checkpointer=checkpointer,
        cfg=zen_cfg)
    exp.train()


if __name__ == '__main__':
    config_hydra_store()
    torch.set_float32_matmul_precision("medium")

    # we use 1.2 so we can use .cache at the root dir
    # and change into the output directory at main()
    zen(main).hydra_main(
        config_name="main",
        version_base="1.2",
        config_path="."
    )
