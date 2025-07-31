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

from proteinzen.runtime.config import config_hydra_store, remove_zen_keys

# from mpi4py import MPI
# # set the node rank based off of what MPI says
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# this is a pretty jenk way around using mpi4py
# since we're really only using mpi to launch and not for distributed training
import socket



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

        with open(os.environ["MPI_HOSTFILE"]) as fp:
            hosts = [l.split()[0] for l in fp]
        node_name = socket.gethostname()
        rank = hosts.index(node_name.split(".")[0])
        os.environ['NODE_RANK'] = str(rank)
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        os.environ['LOCAL_RANK'] = str(local_rank)
        log.info(f"Using devices: {[f'{node_name}:{local_rank}']}")

        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = 1

        # some kinda jenk code to override the defaults
        trainer_cfg = remove_zen_keys(self._exp_cfg.lightning)
        overrides = dict(
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
            num_nodes=int(os.environ['NUM_NODES'])
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
