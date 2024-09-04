""" Train a model """

import logging
import os

import hydra
from hydra_zen import zen
import omegaconf
import torch
import numpy as np

from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

import wandb


from proteinzen.runtime.config import config_hydra_store, remove_zen_keys
from proteinzen.tasks.task import single_task_sampler


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
            logger = WandbLogger(
                **remove_zen_keys(self._exp_cfg.wandb),
            )

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
            ckpt_path=self._exp_cfg.warm_start,
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

    # if hasattr(corrupter, "sidechain_noiser") and isinstance(corrupter.sidechain_noiser, torch.nn.Module):
    #     model.add_module("sidechain_noiser", corrupter.sidechain_noiser.learned_sched)
    kwargs = {}
    if 'freeze_encoder' in zen_cfg:
        kwargs['freeze_encoder'] = zen_cfg['freeze_encoder']
    if 'freeze_decoder' in zen_cfg:
        kwargs['freeze_decoder'] = zen_cfg['freeze_decoder']


    # datamodule and optim are all partial'd __init__s
    # so we instantiate instances of each
    model = lmodule(model, experiment['optim'], **kwargs)
    checkpointer = experiment['checkpointer']
    os.makedirs(model.val_dir, exist_ok=True)
    task_sampler = single_task_sampler(tasks(corrupter))

    if 'only_load_state_dict' in zen_cfg:
        assert zen_cfg['experiment']['warm_start'] is not None
        ckpt = torch.load(zen_cfg['experiment']['warm_start'])
        encoder_dict = {
            k.removeprefix("model.encoder."): v
            for k, v in ckpt['state_dict'].items()
            if k.startswith("model.encoder")
        }
        decoder_dict = {
            k.removeprefix("model.decoder."): v
            for k, v in ckpt['state_dict'].items()
            if k.startswith("model.decoder")
        }
        model.model.encoder.load_state_dict(encoder_dict)
        model.model.decoder.load_state_dict(decoder_dict)
        zen_cfg['experiment']['warm_start'] = None


    # TODO: there's gotta be a nicer way of doing this
    if hasattr(model.model, "lrange_logn_scale") and model.model.lrange_logn_scale > 0:
        _model = model.model
        def batch_by_edge_fn(n):
            lrange = round(
                max(_model.lrange_k, _model.lrange_logn_scale * np.log2(n) + _model.lrange_logn_offset)
            )
            knn = _model.knn_k
            edges_per_node = min(lrange+knn, n)
            return edges_per_node * n
        datamodule_inst = datamodule(task_sampler=task_sampler, batch_by_edge_fn=batch_by_edge_fn)
    else:
        datamodule_inst = datamodule(task_sampler=task_sampler)

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
