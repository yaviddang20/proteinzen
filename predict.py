""" Train a model """

import argparse
import logging
import os
import glob

import hydra
from hydra_zen import zen, load_from_yaml
import omegaconf
import torch

from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

import wandb

from ligbinddiff.runtime.config import config_hydra_store, remove_zen_keys
from ligbinddiff.tasks.task import single_task_sampler
from ligbinddiff.data.io.atom91 import atom91_to_pdb
from ligbinddiff.utils.atom_reps import atom14_to_atom91
from ligbinddiff.data.openfold.residue_constants import restypes


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

    def predict(self):
        callbacks = []

        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = 1

        log.info(f"Using devices: {devices}")
        logger = None

        # some kinda jenk code to override the defaults
        trainer_cfg = remove_zen_keys(self._exp_cfg.lightning)
        overrides = dict(
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
            # strategy='ddp_find_unused_parameters_true',
            # detect_anomaly=True
        )
        trainer_cfg = omegaconf.OmegaConf.to_container(trainer_cfg)
        trainer_cfg.update(overrides)

        trainer = Trainer(
            **trainer_cfg,
        )
        assert self._cfg.domain.domain in ["backbone", "protein", "molecule"], self._cfg.domain
        ret = trainer.predict(
            model=self._model,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )
        if self._cfg.domain.domain in ["backbone", "protein"]:
            os.chdir(self._exp_cfg['samples_dir'])
            for batch in ret:
                samples = batch['samples']
                sample_ids = batch['inputs']['sample_id']
                if self._cfg.domain.domain == 'backbone':
                    for sample, sample_id in zip(samples, sample_ids):
                        sample_len = sample.shape[0]
                        atom91_to_pdb("".join(["A" for _ in range(sample_len)]), sample.numpy(force=True), f"len_{sample_len}_bb_{sample_id}")
                elif self._cfg.domain.domain == 'protein':
                    seqs = batch['seqs']
                    for sample, sample_id, seq in zip(samples, sample_ids, seqs):
                        sample_len = sample.shape[0]
                        seq = "".join([restypes[i] for i in seq.tolist()])
                        atom14 = sample.numpy(force=True)
                        atom91, atom91_mask = atom14_to_atom91(seq, atom14)
                        atom91_to_pdb(seq, atom91, f"len_{sample_len}_protein_{sample_id}")



def main(model,
         corrupter,
         datamodule,
         lmodule,
         experiment,
         tasks,
         zen_cfg):
    # change into the output directory
    # os.chdir(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Experiment started in folder: {os.getcwd()}")

    # datamodule and optim are all partial'd __init__s
    # so we instantiate instances of each
    model = lmodule(model, experiment['optim'])
    task_sampler = single_task_sampler(tasks(corrupter))
    datamodule_inst = datamodule(task_sampler=task_sampler)
    exp = Experiment(
        model=model,
        datamodule=datamodule_inst,
        cfg=zen_cfg)
    exp.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir")
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--out_prefix", default="samples_24hr")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    config_path = os.path.join(
        args.run_dir,
        ".hydra",
        "config.yaml"
    )
    if args.epoch == -1:
        ckpt_path = os.path.join(
            args.run_dir,
            "ckpt/last.ckpt",
        )
    else:
        ckpt_path = list(glob.glob(
            os.path.join(
                args.run_dir,
                f"ckpt/epoch={args.epoch}*.ckpt",
            )
        ))[0]


    cfg = load_from_yaml(config_path)
    cfg['experiment']['warm_start'] = ckpt_path
    cfg['datamodule']['batch_size'] = 2000

    if cfg['domain']['domain'] == "backbone":
        if args.debug:
            cfg['datamodule']['sample_lengths'] = {
                60: 5,
                70: 5,
                80: 5,
                90: 5,
                100: 5,
                110: 5,
                120: 5
            }
        else:
            cfg['datamodule']['sample_lengths'] = {
                i: 10
                for i in range(60, 128+1)
            }

    samples_dir = os.path.join(args.run_dir, args.out_prefix)
    os.makedirs(
        samples_dir,
        exist_ok=True
    )
    cfg['experiment']['samples_dir'] = samples_dir
    if cfg['domain']['domain'] == "molecule":
        cfg['lmodule']['val_dir'] = samples_dir

    zen(main, unpack_kwargs=True)(cfg)
