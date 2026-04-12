""" Train a model """
import logging
import os
import glob
import shutil
from typing import Dict, Any

import hydra
from hydra_zen import zen, load_from_yaml, instantiate
import omegaconf
import torch
import numpy as np
import pandas as pd

from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch import Trainer

from proteinzen.runtime.config import config_sampling_hydra_store
from proteinzen.runtime.lmod import BiomoleculeSamplingModule, PDBWriter

# PyTorch 2.6+ changed default weights_only=True which breaks checkpoint loading
# SIMPLE FIX: Monkey-patch torch.load to use weights_only=False by default
# This restores the old behavior and avoids the ridiculous safe unpickling errors
_original_torch_load = torch.load
def _patched_torch_load(*args, weights_only=None, **kwargs):
    # If weights_only not explicitly set, default to False (old behavior)
    if weights_only is None:
        weights_only = False
    return _original_torch_load(*args, weights_only=weights_only, **kwargs)
torch.load = _patched_torch_load

# A logger for this file
log = logging.getLogger(__name__)


class Experiment:
    def __init__(self,
                 model,
                 sampler,
                 cfg):
        self._cfg = cfg
        self._sampler: LightningDataModule = sampler
        self._model: LightningModule = model

    def predict(self):
        kwargs: Dict[str, Any] = {
            "use_distributed_sampler": False
        }
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
            if len(devices) > 1:
                kwargs['strategy'] = 'ddp_find_unused_parameters_true'
                kwargs['use_distributed_sampler'] = True
        else:
            devices = 1

        log.info(f"Using devices: {devices}")

        pred_writer = PDBWriter(
            output_dir=self._cfg['out_dir'],
            run_cfg=self._cfg
        )
        trainer = Trainer(
            # use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
            callbacks=[pred_writer],
            **kwargs
        )
        trainer.predict(
            model=self._model,
            datamodule=self._sampler,
            ckpt_path=self._cfg['ckpt_path'],
            return_predictions=False
        )


def main(sampler,
         corrupter,
         zen_cfg):
    # change into the output directory
    # os.chdir(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Experiment started in folder: {os.getcwd()}")

    # so we can add extra entries in the config
    zen_cfg = omegaconf.OmegaConf.to_container(zen_cfg)
    assert zen_cfg is not None

    version_num = zen_cfg['version_num']

    run_dir = zen_cfg['model_dir']
    ckpt_list = list(glob.glob(
        os.path.join(
            run_dir,
            f"lightning_logs/version_{version_num}/checkpoints/*.ckpt",
        )
    ))
    epoch_list = []
    has_best = False
    best_ckpt_path = None
    for ckpt_path in ckpt_list:
        fname = ckpt_path.split("/")[-1]
        if fname == "best.ckpt":
            has_best = True
            best_ckpt_path = ckpt_path
        elif fname == "last.ckpt":
            epoch_list.append((ckpt_path, 1e6))
        else:
            epoch = ckpt_path.split("=")[1].split("-")[0]
            epoch_list.append((ckpt_path, int(epoch)))

    epoch_list = sorted(epoch_list, key=lambda x: x[1])
    epoch_list, _ = zip(*epoch_list)
    ckpt_path = epoch_list[zen_cfg["checkpoint_idx"]]
    if has_best:
        ckpt_path = best_ckpt_path
    print(ckpt_path)
    zen_cfg['ckpt_path'] = ckpt_path

    config_path = os.path.join(
        run_dir,
        ".hydra",
        "config.yaml"
    )
    model_cfg = load_from_yaml(config_path)
    # lmodule_init = instantiate(model_cfg['lmodule'])
    model = instantiate(model_cfg['model'])

    # model = lmodule_init(model, corrupter, None)
    model = BiomoleculeSamplingModule(model, corrupter, zen_cfg)

    os.makedirs(zen_cfg['out_dir'], exist_ok=True)
    zen_cfg['samples_dir'] = os.path.join(
        zen_cfg['out_dir'], "samples"
    )
    if os.path.isdir(zen_cfg['samples_dir']):
        shutil.rmtree(zen_cfg['samples_dir'])
    os.makedirs(zen_cfg['samples_dir'])

    with open(os.path.join(zen_cfg['out_dir'], "run.log"), 'w') as fp:
        fp.write(f"Sampling config path: {zen_cfg['sampler']['tasks_yaml']}")
    shutil.copy(
        zen_cfg['sampler']['tasks_yaml'],
        os.path.join(zen_cfg['out_dir'], "config.yaml")
    )


    exp = Experiment(
        model=model,
        sampler=sampler,
        cfg=zen_cfg)
    exp.predict()


if __name__ == '__main__':
    config_sampling_hydra_store()
    torch.set_float32_matmul_precision("medium")
    zen(main, unpack_kwargs=True).hydra_main(
        config_name="main",
        version_base="1.2",
        config_path="."
    )
