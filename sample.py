""" Train a model """
import logging
import os
import glob
import shutil
from typing import Dict, Any
import functools as fn

import hydra
from hydra_zen import zen, load_from_yaml, save_as_yaml, instantiate
import omegaconf
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch import Trainer

from proteinzen.runtime.config import config_sampling_hydra_store
from proteinzen.runtime.lmod import BiomoleculeSamplingModule, PDBWriter

# PyTorch 2.6+ changed default weights_only=True which breaks checkpoint loading
# Monkey-patch torch.load to use weights_only=False by default
# This restores the old behavior and avoids the ridiculous safe unpickling errors
_original_torch_load = torch.load
def _patched_torch_load(*args, weights_only=None, **kwargs):
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
            "use_distributed_sampler": False,
            "inference_mode": self._cfg['inference_mode']
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
         integrator,
         model_wrapper,
         diffeq,
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
        # if fname == "best.ckpt":
        #     has_best = True
        #     best_ckpt_path = ckpt_path
        if fname == "last.ckpt":
            epoch_list.append((ckpt_path, 1e6))
        elif '=' in Path(ckpt_path).name:
            epoch = Path(ckpt_path).name.split("=")[1].split("-")[0]
            epoch_list.append((ckpt_path, int(epoch)))
        else:
            continue

    epoch_list = sorted(epoch_list, key=lambda x: x[1])
    epoch_list, _ = zip(*epoch_list)
    ckpt_path = epoch_list[zen_cfg["checkpoint_idx"]]
    # if has_best:
    #     ckpt_path = best_ckpt_path
    print(ckpt_path)
    zen_cfg['ckpt_path'] = ckpt_path

    config_path = os.path.join(
        run_dir,
        f"lightning_logs/version_{version_num}/config.yaml"
    )
    if not os.path.exists(config_path):
        print(f"config.yaml not found in version dir, defaulting to .hydra/config.yaml")
        config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    model_cfg = load_from_yaml(config_path)
    model = instantiate(model_cfg['model'])

    # create sampling module
    def integrator_init(model):
        return integrator(
            wrapped_model=model_wrapper(model),
            diffeq=diffeq
        )
    model = BiomoleculeSamplingModule(
        model,
        integrator_init=integrator_init,
        run_cfg=zen_cfg,
        strict_weight_loading=zen_cfg.get('strict_weight_loading', True),
    )

    # make output directories
    os.makedirs(zen_cfg['out_dir'], exist_ok=True)
    zen_cfg['samples_dir'] = os.path.join(
        zen_cfg['out_dir'], "samples"
    )
    if os.path.isdir(zen_cfg['samples_dir']):
        shutil.rmtree(zen_cfg['samples_dir'])
    os.makedirs(zen_cfg['samples_dir'])

    traj_dir = os.path.join(zen_cfg['out_dir'], "traj")
    if os.path.isdir(traj_dir):
        shutil.rmtree(traj_dir)

    # record run params
    shutil.copy(
        zen_cfg['sampler']['tasks_yaml'],
        os.path.join(zen_cfg['out_dir'], "tasks_config.yaml")
    )
    zen_cfg_obj = omegaconf.OmegaConf.create(zen_cfg)
    save_as_yaml(
        zen_cfg_obj,
        os.path.join(zen_cfg['out_dir'], "run_config.yaml"),
        resolve=True
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
