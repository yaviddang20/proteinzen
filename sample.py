""" Train a model """
import json
import logging
import os
import glob
import shutil
import time
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
                kwargs['use_distributed_sampler'] = False
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
    traj_dir = os.path.join(zen_cfg['out_dir'], "traj")

    continue_run = zen_cfg.get('continue_run', False)
    if continue_run:
        os.makedirs(zen_cfg['samples_dir'], exist_ok=True)
        dispatcher = sampler.task_dispatcher

        # Lightning's DDP launcher re-execs this whole script once per GPU rank.
        # Only rank 0 scans samples_dir and decides which samples to keep — every
        # rank scanning independently races against other ranks concurrently
        # writing new PDBs into that same directory, so different ranks could
        # compute different-length dispatcher.batches, which breaks Lightning's
        # distributed sampler (it assumes all ranks agree on dataset length).
        is_rank_zero = (
            int(os.environ.get("NODE_RANK", 0)) == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        )
        manifest_path = os.path.join(zen_cfg['out_dir'], "continue_run_kept_indices.json")

        if is_rank_zero:
            try:
                os.remove(manifest_path)  # drop any stale manifest from a prior run
            except FileNotFoundError:
                pass

            # Count already-generated PDBs per task name prefix.
            # File pattern: {task_name}_gpu{rank}_batch{idx}_idx{sid}.pdb
            from collections import defaultdict, Counter
            existing_per_task = defaultdict(int)
            for fname in os.listdir(zen_cfg['samples_dir']):
                if not fname.endswith('.pdb') or '_traj' in fname:
                    continue
                if '_gpu' in fname:
                    task_prefix = fname[:fname.index('_gpu')]
                    existing_per_task[task_prefix] += 1
            # Count how many samples per task are in the full batch list.
            total_per_task = Counter(s['task'] for s in dispatcher.batches)
            # Trim to only the remaining deficit per task.
            per_task_kept = defaultdict(int)
            kept_indices = []
            for i, s in enumerate(dispatcher.batches):
                task = s['task']
                deficit = total_per_task[task] - existing_per_task[task]
                if per_task_kept[task] < deficit:
                    kept_indices.append(i)
                    per_task_kept[task] += 1
            log.info(f"continue_run: keeping {len(kept_indices)}/{len(dispatcher.batches)} samples "
                     f"(existing: {dict(existing_per_task)})")
            tmp_path = f"{manifest_path}.tmp{os.getpid()}"
            with open(tmp_path, "w") as f:
                json.dump(kept_indices, f)
            os.replace(tmp_path, manifest_path)  # atomic — never a partial read
        else:
            # Give rank 0 a head start to delete any stale manifest and finish
            # its own scan before we start polling for the fresh one.
            time.sleep(3.0)
            timeout_s = 600
            waited = 0.0
            while not os.path.exists(manifest_path):
                time.sleep(1.0)
                waited += 1.0
                if waited > timeout_s:
                    raise RuntimeError(
                        f"Timed out waiting for rank 0 to write {manifest_path} for continue_run"
                    )
            with open(manifest_path) as f:
                kept_indices = json.load(f)

        dispatcher.batches = [dispatcher.batches[i] for i in kept_indices]
    else:
        if os.path.isdir(zen_cfg['samples_dir']):
            shutil.rmtree(zen_cfg['samples_dir'])
        os.makedirs(zen_cfg['samples_dir'], exist_ok=True)
        if os.path.isdir(traj_dir):
            shutil.rmtree(traj_dir)
        refold_dir = os.path.join(zen_cfg['out_dir'], "refold_outputs")
        if os.path.isdir(refold_dir):
            shutil.rmtree(refold_dir)

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
