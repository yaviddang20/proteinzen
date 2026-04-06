import os

import torch
from hydra_zen import store, make_custom_builds_fn, make_config, kwargs_of
from hydra.conf import HydraConf, RunDir
import omegaconf
from datetime import timedelta

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from proteinzen.data.datasets.datamodule import BiomoleculeDataModule, BiomoleculeSamplingDataModule

from proteinzen.stoch_interp.multiframe import MultiSE3Interpolant

from proteinzen.model.denoiser import IpaMultiRigidDenoiser


from proteinzen.runtime.lmod import BiomoleculeModule
from proteinzen.runtime.optim import get_std_opt, make_adam, make_muon

if os.environ.get("REPO_ROOT") is None:
    print("Didn't find REPO_ROOT as an env var, so let's try to load it from the repo")
    import pathlib
    # import pprint
    import shlex
    import subprocess
    script_dir = pathlib.Path(__file__).parent.resolve()
    script_dir = os.path.abspath(os.path.join(script_dir, "../.."))

    command = shlex.split(f"bash -c 'set -a && source {script_dir}/env_vars.sh && env'")
    proc = subprocess.Popen(command, stdout = subprocess.PIPE)
    for line in proc.stdout:
        (key, _, value) = line.partition(b"=")
        os.environ[key.decode()] = value.decode().strip()
    proc.communicate()

    # # pprint.pprint(dict(os.environ))
print("REPO_ROOT:", os.environ.get("REPO_ROOT"))

# targets ZenStore
# pylint: disable=not-callable

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)
builds = make_custom_builds_fn(populate_full_signature=True)


def config_hydra_store():
    # change hydra conf defaults to avoid run collisions
    store(
        HydraConf(
            run=RunDir(dir="./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S-%f}")
    ))

    ## switches to allow for hot-swapping between paradigms and domains
    paradigm_store = store(group="paradigm")
    paradigm_store({"paradigm": "multiframefm"}, name="multiframefm")

    domain_store = store(group="domain")
    domain_store({"domain": "protein"}, name="protein")

    corruption_store = store(group="corrupter")
    corruption_store(
        MultiSE3Interpolant,
        name="multiframefm_protein")

    datamodule_store = store(group="datamodule")
    datamodule_store(
        pbuilds(
            BiomoleculeDataModule,
            batch_size=1,
            num_workers=8,
        ),
        name="default"
    )

    dataset_store = store(group="dataset")
    dataset_store(
        {
            "config": f"{os.environ.get('REPO_ROOT')}/configs/train/data/afdb_128.yaml",
            "val_config": None,
            "include_h": False,
        },
        name="afdb_128")

    lmodule_store = store(group="lmodule")
    lmodule_store(
        pbuilds(BiomoleculeModule),
        name="protein"
    )

    # latent_fm_wrapper = ChimeraLatentWrapper
    model_store = store(group="model")
    model_store(IpaMultiRigidDenoiser, name="multiframefm_protein")

    tasks_store = store(group="tasks")
    tasks_store(
        {"config": f"{os.environ.get('REPO_ROOT')}/configs/train/tasks/train_128.yaml"},
        name="train_128"
    )

    exp_store = store(group="experiment")
    exp_store({
        "warm_start": None,
    }, name="default")
    lightning_store = exp_store(group="experiment/lightning")
    lightning_store(
        pbuilds(
            Trainer,
            min_epochs=1,
            max_epochs=-1,
            check_val_every_n_epoch=1,
            log_every_n_steps=50,
            use_distributed_sampler=False,
            # gradient_clip_val=1.0,
        ), name="default")

    optim_store = exp_store(group="experiment/optim")
    optim_store(pbuilds(make_adam, lr=1e-4), name="adam")
    optim_store(pbuilds(get_std_opt, d_model=128), name="noam")
    optim_store(pbuilds(make_muon, lr_muon=1e-3, lr_adam=1e-4), name="muon")

    exp_store(
        ModelCheckpoint,
        save_top_k=-1,
        save_on_train_epoch_end=True,
        save_last=True,
        train_time_interval=timedelta(days=1),
        group="experiment/checkpointer",
        name="protein")

    exp_store(
        {"offline": True},
        group="experiment/wandb",
        name="default"
    )

    # breaking the hydra_zen/hydra/omegaconf abstraction here a little
    # but this allows for nice swappable configs
    ExperimentConfig = make_config(
        debug=True,
        hydra_defaults=[
            {"paradigm": "multiframefm"},
            {"domain": "protein"},
            {"datamodule": "default"},
            {"dataset": "afdb_128"},
            {"lmodule": "${domain}"},
            {"corrupter": "${paradigm}_${domain}"},
            {"model": "${paradigm}_${domain}"},
            {"tasks": "train_128"},
            {"experiment": "default"},
            {"experiment/optim": "adam"},
            {"experiment/lightning": "default"},
            {"experiment/checkpointer": "${domain}"},
            {"experiment/wandb": "default"},
            '_self_'
        ],
    )

    store(ExperimentConfig, name="main")
    store.add_to_hydra_store()


def remove_zen_keys(cfg, keys=['_target_', '_partial_']):
    def remove_key(cfg):
        for key in keys:
            if hasattr(cfg, key):
                delattr(cfg, key)
            elif isinstance(cfg, dict) and key in cfg:
                del cfg[key]

        if isinstance(cfg, (omegaconf.DictConfig, dict)):
            for value in cfg.values():
                remove_key(value)
        elif isinstance(cfg, omegaconf.ListConfig):
            for value in cfg:
                remove_key(value)
        return cfg

    return remove_key(cfg.copy())


def config_sampling_hydra_store():
    # change hydra conf defaults to avoid run collisions
    store(
        HydraConf(
            run=RunDir(dir="./sampling_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S-%f}")
    ))

    sampler_store = store(group="sampler")
    sampler_store(
        builds(
            BiomoleculeSamplingDataModule,
            include_h=False,
        ),
        name="default"
    )
    corrupter_store = store(group="corrupter")
    corrupter_store(
        MultiSE3Interpolant,
        name="default"
    )

    SamplingConfig = make_config(
        model_dir="",
        out_dir="samples",
        save_traj=False,
        output_motif_chains=False,
        checkpoint_idx=-1,
        hydra_defaults=[
            {"sampler": "default"},
            {"corrupter": "default"},
            '_self_'
        ],
    )

    store(SamplingConfig, name="main")
    store.add_to_hydra_store()