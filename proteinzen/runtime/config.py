import os

import torch
from hydra_zen import store, just, make_custom_builds_fn, make_config, kwargs_of
from hydra.conf import HydraConf, RunDir
import omegaconf
from datetime import timedelta

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from proteinzen.data.datasets.datamodule import FramediffDataModule, SamplingDataModule

from proteinzen.stoch_interp.atom14_nonequiv import Atom14Interpolant as NonEquivAtom14Interpolant
from proteinzen.stoch_interp.multiframe import MultiSE3Interpolant, SE3InterpolantConfig

from proteinzen.model.denoiser.protein.atom14_nonequiv import AtomDenoiser
from proteinzen.model.denoiser.protein.dense_multiframe import IpaMultiRigidDenoiser

from proteinzen.harness.fm.atom14_nonequiv import Atom14Interpolation as NonEquivAtom14Interpolation
from proteinzen.harness.fm.multiframe import MultiFrameInterpolation

from proteinzen.runtime.lmod import ProteinModule
from proteinzen.runtime.training.unconditional import UnconditionalGeneration
from proteinzen.runtime.training.motif_scaffold import BackboneMotifScaffolding, ResidueMotifScaffolding, InverseRotamerMotifScaffolding
from proteinzen.runtime.training.folding import Folding
from proteinzen.runtime.training.diffusion_forcing import DiffusionForcing
from proteinzen.runtime.training.sidechain_design import SidechainDesign
from proteinzen.runtime.optim import get_std_opt

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

    # pprint.pprint(dict(os.environ))
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
    paradigm_store({"paradigm": "nonequiv_atom14fm"}, name="nonequiv_atom14fm")
    paradigm_store({"paradigm": "multiframefm"}, name="multiframefm")

    domain_store = store(group="domain")
    domain_store({"domain": "protein"}, name="protein")

    corruption_store = store(group="corrupter")
    corruption_store(
        NonEquivAtom14Interpolant,
        name="nonequiv_atom14fm_protein")
    corruption_store(
        MultiSE3Interpolant,
        cfg=builds(SE3InterpolantConfig),
        name="multiframefm_protein")

    datamodule_store = store(group="datamodule")
    datamodule_store(
        pbuilds(
            FramediffDataModule,
            data_dir=f"{os.environ.get('REPO_ROOT')}/data/cath",
            batch_size=3000,
            num_workers=4,
            use_val_split=True,
            min_ordered_percent=0.0
        ),
        name="cath")
    datamodule_store(
        pbuilds(
            FramediffDataModule,
            data_dir=f"{os.environ.get('REPO_ROOT')}/data/framediff_clustered",
            batch_size=3000,
            num_workers=4
        ),
        name="framediff")
    datamodule_store(
        pbuilds(
            FramediffDataModule,
            data_dir=f"{os.environ.get('REPO_ROOT')}/data/framediff95",
            batch_size=3000,
            num_workers=4
        ),
        name="framediff95")
    datamodule_store(
        pbuilds(
            FramediffDataModule,
            data_dir=f"{os.environ.get('REPO_ROOT')}/data/frameflow",
            batch_size=3000,
            num_workers=4
        ),
        name="frameflow")
    datamodule_store(
        pbuilds(
            FramediffDataModule,
            data_dir=f"{os.environ.get('REPO_ROOT')}/data/afdb_128",
            batch_size=3000,
            num_workers=4
        ),
        name="afdb_128")
    datamodule_store(
        pbuilds(
            FramediffDataModule,
            data_dir=f"{os.environ.get('REPO_ROOT')}/data/afdb_512",
            batch_size=3000,
            num_workers=4
        ),
        name="afdb_512")
    datamodule_store(
        pbuilds(
            FramediffDataModule,
            data_dir=f"{os.environ.get('REPO_ROOT')}/data/afdb_512_clusters",
            batch_size=3000,
            num_workers=8
        ),
        name="afdb_512_clusters")

    lmodule_store = store(group="lmodule")
    lmodule_store(
        pbuilds(ProteinModule),
        name="protein"
    )

    # latent_fm_wrapper = ChimeraLatentWrapper
    model_store = store(group="model")
    model_store(AtomDenoiser, name="nonequiv_atom14fm_protein")
    model_store(IpaMultiRigidDenoiser, name="multiframefm_protein")

    harness_store = store(group="harness")
    harness_store(pbuilds(NonEquivAtom14Interpolation), name="nonequiv_atom14fm_protein")
    harness_store(pbuilds(MultiFrameInterpolation), name="multiframefm_protein")

    tasks_store = store(group="tasks")
    tasks_store({
        'unconditional_freq': 1.0,
        'backbone_motif_scaffolding_freq': 0.0,
        'residue_motif_scaffolding_freq': 0.0,
        'inverse_rotamer_motif_scaffolding_freq': 0.0,
        'folding_freq': 0.0,
        'diffusion_forcing_freq': 0.0,
        'sidechain_design_freq': 0.0,
    }, name='default')
    tasks_store(group='tasks/unconditional')(builds(UnconditionalGeneration), name='default')
    tasks_store(group='tasks/backbone_motif_scaffolding')(builds(BackboneMotifScaffolding), name='default')
    tasks_store(group='tasks/residue_motif_scaffolding')(builds(ResidueMotifScaffolding), name='default')
    tasks_store(group='tasks/inverse_rotamer_motif_scaffolding')(builds(InverseRotamerMotifScaffolding), name='default')
    tasks_store(group='tasks/folding')(builds(Folding), name='default')
    tasks_store(group='tasks/diffusion_forcing')(builds(DiffusionForcing), name='default')
    tasks_store(group='tasks/sidechain_design')(builds(SidechainDesign), name='default')

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
    optim_store(pbuilds(torch.optim.Adam, lr=1e-4), name="adam")
    optim_store(pbuilds(get_std_opt, d_model=128), name="noam")

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
            {"paradigm": "diffusion"},
            {"domain": "bb"},
            {"datamodule": "cath"},
            {"lmodule": "${domain}"},
            {"corrupter": "${paradigm}_${domain}"},
            {"model": "${paradigm}_${domain}"},
            {"harness": "${paradigm}_${domain}"},
            {"tasks": "default"},
            {"tasks/unconditional": "default"},
            {"tasks/backbone_motif_scaffolding": "default"},
            {"tasks/residue_motif_scaffolding": "default"},
            {"tasks/inverse_rotamer_motif_scaffolding": "default"},
            {"tasks/folding": "default"},
            {"tasks/diffusion_forcing": "default"},
            {"tasks/sidechain_design": "default"},
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

    # store(
    #     {
    #         "model_dir": "",
    #         "out_prefix": "samples",
    #         "save_traj": False,
    #         "debug": False,
    #         "checkpoint_idx": -1
    #     },
    #     name='default'
    # )

    sampler_store = store(group="sampler")
    sampler_store(
        builds(
            SamplingDataModule
        ),
        name="default"
    )

    SamplingConfig = make_config(
        model_dir="",
        out_prefix="samples",
        save_traj=False,
        checkpoint_idx=-1,
        hydra_defaults=[
            {"sampler": "default"},
            '_self_'
        ],
    )

    store(SamplingConfig, name="main")
    store.add_to_hydra_store()