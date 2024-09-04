import os

import torch
from hydra_zen import store, just, builds, make_custom_builds_fn, make_config, kwargs_of
from hydra.conf import HydraConf, RunDir
import omegaconf
from datetime import timedelta

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from proteinzen.data.datasets.datamodule import ProteinDataModule, FramediffDataModule, GeomDataModule

from proteinzen.stoch_interp.interpolate.se3 import SE3Interpolant, SE3InterpolantConfig
from proteinzen.stoch_interp.interpolate.protein import ProteinInterpolant, ProteinDirichletInterpolant, ProteinDirichletChiInterpolant, ProteinDirichletMultiChiInterpolant, ProteinFisherInterpolant, ProteinCatFlowInterpolant, ProteinFisherMultiChiInterpolant
from proteinzen.stoch_interp.interpolate.molecule import HarmonicPriorInterpolant
from proteinzen.stoch_interp.interpolate.torsion import TorsionInterpolant
from proteinzen.stoch_interp.interpolate.dirichlet import DirichletConditionalFlow

from proteinzen.model.denoiser.bb.frames import GraphIpaFrameDenoiser, DynamicGraphIpaFrameDenoiser
from proteinzen.model.denoiser.protein.frames_seq import DynamicGraphIpaFrameSeqDenoiser
from proteinzen.model.denoiser.protein.frames_seq_chi import DynamicGraphIpaFrameDirichletChiDenoiser
from proteinzen.model.denoiser.protein.frames_seq_multichi import DynamicGraphIpaFrameSeqMultiChiDenoiser
from proteinzen.model.denoiser.sidechain.ipmp_latent import IPMPDenoiser
from proteinzen.model.denoiser.molecule.tfn_r3 import MoleculeDenoiser
from proteinzen.model.denoiser.molecule.torsional import MoleculeTorsionDenoiser

from proteinzen.model.design.ipmp import IPMPEncoder, IPMPDecoder
from proteinzen.model.design.ipmp_seq_only import IPMPDenoiser as DirichletIPMPDenoiser
from proteinzen.model.wrappers.sidechain import IPMPLatentSidechainWrapper
from proteinzen.model.wrappers.protein import IPMPLatentWrapper, TFNLatentWrapper, ChimeraLatentWrapper, IPMPDenseLatentWrapper, TFNDenseLatentWrapper

from proteinzen.tasks.fm.bb import BackboneFrameInterpolation
from proteinzen.tasks.fm.protein import ProteinInterpolation, ProteinSeqInterpolation, ProteinSeqMultiChiInterpolation
from proteinzen.tasks.fm.molecule import HarmonicFlowMatching, TorsionalFlowMatching
from proteinzen.tasks.fm.sidechain import DirichletFlowMatching

from proteinzen.runtime.lmod import BackboneModule, SidechainModule, ProteinModule, MoleculeModule

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


def config_hydra_store():
    # change hydra conf defaults to avoid run collisions
    store(
        HydraConf(
            run=RunDir(dir="./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S-%f}")
    ))

    ## switches to allow for hot-swapping between paradigms and domains
    paradigm_store = store(group="paradigm")
    paradigm_store({"paradigm": "diffusion"}, name="diffusion")
    paradigm_store({"paradigm": "fm"}, name="fm")
    paradigm_store({"paradigm": "densefm"}, name="densefm")
    paradigm_store({"paradigm": "dirichlet"}, name="dirichlet")
    paradigm_store({"paradigm": "fisher"}, name="fisher")
    paradigm_store({"paradigm": "catflow"}, name="catflow")

    domain_store = store(group="domain")
    domain_store({"domain": "backbone"}, name="bb")
    domain_store({"domain": "sidechain"}, name="sidechain")
    domain_store({"domain": "protein"}, name="protein")
    domain_store({"domain": "protein_multichi"}, name="protein_multichi")
    domain_store({"domain": "molecule"}, name="molecule")

    corruption_store = store(group="corrupter")
    corruption_store(
        ProteinInterpolant,
        se3_cfg=builds(SE3InterpolantConfig),
        name="fm_sidechain")
    corruption_store(
        SE3Interpolant,
        cfg=builds(SE3InterpolantConfig),
        name="fm_bb")
    corruption_store(
        ProteinInterpolant,
        se3_cfg=builds(SE3InterpolantConfig),
        name="fm_protein")
    corruption_store(
        ProteinInterpolant,
        se3_cfg=builds(SE3InterpolantConfig),
        name="densefm_protein")
    corruption_store(
        ProteinDirichletInterpolant,
        se3_cfg=builds(SE3InterpolantConfig),
        name="dirichlet_protein")
    corruption_store(
        ProteinFisherInterpolant,
        se3_cfg=builds(SE3InterpolantConfig),
        name="fisher_protein")
    corruption_store(
        ProteinCatFlowInterpolant,
        se3_cfg=builds(SE3InterpolantConfig),
        name="catflow_protein")
    # corruption_store(
    #     ProteinDirichletChiInterpolant,
    #     se3_cfg=builds(SE3InterpolantConfig),
    #     name="dirichlet_protein")
    corruption_store(
        ProteinDirichletMultiChiInterpolant,
        se3_cfg=builds(SE3InterpolantConfig),
        name="dirichlet_protein_multichi")
    corruption_store(
        ProteinFisherMultiChiInterpolant,
        se3_cfg=builds(SE3InterpolantConfig),
        name="fisher_protein_multichi")
    # corruption_store(
    #     HarmonicPriorInterpolant,
    #     name="fm_molecule")
    corruption_store(
        TorsionInterpolant,
        name="fm_molecule")
    corruption_store(
        DirichletConditionalFlow,
        name="dirichlet_sidechain")

    datamodule_store = store(group="datamodule")
    datamodule_store(
        pbuilds(
            ProteinDataModule,
            data_dir=f"{os.environ.get('REPO_ROOT')}/data/cath",
            batch_size=3000,
            num_workers=4
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
            GeomDataModule,
            data_dir=f"{os.environ.get('REPO_ROOT')}/data/geom_drugs",
            batch_size=3000,
            num_workers=4
        ),
        name="geom")

    lmodule_store = store(group="lmodule")
    lmodule_store(
        pbuilds(BackboneModule),
        name="bb"
    )
    lmodule_store(
        pbuilds(SidechainModule),
        name="sidechain"
    )
    lmodule_store(
        pbuilds(MoleculeModule),
        name="molecule"
    )
    lmodule_store(
        pbuilds(ProteinModule),
        name="protein"
    )
    lmodule_store(
        pbuilds(ProteinModule),
        name="protein_multichi"
    )

    # latent_fm_wrapper = ChimeraLatentWrapper
    latent_fm_wrapper = TFNLatentWrapper
    model_store = store(group="model")
    # model_store(TFNLatentWrapper, name="fm_sidechain")
    model_store(latent_fm_wrapper, name="fm_sidechain")
    model_store(GraphIpaFrameDenoiser, name="diffusion_bb")
    # model_store(GraphIpaFrameDenoiser, name="fm_bb")
    model_store(DynamicGraphIpaFrameDenoiser, name="fm_bb")
    # model_store(IpaScoreWrapper, name="fm_bb")
    # model_store(IPMPLatentWrapper, name="fm_protein")
    # model_store(TFNLatentWrapper, name="fm_protein")
    model_store(latent_fm_wrapper, name="fm_protein")
    model_store(IPMPDenseLatentWrapper, name="densefm_protein")
    # model_store(TFNDenseLatentWrapper, name="fm_protein")
    model_store(DynamicGraphIpaFrameSeqDenoiser, name="dirichlet_protein")
    model_store(DynamicGraphIpaFrameSeqDenoiser, name="fisher_protein")
    model_store(DynamicGraphIpaFrameSeqDenoiser, name="catflow_protein")
    # model_store(DynamicGraphIpaFrameDirichletChiDenoiser, name="dirichlet_protein")
    model_store(DynamicGraphIpaFrameSeqMultiChiDenoiser, name="dirichlet_protein_multichi")
    model_store(DynamicGraphIpaFrameSeqMultiChiDenoiser, name="fisher_protein_multichi")
    #model_store(MoleculeDenoiser, name="fm_molecule")
    model_store(MoleculeTorsionDenoiser, name="fm_molecule")
    # model_store(
    #     DensityLatentSidechainWrapper,
    #     name="diffusion_sidechain")
    model_store(
        IPMPLatentSidechainWrapper,
        name="diffusion_sidechain")
    # model_store(
    #     BilevelIPMPLatentSidechainWrapper,
    #     name="diffusion_sidechain")
    # model_store(
    #     IPALatentSidechainWrapper,
    #     name="diffusion_sidechain")
    model_store(
        DirichletIPMPDenoiser,
        name="dirichlet_sidechain")

    task_store = store(group="tasks")
    task_store(pbuilds(DirichletFlowMatching), name="dirichlet_sidechain")
    task_store(pbuilds(ProteinInterpolation), name="fm_sidechain")
    task_store(pbuilds(BackboneFrameInterpolation), name="fm_bb")
    task_store(pbuilds(ProteinInterpolation), name="fm_protein")
    task_store(pbuilds(ProteinInterpolation), name="densefm_protein")
    task_store(pbuilds(ProteinSeqInterpolation), name="dirichlet_protein")
    task_store(pbuilds(ProteinSeqInterpolation), name="fisher_protein")
    # task_store(pbuilds(ProteinDirichletChiInterpolation), name="dirichlet_protein")
    task_store(pbuilds(ProteinSeqMultiChiInterpolation), name="dirichlet_protein_multichi")
    task_store(pbuilds(ProteinSeqMultiChiInterpolation), name="fisher_protein_multichi")
    # task_store(pbuilds(HarmonicFlowMatching), name="fm_molecule")
    task_store(pbuilds(TorsionalFlowMatching), name="fm_molecule")

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
        name="bb")
    exp_store(
        ModelCheckpoint,
        save_top_k=-1,
        save_on_train_epoch_end=True,
        save_last=True,
        train_time_interval=timedelta(hours=4),
        group="experiment/checkpointer",
        name="sidechain")
    exp_store(
        ModelCheckpoint,
        save_top_k=-1,
        save_on_train_epoch_end=True,
        save_last=True,
        train_time_interval=timedelta(days=1),
        group="experiment/checkpointer",
        name="protein")
    exp_store(
        ModelCheckpoint,
        save_top_k=-1,
        save_on_train_epoch_end=True,
        save_last=True,
        train_time_interval=timedelta(days=1),
        group="experiment/checkpointer",
        name="protein_multichi")
    exp_store(
        pbuilds(
            ModelCheckpoint,
            dirpath="ckpt",
            every_n_epochs=1,
            save_on_train_epoch_end=False,
            save_last=True,
            save_top_k=3,
            monitor="valid/conf_min_rmsd",
            mode="max"
        ),
        group="experiment/checkpointer",
        name="molecule")

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
            {"tasks": "${paradigm}_${domain}"},
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