import torch
from hydra_zen import store, just, builds, make_custom_builds_fn, make_config, kwargs_of
import omegaconf

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from ligbinddiff.data.datasets.datamodule import ProteinDataModule, FramediffDataModule, GeomDataModule

from ligbinddiff.diffusion.noisers.se3_diffuser import SE3Diffuser
from ligbinddiff.diffusion.noisers.latent import SidechainDiffuser
from ligbinddiff.stoch_interp.interpolate.se3 import SE3Interpolant, SE3InterpolantConfig
from ligbinddiff.stoch_interp.interpolate.protein import ProteinInterpolant, ProteinDirichletInterpolant, ProteinDirichletChiInterpolant, ProteinDirichletMultiChiInterpolant
from ligbinddiff.stoch_interp.interpolate.molecule import HarmonicPriorInterpolant
from ligbinddiff.stoch_interp.interpolate.dirichlet import DirichletConditionalFlow

from ligbinddiff.model.denoiser.bb.frames import GraphIpaFrameDenoiser, DynamicGraphIpaFrameDenoiser
from ligbinddiff.model.denoiser.protein.frames_dirichlet import DynamicGraphIpaFrameDirichletDenoiser
from ligbinddiff.model.denoiser.protein.frames_dirichlet_chi import DynamicGraphIpaFrameDirichletChiDenoiser
from ligbinddiff.model.denoiser.protein.frames_dirichlet_multichi import DynamicGraphIpaFrameDirichletMultiChiDenoiser
from ligbinddiff.model.denoiser.sidechain.ipmp_latent import IPMPDenoiser
from ligbinddiff.model.denoiser.molecule.tfn import MoleculeDenoiser
from ligbinddiff.stoch_interp.flow_matchers.frames import GraphFrameFlow

from ligbinddiff.model.design.ipmp import IPMPEncoder, IPMPDecoder
from ligbinddiff.model.design.ipmp_seq_only import IPMPDenoiser as DirichletIPMPDenoiser
from ligbinddiff.model.autoencoder.ipa import IPAEncoder, IPADecoder
from ligbinddiff.model.wrappers.sidechain import IPMPLatentSidechainWrapper, IPALatentSidechainWrapper, DensityLatentSidechainWrapper, BilevelIPMPLatentSidechainWrapper
from ligbinddiff.model.wrappers.protein import IPMPLatentWrapper

from ligbinddiff.tasks.diffusion.bb import BackboneFrameNoising
from ligbinddiff.tasks.diffusion.sidechain import DesignLatentSidechainNoising
from ligbinddiff.tasks.fm.bb import BackboneFrameInterpolation
from ligbinddiff.tasks.fm.protein import ProteinInterpolation, ProteinDirichletInterpolation, ProteinDirichletChiInterpolation, ProteinDirichletMultiChiInterpolation
from ligbinddiff.tasks.fm.molecule import HarmonicFlowMatching
from ligbinddiff.tasks.fm.sidechain import DirichletFlowMatching

from ligbinddiff.runtime.lmod import BackboneModule, SidechainModule, ProteinModule, MoleculeModule

from ligbinddiff.runtime.optim import get_std_opt

# targets ZenStore
# pylint: disable=not-callable

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

se3_diffusion_cfg = {
    "diffuse_rot": True,
    "diffuse_trans": True,
    "so3": {
        "schedule": "logarithmic",
        "min_sigma": 0.1,
        "max_sigma": 1.5,
        "num_sigma": 1000,
        "num_omega": 1000,
        "cache_dir": '.cache/',
        "use_cached_score": False,
    },
    "r3": {
        "min_b": 0.1,
        "max_b": 20,
        "coordinate_scaling": 0.1,
    }
}

def config_hydra_store():
    ## switches to allow for hot-swapping between paradigms and domains
    paradigm_store = store(group="paradigm")
    paradigm_store({"paradigm": "diffusion"}, name="diffusion")
    paradigm_store({"paradigm": "fm"}, name="fm")
    paradigm_store({"paradigm": "dirichlet"}, name="dirichlet")

    domain_store = store(group="domain")
    domain_store({"domain": "backbone"}, name="bb")
    domain_store({"domain": "sidechain"}, name="sidechain")
    domain_store({"domain": "protein"}, name="protein")
    domain_store({"domain": "molecule"}, name="molecule")

    corruption_store = store(group="corrupter")
    corruption_store(
        SE3Diffuser,
        se3_conf=se3_diffusion_cfg,
        name="diffusion_bb")
    corruption_store(
        SidechainDiffuser,
        name="diffusion_sidechain")
    corruption_store(
        SE3Interpolant,
        cfg=builds(SE3InterpolantConfig),
        name="fm_bb")
    corruption_store(
        ProteinInterpolant,
        se3_cfg=builds(SE3InterpolantConfig),
        name="fm_protein")
    # corruption_store(
    #     ProteinDirichletInterpolant,
    #     se3_cfg=builds(SE3InterpolantConfig),
    #     name="dirichlet_protein")
    # corruption_store(
    #     ProteinDirichletChiInterpolant,
    #     se3_cfg=builds(SE3InterpolantConfig),
    #     name="dirichlet_protein")
    corruption_store(
        ProteinDirichletMultiChiInterpolant,
        se3_cfg=builds(SE3InterpolantConfig),
        name="dirichlet_protein")
    corruption_store(
        HarmonicPriorInterpolant,
        name="fm_molecule")
    corruption_store(
        DirichletConditionalFlow,
        name="dirichlet_sidechain")

    datamodule_store = store(group="datamodule")
    datamodule_store(
        pbuilds(
            ProteinDataModule,
            data_dir="/wynton/home/kortemme/alexjli/projects/ligbinddiff/data/cath",
            batch_size=3000,
            num_workers=4
        ),
        name="cath")
    datamodule_store(
        pbuilds(
            FramediffDataModule,
            data_dir="/wynton/home/kortemme/alexjli/projects/ligbinddiff/data/framediff_clustered",
            batch_size=3000,
            num_workers=4
        ),
        name="framediff")
    datamodule_store(
        pbuilds(
            FramediffDataModule,
            data_dir="/wynton/home/kortemme/alexjli/projects/ligbinddiff/data/frameflow",
            batch_size=3000,
            num_workers=4
        ),
        name="frameflow")
    datamodule_store(
        pbuilds(
            GeomDataModule,
            data_dir="/wynton/home/kortemme/alexjli/projects/ligbinddiff/data/geom_drugs",
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

    model_store = store(group="model")
    model_store(GraphIpaFrameDenoiser, name="diffusion_bb")
    # model_store(GraphIpaFrameDenoiser, name="fm_bb")
    model_store(DynamicGraphIpaFrameDenoiser, name="fm_bb")
    model_store(IPMPLatentWrapper, name="fm_protein")
    # model_store(DynamicGraphIpaFrameDirichletDenoiser, name="dirichlet_protein")
    # model_store(DynamicGraphIpaFrameDirichletChiDenoiser, name="dirichlet_protein")
    model_store(DynamicGraphIpaFrameDirichletMultiChiDenoiser, name="dirichlet_protein")
    model_store(MoleculeDenoiser, name="fm_molecule")
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
    task_store(pbuilds(BackboneFrameNoising), name="diffusion_bb")
    task_store(pbuilds(DesignLatentSidechainNoising), name="diffusion_sidechain")
    task_store(pbuilds(DirichletFlowMatching), name="dirichlet_sidechain")
    task_store(pbuilds(BackboneFrameInterpolation), name="fm_bb")
    task_store(pbuilds(ProteinInterpolation), name="fm_protein")
    # task_store(pbuilds(ProteinDirichletInterpolation), name="dirichlet_protein")
    # task_store(pbuilds(ProteinDirichletChiInterpolation), name="dirichlet_protein")
    task_store(pbuilds(ProteinDirichletMultiChiInterpolation), name="dirichlet_protein")
    task_store(pbuilds(HarmonicFlowMatching), name="fm_molecule")

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
        ), name="default")

    optim_store = exp_store(group="experiment/optim")
    optim_store(pbuilds(torch.optim.Adam, lr=1e-4), name="adam")
    optim_store(pbuilds(get_std_opt, d_model=128), name="noam")

    exp_store(
        pbuilds(
            ModelCheckpoint,
            dirpath="ckpt",
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_last=True,
            save_top_k=3,
            monitor="valid/non_coil_percent",
            mode="max"
        ),
        group="experiment/checkpointer",
        name="bb")
    exp_store(
        pbuilds(
            ModelCheckpoint,
            dirpath="ckpt",
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_last=True,
            save_top_k=3,
            monitor="valid/non_coil_percent",
            mode="max"
        ),
        group="experiment/checkpointer",
        name="sidechain")
    exp_store(
        pbuilds(
            ModelCheckpoint,
            dirpath="ckpt",
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_last=True,
            save_top_k=3,
            monitor="valid/non_coil_percent",
            mode="max"
        ),
        group="experiment/checkpointer",
        name="protein")
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
        debug=False,
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


if __name__ == '__main__':
    from hydra_zen import zen

    def main(model, corruption, trainer, datamodule, zen_cfg):
        print(model)
        print(corruption)
        print(trainer)
        print(datamodule)
        print(zen_cfg)

    def preprocess_cfg(cfg):
        print(cfg)
        return cfg

    config_hydra_store()

    zen(main, pre_call=preprocess_cfg).hydra_main(
        config_name="main",
        version_base="1.1"
    )
