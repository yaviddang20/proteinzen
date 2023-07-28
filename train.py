""" Train a model """

import json
import logging
import os
from functools import partial

import dgl
import numpy as np
import torch
from lightning import Fabric
from hydra.utils import get_original_cwd
from hydra_zen import make_config, make_custom_builds_fn, zen
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

from se3_transformer.model.fiber import Fiber

from ligbinddiff.data.dataloaders.cath import CATHDataset, ProteinGraphDataset, BatchSampler
from ligbinddiff.diffusion.density_diff import LearnableVPDensityDiffuser, LinearDiscreteVPDensityDiffuser
from ligbinddiff.model.seq_des.density.se3_transformer.se3_transformer import DensityDenoiser
from ligbinddiff.runtime.training import cath_train_loop
from ligbinddiff.runtime.inference import cath_inference_loop

from ligbinddiff.utils.fiber import gen_compact_nmax_fiber, gen_full_n_channel_fiber

# A logger for this file
log = logging.getLogger(__name__)

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)
ExperimentConfig = make_config(
    job_id=-1,
    n_max=5,
    l_max=1,
    n_h_channels=32,
    dataset_init=pbuilds(ProteinGraphDataset),
    denoiser_init=pbuilds(DensityDenoiser),
    diffuser_init=pbuilds(LinearDiscreteVPDensityDiffuser),
    optimizer_init=pbuilds(optim.Adam),
    batch_size=500,
    num_workers=4,
    num_epoch=100,
    num_warmup_epoch=0,
    comment="",
    training_curve_path="training_curve.json",
    checkpoint_prefix="diffuser",
    resume_path=None
)


def build_model_and_optim(
        fabric,
        denoiser_init,
        diffuser_init,
        optimizer_init,
        l_max,
        n_max,
        n_input_channels,
        n_h_channels,
        resume_path,
        checkpoint_prefix
    ):
    # pre-set based on input features set
    fiber_in = Fiber({
        0: 6,
        1: 3
    })
    fiber_edge = Fiber({
        0: 32,
        1: 1
    })
    fiber_node = gen_full_n_channel_fiber(l_max, n_h_channels)
    fiber_density = gen_compact_nmax_fiber(n_max) * n_input_channels

    denoiser = denoiser_init(
        fiber_in=fiber_in,
        fiber_node=fiber_node,
        fiber_density=fiber_density,
        fiber_edge=fiber_edge,
    )
    diffuser = diffuser_init(denoiser=denoiser, n_channels=n_input_channels)
    optimizer = optimizer_init(diffuser.parameters())
    diffuser, optimizer = fabric.setup(diffuser, optimizer)

    if resume_path:
        log.info(f"Loading state dict from {resume_path}")
        state_dict = torch.load(os.path.join(resume_path, f"{checkpoint_prefix}_last.pt"))
        diffuser.load_state_dict(state_dict["diffuser"])
        optimizer.load_state_dict(state_dict["optimizer"])

    # fabric.clip_gradients(diffuser, optimizer, clip_val=2.0)
    return diffuser, optimizer


def build_dataloaders(fabric, n_max, dataset_init, batch_size, num_workers):
    dataloader = lambda x: DataLoader(
        x,
        num_workers=num_workers,
        batch_sampler=BatchSampler(SequentialSampler(x), batch_size=batch_size),
        collate_fn=dgl.batch)
    top_folder = get_original_cwd()
    cath = CATHDataset(path=os.path.join(top_folder, "data/cath/chain_set.jsonl"),
                       splits_path=os.path.join(top_folder, "data/cath/chain_set_splits.json"))
    dataset = partial(ProteinGraphDataset, density_nmax=n_max, channel_atoms=True, bb_density=False)
    trainset, valset = map(dataset,
                           (cath.train, cath.val))
    train_dataloader, val_dataloader = map(dataloader,
                                           (trainset, valset))
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader, use_distributed_sampler=False)
    return train_dataloader, val_dataloader


def main(denoiser_init,
         l_max,
         n_max,
         n_h_channels,
         diffuser_init,
         optimizer_init,
         dataset_init,
         batch_size,
         num_workers,
         num_epoch=100,
         num_warmup_epoch=0,
         training_curve_path="training_curve.json",
         checkpoint_prefix="diffuser",
         resume_path=None
    ):
    if torch.cuda.is_available():
        device = "cuda"
        torch.set_float32_matmul_precision('medium')
    else:
        device = "cpu"

    # torch.autograd.set_detect_anomaly(True)

    fabric = Fabric(accelerator=device)
    fabric.launch()

    train_dataloader, val_dataloader = build_dataloaders(fabric, n_max, dataset_init, batch_size, num_workers)
    n_input_channels = train_dataloader.dataset.num_channels
    diffuser, optimizer = build_model_and_optim(
        fabric,
        denoiser_init,
        diffuser_init,
        optimizer_init,
        l_max,
        n_max,
        n_input_channels,
        n_h_channels,
        resume_path,
        checkpoint_prefix=checkpoint_prefix
    )

    if resume_path:
        with open(os.path.join(resume_path, training_curve_path)) as fp:
            training_curve = json.load(fp)
            start_epoch = max((e["epoch"] for e in training_curve)) + 1
            best_val_loss = min((e["val_loss"] for e in training_curve))
        log.info(f"Resuming training from epoch {start_epoch}")
    else:
        training_curve = []
        start_epoch = 0
        best_val_loss = np.inf

    for epoch in range(start_epoch, num_epoch):
        log.info(f"Epoch {epoch}")
        train_dict = cath_train_loop(
            diffuser, train_dataloader, optimizer, fabric, train=True, warmup=(epoch < num_warmup_epoch))
        train_loss = train_dict["epoch_loss"]
        train_denoise = train_dict["denoising_loss"]
        train_ref_noise = train_dict["ref_noise"]
        train_seq = train_dict["seq_loss"]
        train_rmsd = train_dict["rmsd_loss"]
        log.info(f"Epoch {epoch}: train {train_loss} {train_denoise} {train_seq} {train_rmsd} | ref noise {train_ref_noise}")

        state_dict = {
            "diffuser": diffuser.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(state_dict, f"{checkpoint_prefix}_last.pt")

        with torch.no_grad():
            val_dict = cath_inference_loop(
                diffuser, val_dataloader, num_steps=100, truncate=10)
        # val_loss = val_dict["epoch_loss"]
        val_denoise = val_dict["denoising_loss"]
        # val_ref_noise = val_dict["ref_noise"]
        val_seq = val_dict["seq_loss"]
        val_seq_recov = val_dict["seq_recov"]
        val_rmsd = val_dict["rmsd_loss"]
        log.info(f"Epoch {epoch}: val {val_denoise} {val_seq} {val_seq_recov}  {val_rmsd}")
        val_loss = val_denoise + val_seq + val_rmsd

        epoch_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_denoise": train_denoise,
            "train_loss_seq": train_seq,
            "train_loss_atom91_rmsd": train_rmsd,
            "train_ref_noise": train_ref_noise,
            "val_loss": val_loss,
            "val_loss_denoise": val_denoise,
            "val_loss_seq": val_seq,
            "val_seq_recov": val_seq_recov,
            "val_loss_atom91_rmsd": val_rmsd,
        }

        training_curve.append(epoch_dict)
        with open(training_curve_path, 'w') as fp:
            json.dump(training_curve, fp)

        state_dict = {
            "diffuser": diffuser.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state_dict, f"{checkpoint_prefix}_best.pt")


task_function = zen(main)

if __name__ == '__main__':
    from hydra_zen import ZenStore

    store = ZenStore(deferred_hydra_store=False)
    store(ExperimentConfig, name="diffuser_train")

    task_function.hydra_main(
        config_name="diffuser_train",
        version_base="1.1",
        config_path="."
    )
