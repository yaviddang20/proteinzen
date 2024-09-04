""" Train a model """

import argparse
import logging
import io
import os
import glob

import hydra
from hydra_zen import zen, load_from_yaml
import omegaconf
import torch
import pandas as pd
import tqdm

from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch import Trainer

from proteinzen.runtime.config import config_hydra_store, remove_zen_keys
from proteinzen.data.datasets.datamodule import FramediffDataModule
from proteinzen.tasks.task import single_task_sampler


# A logger for this file
log = logging.getLogger(__name__)


def main(model,
         corrupter,
         datamodule,
         # lmodule,
         experiment,
         tasks,
         zen_cfg):
    device = "cuda:0"
    # change into the output directory
    # os.chdir(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Experiment started in folder: {os.getcwd()}")

    # datamodule and optim are all partial'd __init__s
    # so we instantiate instances of each
    # model = lmodule(model, experiment['optim'])
    task_sampler = single_task_sampler(tasks(corrupter))
    datamodule_inst = FramediffDataModule(
        task_sampler=task_sampler,
        **remove_zen_keys(zen_cfg['datamodule']))

    assert zen_cfg['experiment']['warm_start'] is not None
    ckpt = torch.load(zen_cfg['experiment']['warm_start'])
    state_dict = ckpt['state_dict']
    state_dict = {k.removeprefix("model."): v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    if not os.path.exists(zen_cfg['experiment']['out_dir']):
        os.makedirs(zen_cfg['experiment']['out_dir'], exist_ok=True)

    filepaths = []
    for batch in tqdm.tqdm(datamodule_inst.train_dataloader()):
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model.encoder(batch)
        for i, pdb_name in enumerate(batch['name']):
            select = (batch['residue'].batch == i)
            latent_mu = outputs['latent_mu'][select].detach().cpu()
            latent_logvar = outputs['latent_logvar'][select].detach().cpu()
            save_dict = {
                'latent_mu': latent_mu,
                'latent_logvar': latent_logvar
            }
            save_path = os.path.abspath(os.path.join(zen_cfg['experiment']['out_dir'], pdb_name + ".pt"))
            torch.save(save_dict, save_path)
            filepaths.append((pdb_name, save_path))

    pdb_names, cache_paths = zip(*filepaths)
    df_list = {
        "pdb_name": pdb_names,
        "latent_cache_file": cache_paths
    }

    df = pd.DataFrame(df_list)
    df.to_csv(os.path.join(zen_cfg['experiment']['out_dir'], "metadata.csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--checkpoint_idx", default=-1, type=int)
    args = parser.parse_args()

    config_path = os.path.join(
        args.run_dir,
        ".hydra",
        "config.yaml"
    )

    ckpt_list = list(glob.glob(
        os.path.join(
            args.run_dir,
            "lightning_logs/version_0/checkpoints/*.ckpt",
        )
    ))
    epoch_list = []
    for ckpt_path in ckpt_list:
        if ckpt_path.split("/")[-1] == "last.ckpt":
            epoch_list.append((ckpt_path, 1e6))
        else:
            epoch = ckpt_path.split("=")[1].split("-")[0]
            epoch_list.append((ckpt_path, int(epoch)))

    epoch_list = sorted(epoch_list, key=lambda x: x[1])
    epoch_list, _ = zip(*epoch_list)
    ckpt_path = epoch_list[args.checkpoint_idx]
    print(ckpt_path)

    cfg = load_from_yaml(config_path)
    cfg['experiment']['warm_start'] = ckpt_path
    cfg['datamodule']['batch_size'] = 4000
    cfg['datamodule']['sample_from_clusters'] = False
    cfg['datamodule']['predict_on_train'] = True
    cfg['datamodule']['data_dir'] = args.data_dir

    os.makedirs(
        args.out_dir,
        exist_ok=True
    )
    cfg['experiment']['out_dir'] = args.out_dir

    zen(main, unpack_kwargs=True)(cfg)
