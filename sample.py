""" Train a model """

import argparse
import json
import logging
import io
import os
import glob

import hydra
from hydra_zen import zen, load_from_yaml, instantiate
import omegaconf
import torch
import numpy as np
import pandas as pd

from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

import wandb

from proteinzen.runtime.config import config_sampling_hydra_store
from proteinzen.data.io.atom91 import atom91_to_pdb, atom91_to_chain, chains_to_model, models_to_struct, save_struct
from proteinzen.utils.atom_reps import atom14_to_atom91
from proteinzen.data.openfold.residue_constants import restypes
from proteinzen.data.io.protein import PDB_CHAIN_IDS
from proteinzen.runtime.lmod import ProteinModule


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
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = 1

        log.info(f"Using devices: {devices}")

        trainer = Trainer(
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
        )
        ret = trainer.predict(
            model=self._model,
            datamodule=self._sampler,
            ckpt_path=self._cfg['ckpt_path']
        )
        os.chdir(self._cfg['samples_dir'])

        samples_metadata = {}

        curr_sample_id = 0
        for batch in ret:
            for sample_data in batch:
                sample_coords = sample_data['sample_coord']
                sample_len = sample_coords.shape[0]
                seq = sample_data['seq']
                sample_task = sample_data['input']['task'].task_name
                # sample_task = sample_data['input']['task']['task_name']

                # seq_lt = "".join([restypes[i] for i in seq.tolist()])
                # atom91, _ = atom14_to_atom91(seq_lt, sample_coords.numpy(force=True))
                # sample_name = f"len_{sample_len}_protein_{curr_sample_id}"
                # atom91_to_pdb(seq_lt, atom91, sample_name)

                chain_idx = sample_data['chain_idx']
                chains = []

                sample_name = f"len_{sample_len}_protein_{curr_sample_id}" #.pdb"
                for i in range(int(chain_idx.max()) + 1):
                    select = (chain_idx == i)
                    chain_seq = seq[select]
                    chain_seq_lt = "".join([restypes[i] for i in chain_seq.tolist()])
                    chain_atom91, _ = atom14_to_atom91(chain_seq_lt, sample_coords[select].numpy(force=True))
                    # atom91_to_pdb(seq_lt, atom91, sample_name)
                    chain_i = atom91_to_chain(
                        chain_seq_lt,
                        chain_atom91,
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
                    )
                    chains.append(chain_i)
                model = chains_to_model(chains, model_id=0)
                save_struct(models_to_struct([model]), sample_name + ".pdb")

                if self._cfg['save_traj']:
                    clean_traj = sample_data['clean_traj'] # batch['all_R_clean_trajs']
                    clean_traj_seq = sample_data['clean_traj_seq']
                    prot_traj = sample_data['prot_traj']
                    prot_traj_seq = sample_data['prot_traj_seq']

                    clean_traj_name = f"len_{sample_len}_protein_{curr_sample_id}_clean_traj.pdb"
                    prot_traj_name = f"len_{sample_len}_protein_{curr_sample_id}_prot_traj.pdb"
                    clean_models = []
                    prot_models = []

                    for t, _ in enumerate(clean_traj):
                        clean = clean_traj[t]
                        clean_seq = clean_traj_seq[t]
                        # print(clean_seq)
                        clean_seq = "".join([restypes[j] for j in clean_seq.tolist()])
                        clean_atom91, _ = atom14_to_atom91(clean_seq, clean.numpy(force=True))
                        clean_chain = atom91_to_chain(
                            clean_seq,
                            clean_atom91,
                            "A"
                        )
                        clean_model = chains_to_model([clean_chain], model_id=t)
                        clean_models.append(clean_model)

                        prot = prot_traj[t]
                        prot_seq = prot_traj_seq[t]
                        # print(prot_seq)
                        prot_seq = "".join([restypes[j] for j in prot_seq.tolist()])
                        prot_atom91, _ = atom14_to_atom91(prot_seq, prot.numpy(force=True))
                        prot_chain = atom91_to_chain(
                            prot_seq,
                            prot_atom91,
                            "A"
                        )
                        prot_model = chains_to_model([prot_chain], model_id=t)
                        prot_models.append(prot_model)
                    save_struct(models_to_struct(reversed(clean_models)), clean_traj_name)
                    save_struct(models_to_struct(reversed(prot_models)), prot_traj_name)

                task_kwargs = sample_data['input']['task'].kwargs
                # task_kwargs = sample_data['input']['task']
                sample_path = os.path.join(self._cfg['samples_dir'], sample_name + ".pdb")
                samples_metadata[sample_name] = {
                    "path": sample_path,
                    "task": sample_task,
                    "name": task_kwargs['name'] if 'name' in task_kwargs else None,
                    "length": sample_len,
                    "fixed_res_idx": [i+1 for i in sample_data['fixed_res_idx'].tolist()],  # 1-indexed chain for pyrosetta
                    "fixed_res_chain": [PDB_CHAIN_IDS[int(i)] for i in sample_data['fixed_res_chain_idx']],
                    "fixed_bb_res_idx": [i+1 for i in sample_data['fixed_bb_res_idx'].tolist()],  # 1-indexed chain for pyrosetta
                    "fixed_bb_chain": [PDB_CHAIN_IDS[int(i)] for i in sample_data['fixed_bb_chain_idx']],
                    "fixed_seq_res_idx": [i+1 for i in sample_data['fixed_seq_res_idx'].tolist()],  # 1-indexed chain for pyrosetta
                    "fixed_seq_chain": [PDB_CHAIN_IDS[int(i)] for i in sample_data['fixed_seq_chain_idx']],
                }
                curr_sample_id += 1

        with open("../samples_metadata.json", 'w') as fp:
            json.dump(samples_metadata, fp)

        pmpnn_fixed_pos_dict = {}
        for name, metadata in samples_metadata.items():
            entry = {
                chain: []
                for chain in set(metadata['fixed_seq_chain'])
            }
            for pos, pos_chain in zip(metadata['fixed_seq_res_idx'], metadata['fixed_seq_chain']):
                entry[pos_chain].append(pos)
            pmpnn_fixed_pos_dict[name] = entry

        with open("../pmpnn_fixed_pos_dict.jsonl", 'w') as fp:
            json.dump(pmpnn_fixed_pos_dict, fp)


def main(sampler,
         corrupter,
         zen_cfg):
    # change into the output directory
    # os.chdir(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Experiment started in folder: {os.getcwd()}")

    # so we can add extra entries in the config
    zen_cfg = omegaconf.OmegaConf.to_container(zen_cfg)
    assert zen_cfg is not None

    run_dir = zen_cfg['model_dir']
    ckpt_list = list(glob.glob(
        os.path.join(
            run_dir,
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
    ckpt_path = epoch_list[zen_cfg["checkpoint_idx"]]
    print(ckpt_path)
    zen_cfg['ckpt_path'] = ckpt_path

    config_path = os.path.join(
        run_dir,
        ".hydra",
        "config.yaml"
    )
    model_cfg = load_from_yaml(config_path)
    lmodule_init = instantiate(model_cfg['lmodule'])
    harness_init = instantiate(model_cfg['harness'])
    # corrupter = instantiate(model_cfg['corrupter'])
    model = instantiate(model_cfg['model'])

    harness = harness_init(corrupter, None)
    model = lmodule_init(model, None, harness)

    zen_cfg['samples_dir'] = os.path.join(
        run_dir, zen_cfg['out_prefix'], "samples"
    )
    os.makedirs(zen_cfg['samples_dir'], exist_ok=True)

    with open(os.path.join(run_dir, zen_cfg['out_prefix'], "run.log"), 'w') as fp:
        fp.write(f"Sampling config path: {zen_cfg['sampler']['tasks_yaml']}")

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
