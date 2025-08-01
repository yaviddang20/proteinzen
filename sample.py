""" Train a model """

import argparse
import json
import logging
import io
import os
import glob
from dataclasses import replace

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

from proteinzen.boltz.data.types import Structure

from proteinzen.data.featurize.sampling import construct_atoms
from proteinzen.runtime.config import config_sampling_hydra_store
from proteinzen.data.featurize.tokenize import Tokenized, update_structure
from proteinzen.data.featurize.sampling import construct_atoms
# from proteinzen.data.write.mmcif import to_mmcif
from proteinzen.data.write.pdb import to_pdb


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
                # sample_coords = sample_data['sample_coord']
                sample_output = sample_data['output_data']
                # TODO: idek why i have to do this...
                sample_output['structure']['mask'] = np.ones_like(sample_output['structure']['mask'].astype(bool))
                struct = Structure(**sample_output['structure'])
                sample_output = Tokenized(
                    tokens=sample_output['tokens'],
                    rigids=sample_output['rigids'],
                    bonds=sample_output['bonds'],
                    structure=struct,
                )
                struct = construct_atoms(sample_output, struct)
                sample_len = sample_output.tokens.shape[0]
                sample_name = f"len_{sample_len}_protein_{curr_sample_id}" #.pdb"
                struct = update_structure(struct, sample_output.rigids['tensor7'])

                if self._cfg['output_motif_chains']:
                    # we rename the motif chain to something different
                    # so that it will be separated when outputted
                    # it doesn't particularly matter what this letter is since it'll be
                    # overwritten by to_pdb
                    num_chains = len(struct.chains)
                    struct.chains['asym_id'] = np.arange(num_chains)

                    def get_next_free_chain_name(seen_names):
                        CHAIN_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                        for c in CHAIN_ALPHABET:
                            if c not in seen_names:
                                return c
                        raise ValueError("output has too many chains to be represented in .pdb format")

                    seen_names = []
                    new_chain_names = []
                    for chain in struct.chains:
                        if chain['name'] not in seen_names:
                            seen_names.append(chain['name'])
                            new_chain_names.append(chain['name'])
                        else:
                            chain_rename = get_next_free_chain_name(seen_names)
                            seen_names.append(chain_rename)
                            new_chain_names.append(chain_rename)
                    struct.chains['name'] = np.array(new_chain_names)

                    for chain in struct.chains:
                        res_start = chain["res_idx"]
                        res_end = chain["res_idx"] + chain["res_num"]
                        residues = struct.residues[res_start:res_end]
                        residues['res_idx'] = np.arange(chain["res_num"])
                else:
                    # we basically detect which chain is the motif by any duplicate chains
                    # we rely on the fact that the motif is appended to the generated residues
                    # so that it will always be second
                    seen_asym_id = []
                    chain_mask = []
                    for chain in struct.chains:
                        if chain['asym_id'] not in seen_asym_id:
                            seen_asym_id.append(chain['asym_id'])
                            chain_mask.append(True)
                        else:
                            chain_mask.append(False)
                    struct = replace(struct, mask=np.array(chain_mask))
                    # print(struct, chain_mask)

                pdb_str = to_pdb(struct)
                with open(sample_name + ".pdb", 'w') as fp:
                    fp.write(pdb_str)

                if self._cfg['save_traj']:
                    clean_traj = sample_data['clean_traj']
                    prot_traj = sample_data['prot_traj']

                    clean_traj_name = f"len_{sample_len}_protein_{curr_sample_id}_clean_traj.pdb"
                    prot_traj_name = f"len_{sample_len}_protein_{curr_sample_id}_prot_traj.pdb"
                    clean_model_strs = []
                    prot_model_strs = []

                    for i, traj_data in enumerate(clean_traj):
                        traj_struct = Structure(**traj_data['structure'])
                        traj_output = Tokenized(
                            tokens=traj_data['tokens'],
                            rigids=traj_data['rigids'],
                            bonds=traj_data['bonds'],
                            structure=traj_struct,

                        )
                        traj_struct = update_structure(traj_struct, traj_output.rigids['tensor7'])
                        pdb_str = to_pdb(traj_struct)
                        clean_model_strs.append(f"MODEL        {i}\n")
                        clean_model_strs.append(pdb_str.split("END")[0])
                        clean_model_strs.append(f"ENDMDL       \n")
                    clean_model_strs.append("END\n")

                    with open(clean_traj_name, 'w') as fp:
                        for pdb_str in clean_model_strs:
                            fp.write(pdb_str)

                    for i, traj_data in enumerate(prot_traj):
                        traj_struct = Structure(**traj_data['structure'])
                        traj_output = Tokenized(
                            tokens=traj_data['tokens'],
                            rigids=traj_data['rigids'],
                            bonds=traj_data['bonds'],
                            structure=traj_struct,

                        )
                        traj_struct = update_structure(traj_struct, traj_output.rigids['tensor7'])
                        pdb_str = to_pdb(traj_struct)
                        prot_model_strs.append(f"MODEL        {i}\n")
                        prot_model_strs.append(pdb_str.split("END")[0])
                        prot_model_strs.append(f"ENDMDL       \n")
                    prot_model_strs.append("END\n")
                    with open(prot_traj_name, 'w') as fp:
                        for pdb_str in prot_model_strs:
                            fp.write(pdb_str)


                sample_path = os.path.abspath(sample_name + ".pdb")

                chain_data = struct.chains
                chain_mapping = {
                    c['asym_id']: c['name']
                    for c in chain_data
                }

                samples_metadata[sample_name] = {
                    "path": sample_path,
                    # "task": sample_task,
                    "name": sample_data['name'] if 'name' in sample_data else None,
                    "length": sample_len,
                    "fixed_bb_res_idx": [i+1 for i in sample_data['fixed_bb_res_idx'].tolist()],  # 1-indexed chain for pyrosetta
                    "fixed_bb_chain": [chain_mapping[int(i)] for i in sample_data['fixed_bb_chain_idx']],
                    "fixed_seq_res_idx": [i+1 for i in sample_data['fixed_seq_res_idx'].tolist()],  # 1-indexed chain for pyrosetta
                    "fixed_seq_chain": [chain_mapping[int(i)] for i in sample_data['fixed_seq_chain_idx']],
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
    model = instantiate(model_cfg['model'])

    model = lmodule_init(model, corrupter, None)

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
