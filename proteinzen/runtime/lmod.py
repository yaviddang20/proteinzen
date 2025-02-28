import logging
import math
import os
from functools import partial
import copy

import numpy as np
import pandas as pd
import torch
import tree
import lightning as L
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import rdGeometry

from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


from proteinzen.tasks import TaskList
from proteinzen.data.io.atom91 import atom91_to_pdb
from proteinzen.runtime.optim import get_std_opt

from .utils import gen_pbar_str
from .ema import EMAModel

class MedianMetric(Metric):
    is_differentiable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("seq_recov", default=[], dist_reduce_fx="cat")

    def update(self, seq_recov: torch.Tensor) -> None:
        self.seq_recov.append(seq_recov)

    def compute(self) -> torch.Tensor:
        return torch.median(dim_zero_cat(self.seq_recov))


def t_stratified_loss(batch_t, batch_loss, num_bins=4, loss_name=None):
    """Stratify loss by binning t."""
    batch_t = batch_t.numpy(force=True)
    batch_loss = batch_loss.numpy(force=True)
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins+1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = 'loss'
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin+1]
        t_range = f'{loss_name} t=[{bin_start:.2f},{bin_end:.2f})'
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


class BackboneModule(L.LightningModule):
    def __init__(self, model, optim, val_dir="validation"):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        self.optim = optim
        self.val_dir = val_dir
        self.validation_epoch_metrics = []

        self.median_metric = MedianMetric()

    def on_train_epoch_start(self):
        if hasattr(self.trainer.train_dataloader.batch_sampler, "epoch"):
            self.trainer.train_dataloader.batch_sampler.set_epoch(self.trainer.current_epoch)

    def training_step(self, batch):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        loss_dict = task.compile_task_losses(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict = {
            ("train/" + key): value
            for key, value in
            sorted(log_dict.items(), key = lambda x: x[0])
        }
        t = batch['t']
        for loss_name, loss_list in loss_dict.items():
            if loss_name in ['loss', 'frameflow_loss']:
                continue
            stratified_losses = t_stratified_loss(
                t, loss_list, loss_name=loss_name)
            stratified_losses = {f"train/{k}": torch.as_tensor(v, device=log_dict['train/loss'].device) for k,v in stratified_losses.items()}
            self.log_dict(
                stratified_losses,
                prog_bar=False,
                logger=True,
                on_step=None,
                on_epoch=True,
                batch_size=batch.num_graphs,
                sync_dist=False)

        self.log_dict(
            log_dict,
            on_step=None,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch.num_graphs)
        # self._log.info(gen_pbar_str(loss_dict))
        return loss_dict

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=True,#False,
            rank_zero_only=False,#True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )
    # def validation_step(self, batch, batch_idx):
    #     task: TaskList = batch['task']

    #     with torch.no_grad():
    #         outputs = task.run_predicts(self.model, batch)
    #         samples = outputs['samples']
    #         sample_ids = outputs['inputs']['sample_id']
    #         log_metrics = []
    #         for sample, sample_id in zip(samples, sample_ids):
    #             sample_len = sample.shape[0]
    #             out_path = os.path.join(
    #                 self.val_dir,
    #                 f"len_{sample_len}_bb_{sample_id}"
    #             )
    #             atom91_to_pdb("".join(["A" for _ in range(sample_len)]), sample.numpy(force=True), out_path)

    #             ss_metrics = metrics.calc_mdtraj_metrics(out_path + ".pdb")
    #             clash_metrics = metrics.calc_ca_ca_metrics(sample[:, 1])
    #             log_metrics.append((ss_metrics | clash_metrics))
    #     log_df = pd.DataFrame(log_metrics)
    #     self.validation_epoch_metrics.append(log_df)

    # def on_validation_epoch_end(self):
    #     val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
    #     for metric_name, metric_val in sorted(
    #         val_epoch_metrics.mean().to_dict().items(),
    #         key=lambda x: x[0]
    #     ):
    #         self.log(
    #             f'valid/{metric_name}',
    #             metric_val,
    #             on_step=False,
    #             on_epoch=True,
    #             prog_bar=False,
    #             batch_size=len(val_epoch_metrics),
    #         )
    #     self.validation_epoch_metrics.clear()

    def test_step(self, batch, batch_idx):
        task: TaskList = batch.task
        sample_outputs = task.run_predicts(self.model, batch)
        sample_loss_dict = task.compile_task_losses(batch, sample_outputs)
        sample_log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            sample_loss_dict
        )
        sample_log_dict = {"sample_" + k: v for k,v in sample_log_dict.items()}

        self.log_dict(
            sample_log_dict,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)

        self.median_metric.update(sample_loss_dict['autoenc_per_seq_recov'])
        self.log("median_seq_recov", self.median_metric, on_step=False, on_epoch=True)
        # self._log.info(gen_pbar_str(sample_loss_dict))
        return sample_loss_dict

    def predict_step(self, batch, batch_idx):
        task: TaskList = batch['task']
        outputs = task.run_predicts(self.model, batch)
        return outputs


    def forward(self, batch):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        return outputs

    def configure_optimizers(self):
        return self.optim(self.model.parameters())


class SidechainModule(L.LightningModule):
    def __init__(self, model, optim, val_dir="validation"):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        self.optim = optim
        self.val_dir = val_dir
        self.validation_epoch_metrics = []

        self.one_shot_median_metric = MedianMetric()
        self.sample_median_metric = MedianMetric()

    def training_step(self, batch):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        loss_dict = task.compile_task_losses(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict = {
            ("train/" + key): value
            for key, value in
            sorted(log_dict.items(), key = lambda x: x[0])
        }

        if 'train/percent_masked' in log_dict and 'train/per_seq_recov' in log_dict:
            if log_dict['train/percent_masked'] == 1.0:
                loss_name = "train/per_seq_recov_mask_all"
                pt_loss_name = "train/pt_per_seq_recov_mask_all"
            elif log_dict['train/percent_masked'] == 0.0:
                loss_name = "train/per_seq_recov_mask_none"
                pt_loss_name = "train/pt_per_seq_recov_mask_none"
            else:
                loss_name = "train/per_seq_recov_mask_partial"
                pt_loss_name = "train/pt_per_seq_recov_mask_partial"
            self.log(
                loss_name,
                log_dict['train/per_seq_recov'],
                prog_bar=False,
                logger=True,
                on_step=None,
                on_epoch=True,
                batch_size=batch.num_graphs,
                sync_dist=False
            )
            if 'train/pt_per_seq_recov' in log_dict:
                self.log(
                    pt_loss_name,
                    log_dict['train/pt_per_seq_recov'],
                    prog_bar=False,
                    logger=True,
                    on_step=None,
                    on_epoch=True,
                    batch_size=batch.num_graphs,
                    sync_dist=False
                )

        self.log_dict(
            log_dict,
            on_step=None,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)
        # self._log.info(gen_pbar_str(loss_dict))
        return loss_dict

    # def on_after_backward(self):
    #     if self.trainer.global_step % 1 == 0:
    #         for name, p in self.named_parameters():
    #             if hasattr(p.grad, "data"):
    #                 grad = p.grad.data
    #                 print(name, p.shape, torch.mean(grad ** 2))
    #     exit()


    def validation_step(self, batch, batch_idx):
        task: TaskList = batch['task']

        outputs = task.run_evals(self.model, batch)
        loss_dict = task.compile_task_losses(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict = {"val_" + k: v for k,v in log_dict.items()}

        self.log_dict(
            log_dict,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)
        # self._log.info(gen_pbar_str(loss_dict))

        batch = copy.copy(batch)
        batch['residue']['res_noising_mask'] = torch.ones_like(batch['residue']['noising_mask'])
        batch['residue']['seq_noising_mask'] = torch.ones_like(batch['residue']['noising_mask'])
        outputs = task.run_evals(self.model, batch)
        loss_dict = task.compile_task_losses(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict = {"val_no_mask_" + k: v for k,v in log_dict.items()}

        self.log_dict(
            log_dict,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)

        batch = copy.copy(batch)
        batch['residue']['res_noising_mask'] = torch.ones_like(batch['residue']['noising_mask'])
        batch['residue']['seq_noising_mask'] = torch.zeros_like(batch['residue']['noising_mask'])
        outputs = task.run_evals(self.model, batch)
        loss_dict = task.compile_task_losses(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict = {"val_all_mask_" + k: v for k,v in log_dict.items()}

        self.log_dict(
            log_dict,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)
        # self._log.info(gen_pbar_str(loss_dict))

        # sample_outputs = task.run_evals(self.model, batch)
        # sample_loss_dict = task.compile_task_losses(batch, sample_outputs)
        # sample_log_dict = tree.map_structure(
        #     lambda x: torch.mean(x) if torch.is_tensor(x) else x,
        #     sample_loss_dict
        # )
        # sample_log_dict = {"sample_" + k: v for k,v in sample_log_dict.items()}
        # self.log_dict(
        #     sample_log_dict,
        #     logger=True,
        #     batch_size=batch.num_graphs,
        #     sync_dist=True)
        # sample_loss_dict = {"sample_" + k: v for k,v in sample_loss_dict.items()}
        # self._log.info(gen_pbar_str(sample_loss_dict))
        return loss_dict


    def test_step(self, batch, batch_idx):
        task: TaskList = batch['task']
        assert not self.model.training
        outputs = task.run_evals(self.model, batch)
        loss_dict = task.compile_task_losses(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict = {"val_" + k: v for k,v in log_dict.items()}
        self.one_shot_median_metric.update(loss_dict['per_seq_recov'])
        self.log("median_seq_recov", self.one_shot_median_metric, on_step=False, on_epoch=True)
        # self._log.info(gen_pbar_str(loss_dict))

        # sample_outputs = task.run_predicts(self.model, batch)
        # sample_loss_dict = task.compile_task_losses(batch, sample_outputs)
        # sample_log_dict = tree.map_structure(
        #     lambda x: torch.mean(x) if torch.is_tensor(x) else x,
        #     sample_loss_dict
        # )
        # sample_log_dict = {"sample_" + k: v for k,v in sample_log_dict.items()}


        # self.log_dict(
        #     sample_log_dict,
        #     logger=True,
        #     batch_size=batch.num_graphs,
        #     sync_dist=True)

        # self.sample_median_metric.update(sample_loss_dict['per_seq_recov'])
        # self.log("sample_median_seq_recov", self.sample_median_metric, on_step=False, on_epoch=True)
        # self._log.info(gen_pbar_str(sample_loss_dict))
        return loss_dict

    def predict_step(self, batch, batch_idx):
        task: TaskList = batch['task']
        outputs = task.run_predicts(self.model, batch)
        return outputs


    def forward(self, batch):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        return outputs

    def configure_optimizers(self):
        return self.optim(self.model.parameters())


class MoleculeModule(L.LightningModule):
    def __init__(self, model, optim, val_dir="validation"):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        self.optim = optim
        self.val_dir = val_dir
        self.validation_epoch_metrics = []

        self.median_metric = MedianMetric()

    def training_step(self, batch):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        loss_dict = task.compile_task_losses(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            loss_dict
        )

        self.log_dict(
            log_dict,
            on_step=None,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)
        # self._log.info(gen_pbar_str(loss_dict))
        return loss_dict

    def validation_step(self, batch, batch_idx):
        task: TaskList = batch['task']

        def set_conf(conf, atom_pos):
            mol = conf.GetOwningMol()
            atom_mat_idx = 0
            for atom_mol_idx, atom in enumerate(mol.GetAtoms()):
                if atom.GetSymbol() == 'H':
                    continue
                atom_coord = atom_pos[atom_mat_idx]
                point = rdGeometry.Point3D(*atom_coord.tolist())
                conf.SetAtomPosition(atom_mol_idx, point)
                atom_mat_idx += 1

        with torch.no_grad():
            outputs = task.run_predicts(self.model, batch)
            samples = outputs['samples']
            clean_trajs = outputs['clean_trajs']
            mol_trajs = outputs['mol_trajs']
            rd_mols = batch['rd_mol']
            log_metrics = []
            for idx, (sample, clean_traj, mol_traj, rd_mol_list) in enumerate(zip(samples, clean_trajs, mol_trajs, rd_mols)):
                # "copying" a molecule
                sample_mol = Chem.Mol(rd_mol_list[0])
                sample_mol = Chem.RemoveHs(sample_mol)
                sample_conf = sample_mol.GetConformer()
                set_conf(sample_conf, sample)

                all_confs_mol = Chem.Mol(rd_mol_list[0])
                all_confs_mol = Chem.RemoveHs(all_confs_mol)
                all_confs_mol.RemoveAllConformers()
                all_confs_mol.AddConformer(sample_conf)
                for conf_id, mol in enumerate(rd_mol_list):
                    if conf_id < 10:
                        out_path = os.path.join(
                            self.val_dir,
                            f"sample_{idx}_gt_conf_{conf_id}.sdf"
                        )
                        sd_writer = Chem.SDWriter(out_path)
                        sd_writer.write(mol, confId=0)
                    mol = Chem.RemoveHs(mol)
                    all_confs_mol.AddConformer(mol.GetConformer())

                rmslist = []
                AllChem.AlignMolConformers(all_confs_mol, RMSlist=rmslist)


                out_path = os.path.join(
                    self.val_dir,
                    f"sample_{idx}.sdf"
                )
                sd_writer = Chem.SDWriter(out_path)
                sd_writer.write(sample_conf.GetOwningMol(), confId=0)

                # write the clean trajectory to a PDB file
                traj_len = clean_traj.shape[0]
                out_path = os.path.join(
                    self.val_dir,
                    f"sample_{idx}_clean_traj.pdb"
                )
                with open(out_path, 'w') as fp:
                    for traj_idx in range(traj_len):
                        set_conf(sample_conf, clean_traj[traj_idx])
                        fp.write("MODEL" + "".join([" "] * 8) + f"{idx}\n")
                        fp.write(Chem.MolToPDBBlock(Chem.RemoveHs(sample_conf.GetOwningMol()), flavor=1))
                        fp.write(f"ENDMDL\n")
                # write the mol trajectory to a PDB file
                traj_len = mol_traj.shape[0]
                out_path = os.path.join(
                    self.val_dir,
                    f"sample_{idx}_mol_traj.pdb"
                )
                with open(out_path, 'w') as fp:
                    for traj_idx in range(traj_len):
                        set_conf(sample_conf, mol_traj[traj_idx])
                        fp.write("MODEL" + "".join([" "] * 8) + f"{idx}\n")
                        fp.write(Chem.MolToPDBBlock(Chem.RemoveHs(sample_conf.GetOwningMol()), flavor=1))
                        fp.write(f"ENDMDL\n")


                # "copying" a molecule
                sample_gt_mol = Chem.Mol(rd_mol_list[0])
                sample_gt_conf = sample_gt_mol.GetConformer()
                sample_gt = batch['ligand']['atom_pos']
                set_conf(sample_gt_conf, sample_gt)
                sample_gt_mol = Chem.RemoveHs(sample_gt_mol)

                out_path = os.path.join(
                    self.val_dir,
                    f"sample_{idx}_gt.sdf"
                )
                sd_writer = Chem.SDWriter(out_path)
                sd_writer.write(sample_gt_mol, confId=0)


                log_metrics.append({
                    "conf_min_rmsd": min(rmslist),
                    "conf_max_rmsd": max(rmslist),
                    "conf_mean_rmsd": np.mean(rmslist)
                })

        log_df = pd.DataFrame(log_metrics)
        self.validation_epoch_metrics.append(log_df)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                print(name, parameter.grad.norm())

    def on_validation_epoch_end(self):
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
            self.log(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

    def test_step(self, batch, batch_idx):
        task: TaskList = batch.task
        sample_outputs = task.run_predicts(self.model, batch)
        sample_loss_dict = task.compile_task_losses(batch, sample_outputs)
        sample_log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            sample_loss_dict
        )
        sample_log_dict = {"sample_" + k: v for k,v in sample_log_dict.items()}

        self.log_dict(
            sample_log_dict,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)

        self.median_metric.update(sample_loss_dict['autoenc_per_seq_recov'])
        self.log("median_seq_recov", self.median_metric, on_step=False, on_epoch=True)
        # self._log.info(gen_pbar_str(sample_loss_dict))
        return sample_loss_dict

    def predict_step(self, batch, batch_idx):
        task: TaskList = batch['task']

        with torch.no_grad():
            outputs = task.run_predicts(self.model, batch)
            samples = outputs['samples']
            clean_trajs = outputs['clean_trajs']
            mol_trajs = outputs['mol_trajs']
            rd_mols = batch['rd_mol']
            log_metrics = []
            for idx, (sample, clean_traj, mol_traj, rd_mol_list) in enumerate(zip(samples, clean_trajs, mol_trajs, rd_mols)):
                # "copying" a molecule
                sample_mol = Chem.MolFromSmiles(Chem.MolToSmiles(rd_mol_list[0]))
                sample_mol = Chem.AddHs(sample_mol)
                rdDistGeom.EmbedMolecule(sample_mol)
                sample_mol = Chem.RemoveHs(sample_mol)
                sample_conf = sample_mol.GetConformer()
                for atom_idx in range(sample.shape[0]):
                    atom_coord = sample[atom_idx]
                    point = rdGeometry.Point3D(*atom_coord.tolist())
                    sample_conf.SetAtomPosition(atom_idx, point)

                out_path = os.path.join(
                    self.val_dir,
                    f"sample_{idx}.sdf"
                )
                sd_writer = Chem.SDWriter(out_path)
                sd_writer.write(sample_mol, confId=0)

                all_confs_mol = Chem.MolFromSmiles(Chem.MolToSmiles(rd_mol_list[0]))
                all_confs_mol.AddConformer(sample_conf)
                all_confs_mol = Chem.AddHs(all_confs_mol)
                for conf_id, mol in enumerate(rd_mol_list[:10]):
                    out_path = os.path.join(
                        self.val_dir,
                        f"sample_{idx}_gt_conf_{conf_id}.sdf"
                    )
                    sd_writer = Chem.SDWriter(out_path)
                    sd_writer.write(mol, confId=0)
                    mol = Chem.AddHs(mol)
                    all_confs_mol.AddConformer(mol.GetConformer())

                rmslist = []
                AllChem.AlignMolConformers(all_confs_mol, RMSlist=rmslist)

                # write the clean trajectory to a PDB file
                traj_len = clean_traj.shape[0]
                out_path = os.path.join(
                    self.val_dir,
                    f"sample_{idx}_clean_traj.pdb"
                )
                with open(out_path, 'w') as fp:
                    sample_conf = sample_mol.GetConformer(0)
                    for traj_idx in range(traj_len):
                        for atom_idx in range(sample.shape[0]):
                            atom_coord = clean_traj[traj_idx][atom_idx]
                            point = rdGeometry.Point3D(*atom_coord.tolist())
                            sample_conf.SetAtomPosition(atom_idx, point)
                        fp.write("MODEL" + "".join([" "] * 8) + f"{idx}\n")
                        fp.write(Chem.MolToPDBBlock(sample_mol, flavor=1))
                        fp.write(f"ENDMDL\n")
                # write the mol trajectory to a PDB file
                traj_len = mol_traj.shape[0]
                out_path = os.path.join(
                    self.val_dir,
                    f"sample_{idx}_mol_traj.pdb"
                )
                with open(out_path, 'w') as fp:
                    sample_conf = sample_mol.GetConformer(0)
                    for traj_idx in range(traj_len):
                        for atom_idx in range(sample.shape[0]):
                            atom_coord = mol_traj[traj_idx][atom_idx]
                            point = rdGeometry.Point3D(*atom_coord.tolist())
                            sample_conf.SetAtomPosition(atom_idx, point)
                        fp.write("MODEL" + "".join([" "] * 8) + f"{idx}\n")
                        fp.write(Chem.MolToPDBBlock(sample_mol, flavor=1))
                        fp.write(f"ENDMDL\n")

                log_metrics.append({
                    "conf_min_rmsd": min(rmslist),
                    "conf_max_rmsd": max(rmslist),
                    "conf_mean_rmsd": np.mean(rmslist)
                })



    def forward(self, batch):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        return outputs

    def configure_optimizers(self):
        return self.optim(self.model.parameters())


class ProteinModule(L.LightningModule):
    def __init__(self,
                 model,
                 optim,
                 val_dir="validation",
                 freeze_encoder=False,
                 freeze_decoder=False,
                 finetune_seq_head=False,
                 use_amp=False,
                 use_cosine_lr_sched=False,
                 cosine_warmup_steps=0,
                 cosine_total_steps=1e6,
                 use_ema=False,
                 ema_decay=0.999,
                 use_posthoc_ema=False,
    ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        self.optim = optim
        self.val_dir = val_dir
        self.use_amp = use_amp
        self.use_cosine_lr_sched = use_cosine_lr_sched
        self.cosine_warmup_steps = cosine_warmup_steps
        self.cosine_total_steps = cosine_total_steps
        self.finetune_seq_head = finetune_seq_head
        self.use_ema = use_ema
        self.use_posthoc_ema = use_posthoc_ema
        if self.use_amp:
            torch.set_float32_matmul_precision("medium")

        if use_ema:
            self.ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        else:
            self.ema = None

        if use_posthoc_ema:
            self.ema_long = EMAModel(model, gamma=6.94)
            self.ema_short = EMAModel(model, gamma=16.97)
        else:
            self.ema_long = None
            self.ema_short = None


        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder

        self.median_metric = MedianMetric()

        if hasattr(model, "discriminator"):
            self.automatic_optimization = False

        # ckpt = torch.load(path, map_location='cpu')
        # self.model.encoder.load_state_dict({
        #     key[len("model.encoder."):]: ckpt['state_dict'][key]
        #     for key in ckpt['state_dict'].keys()
        #     if key.startswith("model.encoder")
        # })
        # self.model.decoder.load_state_dict({
        #     key[len("model.decoder."):]: ckpt['state_dict'][key]
        #     for key in ckpt['state_dict'].keys()
        #     if key.startswith("model.decoder")
        # })

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if hasattr(self.trainer.train_dataloader.batch_sampler, "epoch"):
            self.trainer.train_dataloader.batch_sampler.set_epoch(self.trainer.current_epoch)

    def training_step(self, batch):
        task: TaskList = batch.task

        if self.ema is not None and self.global_step > 0:
            self.ema.update_parameters(self.model)
        if self.ema_long is not None and self.global_step > 0:
            self.ema_long.update_parameters(self.model, self.global_step-1)
        if self.ema_short is not None and self.global_step > 0:
            self.ema_short.update_parameters(self.model, self.global_step-1)


        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = task.run_evals(self.model, batch)
            loss_dict = task.compile_task_losses(batch, outputs)
        else:
            outputs = task.run_evals(self.model, batch)
            loss_dict = task.compile_task_losses(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.round(torch.mean(x), decimals=3) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict = {
            ("train/" + key): value
            for key, value in
            sorted(log_dict.items(), key = lambda x: x[0])
        }
        t = batch['t']
        for loss_name, loss_list in loss_dict.items():
            if loss_name in ['loss', 'frameflow_loss']:
                continue
            if not loss_name.startswith("pt_") and not loss_name.startswith("latent_"):
                continue
            stratified_losses = t_stratified_loss(
                t, loss_list, loss_name=loss_name)
            stratified_losses = {
                f"train/{k}": torch.round(torch.as_tensor(v, device=log_dict['train/loss'].device), decimals=3)
                for k,v in stratified_losses.items()
            }
            self.log_dict(
                stratified_losses,
                prog_bar=False,
                logger=True,
                on_step=None,
                on_epoch=True,
                batch_size=batch.num_graphs,
                sync_dist=False)

        if 'train/percent_masked' in log_dict and 'train/per_seq_recov' in log_dict:
            if log_dict['train/percent_masked'] == 1.0:
                loss_name = "train/per_seq_recov_mask_all"
                pt_loss_name = "train/pt_per_seq_recov_mask_all"
            elif log_dict['train/percent_masked'] == 0.0:
                loss_name = "train/per_seq_recov_mask_none"
                pt_loss_name = "train/pt_per_seq_recov_mask_none"
            else:
                loss_name = "train/per_seq_recov_mask_partial"
                pt_loss_name = "train/pt_per_seq_recov_mask_partial"
            self.log(
                loss_name,
                torch.round(log_dict['train/per_seq_recov'], decimals=3),
                prog_bar=False,
                logger=True,
                on_step=None,
                on_epoch=True,
                batch_size=batch.num_graphs,
                sync_dist=False
            )
            if 'train/pt_per_seq_recov' in log_dict:
                self.log(
                    pt_loss_name,
                    torch.round(log_dict['train/pt_per_seq_recov'], decimals=3),
                    prog_bar=False,
                    logger=True,
                    on_step=None,
                    on_epoch=True,
                    batch_size=batch.num_graphs,
                    sync_dist=False
                )


        if hasattr(self.model, "discriminator"):
            optim_G, optim_D = self.optimizers()
            loss_G = loss_dict['loss']
            optim_G.zero_grad()
            self.manual_backward(loss_G)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optim_G.step()

            # this is breaking the task abstraction, but i should really
            # change this setup at some point anyway
            loss_D, discrim_gt_loss, discrim_gt_score = task.task_list[0].run_discrim(self.model, batch, outputs)
            optim_D.zero_grad()
            self.manual_backward(loss_D)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optim_D.step()

            log_dict['train/discrim_gt_score'] = torch.round(discrim_gt_score.mean(), decimals=3)
            log_dict['train/discrim_gt_loss'] = torch.round(discrim_gt_loss.mean(), decimals=3)


            self.log_dict(
                log_dict,
                on_step=None,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch.num_graphs,
                sync_dist=True)

            return

        else:
            self.log_dict(
                log_dict,
                on_step=None,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch.num_graphs,
                sync_dist=True)

            # self._log.info(gen_pbar_str(loss_dict))
            return loss_dict

    def validation_step(self, batch, batch_idx):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        loss_dict = task.compile_task_losses(batch, outputs)

        log_dict = tree.map_structure(
            lambda x: torch.round(torch.mean(x), decimals=3) if torch.is_tensor(x) else x,
            loss_dict
        )
        log_dict = {
            ("val/" + key): value
            for key, value in
            sorted(log_dict.items(), key = lambda x: x[0])
        }
        t = batch['t']
        for loss_name, loss_list in loss_dict.items():
            if loss_name in ['loss', 'frameflow_loss']:
                continue
            if not loss_name.startswith("pt_") and not loss_name.startswith("latent_"):
                continue
            stratified_losses = t_stratified_loss(
                t, loss_list, loss_name=loss_name)
            stratified_losses = {
                f"val/{k}": torch.round(torch.as_tensor(v, device=log_dict['val/loss'].device), decimals=3)
                for k,v in stratified_losses.items()
            }
            self.log_dict(
                stratified_losses,
                prog_bar=False,
                logger=True,
                on_step=None,
                on_epoch=True,
                batch_size=batch.num_graphs,
                sync_dist=False)

        if 'val/percent_masked' in log_dict and 'val/per_seq_recov' in log_dict:
            if log_dict['val/percent_masked'] == 1.0:
                loss_name = "val/per_seq_recov_mask_all"
                pt_loss_name = "val/pt_per_seq_recov_mask_all"
            elif log_dict['val/percent_masked'] == 0.0:
                loss_name = "val/per_seq_recov_mask_none"
                pt_loss_name = "val/pt_per_seq_recov_mask_none"
            else:
                loss_name = "val/per_seq_recov_mask_partial"
                pt_loss_name = "val/pt_per_seq_recov_mask_partial"
            self.log(
                loss_name,
                torch.round(log_dict['val/per_seq_recov'], decimals=3),
                prog_bar=False,
                logger=True,
                on_step=None,
                on_epoch=True,
                batch_size=batch.num_graphs,
                sync_dist=False
            )
            if 'val/pt_per_seq_recov' in log_dict:
                self.log(
                    pt_loss_name,
                    torch.round(log_dict['val/pt_per_seq_recov'], decimals=3),
                    prog_bar=False,
                    logger=True,
                    on_step=None,
                    on_epoch=True,
                    batch_size=batch.num_graphs,
                    sync_dist=False
                )


        if hasattr(self.model, "discriminator"):
            loss_G = loss_dict['loss']
            # this is breaking the task abstraction, but i should really
            # change this setup at some point anyway
            loss_D, discrim_gt_loss, discrim_gt_score = task.task_list[0].run_discrim(self.model, batch, outputs)

            log_dict['val/discrim_gt_score'] = torch.round(discrim_gt_score.mean(), decimals=3)
            log_dict['val/discrim_gt_loss'] = torch.round(discrim_gt_loss.mean(), decimals=3)


        self.log_dict(
            log_dict,
            on_step=None,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)

        # self._log.info(gen_pbar_str(loss_dict))
        return loss_dict

    def test_step(self, batch, batch_idx):
        task: TaskList = batch.task
        if self.use_ema:
            sample_outputs = task.run_predicts(self.ema, batch)
        else:
            sample_outputs = task.run_predicts(self.model, batch)

        sample_loss_dict = task.compile_task_losses(batch, sample_outputs)
        sample_log_dict = tree.map_structure(
            lambda x: torch.mean(x) if torch.is_tensor(x) else x,
            sample_loss_dict
        )
        sample_log_dict = {"sample_" + k: v for k,v in sample_log_dict.items()}

        self.log_dict(
            sample_log_dict,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)

        self.median_metric.update(sample_loss_dict['autoenc_per_seq_recov'])
        self.log("median_seq_recov", self.median_metric, on_step=False, on_epoch=True)
        print(gen_pbar_str(sample_loss_dict))
        # self._log.info(gen_pbar_str(sample_loss_dict))
        return sample_loss_dict

    def predict_step(self, batch, batch_idx):
        task: TaskList = batch['task']
        if self.use_ema:
            outputs = task.run_predicts(self.ema.module, batch, device=self.device)
        else:
            outputs = task.run_predicts(self.model, batch, device=self.device)
        return outputs


    def forward(self, batch):
        task: TaskList = batch.task
        outputs = task.run_evals(self.model, batch)
        return outputs

    def configure_optimizers(self):
        if hasattr(self.model, "discriminator"):
            optim_G = self.optim(
                list(self.model.encoder.parameters()) +
                list(self.model.decoder.parameters()) +
                list(self.model.denoiser.parameters())
            )
            optim_D = self.optim(self.model.discriminator.parameters())
            return optim_G, optim_D
        elif self.finetune_seq_head:
            optimizer = self.optim(
                list(self.model.decoder.seq_head.parameters())
            )
            return optimizer
        else:
            if self.freeze_encoder and self.freeze_decoder:
                print("Freezing autoenc parameters")
                optimizer = self.optim(
                    list(self.model.denoiser.parameters())
                )
            elif self.freeze_encoder:
                print("Freezing encoder parameters")
                optimizer = self.optim(
                    list(self.model.denoiser.parameters())
                    + list(self.model.decoder.parameters())
                )
            elif self.freeze_decoder:
                print("Freezing decoder parameters")
                optimizer = self.optim(
                    list(self.model.denoiser.parameters())
                    + list(self.model.encoder.parameters())
                )
            else:
                optimizer = self.optim(self.model.parameters())

            if self.use_cosine_lr_sched:
                scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.cosine_warmup_steps,
                    num_training_steps=int(self.cosine_total_steps),
                    num_cycles=1
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': scheduler
                }
            else:
                return optimizer


# from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return 0.0
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)