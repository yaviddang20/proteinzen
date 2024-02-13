import logging
import os
from functools import partial

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


from ligbinddiff.tasks import TaskList
from ligbinddiff.utils import metrics
from ligbinddiff.data.io.atom91 import atom91_to_pdb
from ligbinddiff.runtime.optim import get_std_opt

from .utils import gen_pbar_str

class MedianMetric(Metric):
    is_differentiable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("seq_recov", default=[], dist_reduce_fx="cat")

    def update(self, seq_recov: torch.Tensor) -> None:
        self.seq_recov.append(seq_recov)

    def compute(self) -> torch.Tensor:
        return torch.median(dim_zero_cat(self.seq_recov))


class BackboneModule(L.LightningModule):
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
        log_dict = {
            ("train/" + key): value
            for key, value in
            sorted(log_dict.items(), key = lambda x: x[0])
        }

        self.log_dict(
            log_dict,
            on_step=None,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)
        self._log.info(gen_pbar_str(loss_dict))
        return loss_dict

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
        self._log.info(gen_pbar_str(sample_loss_dict))
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

        self.median_metric = MedianMetric()

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

        self.log_dict(
            log_dict,
            on_step=None,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.num_graphs,
            sync_dist=True)
        self._log.info(gen_pbar_str(loss_dict))
        return loss_dict

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
        self._log.info(gen_pbar_str(loss_dict))

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
        self._log.info(gen_pbar_str(sample_loss_dict))
        return loss_dict


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
        self._log.info(gen_pbar_str(sample_loss_dict))
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
        self._log.info(gen_pbar_str(loss_dict))
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
                sample_conf = sample_mol.GetConformer()
                set_conf(sample_conf, sample)
                sample_mol = Chem.RemoveHs(sample_mol)

                all_confs_mol = Chem.Mol(rd_mol_list[0])
                all_confs_mol = Chem.RemoveHs(all_confs_mol)
                all_confs_mol.RemoveAllConformers()
                all_confs_mol.AddConformer(sample_mol.GetConformer())
                for conf_id, mol in enumerate(rd_mol_list[:10]):
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
                sd_writer.write(sample_mol, confId=0)

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
        self._log.info(gen_pbar_str(sample_loss_dict))
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
    def __init__(self, model, optim, val_dir="validation"):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.model = model
        self.optim = optim
        self.val_dir = val_dir

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
        self._log.info(gen_pbar_str(loss_dict))
        return loss_dict

    # def validation_step(self, batch, batch_idx):
    #     task: TaskList = batch.task
    #     outputs = task.run_evals(self.model, batch)
    #     loss_dict = task.compile_task_losses(batch, outputs)

    #     log_dict = tree.map_structure(
    #         lambda x: torch.mean(x) if torch.is_tensor(x) else x,
    #         loss_dict
    #     )
    #     log_dict = {"val_" + k: v for k,v in log_dict.items()}

    #     self.log_dict(
    #         log_dict,
    #         logger=True,
    #         batch_size=batch.num_graphs,
    #         sync_dist=True)
    #     self._log.info(gen_pbar_str(loss_dict))

    #     sample_outputs = task.run_predicts(self.model, batch)
    #     sample_loss_dict = task.compile_task_losses(batch, sample_outputs)
    #     sample_log_dict = tree.map_structure(
    #         lambda x: torch.mean(x) if torch.is_tensor(x) else x,
    #         sample_loss_dict
    #     )
    #     sample_log_dict = {"sample_" + k: v for k,v in sample_log_dict.items()}
    #     self.log_dict(
    #         sample_log_dict,
    #         logger=True,
    #         batch_size=batch.num_graphs,
    #         sync_dist=True)
    #     self._log.info(gen_pbar_str(sample_loss_dict))

    #     return loss_dict

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
        self._log.info(gen_pbar_str(sample_loss_dict))
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
