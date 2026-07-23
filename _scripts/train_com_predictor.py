"""
CoM predictor: frozen pretrained IpaMultiRigidDenoiser backbone + small CoMHead.

Uses the same training data pipeline as the plinder model (MMCIFDataset +
ProteinConditionedGenerateLigand). The batch already has:
  - rigids_1  : crystal GT (protein + ligand in crystal position)
  - rigids_t  : noised at random t (protein fixed, ligand noised)
  - rigids_noising_mask : True = ligand, False = protein

GT CoM comes directly from rigids_1 — no input_data parsing needed.

Usage
-----
python _scripts/train_com_predictor.py \
    --model-dir outputs/plinder_protein_cond/train \
    --dataset-config configs/train/data/plinder_protein_cond.yaml \
    --val-dataset-config configs/train/data/plinder_val.yaml \
    --out-dir outputs/com_predictor
"""

import argparse
import glob
import inspect
import os
from pathlib import Path

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.fabric.plugins.environments import LightningEnvironment
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra_zen import instantiate, load_from_yaml

from proteinzen.data.datasets.datamodule import BiomoleculeDataModule
from proteinzen.stoch_interp.multiframe import MultiSE3Interpolant
from proteinzen.model.utils import gather_helper
from proteinzen.openfold.utils import rigid_utils as ru

# PyTorch 2.6+ safe-unpickling workaround (same as sample.py)
_orig_load = torch.load
def _patched_load(*args, weights_only=None, **kwargs):
    if weights_only is None:
        weights_only = False
    return _orig_load(*args, weights_only=weights_only, **kwargs)
torch.load = _patched_load


# ── CoM head ──────────────────────────────────────────────────────────────────

class CoMHead(nn.Module):
    def __init__(self, c_s: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_s),
            nn.SiLU(),
            nn.Linear(c_s, 4),  # [delta_x, delta_y, delta_z, confidence]
        )

    def forward(
        self,
        node_embed: torch.Tensor,          # (B, N_tok, c_s)
        token_noising_mask: torch.Tensor,  # (B, N_tok) bool — True = ligand
        token_trans: torch.Tensor,         # (B, N_tok, 3) — noised translations
    ) -> torch.Tensor:                     # (B, 3)
        out = self.mlp(node_embed)
        deltas = out[..., :3]
        confs  = out[..., 3].masked_fill(~token_noising_mask, -1e9)
        weights = F.softmax(confs, dim=-1)
        return (weights[..., None] * (token_trans + deltas)).sum(1)


# ── lightning module ───────────────────────────────────────────────────────────

class CoMPredictorLit(pl.LightningModule):
    def __init__(self, backbone: nn.Module, corrupter, c_s: int, lr: float = 1e-3,
                 cosine_annealing_T_max: int = 0):
        super().__init__()
        self.backbone = backbone
        self.corrupter = corrupter
        self.com_head = CoMHead(c_s)
        self.lr = lr
        self.cosine_annealing_T_max = cosine_annealing_T_max

    def _apply_etkdg(self, batch):
        etkdg_pos = batch['rigids']['etkdg_pos']           # (B, N, 3)
        noising_mask = batch['rigids']['rigids_noising_mask'].bool()  # (B, N)
        rigids_mask  = batch['rigids']['rigids_mask'].bool()

        lig_mask = noising_mask & rigids_mask

        # Per-sample flag: ETKDG succeeded if any ligand atom has non-zero position
        etkdg_valid = (etkdg_pos * lig_mask[..., None]).abs().sum(dim=(1, 2)) > 0  # (B,)

        lig_center = (etkdg_pos * lig_mask[..., None]).sum(1) \
                     / lig_mask.float().sum(1, keepdim=True).clamp(min=1)
        etkdg_centered = etkdg_pos - lig_center[:, None, :]

        old = ru.Rigid.from_tensor_7(batch['rigids']['rigids_t'])
        old_trans = old.get_trans()
        # Only replace translations for samples where ETKDG succeeded
        replace_mask = (noising_mask & etkdg_valid[:, None])[..., None]
        new_trans = torch.where(replace_mask, etkdg_centered, old_trans)
        batch['rigids']['rigids_t'] = ru.Rigid(rots=old.get_rots(), trans=new_trans).to_tensor_7()
        batch['rigids']['trans_t']  = new_trans
        return batch

    def _step(self, batch, split: str) -> torch.Tensor:
        batch['t'] = torch.ones_like(batch['t'])
        batch['trans_t'] = batch['t']
        batch['rot_t'] = batch['t']
        batch = self.corrupter.corrupt_dense_batch(batch, identity_rot_noise=False)
        batch = self._apply_etkdg(batch)

        token_to_rep = batch['token']['token_to_rep_rigid']  # (B, N_tok)

        # Token-level noising mask: True = ligand, False = protein
        token_noising_mask = gather_helper(
            batch['rigids']['rigids_noising_mask'][..., None].float(), token_to_rep
        ).squeeze(-1).bool()  # (B, N_tok)

        # mask protein↔ligand geometric information throughout the backbone
        cross_type_mask = token_noising_mask.unsqueeze(-1) ^ token_noising_mask.unsqueeze(-2)
        batch['cross_type_mask'] = cross_type_mask

        with torch.no_grad():
            out = self.backbone(batch)

        node_embed = out['node_embed']  # (B, N_tok, c_s)

        # GT CoM: mean crystal ligand translation from rigids_1
        rigids_1_trans = ru.Rigid.from_tensor_7(batch['rigids']['rigids_1']).get_trans()
        token_trans_gt  = gather_helper(rigids_1_trans, token_to_rep)
        gt_com = (token_trans_gt * token_noising_mask[..., None]).sum(1) \
                 / token_noising_mask.float().sum(1, keepdim=True).clamp(min=1)

        # Noised token translations for CoM head (from rigids_t)
        token_trans = gather_helper(
            ru.Rigid.from_tensor_7(batch['rigids']['rigids_t']).get_trans(), token_to_rep
        )

        com_pred = self.com_head(node_embed, token_noising_mask, token_trans)
        loss = F.mse_loss(com_pred, gt_com)
        self.log(f'{split}/mse',  loss,        prog_bar=False, sync_dist=True, on_epoch=True)
        self.log(f'{split}/rmse', loss.sqrt(), prog_bar=True,  sync_dist=True, on_epoch=True)
        return loss

    def training_step(self, batch, _):   return self._step(batch, 'train')
    def validation_step(self, batch, _): return self._step(batch, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.com_head.parameters(), lr=self.lr, weight_decay=1e-2)
        if self.cosine_annealing_T_max > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cosine_annealing_T_max
            )
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
        return optimizer


# ── checkpoint / config helpers ───────────────────────────────────────────────

def resolve_ckpt(run_dir: str, version_num: int, checkpoint_idx: int) -> str:
    pattern = os.path.join(run_dir, f"lightning_logs/version_{version_num}/checkpoints/*.ckpt")
    ckpt_list = glob.glob(pattern)
    epoch_list = []
    for p in ckpt_list:
        fname = Path(p).name
        if fname == 'last.ckpt':
            epoch_list.append((p, int(1e6)))
        elif '=' in fname:
            epoch_list.append((p, int(fname.split('=')[1].split('-')[0])))
    if not epoch_list:
        raise FileNotFoundError(f"No checkpoints found in {pattern}")
    epoch_list.sort(key=lambda x: x[1])
    return epoch_list[checkpoint_idx][0]


def load_backbone_and_corrupter(run_dir: str, version_num: int, checkpoint_idx: int, load_weights: bool = True):
    ckpt_path = resolve_ckpt(run_dir, version_num, checkpoint_idx)
    print(f"Loading proteinzen checkpoint: {ckpt_path}")

    config_path = os.path.join(run_dir, f"lightning_logs/version_{version_num}/config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    cfg = load_from_yaml(config_path)

    backbone = instantiate(cfg['model'])

    if load_weights:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = {k[len('model.'):]: v
                 for k, v in ckpt['state_dict'].items()
                 if k.startswith('model.')}
        backbone.load_state_dict(state, strict=False)
    backbone.requires_grad_(False)

    valid_params = set(inspect.signature(MultiSE3Interpolant.__init__).parameters) - {'self'}
    corrupter_kwargs = {k: v for k, v in cfg['corrupter'].items() if k in valid_params}
    # Old configs have center_on_motif=True but no center_on_motif_then_hotspots.
    # Current __init__ defaults center_on_motif_then_hotspots=True, which would
    # trigger the mutual-exclusion assertion. Default it to False here.
    if corrupter_kwargs.get('center_on_motif', False) and 'center_on_motif_then_hotspots' not in corrupter_kwargs:
        corrupter_kwargs['center_on_motif_then_hotspots'] = False
    corrupter = MultiSE3Interpolant(**corrupter_kwargs)

    c_s = cfg['model'].get('c_s', 768)
    return backbone, corrupter, c_s


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir',           required=True,
                    help='Run dir with lightning_logs/version_N inside')
    ap.add_argument('--version-num',         type=int, default=0)
    ap.add_argument('--checkpoint-idx',      type=int, default=-1,
                    help='-1 = last checkpoint (by epoch)')
    ap.add_argument('--resume-ckpt',         default=None,
                    help='Path to a CoM predictor .ckpt to resume from; '
                         'backbone weights are taken from this ckpt instead of --model-dir')
    ap.add_argument('--dataset-config',      required=True,
                    help='Path to train dataset config yaml (e.g. configs/train/data/plinder_protein_cond.yaml)')
    ap.add_argument('--val-dataset-config',  default=None,
                    help='Path to val dataset config yaml (e.g. configs/train/data/plinder_val.yaml)')
    ap.add_argument('--out-dir',             default='outputs/com_predictor')
    ap.add_argument('--lr',                  type=float, default=1e-3)
    ap.add_argument('--batch-size',          type=int,   default=4)
    ap.add_argument('--num-workers',         type=int,   default=4)
    ap.add_argument('--max-epochs',          type=int,   default=-1)
    ap.add_argument('--cosine-annealing-T-max', type=int, default=500,
                    help='CosineAnnealingLR T_max in epochs; 0 = disabled (constant lr)')
    ap.add_argument('--seed',                type=int,   default=42)
    args = ap.parse_args()

    pl.seed_everything(args.seed)

    # If resuming from a CoM predictor ckpt, skip loading proteinzen weights —
    # Lightning will restore the full model (backbone + com_head) from resume_ckpt.
    load_weights = args.resume_ckpt is None
    backbone, corrupter, c_s = load_backbone_and_corrupter(
        args.model_dir, args.version_num, args.checkpoint_idx, load_weights=load_weights
    )
    if args.resume_ckpt:
        print(f"Resuming from CoM predictor ckpt: {args.resume_ckpt}")
    print(f"Backbone loaded, c_s={c_s}")
    assert all(not p.requires_grad for p in backbone.parameters()), \
        "Backbone should be fully frozen"

    train_cfg = load_from_yaml(args.dataset_config)
    train_ds  = instantiate(train_cfg)

    val_ds = None
    if args.val_dataset_config:
        val_cfg = load_from_yaml(args.val_dataset_config)
        val_ds  = instantiate(val_cfg)

    datamodule = BiomoleculeDataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = CoMPredictorLit(backbone=backbone, corrupter=corrupter, c_s=c_s, lr=args.lr,
                            cosine_annealing_T_max=args.cosine_annealing_T_max)

    ckpt_cb = ModelCheckpoint(
        monitor='val/rmse', mode='min', save_top_k=3, save_last=True,
        filename='epoch={epoch:03d}-val_rmse={val/rmse:.3f}',
        auto_insert_metric_name=False,
    )
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else 1
    trainer = pl.Trainer(
        default_root_dir=args.out_dir,
        max_epochs=args.max_epochs,
        callbacks=[ckpt_cb],
        log_every_n_steps=10,
        devices=devices,
        strategy=DDPStrategy(cluster_environment=LightningEnvironment(), find_unused_parameters=True),
        use_distributed_sampler=False,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_ckpt)


if __name__ == '__main__':
    main()
