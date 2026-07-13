"""
Standalone CoM predictor: given protein structure + ligand SMILES (ETKDG conformer),
predict the 3D center of mass of the ligand in the protein frame.

Architecture
------------
- Ligand encoder  : atom-type embeddings + pairwise distance RBF → transformer → mean pool → global vec
- Protein encoder : residue-type embeddings + pairwise distance bias → transformer, conditioned on ligand global
- Per-token head  : MLP → (delta_i: 3D offset, w_i: scalar confidence)
- CoM             : sum_i( softmax(w_i) * (ca_i + delta_i) )
- Loss            : MSE(CoM_pred, CoM_gt)

Usage
-----
python _scripts/train_com_predictor.py \
    --manifest /path/to/plinder_pocket_processed/train/manifest.json \
    --val-manifest /path/to/plinder_pocket_processed/val/manifest.json \
    --out-dir outputs/com_predictor
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, Dataset

from proteinzen.boltz.data import const
from proteinzen.runtime.sampling.protein_pocket import _crop_protein_to_pocket, load_structure_from_npz

RDLogger.DisableLog("rdApp.*")

# ── constants ─────────────────────────────────────────────────────────────────

N_RBF       = 20
RBF_D_MAX   = 32.0
TOKEN_VOCAB = 33    # proteinzen token vocab (const.tokens)
ELEM_VOCAB  = 119   # atomic numbers 0-118


# ── helpers ───────────────────────────────────────────────────────────────────

def rbf(d: torch.Tensor, n: int = N_RBF, d_max: float = RBF_D_MAX) -> torch.Tensor:
    centers = torch.linspace(0.0, d_max, n, device=d.device, dtype=d.dtype)
    sigma   = d_max / (n - 1) * 1.5
    return torch.exp(-0.5 * ((d.unsqueeze(-1) - centers) / sigma) ** 2)


def pairwise_rbf(coords: torch.Tensor, n_heads: int, rbf_proj: nn.Linear) -> torch.Tensor:
    """coords: (B, N, 3) → attn bias (B*n_heads, N, N) ready for nn.Transformer."""
    diff  = coords.unsqueeze(2) - coords.unsqueeze(1)   # (B, N, N, 3)
    dists = diff.norm(dim=-1)                            # (B, N, N)
    bias  = rbf_proj(rbf(dists))                         # (B, N, N, n_heads)
    B, N  = dists.shape[:2]
    return bias.permute(0, 3, 1, 2).reshape(B * n_heads, N, N)


# ── model modules ─────────────────────────────────────────────────────────────

class LigandEncoder(nn.Module):
    def __init__(self, hidden: int, n_layers: int, n_heads: int):
        super().__init__()
        self.embed    = nn.Embedding(ELEM_VOCAB, hidden)
        self.rbf_proj = nn.Linear(N_RBF, n_heads, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out      = nn.Linear(hidden, hidden)

    def forward(self, elements: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        elements : (B, M) int
        coords   : (B, M, 3) — ETKDG, arbitrary frame, centered at 0
        mask     : (B, M) bool — True = real atom
        → (B, hidden)
        """
        x    = self.embed(elements)
        B, M = x.shape[:2]
        bias = pairwise_rbf(coords, self.rbf_proj.out_features, self.rbf_proj)
        x    = self.encoder(x, mask=bias, src_key_padding_mask=~mask)
        x    = (x * mask.float().unsqueeze(-1)).sum(1) / mask.float().sum(1, keepdim=True).clamp(min=1)
        return self.out(x)   # (B, hidden)


class ProteinEncoder(nn.Module):
    def __init__(self, hidden: int, n_layers: int, n_heads: int):
        super().__init__()
        self.embed    = nn.Embedding(TOKEN_VOCAB, hidden)
        self.rbf_proj = nn.Linear(N_RBF, n_heads, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.encoder  = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        res_types:  torch.Tensor,   # (B, N) int
        ca_coords:  torch.Tensor,   # (B, N, 3)
        res_mask:   torch.Tensor,   # (B, N) bool
        lig_global: torch.Tensor,   # (B, hidden)
    ) -> torch.Tensor:              # → (B, N, hidden)
        x    = self.embed(res_types) + lig_global.unsqueeze(1)
        bias = pairwise_rbf(ca_coords, self.rbf_proj.out_features, self.rbf_proj)
        return self.encoder(x, mask=bias, src_key_padding_mask=~res_mask)


class CoMPredictorModel(nn.Module):
    def __init__(self, hidden: int = 128, prot_layers: int = 4, lig_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.lig_enc  = LigandEncoder(hidden, lig_layers, n_heads)
        self.prot_enc = ProteinEncoder(hidden, prot_layers, n_heads)
        self.head     = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4),   # [delta_x, delta_y, delta_z, confidence]
        )

    def forward(
        self,
        res_types:   torch.Tensor,  # (B, N)
        ca_coords:   torch.Tensor,  # (B, N, 3)
        res_mask:    torch.Tensor,  # (B, N)
        lig_elems:   torch.Tensor,  # (B, M)
        lig_coords:  torch.Tensor,  # (B, M, 3)
        lig_mask:    torch.Tensor,  # (B, M)
    ) -> torch.Tensor:              # → (B, 3)
        lig_global  = self.lig_enc(lig_elems, lig_coords, lig_mask)
        prot_tokens = self.prot_enc(res_types, ca_coords, res_mask, lig_global)

        out    = self.head(prot_tokens)          # (B, N, 4)
        deltas = out[..., :3]                    # (B, N, 3)
        confs  = out[..., 3].masked_fill(~res_mask, float('-inf'))
        weights = F.softmax(confs, dim=-1)       # (B, N)

        return (weights.unsqueeze(-1) * (ca_coords + deltas)).sum(1)  # (B, 3)


# ── dataset ───────────────────────────────────────────────────────────────────

class PlindersCoMDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        max_protein_residues: int = 100,
        seed: int = 42,
    ):
        with open(manifest_path) as f:
            raw = json.load(f)
        self.records        = [r for r in raw if r.get('smiles')]
        self.structures_dir = Path(manifest_path).parent / 'structures'
        self.max_prot_res   = max_protein_residues
        self.rng            = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def _etkdg(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed  = self.rng.randint(0, 2**31 - 1)
        params.numThreads  = 1
        if not AllChem.EmbedMolecule(mol, params) == 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol, numThreads=1)
        mol = AllChem.RemoveHs(mol)
        conf     = mol.GetConformer()
        coords   = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], np.float32)
        elements = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())], np.int64)
        coords  -= coords.mean(0)   # center in own frame
        return coords, elements

    def __getitem__(self, idx: int):
        record = self.records[idx]
        sid    = record['id']
        mid    = sid[1:3]
        npz    = self.structures_dir / mid / f"{sid}.npz"

        try:
            struct = load_structure_from_npz(str(npz), include_h=False)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        if self.max_prot_res:
            struct = _crop_protein_to_pocket(struct, self.max_prot_res)

        protein_id    = const.chain_type_ids['PROTEIN']
        nonpolymer_id = const.chain_type_ids['NONPOLYMER']
        active        = struct.chains[struct.mask]

        # ── protein: Cα position ≈ mean backbone atom coords, residue type ──
        ca_list, rt_list = [], []
        for chain in active:
            if int(chain['mol_type']) != protein_id:
                continue
            r0, rn = int(chain['res_idx']), int(chain['res_num'])
            for res in struct.residues[r0:r0+rn]:
                if not res['is_present']:
                    continue
                a0, an = int(res['atom_idx']), int(res['atom_num'])
                atoms  = struct.atoms[a0:a0+an]
                pres   = atoms['is_present'].astype(bool)
                n_bb   = min(4, an)   # N, CA, C, O
                bb     = atoms[:n_bb]
                bb_pres = bb['is_present'].astype(bool)
                if not bb_pres.any():
                    continue
                ca_list.append(bb['coords'][bb_pres].mean(0))
                rt_list.append(int(res['res_type']) % TOKEN_VOCAB)

        if not ca_list:
            return self.__getitem__((idx + 1) % len(self))

        ca_coords = np.array(ca_list, np.float32)   # (N, 3)
        res_types = np.array(rt_list, np.int64)      # (N,)

        # ── GT ligand CoM (crystal, in protein-centered frame) ──
        lig_list = []
        for chain in active:
            if int(chain['mol_type']) != nonpolymer_id:
                continue
            a0, an = int(chain['atom_idx']), int(chain['atom_num'])
            atoms  = struct.atoms[a0:a0+an]
            pres   = atoms['is_present'].astype(bool) & (atoms['element'] != 1)
            if pres.any():
                lig_list.append(atoms['coords'][pres])

        if not lig_list:
            return self.__getitem__((idx + 1) % len(self))

        gt_com = np.concatenate(lig_list).mean(0).astype(np.float32)

        # center everything on protein Cα COM (consistent with training corruption)
        prot_center  = ca_coords.mean(0)
        ca_coords   -= prot_center
        gt_com      -= prot_center

        # ── ETKDG conformer ──
        etkdg = self._etkdg(record['smiles'])
        if etkdg is None:
            return self.__getitem__((idx + 1) % len(self))
        lig_coords, lig_elems = etkdg

        return {
            'ca_coords':  torch.from_numpy(ca_coords),   # (N, 3)
            'res_types':  torch.from_numpy(res_types),   # (N,)
            'gt_com':     torch.from_numpy(gt_com),      # (3,)
            'lig_coords': torch.from_numpy(lig_coords),  # (M, 3)
            'lig_elems':  torch.from_numpy(lig_elems),   # (M,)
        }


def collate_fn(batch):
    N_max = max(b['ca_coords'].shape[0] for b in batch)
    M_max = max(b['lig_coords'].shape[0] for b in batch)

    ca_coords  = torch.zeros(len(batch), N_max, 3)
    res_types  = torch.zeros(len(batch), N_max, dtype=torch.long)
    res_mask   = torch.zeros(len(batch), N_max, dtype=torch.bool)
    lig_coords = torch.zeros(len(batch), M_max, 3)
    lig_elems  = torch.zeros(len(batch), M_max, dtype=torch.long)
    lig_mask   = torch.zeros(len(batch), M_max, dtype=torch.bool)
    gt_com     = torch.stack([b['gt_com'] for b in batch])

    for i, b in enumerate(batch):
        N = b['ca_coords'].shape[0]
        M = b['lig_coords'].shape[0]
        ca_coords[i, :N]  = b['ca_coords']
        res_types[i, :N]  = b['res_types']
        res_mask[i, :N]   = True
        lig_coords[i, :M] = b['lig_coords']
        lig_elems[i, :M]  = b['lig_elems']
        lig_mask[i, :M]   = True

    return dict(
        ca_coords=ca_coords, res_types=res_types, res_mask=res_mask,
        lig_coords=lig_coords, lig_elems=lig_elems, lig_mask=lig_mask,
        gt_com=gt_com,
    )


# ── lightning module ──────────────────────────────────────────────────────────

class CoMPredictorLit(pl.LightningModule):
    def __init__(self, hidden=128, prot_layers=4, lig_layers=2, n_heads=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = CoMPredictorModel(hidden, prot_layers, lig_layers, n_heads)
        self.lr    = lr

    def _step(self, batch, split: str):
        pred = self.model(
            batch['res_types'], batch['ca_coords'], batch['res_mask'],
            batch['lig_elems'], batch['lig_coords'], batch['lig_mask'],
        )
        loss = F.mse_loss(pred, batch['gt_com'])
        rmse = loss.sqrt()
        self.log(f"{split}/mse",  loss, prog_bar=(split == 'val'), sync_dist=True)
        self.log(f"{split}/rmse", rmse, prog_bar=True,             sync_dist=True)
        return loss

    def training_step(self, batch, _):   return self._step(batch, 'train')
    def validation_step(self, batch, _): return self._step(batch, 'val')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest',     required=True)
    ap.add_argument('--val-manifest', required=True)
    ap.add_argument('--out-dir',      default='outputs/com_predictor')
    ap.add_argument('--max-prot-res', type=int,   default=100)
    ap.add_argument('--hidden',       type=int,   default=128)
    ap.add_argument('--prot-layers',  type=int,   default=4)
    ap.add_argument('--lig-layers',   type=int,   default=2)
    ap.add_argument('--n-heads',      type=int,   default=4)
    ap.add_argument('--lr',           type=float, default=1e-3)
    ap.add_argument('--batch-size',   type=int,   default=32)
    ap.add_argument('--max-epochs',   type=int,   default=100)
    ap.add_argument('--num-workers',  type=int,   default=4)
    ap.add_argument('--seed',         type=int,   default=42)
    args = ap.parse_args()

    pl.seed_everything(args.seed)

    train_ds = PlindersCoMDataset(args.manifest,     args.max_prot_res, args.seed)
    val_ds   = PlindersCoMDataset(args.val_manifest, args.max_prot_res, args.seed)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    model = CoMPredictorLit(args.hidden, args.prot_layers, args.lig_layers, args.n_heads, args.lr)

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor='val/rmse', mode='min', save_top_k=3, save_last=True,
    )
    trainer = pl.Trainer(
        default_root_dir=args.out_dir,
        max_epochs=args.max_epochs,
        callbacks=[ckpt_cb],
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == '__main__':
    main()
