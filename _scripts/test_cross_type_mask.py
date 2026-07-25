"""
Invariance test for cross_type_mask geometric masking.

If masking is correct, translating the ligand by an arbitrary offset should
produce bit-identical node_embed from the backbone (the backbone cannot see
any protein↔ligand geometric information).

As a sanity check we also verify that WITHOUT cross_type_mask the two runs
DO differ, confirming that the mask is actually doing something.

Usage
-----
python _scripts/test_cross_type_mask.py \
    --model-dir outputs/plinder_protein_cond/train \
    --dataset-config configs/train/data/com_pocket.yaml
"""

import argparse
import copy
import sys

import torch
from hydra_zen import instantiate, load_from_yaml

# reuse helpers from train_com_predictor
sys.path.insert(0, '.')
from _scripts.train_com_predictor import load_backbone_and_corrupter

from proteinzen.model.utils import gather_helper
from proteinzen.openfold.utils import rigid_utils as ru
from proteinzen.stoch_interp.multiframe import MultiSE3Interpolant

_orig_load = torch.load
def _patched_load(*args, weights_only=None, **kwargs):
    if weights_only is None:
        weights_only = False
    return _orig_load(*args, weights_only=weights_only, **kwargs)
torch.load = _patched_load


OFFSET = torch.tensor([200.0, 0.0, 0.0])  # large translation applied to ligand


def apply_etkdg(batch):
    etkdg_pos   = batch['rigids']['etkdg_pos']
    noising_mask = batch['rigids']['rigids_noising_mask'].bool()
    rigids_mask  = batch['rigids']['rigids_mask'].bool()
    lig_mask     = noising_mask & rigids_mask

    etkdg_valid  = (etkdg_pos * lig_mask[..., None]).abs().sum(dim=(1, 2)) > 0
    lig_center   = (etkdg_pos * lig_mask[..., None]).sum(1) \
                   / lig_mask.float().sum(1, keepdim=True).clamp(min=1)
    etkdg_centered = etkdg_pos - lig_center[:, None, :]

    old = ru.Rigid.from_tensor_7(batch['rigids']['rigids_t'])
    replace_mask = (noising_mask & etkdg_valid[:, None])[..., None]
    new_trans = torch.where(replace_mask, etkdg_centered, old.get_trans())
    batch['rigids']['rigids_t'] = ru.Rigid(rots=old.get_rots(), trans=new_trans).to_tensor_7()
    batch['rigids']['trans_t']  = new_trans
    return batch, etkdg_valid


def translate_ligand(batch, offset):
    """Return a deep-copied batch with ligand frames shifted by offset."""
    b = copy.deepcopy(batch)
    noising_mask = b['rigids']['rigids_noising_mask'].bool()
    rigids_mask  = b['rigids']['rigids_mask'].bool()
    lig_mask     = noising_mask & rigids_mask

    old = ru.Rigid.from_tensor_7(b['rigids']['rigids_t'])
    dev  = old.get_trans().device
    new_trans = old.get_trans() + offset.to(dev) * lig_mask[..., None].float()
    b['rigids']['rigids_t'] = ru.Rigid(rots=old.get_rots(), trans=new_trans).to_tensor_7()
    b['rigids']['trans_t']  = new_trans
    return b


def run_backbone(backbone, batch, with_cross_mask: bool):
    token_to_rep = batch['token']['token_to_rep_rigid']
    token_noising_mask = gather_helper(
        batch['rigids']['rigids_noising_mask'][..., None].float(), token_to_rep
    ).squeeze(-1).bool()

    b = copy.deepcopy(batch)
    if with_cross_mask:
        b['cross_type_mask'] = token_noising_mask.unsqueeze(-1) ^ token_noising_mask.unsqueeze(-2)
    else:
        b.pop('cross_type_mask', None)

    with torch.no_grad():
        out = backbone(b)
    return out['node_embed']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir',      required=True)
    ap.add_argument('--version-num',    type=int, default=0)
    ap.add_argument('--checkpoint-idx', type=int, default=-1)
    ap.add_argument('--dataset-config', required=True)
    ap.add_argument('--device',         default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--atol',           type=float, default=1e-4,
                    help='Absolute tolerance for masked-run comparison')
    args = ap.parse_args()

    device = torch.device(args.device)

    backbone, corrupter, c_s = load_backbone_and_corrupter(
        args.model_dir, args.version_num, args.checkpoint_idx
    )
    backbone = backbone.to(device).eval()

    train_cfg = load_from_yaml(args.dataset_config)
    ds = instantiate(train_cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                         collate_fn=ds.collate_fn if hasattr(ds, 'collate_fn') else None)

    for batch in loader:
        # move to device
        def to_dev(x):
            if isinstance(x, dict):
                return {k: to_dev(v) for k, v in x.items()}
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return x
        batch = to_dev(batch)

        # same setup as _step
        batch['t'] = torch.ones_like(batch['t'])
        batch['trans_t'] = batch['t']
        batch['rot_t']   = batch['t']
        batch = corrupter.corrupt_dense_batch(batch, identity_rot_noise=False)
        batch, etkdg_valid = apply_etkdg(batch)

        if not etkdg_valid.any():
            print("ETKDG failed for this batch, skipping")
            continue

        # keep only valid samples
        valid_idx = etkdg_valid.nonzero(as_tuple=True)[0]
        batch = {k: ({kk: vv[valid_idx] for kk, vv in v.items()} if isinstance(v, dict) else v[valid_idx])
                 for k, v in batch.items()}

        batch_shifted = translate_ligand(batch, OFFSET)

        # ── Test 1: with cross_type_mask, outputs must be identical ──────────
        embed_orig    = run_backbone(backbone, batch,         with_cross_mask=True)
        embed_shifted = run_backbone(backbone, batch_shifted, with_cross_mask=True)

        max_diff = (embed_orig - embed_shifted).abs().max().item()
        mean_diff = (embed_orig - embed_shifted).abs().mean().item()
        passed = max_diff < args.atol

        print(f"\n[MASKED] ligand translated by {OFFSET.tolist()}")
        print(f"  max |Δnode_embed| = {max_diff:.2e}  (tol={args.atol:.1e})")
        print(f"  mean|Δnode_embed| = {mean_diff:.2e}")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")

        # ── Test 2: without cross_type_mask, outputs must differ (sanity) ────
        embed_orig_nomask    = run_backbone(backbone, batch,         with_cross_mask=False)
        embed_shifted_nomask = run_backbone(backbone, batch_shifted, with_cross_mask=False)

        max_diff_nm = (embed_orig_nomask - embed_shifted_nomask).abs().max().item()
        differs = max_diff_nm > args.atol

        print(f"\n[UNMASKED] ligand translated by {OFFSET.tolist()}")
        print(f"  max |Δnode_embed| = {max_diff_nm:.2e}")
        print(f"  {'PASS ✓ (outputs differ as expected)' if differs else 'FAIL ✗ (outputs identical — mask has no effect?)'}")

        if passed and differs:
            print("\nOverall: PASS — masking correctly blocks ligand position from backbone.")
            sys.exit(0)
        else:
            print("\nOverall: FAIL")
            sys.exit(1)

    print("No valid batches found.")
    sys.exit(1)


if __name__ == '__main__':
    main()
