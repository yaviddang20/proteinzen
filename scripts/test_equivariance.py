"""
SE(3) equivariance/invariance diagnostic for the proteinzen model.

For each output (trans_vf, rot_vf, pred_trans_1, pred_rotmats_1) we test:
  - Invariance under translation:  output(x + d)    == output(x)
  - Equivariance under translation: output(x + d)   == output(x) + d   [for trans outputs]
  - Invariance under rotation:     output(R x)      == output(x)
  - Equivariance under rotation:   output(R x)      == R output(x)     [where applicable]

This lets us diagnose whether the model is doing what it should even if it's
not strictly SE(3)-equivariant architecturally.

Usage:
    python scripts/test_equivariance.py \
        --ckpt  <path/to/last.ckpt> \
        --data_config <configs/train/data/plinder_val.yaml> \
        [--t 0.5] [--seed 42]
"""
import argparse
import copy
import sys
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRot

sys.path.insert(0, str(Path(__file__).parent.parent))
from proteinzen.openfold.utils import rigid_utils as ru


# ─── helpers ─────────────────────────────────────────────────────────────────

def apply_global_se3(batch, R: torch.Tensor, t: torch.Tensor):
    """Deep-copy batch and left-apply global (R, t) to rigids_t."""
    batch = copy.deepcopy(batch)
    rd = batch['rigids']
    rigids = ru.Rigid.from_tensor_7(rd['rigids_t'])

    old_trans = rigids.get_trans()                             # [B, N, 3]
    old_rots  = rigids.get_rots().get_rot_mats()               # [B, N, 3, 3]

    new_trans = (R @ old_trans.unsqueeze(-1)).squeeze(-1) + t  # [B, N, 3]
    new_rots  = R @ old_rots                                   # [B, N, 3, 3]

    rd['rigids_t'] = ru.Rigid(
        rots=ru.Rotation(rot_mats=new_rots), trans=new_trans
    ).to_tensor_7()
    if 'trans_t'   in rd: rd['trans_t']   = new_trans
    if 'rotmats_t' in rd: rd['rotmats_t'] = new_rots
    return batch


def run(model, batch):
    """Single forward pass; returns dict of outputs."""
    with torch.no_grad():
        out = model(batch)
    dr = out['denoised_rigids']
    trans_t  = ru.Rigid.from_tensor_7(batch['rigids']['rigids_t']).get_trans()

    pred_trans_1  = dr.get_trans()                        # [B, N, 3]
    pred_rotmats_1 = dr.get_rots().get_rot_mats()         # [B, N, 3, 3]
    t_scalar = batch['t'][:, 0].view(-1, 1, 1)            # [B, 1, 1]
    trans_vf = (pred_trans_1 - trans_t) / (1 - t_scalar + 1e-6)  # [B, N, 3]
    rot_vf   = out.get('pred_rot_vf')                     # [B, N, 3] or None
    return {
        'pred_trans_1':  pred_trans_1.squeeze(0),         # [N, 3]
        'pred_rotmats_1': pred_rotmats_1.squeeze(0),      # [N, 3, 3]
        'trans_vf':      trans_vf.squeeze(0),             # [N, 3]
        'rot_vf':        rot_vf.squeeze(0) if rot_vf is not None else None,  # [N, 3]
    }


def err_vec(a, b, mask):
    """Per-rigid L2 error between [N,3] tensors, masked."""
    return (a - b).norm(dim=-1)[mask]


def err_mat(a, b, mask):
    """Per-rigid Frobenius error between [N,3,3] tensors, masked."""
    return (a - b).norm(dim=(-2, -1))[mask]


def report(label, errs):
    if errs is None:
        print(f"  {label}: N/A (rot_vf not produced by model)")
        return
    print(f"  {label}: mean={errs.mean():.4f}  max={errs.max():.4f}  Å")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=Path, required=True,
                        help='Checkpoint path — weights do not affect equivariance, but we need the model arch')
    parser.add_argument('--hydra_config', type=Path, required=True,
                        help='Path to .hydra/config.yaml from the training run')
    parser.add_argument('--data_config', type=Path, required=True)
    parser.add_argument('--t', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_dir = args.hydra_config.parent.parent
    out_path = out_dir / f"equivariance_{args.ckpt.stem}.txt"
    out_file = open(out_path, 'w')
    sys.stdout = _Tee(sys.__stdout__, out_file)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import omegaconf
    from hydra.utils import instantiate
    from proteinzen.runtime.lmod import BiomoleculeModule

    print(f"Instantiating model from {args.hydra_config}...")
    cfg = omegaconf.OmegaConf.load(args.hydra_config)
    model_inst    = instantiate(cfg.model)
    corrupter_inst = instantiate(cfg.corrupter)
    # lmodule has _partial_: true in hydra config; bypass it entirely
    lmod_cfg = omegaconf.OmegaConf.to_container(cfg.lmodule, resolve=True)
    lmod_cfg.pop('_target_', None)
    lmod_cfg.pop('_partial_', None)
    lmod = BiomoleculeModule(model=model_inst, corrupter=corrupter_inst,
                             optim={'lr': 1e-4}, **lmod_cfg)

    print(f"Loading weights from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    lmod.load_state_dict(ckpt['state_dict'], strict=False)
    lmod.eval().to(device)
    model = lmod.ema.module if (lmod.use_ema and lmod.ema is not None) else lmod.model
    model.eval()

    # get one real batch
    print("Building dataloader...")
    from proteinzen.data.datasets.datamodule import collate
    from torch.utils.data import DataLoader

    val_cfg = omegaconf.OmegaConf.load(args.data_config)
    val_dataset = instantiate(val_cfg)
    loader = DataLoader(val_dataset, batch_size=args.batch_size,
                        collate_fn=collate, shuffle=True, num_workers=0)
    def to_device(x):
        if isinstance(x, torch.Tensor): return x.to(device)
        if isinstance(x, dict): return {k: to_device(v) for k, v in x.items()}
        if isinstance(x, list): return [to_device(v) for v in x]
        return x

    batch = to_device(next(iter(loader)))
    t_val = torch.full((args.batch_size,), args.t, device=device)
    batch['t']       = t_val.unsqueeze(-1)  # [B, 1]
    batch['trans_t'] = t_val               # [B]
    batch['rot_t']   = t_val               # [B]
    with torch.no_grad():
        batch = lmod.corrupter.corrupt_dense_batch(batch, lmod.identity_rot_noise)

    mask = batch['rigids']['rigids_mask'].squeeze(0).bool()
    print(f"Active rigids: {mask.sum().item()}")

    # random transforms
    R = torch.tensor(SciRot.random().as_matrix(), dtype=torch.float32, device=device)
    d = torch.randn(3, device=device) * 5.0
    eye = torch.eye(3, device=device)
    zero = torch.zeros(3, device=device)

    # forward passes
    o     = run(model, batch)                            # original
    o_td  = run(model, apply_global_se3(batch, eye, d)) # translated
    o_rd  = run(model, apply_global_se3(batch, R, zero))# rotated
    o_se3 = run(model, apply_global_se3(batch, R, d))   # full SE(3)

    def rot_vec(v): return (R @ v.T).T                  # [N,3] → [N,3]
    def rot_mat(M): return R @ M                         # [N,3,3] → [N,3,3]

    sep = '=' * 65
    print(f"\n{sep}")
    print("OUTPUT: pred_trans_1  (predicted clean translation)")
    print(sep)
    report("invariant  under translation  (wrong: should be equivariant)",
           err_vec(o_td['pred_trans_1'], o['pred_trans_1'], mask))
    report("equivariant under translation  (+d, should be ~0)",
           err_vec(o_td['pred_trans_1'], o['pred_trans_1'] + d, mask))
    report("invariant  under rotation     (wrong: should be equivariant)",
           err_vec(o_rd['pred_trans_1'], o['pred_trans_1'], mask))
    report("equivariant under rotation    (R@, should be ~0 if SE3-equiv)",
           err_vec(o_rd['pred_trans_1'], rot_vec(o['pred_trans_1']), mask))

    print(f"\n{sep}")
    print("OUTPUT: pred_rotmats_1  (predicted clean rotation matrices)")
    print(sep)
    report("invariant  under translation  (should be ~0)",
           err_mat(o_td['pred_rotmats_1'], o['pred_rotmats_1'], mask))
    report("invariant  under rotation     (wrong: should be equivariant)",
           err_mat(o_rd['pred_rotmats_1'], o['pred_rotmats_1'], mask))
    report("equivariant under rotation    (R@, should be ~0 if SE3-equiv)",
           err_mat(o_rd['pred_rotmats_1'], rot_mat(o['pred_rotmats_1']), mask))

    print(f"\n{sep}")
    print("OUTPUT: trans_vf  = (pred_trans_1 - trans_t) / (1-t)")
    print(sep)
    report("invariant  under translation  (should be ~0, d cancels in VF)",
           err_vec(o_td['trans_vf'], o['trans_vf'], mask))
    report("equivariant under rotation    (R@, should be ~0 if SE3-equiv)",
           err_vec(o_rd['trans_vf'], rot_vec(o['trans_vf']), mask))
    report("invariant  under rotation     (wrong: should be equivariant)",
           err_vec(o_rd['trans_vf'], o['trans_vf'], mask))

    print(f"\n{sep}")
    print("OUTPUT: rot_vf  (directly predicted rotation vector field)")
    print(sep)
    report("invariant  under translation  (should be ~0)",
           err_vec(o_td['rot_vf'], o['rot_vf'], mask) if o['rot_vf'] is not None else None)
    report("equivariant under rotation    (R@, should be ~0 if SE3-equiv)",
           err_vec(o_rd['rot_vf'], rot_vec(o['rot_vf']), mask) if o['rot_vf'] is not None else None)
    report("invariant  under rotation     (wrong: should be equivariant)",
           err_vec(o_rd['rot_vf'], o['rot_vf'], mask) if o['rot_vf'] is not None else None)

    print(f"\n{sep}")
    print("FULL SE(3) equivariance summary")
    print(sep)
    report("pred_trans_1   equivariant  (R@ + d)",
           err_vec(o_se3['pred_trans_1'], rot_vec(o['pred_trans_1']) + d, mask))
    report("pred_rotmats_1 equivariant  (R@)",
           err_mat(o_se3['pred_rotmats_1'], rot_mat(o['pred_rotmats_1']), mask))
    report("trans_vf       equivariant  (R@, d cancels)",
           err_vec(o_se3['trans_vf'], rot_vec(o['trans_vf']), mask))
    if o['rot_vf'] is not None:
        report("rot_vf         equivariant  (R@)",
               err_vec(o_se3['rot_vf'], rot_vec(o['rot_vf']), mask))
    print(f"{sep}\n")
    sys.stdout = sys.__stdout__
    out_file.close()
    print(f"Results written to {out_path}")


if __name__ == '__main__':
    main()
