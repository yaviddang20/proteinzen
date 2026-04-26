"""
Generate ligand conformers conditioned on protein pockets from Plinder structures.

Loads each requested Plinder system, fixes its protein chain as conditioning
input, and samples the ligand from pure noise using the trained flow-matching
model.  Output PDBs contain both the protein pocket and the generated ligand.

Usage
-----
python scripts/sample_pocket.py \
    --model-dir /path/to/training/run \
    --data-dir   /path/to/plinder/processed/val \
    --out-dir    ./pocket_samples \
    --num-samples 10 \
    [--system-ids 1abc__1.2_B_L 2xyz__1.1_A_L] \
    [--system-ids-file systems.txt] \
    [--num-steps 100] \
    [--version-num 0]

System IDs can be provided inline (--system-ids) or one-per-line in a text
file (--system-ids-file); both can be given together.
"""

import argparse
import glob
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from hydra_zen import load_from_yaml, instantiate
from lightning import Trainer

# PyTorch 2.6+ changed default weights_only=True which breaks checkpoint loading
_original_torch_load = torch.load
def _patched_torch_load(*args, weights_only=None, **kwargs):
    if weights_only is None:
        weights_only = False
    return _original_torch_load(*args, weights_only=weights_only, **kwargs)
torch.load = _patched_torch_load

from proteinzen.runtime.lmod import BiomoleculeSamplingModule, PDBWriter
from proteinzen.runtime.sampling.protein_pocket import ProteinPocketConditionedSampling
from proteinzen.stoch_interp.multiframe import MultiSE3Interpolant
from proteinzen.data.featurize.assembler import collate

log = logging.getLogger(__name__)


def _find_checkpoint(model_dir: Path, version_num: int, checkpoint_idx: int = -1) -> str:
    ckpt_glob = str(model_dir / f"lightning_logs/version_{version_num}/checkpoints/*.ckpt")
    ckpt_list = glob.glob(ckpt_glob)
    if not ckpt_list:
        raise FileNotFoundError(f"No checkpoints found at {ckpt_glob}")

    best_ckpt = None
    epoch_list = []
    for path in ckpt_list:
        fname = os.path.basename(path)
        if fname == "best.ckpt":
            best_ckpt = path
        elif fname == "last.ckpt":
            epoch_list.append((path, int(1e9)))
        else:
            try:
                epoch = int(fname.split("=")[1].split("-")[0])
                epoch_list.append((path, epoch))
            except (IndexError, ValueError):
                epoch_list.append((path, 0))

    if best_ckpt is not None:
        return best_ckpt

    epoch_list.sort(key=lambda x: x[1])
    return epoch_list[checkpoint_idx][0]


def _resolve_system_ids(args) -> list[str]:
    ids = list(args.system_ids or [])
    if args.system_ids_file:
        with open(args.system_ids_file) as f:
            ids.extend(line.strip() for line in f if line.strip())
    if not ids:
        raise ValueError("Provide at least one system ID via --system-ids or --system-ids-file")
    return ids


def _npz_path(data_dir: Path, system_id: str) -> Path:
    mid = system_id[1:3]
    return data_dir / "structures" / mid / f"{system_id}.npz"


def main():
    parser = argparse.ArgumentParser(
        description="Sample ligand conformers conditioned on a Plinder protein pocket."
    )
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Training run directory (contains .hydra/config.yaml and lightning_logs/)")
    parser.add_argument("--version-num", type=int, default=0,
                        help="lightning_logs version number to load checkpoint from")
    parser.add_argument("--checkpoint-idx", type=int, default=-1,
                        help="Index into epoch-sorted checkpoint list (-1 = latest)")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Plinder processed data directory (contains structures/ subdirectory)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for generated PDB files")
    parser.add_argument("--system-ids", nargs="+", default=None,
                        help="Plinder system IDs to sample from")
    parser.add_argument("--system-ids-file", type=str, default=None,
                        help="Text file with one system ID per line")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of ligand conformers to generate per system")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Number of ODE integration steps")
    parser.add_argument("--trans-std", type=float, default=16.0,
                        help="Std of initial ligand translation noise (Å)")
    parser.add_argument("--include-h", action="store_true",
                        help="Keep hydrogen atoms in the structure")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # ── resolve system IDs and npz paths ─────────────────────────────────────
    system_ids = _resolve_system_ids(args)
    log.info(f"Sampling {len(system_ids)} systems × {args.num_samples} conformers each")

    missing = [sid for sid in system_ids if not _npz_path(args.data_dir, sid).exists()]
    if missing:
        log.warning(f"{len(missing)} system(s) not found in {args.data_dir}: {missing[:5]}...")

    # ── load model from training config ──────────────────────────────────────
    config_path = args.model_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No hydra config found at {config_path}")
    model_cfg = load_from_yaml(config_path)
    model = instantiate(model_cfg["model"])

    ckpt_path = _find_checkpoint(args.model_dir, args.version_num, args.checkpoint_idx)
    log.info(f"Using checkpoint: {ckpt_path}")

    corrupter = MultiSE3Interpolant(num_timesteps=args.num_steps)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir()

    run_cfg = {
        "out_dir": str(out_dir),
        "save_traj": False,
        "output_motif_chains": True,
        "identity_rot_noise": False,
        "no_rot_sampling": False,
    }

    module = BiomoleculeSamplingModule(model, corrupter, run_cfg)
    module.strict_loading = False

    # ── build flat list of samples ────────────────────────────────────────────
    all_samples = []
    for system_id in system_ids:
        npz = _npz_path(args.data_dir, system_id)
        if not npz.exists():
            log.warning(f"Skipping {system_id}: npz not found at {npz}")
            continue
        task = ProteinPocketConditionedSampling(
            npz_path=str(npz),
            num_samples=args.num_samples,
            trans_std=args.trans_std,
            include_h=args.include_h,
            name=system_id,
        )
        try:
            for sample in task.sample_data():
                all_samples.append(sample)
        except Exception:
            log.exception(f"Failed to build samples for {system_id}, skipping")

    if not all_samples:
        log.error("No samples to process. Exiting.")
        sys.exit(1)

    log.info(f"Total samples to run: {len(all_samples)}")

    dataloader = DataLoader(
        all_samples,
        batch_size=args.batch_size,
        collate_fn=collate,
        shuffle=False,
    )

    # ── set up trainer ────────────────────────────────────────────────────────
    trainer_kwargs = {"use_distributed_sampler": False}
    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
        if len(devices) > 1:
            trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"
            trainer_kwargs["use_distributed_sampler"] = True
    else:
        devices = 1

    pred_writer = PDBWriter(output_dir=str(out_dir), run_cfg=run_cfg)

    trainer = Trainer(
        enable_progress_bar=True,
        enable_model_summary=False,
        devices=devices,
        callbacks=[pred_writer],
        **trainer_kwargs,
    )

    trainer.predict(
        model=module,
        dataloaders=[dataloader],
        ckpt_path=ckpt_path,
        return_predictions=False,
    )

    log.info(f"Done. PDB files written to {samples_dir}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
