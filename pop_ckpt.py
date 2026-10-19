import argparse
import torch
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Strip optimizer/scheduler state from a checkpoint so it can be used "
                "as a fresh warm_start. Writes to {ckpt_path.parent}/{ckpt_path.stem}_no_opt.ckpt"
)
parser.add_argument("ckpt_path", type=Path, help="Path to the source .ckpt file")
args = parser.parse_args()

ckpt_path = args.ckpt_path
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
ckpt.pop("optimizer_states", None)
ckpt.pop("lr_schedulers", None)
ckpt.pop("callbacks", None)
ckpt.pop("loops", None)
ckpt["optimizer_states"] = []
ckpt["lr_schedulers"] = []
ckpt["epoch"] = 0
ckpt["global_step"] = 0
out_path = ckpt_path.parent / f"{ckpt_path.stem}_no_opt.ckpt"
torch.save(ckpt, out_path)
print(f"Finished saving checkpoint to {out_path}")
