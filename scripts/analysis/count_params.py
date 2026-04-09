"""
Count number of parameters in a model checkpoint.
Usage: python count_params.py --ckpt path/to/checkpoint.ckpt
"""
import argparse
from pathlib import Path
from collections import defaultdict

import torch


def count_params(state_dict: dict) -> None:
    total = 0
    module_totals = defaultdict(int)

    for name, tensor in state_dict.items():
        n = tensor.numel()
        total += n
        # top-level module prefix (e.g. "model.encoder...." -> "model.encoder")
        parts = name.split(".")
        prefix = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        module_totals[prefix] += n

    print(f"\n{'Module':<60} {'Params':>12}")
    print("-" * 74)
    for mod, n in sorted(module_totals.items(), key=lambda x: -x[1]):
        print(f"{mod:<60} {n:>12,}")
    print("-" * 74)
    print(f"{'TOTAL':<60} {total:>12,}")
    print(f"{'TOTAL (M)':<60} {total/1e6:>11.2f}M\n")


def main():
    parser = argparse.ArgumentParser(description="Count params in a checkpoint.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Only count params under this prefix (e.g. 'model')",
    )
    args = parser.parse_args()

    print(f"Loading {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # Lightning checkpoints store weights under 'state_dict'
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print(f"Lightning checkpoint detected. Epoch: {ckpt.get('epoch', '?')}, "
              f"Global step: {ckpt.get('global_step', '?')}")
    else:
        state_dict = ckpt

    if args.prefix:
        state_dict = {
            k: v for k, v in state_dict.items() if k.startswith(args.prefix)
        }
        print(f"Filtered to prefix '{args.prefix}': {len(state_dict)} tensors")

    count_params(state_dict)


if __name__ == "__main__":
    main()
