"""
Benchmark: manual grad accumulation vs Lightning auto accumulate_grad_batches.
Runs both comparisons on single GPU and multi-GPU (DDP).

Usage:
  python grad_accum_benchmark.py --mode single
  python grad_accum_benchmark.py --mode multi [--gpus N]
"""

import contextlib
import json
import statistics
import tempfile
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback


# ─── Model ────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim=256, hidden=1024, out_dim=128, depth=6):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Manual accum LightningModule ─────────────────────────────────────────────
# Uses automatic_optimization=False; handles zero_grad/step internally.
# Timing: start at first micro-step of each group, stop just before opt.step().
# In DDP: uses model.no_sync() on intermediate micro-steps to skip redundant
# all-reduces, matching what Lightning does in auto accum mode.

class ManualLitModel(pl.LightningModule):
    def __init__(self, accum_steps: int, result_file: str | None = None, lr=1e-3):
        super().__init__()
        self.automatic_optimization = False
        self.accum_steps = accum_steps
        self.result_file = result_file
        self.model = MLP()
        self.loss_fn = nn.MSELoss()
        self._step_times: list[float] = []
        self._t0: float | None = None
        self._micro = 0
        self._accum_loss = 0.0

    def _no_sync_ctx(self, is_last: bool):
        """Skip DDP all-reduce on non-final micro-steps."""
        strategy_model = getattr(self.trainer.strategy, "model", None)
        if strategy_model is not None and hasattr(strategy_model, "no_sync") and not is_last:
            return strategy_model.no_sync()
        return contextlib.nullcontext()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch
        is_first = (self._micro % self.accum_steps == 0)
        is_last  = (self._micro % self.accum_steps == self.accum_steps - 1)

        if is_first:
            opt.zero_grad()
            self._t0 = time.perf_counter()
            self._accum_loss = 0.0

        with self._no_sync_ctx(is_last):
            loss = self.loss_fn(self.model(x), y) / self.accum_steps
            self.manual_backward(loss)
        self._accum_loss += loss.item()

        if is_last:
            # Record time before opt.step() — same span as TimingCallback on auto model
            self._step_times.append(time.perf_counter() - self._t0)
            opt.step()
            self.log("train_loss", self._accum_loss, on_step=True, prog_bar=True)

        self._micro += 1

    def on_train_end(self):
        if self.trainer.global_rank == 0 and self.result_file:
            loss = self.trainer.callback_metrics.get("train_loss")
            with open(self.result_file, "w") as f:
                json.dump({
                    "times": self._step_times,
                    "final_loss": loss.item() if loss is not None else float("nan"),
                }, f)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


# ─── Auto accum LightningModule ───────────────────────────────────────────────

class AutoLitModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = MLP()
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self.model(x), y)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


# ─── Timing Callback (auto model) ─────────────────────────────────────────────

class TimingCallback(Callback):
    def __init__(self, result_file=None):
        self.result_file = result_file
        self.step_times: list[float] = []
        self._t0: float | None = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._t0 is None:
            self._t0 = time.perf_counter()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self._t0 is not None:
            self.step_times.append(time.perf_counter() - self._t0)
            self._t0 = None

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank == 0 and self.result_file:
            loss = trainer.callback_metrics.get("train_loss")
            with open(self.result_file, "w") as f:
                json.dump({
                    "times": self.step_times,
                    "final_loss": loss.item() if loss is not None else float("nan"),
                }, f)


# ─── Dataset ──────────────────────────────────────────────────────────────────

def make_dataset(n=8192, in_dim=256, out_dim=128):
    torch.manual_seed(42)
    return TensorDataset(torch.randn(n, in_dim), torch.randn(n, out_dim))


# ─── Unified runner ───────────────────────────────────────────────────────────

def run(
    is_manual: bool,
    accum_steps: int,
    micro_batch: int,
    max_steps: int,
    accelerator: str,
    devices: int,
    strategy: str | None,
    result_file: str,
) -> dict:
    dataset = make_dataset()
    loader = DataLoader(dataset, batch_size=micro_batch, shuffle=True, drop_last=True, num_workers=0)

    if is_manual:
        model = ManualLitModel(accum_steps=accum_steps, result_file=result_file)
        trainer_accum = 1        # model manages accumulation internally
        callbacks = []
    else:
        model = AutoLitModel()
        timing_cb = TimingCallback(result_file=result_file)
        trainer_accum = accum_steps
        callbacks = [timing_cb]

    trainer_kwargs = dict(
        max_steps=max_steps * accum_steps,  # Lightning counts micro-steps
        accumulate_grad_batches=trainer_accum,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,
        callbacks=callbacks,
        precision="32-true",
    )
    if strategy:
        trainer_kwargs["strategy"] = strategy

    Trainer(**trainer_kwargs).fit(model, loader)

    with open(result_file) as f:
        return json.load(f)


# ─── Reporting ────────────────────────────────────────────────────────────────

def summarize(label: str, data: dict):
    times = data["times"][2:]  # drop 2 warm-up steps
    if not times:
        print(f"  {label}: insufficient data")
        return
    mean_t = statistics.mean(times) * 1000
    std_t  = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
    print(f"\n  {label}")
    print(f"    optimizer steps timed : {len(times)}")
    print(f"    mean step wall time   : {mean_t:.2f} ms  (±{std_t:.2f})")
    print(f"    final loss            : {data['final_loss']:.6f}")


def report(label_manual: str, manual: dict, label_auto: str, auto: dict):
    print(f"\n{'─'*56}")
    summarize(label_manual, manual)
    summarize(label_auto,   auto)
    mt = manual["times"][2:]
    at = auto["times"][2:]
    if mt and at:
        ratio = statistics.mean(at) / statistics.mean(mt)
        print(f"\n  Auto/Manual overhead: {ratio:.3f}x")
    print(f"{'─'*56}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="single", choices=["single", "multi"])
    parser.add_argument("--accum", type=int, default=8,  help="accumulation steps")
    parser.add_argument("--batch", type=int, default=64, help="micro-batch size")
    parser.add_argument("--steps", type=int, default=50, help="optimizer steps to time")
    parser.add_argument("--gpus",  type=int, default=None, help="GPUs for multi mode")
    args = parser.parse_args()

    print(f"Micro-batch: {args.batch} | Accum: {args.accum} | "
          f"Effective batch: {args.batch * args.accum} | Steps: {args.steps}")

    if args.mode == "single":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        print(f"Accelerator: {accelerator}\n")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            manual_file = f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            auto_file = f.name

        print("[1/2] Manual accum (single GPU) ...")
        manual_data = run(True,  args.accum, args.batch, args.steps, accelerator, 1, None, manual_file)

        print("\n[2/2] Auto accum (single GPU) ...")
        auto_data   = run(False, args.accum, args.batch, args.steps, accelerator, 1, None, auto_file)

        report("Manual accum — single GPU", manual_data,
               "Auto accum   — single GPU", auto_data)

    elif args.mode == "multi":
        n_gpus = args.gpus or torch.cuda.device_count()
        if n_gpus < 1:
            raise RuntimeError("No CUDA GPUs found.")
        if n_gpus < 2:
            print(f"WARNING: only {n_gpus} GPU — DDP runs but no inter-GPU comms.")
        print(f"GPUs: {n_gpus}\n")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            manual_file = f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            auto_file = f.name

        print(f"[1/2] Manual accum (DDP, {n_gpus} GPUs) ...")
        manual_data = run(True,  args.accum, args.batch, args.steps, "gpu", n_gpus, "ddp", manual_file)

        print(f"\n[2/2] Auto accum (DDP, {n_gpus} GPUs) ...")
        auto_data   = run(False, args.accum, args.batch, args.steps, "gpu", n_gpus, "ddp", auto_file)

        report(f"Manual accum — DDP {n_gpus} GPUs", manual_data,
               f"Auto accum   — DDP {n_gpus} GPUs", auto_data)


if __name__ == "__main__":
    main()
