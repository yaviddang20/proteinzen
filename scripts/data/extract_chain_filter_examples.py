#!/usr/bin/env python3
"""Pull a handful of raw Plinder systems that plinder.py's chain-count filter
rejected (multi-protein-chain and/or multi-ligand systems) out for manual
inspection.

Reads filtered_ids.json from each split written by the pocket-filtered run
(scripts/data/run_plinder_pocket.sh), pools every system_id rejected under a
"chain_filter_*" reason (plinder.py's process_system returns
f"chain_filter_protein{n_protein}_ligand{n_ligand}" whenever n_ligand != 1 or
n_protein is outside the allowed count -- the only reason string with that
prefix), and copies N of them from the raw Plinder systems directory into a
local output folder. Sampling uses a fixed seed over a sorted pool, so
repeat runs are idempotent.

No arguments -- assumes the standard run_plinder_pocket.sh layout relative
to the repo root (derived from this file's own path).
"""
import json
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLINDER_DIR = REPO_ROOT / "plinder" / "2024-06" / "v2"
POCKET_OUTDIR = REPO_ROOT / "plinder_pocket_processed"
SPLITS = ["train", "val", "test"]

OUT_DIR = REPO_ROOT / "plinder_pocket_chain_filter_examples"
N_EXAMPLES = 5
SEED = 0


def main():
    pool = []  # (system_id, reason, split)
    for split in SPLITS:
        f = POCKET_OUTDIR / split / "filtered_ids.json"
        if not f.exists():
            print(f"skipping {split}: {f} not found")
            continue
        filtered_ids = json.loads(f.read_text())
        for reason, ids in filtered_ids.items():
            if not reason.startswith("chain_filter_"):
                continue
            for sid in ids:
                pool.append((sid, reason, split))

    if not pool:
        raise SystemExit(
            "No chain_filter_* rejected systems found under "
            f"{POCKET_OUTDIR}/{{train,val,test}}/filtered_ids.json -- "
            "has run_plinder_pocket.sh been run?"
        )

    # Sort first so sampling is deterministic regardless of on-disk dict/list
    # ordering, then sample with a fixed seed -- same pool in, same 5 out.
    pool.sort()
    n = min(N_EXAMPLES, len(pool))
    chosen = random.Random(SEED).sample(pool, n)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    manifest = {}
    for sid, reason, split in chosen:
        src = PLINDER_DIR / "systems" / sid
        dst = OUT_DIR / sid
        if not src.exists():
            print(f"WARNING: raw system dir missing for {sid} ({src}), skipping")
            continue
        shutil.copytree(src, dst)
        manifest[sid] = {"reason": reason, "split": split}
        print(f"copied {sid}  (reason={reason}, split={split})")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} example systems to {OUT_DIR}")


if __name__ == "__main__":
    main()
