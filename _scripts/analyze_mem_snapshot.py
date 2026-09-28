#!/usr/bin/env python
"""Aggregate a torch CUDA memory snapshot by allocating source line.

Usage:
    python _scripts/analyze_mem_snapshot.py /tmp/pz_mem.pickle [--top 25]
"""

import argparse
import pickle
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--skip-torch", action="store_true", default=True,
                    help="attribute to the innermost frame outside site-packages/torch")
    args = ap.parse_args()

    with open(args.snapshot, "rb") as fh:
        snap = pickle.load(fh)

    by_site = defaultdict(lambda: [0, 0])  # bytes, count
    total = 0
    for seg in snap.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            size = blk.get("size", 0)
            total += size
            frames = blk.get("frames") or []
            site = "<no stack>"
            for f in frames:
                fn = f.get("filename", "")
                if args.skip_torch and ("site-packages/torch" in fn or "<built-in>" in fn):
                    continue
                site = f"{fn}:{f.get('line')} {f.get('name')}"
                break
            else:
                if frames:
                    f = frames[0]
                    site = f"{f.get('filename')}:{f.get('line')} {f.get('name')}"
            by_site[site][0] += size
            by_site[site][1] += 1

    print(f"total active allocated: {total / 2**30:.2f} GiB across {len(by_site)} sites\n")
    rows = sorted(by_site.items(), key=lambda kv: -kv[1][0])[: args.top]
    for site, (nbytes, count) in rows:
        print(f"{nbytes / 2**30:8.3f} GiB  {count:6d} allocs  {site}")


if __name__ == "__main__":
    main()
