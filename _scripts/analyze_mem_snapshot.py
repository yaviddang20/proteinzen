#!/usr/bin/env python
"""Attribute peak CUDA memory to the Python lines that allocated it.

Replays the alloc/free trace in a torch memory snapshot, finds the moment of
peak usage, and reports which source lines own the memory live at that moment.

Usage:
    python _scripts/analyze_mem_snapshot.py /tmp/pz_mem.pickle [--top 25]
"""

import argparse
import pickle
from collections import defaultdict

_ALLOC = {"alloc", "segment_alloc"}
_FREE = {"free_completed", "segment_free"}


def _site(frames, skip_torch=True):
    if not frames:
        return "<no stack>"
    for f in frames:
        fn = f.get("filename", "") or ""
        if skip_torch and ("site-packages/torch" in fn or fn.startswith("<")):
            continue
        return f"{fn}:{f.get('line')} {f.get('name')}"
    f = frames[0]
    return f"{f.get('filename')}:{f.get('line')} {f.get('name')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--keep-torch", action="store_true",
                    help="attribute to the innermost frame even if it is inside torch")
    args = ap.parse_args()

    with open(args.snapshot, "rb") as fh:
        snap = pickle.load(fh)

    traces = snap.get("device_traces") or []
    if not traces:
        print("no device_traces in snapshot — was _record_memory_history called?")
        return

    best = None
    for dev, events in enumerate(traces):
        live = {}
        cur = 0
        peak = 0
        peak_live = None
        for ev in events:
            action = ev.get("action")
            addr = ev.get("addr")
            size = ev.get("size", 0)
            if action in _ALLOC:
                live[addr] = (size, _site(ev.get("frames"), not args.keep_torch))
                cur += size
                if cur > peak:
                    peak = cur
                    peak_live = dict(live)
            elif action in _FREE:
                if addr in live:
                    cur -= live.pop(addr)[0]
        if peak and (best is None or peak > best[1]):
            best = (dev, peak, peak_live, len(events))

    if best is None:
        print("no allocation events found")
        return

    dev, peak, peak_live, n_events = best
    by_site = defaultdict(lambda: [0, 0])
    for size, site in peak_live.values():
        by_site[site][0] += size
        by_site[site][1] += 1

    print(f"device {dev}: peak {peak / 2**30:.2f} GiB live, "
          f"{len(peak_live)} tensors, {n_events} trace events\n")
    for site, (nbytes, count) in sorted(by_site.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(f"{nbytes / 2**30:8.3f} GiB  {count:6d} tensors  {site}")


if __name__ == "__main__":
    main()
