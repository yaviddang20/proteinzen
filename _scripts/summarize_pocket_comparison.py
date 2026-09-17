#!/usr/bin/env python
"""Summarize compare_generated_to_real_pockets.py's output: for each generated
design, its best (max) TM-score against any known real binder pocket for that
ligand, aggregated per ligand code.

Unlike pairwise_tmscore.py's self-comparison matrices (symmetric, diagonal=1,
clusterable), cross_tmscore_<CODE>.npy is a rectangular
(generated_sample x real_reference) matrix -- there's no self-similarity or
diagonal to sanity-check, and "clustering" doesn't apply the same way, so this
is a separate, simpler summary: per generated design, how close is its pocket
to the single most similar real reference pocket.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SUCCESS_THRESHOLD = 0.5  # Zhang & Skolnick "same fold" cutoff, same one used
                          # for the generation-diversity clustering summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comparison-dir", type=Path, required=True,
                   help="Output dir from compare_generated_to_real_pockets.py.")
    p.add_argument("--codes", nargs="+", default=None,
                   help="Ligand codes to summarize. Default: auto-discover from cross_tmscore_<CODE>.npy filenames.")
    args = p.parse_args()

    codes = args.codes or sorted(
        f.stem.replace("cross_tmscore_", "") for f in args.comparison_dir.glob("cross_tmscore_*.npy")
    )
    if not codes:
        raise SystemExit(f"no cross_tmscore_*.npy found in {args.comparison_dir}")

    lines = [
        "Generated-vs-real-binder-pocket comparison (GTalign)",
        f"comparison_dir: {args.comparison_dir}",
        f"'resembles_known' = fraction of generated designs whose best match to ANY "
        f"real reference pocket has TM-score >= {SUCCESS_THRESHOLD}",
        "",
        f"{'ligand':<6} {'n_gen':>6} {'n_ref':>6} {'finite_pairs':>15} "
        f"{'best_mean':>10} {'best_min':>9} {'best_max':>9} {'resembles_known':>16}",
    ]
    for code in codes:
        mat = np.load(args.comparison_dir / f"cross_tmscore_{code}.npy")
        n_gen, n_ref = mat.shape
        best_per_query = np.nanmax(mat, axis=1)
        finite = best_per_query[np.isfinite(best_per_query)]
        resembles_frac = (finite >= SUCCESS_THRESHOLD).mean() if finite.size else float("nan")
        finite_pairs = int(np.isfinite(mat).sum())
        pairs_str = f"{finite_pairs}/{n_gen * n_ref}"
        lines.append(
            f"{code:<6} {n_gen:>6} {n_ref:>6} {pairs_str:>15} "
            f"{finite.mean():>10.3f} {finite.min():>9.3f} {finite.max():>9.3f} "
            f"{resembles_frac*100:>15.1f}%"
        )

    text = "\n".join(lines) + "\n"
    (args.comparison_dir / "pocket_comparison_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
