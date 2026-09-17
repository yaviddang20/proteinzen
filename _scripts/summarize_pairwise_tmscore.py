#!/usr/bin/env python
"""Summarize pairwise_tmscore.py's output: per-ligand diversity stats and
cluster counts, from the tmscore_matrix_<CODE>.npy files it writes.

Two summaries, written into --tmscore-dir alongside the matrices:
  - diversity_summary.txt: basic pairwise stats (finite pairs, mean/min/max
    TM-score, symmetry/diagonal sanity checks).
  - cluster_summary.txt: number of clusters at a few TM-score thresholds,
    via two different definitions --
      * connected components (what Plinder's own "weak_component" columns
        use): edge if TM-score >= threshold, group anything reachable via a
        CHAIN of edges. Prone to a single-linkage "chaining" artifact --
        a long chain of moderately-similar pairs can merge the whole set
        into one giant component even when most individual pairs aren't
        alike. Included for reference, not recommended as "# unique".
      * greedy set-cover (what MMseqs2/CD-HIT actually do): repeatedly pick
        the still-unclustered structure with the most unclustered
        neighbors as a new cluster center, absorb it + its neighbors,
        repeat. This is the number worth trusting as "# distinct designs".
  0.5 (Zhang & Skolnick's "same fold" cutoff, also GTalign's own
  --cls-threshold default) is included among the thresholds; run with a
  different --thresholds list if 0.5 isn't the right call for your data
  (e.g. if your mean pairwise TM-score sits right on top of it).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def connected_components(adj: np.ndarray) -> int:
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    n_components = 0
    for i in range(n):
        if seen[i]:
            continue
        n_components += 1
        stack = [i]
        seen[i] = True
        while stack:
            u = stack.pop()
            for v in np.nonzero(adj[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return n_components


def greedy_set_cover(adj: np.ndarray) -> int:
    n = adj.shape[0]
    assigned = np.zeros(n, dtype=bool)
    n_clusters = 0
    while not assigned.all():
        remaining = ~assigned
        degree = (adj & remaining[None, :]).sum(axis=1)
        degree[assigned] = -1
        center = int(np.argmax(degree))
        n_clusters += 1
        assigned[center] = True
        neighbors = np.nonzero(adj[center] & remaining)[0]
        assigned[neighbors] = True
    return n_clusters


def diversity_summary(tmscore_dir: Path, codes: list[str]) -> str:
    lines = [
        "All-by-all GTalign structural diversity",
        f"tmscore_dir: {tmscore_dir}",
        "TM-score symmetrized via max(mat[i,j], mat[j,i])",
        "",
        f"{'ligand':<6} {'n':>4} {'finite_pairs':>15} {'diag=1.0':>9} "
        f"{'mean':>7} {'min':>7} {'max':>7} {'symmetric':>10}",
    ]
    for code in codes:
        mat = np.load(tmscore_dir / f"tmscore_matrix_{code}.npy")
        n = mat.shape[0]
        off_diag = ~np.eye(n, dtype=bool)
        finite = mat[off_diag & np.isfinite(mat)]
        diag = np.diag(mat)
        pairs_str = f"{finite.size}/{n * (n - 1)}"
        lines.append(
            f"{code:<6} {n:>4} {pairs_str:>15} {str(np.allclose(diag, 1.0)):>9} "
            f"{finite.mean():>7.3f} {finite.min():>7.3f} {finite.max():>7.3f} "
            f"{str(np.allclose(mat, mat.T, equal_nan=True)):>10}"
        )
    return "\n".join(lines) + "\n"


def cluster_summary(tmscore_dir: Path, codes: list[str], thresholds: list[float]) -> str:
    header = f"{'ligand':<6}"
    for t in thresholds:
        header += f"  cc@{t:<4}  greedy@{t:<4}"
    lines = [
        "Cluster counts from pairwise TM-score matrices",
        f"tmscore_dir: {tmscore_dir}",
        "cc = connected components (chaining-prone, see module docstring)",
        "greedy = greedy set-cover a la MMseqs2/CD-HIT (recommended '# unique')",
        "",
        header,
    ]
    for code in codes:
        mat = np.load(tmscore_dir / f"tmscore_matrix_{code}.npy")
        mat_filled = np.where(np.isfinite(mat), mat, 0.0)  # missing pairs treated as "not similar"
        row = f"{code:<6}"
        for t in thresholds:
            adj = mat_filled >= t
            np.fill_diagonal(adj, False)
            cc = connected_components(adj)
            gc = greedy_set_cover(adj)
            row += f"  {cc:>7}  {gc:>10}"
        lines.append(row)
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tmscore-dir", type=Path, required=True,
                   help="Output dir from pairwise_tmscore.py (contains tmscore_matrix_<CODE>.npy).")
    p.add_argument("--codes", nargs="+", default=None,
                   help="Ligand codes to summarize. Default: auto-discover from tmscore_matrix_*.npy filenames.")
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.4, 0.5, 0.6])
    args = p.parse_args()

    codes = args.codes or sorted(
        f.stem.replace("tmscore_matrix_", "") for f in args.tmscore_dir.glob("tmscore_matrix_*.npy")
    )
    if not codes:
        raise SystemExit(f"no tmscore_matrix_*.npy found in {args.tmscore_dir}")

    div_text = diversity_summary(args.tmscore_dir, codes)
    (args.tmscore_dir / "diversity_summary.txt").write_text(div_text)
    print(div_text)

    clu_text = cluster_summary(args.tmscore_dir, codes, args.thresholds)
    (args.tmscore_dir / "cluster_summary.txt").write_text(clu_text)
    print(clu_text)


if __name__ == "__main__":
    main()
