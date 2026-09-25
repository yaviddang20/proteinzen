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


def greedy_set_cover(adj: np.ndarray) -> list[int]:
    """Returns the chosen cluster-center indices (len() of this = cluster count)."""
    n = adj.shape[0]
    assigned = np.zeros(n, dtype=bool)
    centers = []
    while not assigned.all():
        remaining = ~assigned
        degree = (adj & remaining[None, :]).sum(axis=1)
        degree[assigned] = -1
        center = int(np.argmax(degree))
        centers.append(center)
        assigned[center] = True
        neighbors = np.nonzero(adj[center] & remaining)[0]
        assigned[neighbors] = True
    return centers


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


def load_successful_names(results_json: Path, success_key: str) -> set[str]:
    """results.json (from eval_pallatom_ligand_cond.py) -> set of sample_ids
    where r[success_key] is truthy. Field names verified directly against a
    real results.json: sample_id, ligand_code, fold_success/pocket_success/
    pose_success/rfd3_success/complexa_success (+ _mpnn variants)."""
    records = json.loads(results_json.read_text())
    missing_key = [r["sample_id"] for r in records if success_key not in r]
    if missing_key:
        raise SystemExit(f"--success-key {success_key!r} not found on {len(missing_key)} record(s) "
                          f"(e.g. {missing_key[0]}) -- check the key name against results.json.")
    return {r["sample_id"] for r in records if r.get(success_key)}


def _restrict_to_successes(mat: np.ndarray, names: list[str], successful_names: set[str] | None):
    """successful_names=None -> no filtering (cluster among ALL samples).
    Otherwise, subset mat/names down to only the names present in successful_names --
    a cluster made entirely of failed designs isn't a "unique success" and
    shouldn't be counted/saved as one."""
    if successful_names is None:
        return mat, names
    keep = [i for i, n in enumerate(names) if n in successful_names]
    if not keep:
        return np.zeros((0, 0)), []
    sub = mat[np.ix_(keep, keep)]
    return sub, [names[i] for i in keep]


def cluster_summary(tmscore_dir: Path, codes: list[str], thresholds: list[float],
                     successful_names_by_code: dict[str, set[str]] | None = None) -> str:
    filtered = successful_names_by_code is not None
    header = f"{'ligand':<6} {'n':>4}"
    for t in thresholds:
        header += f"  cc@{t:<4}  greedy@{t:<4}"
    lines = [
        "Cluster counts from pairwise TM-score matrices"
        + (" -- SUCCESSFUL SAMPLES ONLY (matches RFD3/Complexa's '# unique successes')" if filtered else ""),
        f"tmscore_dir: {tmscore_dir}",
        "cc = connected components (chaining-prone, see module docstring)",
        "greedy = greedy set-cover a la MMseqs2/CD-HIT (recommended '# unique')",
        "",
        header,
    ]
    for code in codes:
        mat = np.load(tmscore_dir / f"tmscore_matrix_{code}.npy")
        names = json.loads((tmscore_dir / f"tmscore_names_{code}.json").read_text())
        succ = successful_names_by_code.get(code) if filtered else None
        mat, names = _restrict_to_successes(mat, names, succ)
        n = len(names)
        row = f"{code:<6} {n:>4}"
        if n == 0:
            row += "".join(f"  {'-':>7}  {'-':>10}" for _ in thresholds)
            lines.append(row)
            continue
        mat_filled = np.where(np.isfinite(mat), mat, 0.0)  # missing pairs treated as "not similar"
        for t in thresholds:
            adj = mat_filled >= t
            np.fill_diagonal(adj, False)
            cc = connected_components(adj)
            gc = len(greedy_set_cover(adj))
            row += f"  {cc:>7}  {gc:>10}"
        lines.append(row)
    return "\n".join(lines) + "\n"


def save_cluster_representatives(tmscore_dir: Path, samples_dir: Path, out_dir: Path,
                                  codes: list[str], threshold: float,
                                  successful_names_by_code: dict[str, set[str]] | None = None) -> None:
    """For each ligand, greedy-set-cover cluster at `threshold` and copy one
    representative PDB per cluster into out_dir/<code>/ -- for visualizing
    "how many genuinely distinct solutions did we generate" rather than
    scrolling through near-duplicate samples. Pass successful_names_by_code to
    cluster only among successes -- a cluster of only-failures isn't a
    "novel"/"unique" result worth a representative."""
    import shutil
    filtered = successful_names_by_code is not None

    for code in codes:
        mat = np.load(tmscore_dir / f"tmscore_matrix_{code}.npy")
        names = json.loads((tmscore_dir / f"tmscore_names_{code}.json").read_text())
        succ = successful_names_by_code.get(code) if filtered else None
        mat, names = _restrict_to_successes(mat, names, succ)
        if not names:
            print(f"{code}: 0 successful samples -- nothing to cluster/save")
            continue
        mat_filled = np.where(np.isfinite(mat), mat, 0.0)
        adj = mat_filled >= threshold
        np.fill_diagonal(adj, False)
        centers = greedy_set_cover(adj)

        code_out = out_dir / code
        code_out.mkdir(parents=True, exist_ok=True)
        saved, missing = 0, []
        for idx in centers:
            name = names[idx]
            src = samples_dir / f"{name}.pdb"
            if src.exists():
                shutil.copy(src, code_out / f"{name}.pdb")
                saved += 1
            else:
                missing.append(name)
        tag = " (successful samples only)" if filtered else ""
        print(f"{code}: {saved}/{len(centers)} cluster representatives{tag} (@TM>={threshold}) -> {code_out}"
              + (f"  [{len(missing)} source PDB(s) not found, e.g. {missing[0]}]" if missing else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tmscore-dir", type=Path, required=True,
                   help="Output dir from pairwise_tmscore.py (contains tmscore_matrix_<CODE>.npy).")
    p.add_argument("--codes", nargs="+", default=None,
                   help="Ligand codes to summarize. Default: auto-discover from tmscore_matrix_*.npy filenames.")
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.4, 0.5, 0.6])
    p.add_argument("--save-representatives", action="store_true",
                   help="Also copy one representative PDB per greedy-set-cover cluster "
                        "(at --rep-threshold) into --tmscore-dir/cluster_representatives/<code>/, "
                        "for visualizing '# genuinely distinct designs' rather than scrolling "
                        "through near-duplicates. Requires --samples-dir.")
    p.add_argument("--samples-dir", type=Path, default=None,
                   help="Directory of the original sample PDBs (same one passed to "
                        "pairwise_tmscore.py's --samples-dir). Required with --save-representatives.")
    p.add_argument("--rep-threshold", type=float, default=0.5,
                   help="TM-score clustering threshold for --save-representatives (default: 0.5, "
                        "Zhang & Skolnick's 'same fold' cutoff / GTalign's own --cls-threshold default).")
    p.add_argument("--representatives-out", type=Path, default=None,
                   help="Where to write representative PDBs (default: --tmscore-dir/cluster_representatives).")
    p.add_argument("--results-json", type=Path, default=None,
                   help="eval_pallatom_ligand_cond.py's results.json. If given, clustering (both "
                        "cluster_summary.txt and --save-representatives) is restricted to samples "
                        "where --success-key is true -- a cluster made entirely of failed designs "
                        "isn't a 'unique success' and shouldn't be counted as one (matches how "
                        "RFD3/Complexa's own '# unique successes' is actually defined). Without "
                        "this, clustering runs over ALL samples regardless of pass/fail.")
    p.add_argument("--success-key", default="fold_success",
                   help="Which results.json field counts as 'success' when --results-json is given. "
                        "One of fold_success/pocket_success/pose_success/rfd3_success/complexa_success "
                        "(or the _mpnn variants). Default: fold_success.")
    args = p.parse_args()

    if args.save_representatives and args.samples_dir is None:
        raise SystemExit("--save-representatives requires --samples-dir")

    codes = args.codes or sorted(
        f.stem.replace("tmscore_matrix_", "") for f in args.tmscore_dir.glob("tmscore_matrix_*.npy")
    )
    if not codes:
        raise SystemExit(f"no tmscore_matrix_*.npy found in {args.tmscore_dir}")

    successful_names_by_code = None
    if args.results_json is not None:
        all_successful = load_successful_names(args.results_json, args.success_key)
        # split the flat set into per-code sets by cross-referencing tmscore_names_<code>.json
        successful_names_by_code = {}
        for code in codes:
            names = json.loads((args.tmscore_dir / f"tmscore_names_{code}.json").read_text())
            successful_names_by_code[code] = {n for n in names if n in all_successful}

    div_text = diversity_summary(args.tmscore_dir, codes)
    (args.tmscore_dir / "diversity_summary.txt").write_text(div_text)
    print(div_text)

    clu_text = cluster_summary(args.tmscore_dir, codes, args.thresholds, successful_names_by_code)
    (args.tmscore_dir / "cluster_summary.txt").write_text(clu_text)
    print(clu_text)

    if args.save_representatives:
        rep_out = args.representatives_out or (args.tmscore_dir / "cluster_representatives")
        save_cluster_representatives(args.tmscore_dir, args.samples_dir, rep_out, codes,
                                      args.rep_threshold, successful_names_by_code)


if __name__ == "__main__":
    main()
