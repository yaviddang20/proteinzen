#!/usr/bin/env python
"""Pairwise TM-score matrices via GTalign (GPU-accelerated), for measuring
structural diversity among generated samples.

Two modes:
  - whole-dir (default): all-vs-all over every .pdb in --samples-dir
  - --group-by-ligand-code: split by ligand CCD code (make_ligand_cond_yaml.py /
    sample.py's naming convention, same as eval_pallatom_ligand_cond.py's
    _ligand_code_for) and run a separate all-vs-all within each group

Requires GTalign installed separately -- there is no official container;
the easiest route is conda/micromamba:
    micromamba install -n <env> -c minmarg gtalign_gpu
See https://github.com/minmarg/gtalign_alpha for source/build-from-scratch options.

JSON schema verified directly against real GTalign 0.19.00 output (--outfmt=1),
not guessed:
    {"gtalign_search": {
        "query": {"description": "<path>/<name>.pdb Chn:A", "length": N},
        "search_results": [
            {"hit_record": {
                "reference_description": "<path>/<name>.pdb Chn:A",
                "alignment": {"tmscore_query": F, "tmscore_refn": F, "rmsd": F, ...}
            }},
            ...
        ]
    }}
tmscore_query is normalized by the query's length, tmscore_refn by the
reference's length -- identical when both structures are the same length
(true for fixed-length-128 Pallatom-style samples), which is why the default
here is max(tmscore_query, tmscore_refn), matching GTalign's own --sort=0
("greater TM-score of the two") convention. Pass --tm-field query|refn to
pin one explicitly.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_pallatom_ligand_cond import _ligand_code_for  # noqa: E402


def _name_from_description(desc: str) -> str:
    """'<path>/<name>.pdb Chn:A' -> '<name>' (matches Path(pdb).stem)."""
    desc = desc.split(" Chn:")[0]
    return Path(desc).stem


def run_gtalign(query_dir: Path, out_dir: Path, gtalign_bin: str, nhits: int, ref_dir: Path = None):
    """ref_dir=None (default): self-vs-self all-pairwise (--rfs=query_dir). Pass a
    different ref_dir for a cross-set comparison (e.g. generated designs' pockets
    vs a fixed set of real reference pockets) -- see compare_generated_to_real_pockets.py."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        gtalign_bin, "-v",
        f"--qrs={query_dir}", f"--rfs={ref_dir if ref_dir is not None else query_dir}",
        "-s", "0",                # report ALL pairs, not just TM-score>=0.5 (the default)
        f"--nhits={nhits}",       # must cover every structure or results get truncated
        "--outfmt=1",             # JSON
        "-o", str(out_dir),
    ]
    subprocess.run(cmd, check=True)


def stage_subset(paths: list[Path], staging_dir: Path):
    """GTalign takes a directory as input -- symlink just the wanted files into
    a fresh staging dir so unrelated samples never enter the comparison."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    for p in paths:
        (staging_dir / p.name).symlink_to(p.resolve())


def parse_results(out_dir: Path, inspect: bool, tm_field: str | None):
    """Parse GTalign's per-query JSON result files (--outfmt=1) into
    {(query, ref): tm_score}. Schema verified directly against real output --
    see module docstring."""
    pairwise: dict[tuple[str, str], float] = {}
    inspected = False
    for f in sorted(out_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        search = data.get("gtalign_search")
        if not search:
            continue

        if inspect and not inspected:
            print(f"--- inspecting {f} ---")
            print(json.dumps(data, indent=2)[:3000])
            inspected = True

        query_name = _name_from_description(search["query"]["description"])
        for entry in search.get("search_results", []):
            hit = entry.get("hit_record", {})
            aln = hit.get("alignment", {})
            ref_name = _name_from_description(hit["reference_description"])
            tq, tr = aln.get("tmscore_query"), aln.get("tmscore_refn")
            if tm_field == "query":
                val = tq
            elif tm_field == "refn":
                val = tr
            else:
                vals = [v for v in (tq, tr) if isinstance(v, (int, float))]
                val = max(vals) if vals else None
            if val is not None:
                pairwise[(query_name, ref_name)] = float(val)

    return pairwise


def to_matrix(pairwise: dict, names: list[str], symmetrize: bool = True) -> np.ndarray:
    """TM-score is only exactly symmetric in theory for equal-length structures --
    in practice the heuristic superposition search can find a different best
    alignment depending on direction, so q->r and r->q can differ slightly even
    then (confirmed empirically, e.g. 0.440 vs 0.503 for a real equal-length
    pair). symmetrize=True takes max(mat[i,j], mat[j,i]), matching GTalign's own
    "greater of the two" (--sort=0) convention -- the right choice for feeding
    this into clustering downstream."""
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)
    for (q, r), v in pairwise.items():
        if q in idx and r in idx:
            mat[idx[q], idx[r]] = v
    if symmetrize:
        with np.errstate(invalid="ignore"):
            mat = np.fmax(mat, mat.T)
    return mat


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--group-by-ligand-code", action="store_true",
                   help="Split by ligand CCD code and run a separate all-vs-all within each "
                        "group, instead of one pass mixing every sample together.")
    p.add_argument("--gtalign-bin", default="gtalign")
    p.add_argument("--nhits", type=int, default=None,
                   help="Default: number of structures in the group, so nothing truncates.")
    p.add_argument("--tm-field", choices=["query", "refn"], default=None,
                   help="Pin TM-score to query-length- or reference-length-normalized. "
                        "Default: max of the two (matches GTalign's own --sort=0).")
    p.add_argument("--inspect", action="store_true",
                   help="Print the raw JSON of the first parsed result file, for sanity-checking.")
    args = p.parse_args()

    all_pdbs = sorted(args.samples_dir.glob("*.pdb"))
    if not all_pdbs:
        raise SystemExit(f"no .pdb files found in {args.samples_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = {"all": all_pdbs}
    if args.group_by_ligand_code:
        groups = {}
        for pdb in all_pdbs:
            groups.setdefault(_ligand_code_for(pdb), []).append(pdb)

    for group_name, pdbs in groups.items():
        names = [pdb.stem for pdb in pdbs]
        nhits = args.nhits or len(pdbs)
        staging = args.out_dir / f"_staged_{group_name}"
        gtalign_out = args.out_dir / f"gtalign_raw_{group_name}"
        stage_subset(pdbs, staging)
        print(f"[{group_name}] running gtalign on {len(pdbs)} structures...")
        run_gtalign(staging, gtalign_out, args.gtalign_bin, nhits)

        pairwise = parse_results(gtalign_out, inspect=args.inspect, tm_field=args.tm_field)
        if not pairwise:
            print(f"[{group_name}] WARNING: parsed 0 pairs -- check --inspect output above, "
                  f"the JSON schema likely doesn't match what this parser expects yet.")
            continue
        mat = to_matrix(pairwise, names)

        np.save(args.out_dir / f"tmscore_matrix_{group_name}.npy", mat)
        (args.out_dir / f"tmscore_names_{group_name}.json").write_text(json.dumps(names, indent=2))

        off_diag = ~np.eye(len(names), dtype=bool)
        finite = mat[off_diag & np.isfinite(mat)]
        if finite.size:
            print(f"[{group_name}] n={len(names)}  pairs={finite.size}  "
                  f"mean TM-score={finite.mean():.3f}  min={finite.min():.3f}  max={finite.max():.3f}")

    print(f"\nDone. Matrices + name lists written to {args.out_dir}")
    print("(_staged_*/gtalign_raw_* subdirs left in place for debugging -- "
          "safe to delete once you trust the parsed matrices)")


if __name__ == "__main__":
    main()
