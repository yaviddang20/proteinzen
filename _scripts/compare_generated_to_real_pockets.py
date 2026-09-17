#!/usr/bin/env python
"""Cross-comparison: generated designs' pockets vs. real known-binder pockets,
via GTalign, per ligand code.

Prerequisites (both already-built as separate steps):
  1. Real reference pockets: extract_pocket_structures.py real-representatives
     -- one pocket-only PDB per Plinder pocket_lddt cluster representative,
     grouped under <ref-pockets-dir>/<CODE>/.
  2. Generated designs' pockets: extract_pocket_structures.py generated-samples
     -- same 6A-cutoff extraction applied to our own samples, grouped under
     <gen-pockets-dir>/<CODE>/ (same _ligand_code_for convention).

This script runs GTalign with --qrs=<generated pockets for CODE>
--rfs=<real pockets for CODE> (a genuine cross-set comparison, not a shared-pool
all-vs-all like pairwise_tmscore.py), producing one rectangular
(generated_sample x real_system) TM-score matrix per ligand code -- for each
generated design, "does its pocket resemble any known real binder pocket for
this ligand."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from pairwise_tmscore import run_gtalign, parse_results  # noqa: E402


def cross_matrix(pairwise: dict, query_names: list[str], ref_names: list[str]) -> np.ndarray:
    qidx = {n: i for i, n in enumerate(query_names)}
    ridx = {n: i for i, n in enumerate(ref_names)}
    mat = np.full((len(query_names), len(ref_names)), np.nan)
    for (q, r), v in pairwise.items():
        if q in qidx and r in ridx:
            mat[qidx[q], ridx[r]] = v
    return mat


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen-pockets-dir", type=Path, required=True,
                   help="Output of `extract_pocket_structures.py generated-samples`.")
    p.add_argument("--ref-pockets-dir", type=Path, required=True,
                   help="Output of `extract_pocket_structures.py real-representatives`.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--gtalign-bin", required=True,
                   help="e.g. /mnt/scratch/user/daviyang/micromamba/envs/gtalign_test/bin/gtalign")
    p.add_argument("--nhits", type=int, default=None,
                   help="Default: number of real reference structures for that ligand.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    codes = sorted(p.name for p in args.gen_pockets_dir.iterdir() if p.is_dir())

    for code in codes:
        gen_dir = args.gen_pockets_dir / code
        ref_dir = args.ref_pockets_dir / code
        if not ref_dir.is_dir():
            print(f"[{code}] no real reference pockets found at {ref_dir} -- skipping")
            continue

        gen_files = sorted(gen_dir.glob("*.pdb"))
        ref_files = sorted(ref_dir.glob("*.pdb"))
        query_names = [f.stem for f in gen_files]
        ref_names = [f.stem for f in ref_files]
        nhits = args.nhits or len(ref_files)

        gtalign_out = args.out_dir / f"gtalign_raw_{code}"
        print(f"[{code}] {len(gen_files)} generated pockets x {len(ref_files)} real reference pockets")
        run_gtalign(gen_dir, gtalign_out, args.gtalign_bin, nhits, ref_dir=ref_dir)

        pairwise = parse_results(gtalign_out, inspect=False, tm_field=None)
        if not pairwise:
            print(f"[{code}] WARNING: parsed 0 pairs")
            continue
        mat = cross_matrix(pairwise, query_names, ref_names)

        np.save(args.out_dir / f"cross_tmscore_{code}.npy", mat)
        (args.out_dir / f"cross_tmscore_{code}_query_names.json").write_text(json.dumps(query_names, indent=2))
        (args.out_dir / f"cross_tmscore_{code}_ref_names.json").write_text(json.dumps(ref_names, indent=2))

        # per generated sample: best match among all known real pockets for this ligand
        best_per_query = np.nanmax(mat, axis=1)
        finite = best_per_query[np.isfinite(best_per_query)]
        if finite.size:
            print(f"[{code}] best-match-to-any-real-pocket TM-score: "
                  f"mean={finite.mean():.3f}  min={finite.min():.3f}  max={finite.max():.3f}")

    print(f"\nDone. Matrices written to {args.out_dir}")


if __name__ == "__main__":
    main()
