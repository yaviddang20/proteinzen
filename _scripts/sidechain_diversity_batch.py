"""
Batch driver for sidechain_diversity.py: takes a list of input PDB paths,
splits it into --num-chunks pieces (one per SLURM array task), and runs
--num-workers parallel processes over this chunk's share via multiprocessing.

Usage (called once per SLURM array task — see sidechain_diversity.sbatch):
    python sidechain_diversity_batch.py \
        --pdb-list all_structures.txt \
        --out-root ./diversity_out \
        --chunk-index $SLURM_ARRAY_TASK_ID --num-chunks 20 \
        --num-workers 96

Pass --custom-ligand-sdf-dir to also register on-the-fly Rosetta residue
types (built from CCD-derived SDFs, one per ligand code) for structures
whose ligand isn't in Rosetta's default fa_standard params library.
"""
import argparse
import multiprocessing
from pathlib import Path

import pyrosetta

from sidechain_diversity import generate_ensemble

_ARGS = None
_CUSTOM_RTS = None


def _build_custom_residue_type_set(sdf_dir: Path):
    from pyrosetta.rosetta.core.chemical import PoseResidueTypeSet, ChemicalManager
    from pyrosetta.rosetta.core.chemical.sdf import SDFParser, convert_to_ResidueType
    from pyrosetta.rosetta.std import ifstream

    base_rts = ChemicalManager.get_instance().residue_type_set("fa_standard")
    pose_rts = PoseResidueTypeSet(base_rts)
    parser = SDFParser()

    n_ok, n_failed = 0, 0
    for sdf_path in sorted(sdf_dir.glob("*.sdf")):
        try:
            strm = ifstream(str(sdf_path))
            mols = parser.parse(strm)
            mrt = convert_to_ResidueType(mols, "fa_standard")
            if mrt is None:
                n_failed += 1
                continue
            pose_rts.add_base_residue_type(mrt)
            n_ok += 1
        except Exception:
            n_failed += 1
    print(f"Custom ligand residue types: {n_ok} registered, {n_failed} failed to convert", flush=True)
    return pose_rts


def _worker_init(args):
    global _ARGS, _CUSTOM_RTS
    _ARGS = args
    pyrosetta.init("-mute all")
    if args.custom_ligand_sdf_dir is not None:
        _CUSTOM_RTS = _build_custom_residue_type_set(args.custom_ligand_sdf_dir)


def _process_one(pdb_path: str):
    args = _ARGS
    pdb_path = Path(pdb_path)
    out_dir = Path(args.out_root) / pdb_path.stem
    if (out_dir / "ensemble_stats.json").exists() and not args.overwrite:
        return pdb_path.name, "skipped (exists)"
    try:
        generate_ensemble(
            pdb_path, out_dir,
            n_samples=args.n_samples, ntrials=args.ntrials,
            temperature=args.temperature, cutoff=args.cutoff,
            max_retries=args.max_retries, min_frac_changed=args.min_frac_changed,
            residue_type_set=_CUSTOM_RTS,
        )
        return pdb_path.name, "ok"
    except Exception as e:
        return pdb_path.name, f"error: {type(e).__name__}: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb-list", type=Path, required=True, help="Text file, one input PDB path per line")
    parser.add_argument("--out-root", type=Path, required=True, help="One subdirectory per input PDB gets created here")
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--ntrials", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--min-frac-changed", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--custom-ligand-sdf-dir", type=Path, default=None,
                         help="Dir of <ligand_ccd_code>.sdf files to register as extra Rosetta residue types")
    args = parser.parse_args()

    all_paths = [line.strip() for line in args.pdb_list.read_text().splitlines() if line.strip()]
    my_paths = all_paths[args.chunk_index::args.num_chunks]
    print(f"Chunk {args.chunk_index}/{args.num_chunks}: {len(my_paths)}/{len(all_paths)} structures, "
          f"{args.num_workers} workers")

    args.out_root.mkdir(parents=True, exist_ok=True)
    with multiprocessing.Pool(processes=args.num_workers, initializer=_worker_init, initargs=(args,)) as pool:
        for name, status in pool.imap_unordered(_process_one, my_paths, chunksize=1):
            print(f"  {name}: {status}", flush=True)
