"""
Backfill pose_resnum_map (Rosetta pose index -> [chain, auth_resnum]) into
the ensemble_stats.json of every already-generated sidechain-diversity
ensemble. Needed because packable_resnums/first_shell_resnums/etc in
ensemble_stats.json are Rosetta's internal sequential pose indices, which
only equal the author residue numbering (used everywhere else -- training
data, chain_map.json) when there are no missing residues or multi-chain
gaps -- not safe to assume in general. Re-derives this from the same
staged input PDB used during generation, using pose.pdb_info(), so no
sidechain sampling is redone.

Usage:
    python backfill_pose_resnum_maps.py \
        --ensembles-root plinder_pocket_processed_sc_ensembles \
        --staging-dir plinder_pocket_sc_ensemble_staging_pdbs \
        --custom-ligand-sdf-dir sc_diversity_custom_ligand_sdfs \
        --num-workers 16
"""
import argparse
import json
import multiprocessing
from pathlib import Path

_ARGS = None
_CUSTOM_RTS = None


def _worker_init(args):
    global _ARGS, _CUSTOM_RTS
    _ARGS = args
    import pyrosetta
    pyrosetta.init("-mute all")
    if args.custom_ligand_sdf_dir is not None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from sidechain_diversity_batch import _build_custom_residue_type_set
        _CUSTOM_RTS = _build_custom_residue_type_set(args.custom_ligand_sdf_dir)


def _process_one(system_id: str):
    args = _ARGS
    stats_path = Path(args.ensembles_root) / system_id / "ensemble_stats.json"
    pdb_path = Path(args.staging_dir) / f"{system_id}.pdb"
    if not stats_path.exists() or not pdb_path.exists():
        return system_id, "missing stats or staged pdb"
    try:
        import pyrosetta
        if _CUSTOM_RTS is not None:
            pose = pyrosetta.rosetta.core.import_pose.pose_from_file(_CUSTOM_RTS, str(pdb_path))
        else:
            pose = pyrosetta.pose_from_pdb(str(pdb_path))
        pdb_info = pose.pdb_info()
        pose_resnum_map = {
            i: [pdb_info.chain(i), pdb_info.number(i)]
            for i in range(1, pose.total_residue() + 1)
        }
        stats = json.loads(stats_path.read_text())
        stats["pose_resnum_map"] = pose_resnum_map
        stats_path.write_text(json.dumps(stats))
        return system_id, None
    except Exception as e:
        return system_id, f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensembles-root", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--custom-ligand-sdf-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    system_ids = [
        d.name for d in args.ensembles_root.iterdir()
        if d.is_dir() and (d / "ensemble_stats.json").exists()
    ]
    print(f"{len(system_ids)} systems with an ensemble")

    n_ok, failed = 0, []
    with multiprocessing.Pool(processes=args.num_workers, initializer=_worker_init, initargs=(args,)) as pool:
        for system_id, err in pool.imap_unordered(_process_one, system_ids, chunksize=8):
            if err is None:
                n_ok += 1
            else:
                failed.append((system_id, err))

    print(f"Backfilled pose_resnum_map for {n_ok}/{len(system_ids)} systems, {len(failed)} failed")
    if failed:
        (args.ensembles_root / "_pose_resnum_map_failures.json").write_text(json.dumps(failed, indent=2))
        print(f"Failure details -> {args.ensembles_root / '_pose_resnum_map_failures.json'}")
