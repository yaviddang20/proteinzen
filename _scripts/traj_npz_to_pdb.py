"""Convert a _traj.npz file (saved by run_epoch_sample) to multi-model PDB(s).

Usage:
    python traj_npz_to_pdb.py path/to/sample_traj.npz [--traj prot|clean|both] [--out-dir DIR]

Outputs:
    <stem>_traj_noise.pdb  and/or  <stem>_traj_clean.pdb
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def npz_to_pdb(npz_path: Path, which: str, out_dir: Path):
    from proteinzen.runtime.lmod import _build_all_atom_records, _write_model_block

    d = np.load(npz_path)
    rigids_mask = d["rigids_mask"]
    ref_elements = d["ref_elements"]
    is_atom_mask = d["is_atom_mask"]
    sc_idx       = d["sc_idx"]
    to_tok       = d["to_tok"]
    seq_idx      = d["seq_idx"]
    res_type     = d["res_type"]
    asym_id      = d["asym_id"]
    res_idx      = d["res_idx"]

    stem = npz_path.stem.replace("_traj", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    trajs = []
    if which in ("prot", "both"):
        trajs.append((d["prot_traj"], out_dir / f"{stem}_traj_noise.pdb"))
    if which in ("clean", "both"):
        trajs.append((d["clean_traj"], out_dir / f"{stem}_traj_clean.pdb"))

    for traj_frames, out_path in trajs:
        with open(out_path, "w") as f:
            for step_idx, frame in enumerate(traj_frames):
                records = _build_all_atom_records(
                    frame, rigids_mask, ref_elements, is_atom_mask,
                    sc_idx, to_tok, seq_idx, res_type, asym_id,
                    token_residue_idx=res_idx,
                )
                _write_model_block(f, records, step_idx + 1)
            f.write("END\n")
        print(f"Written: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path, help="Path to _traj.npz file")
    parser.add_argument("--traj", choices=["prot", "clean", "both"], default="both",
                        help="Which trajectory to export (default: both)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: same directory as input npz)")
    args = parser.parse_args()

    if not args.npz.exists():
        sys.exit(f"File not found: {args.npz}")

    out_dir = args.out_dir or args.npz.parent
    npz_to_pdb(args.npz, args.traj, out_dir)


if __name__ == "__main__":
    main()
