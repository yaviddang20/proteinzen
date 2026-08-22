"""
Convert plinder_pocket_placer processed .npz structures to PDB files, so you can
open them in PyMOL/ChimeraX and sanity-check the multi-chain fusion.

Also finds likely-fused (originally 2-3 protein chain) systems using a heuristic:
after fusion, the merged protein chain's auth_indices (original PDB residue
numbering) is a straight concatenation of each constituent chain's own numbering.
A genuinely single-chain system's auth numbering increases ~monotonically; a
fused system typically shows a "reset" partway through, where one original
chain's numbering ends and the next chain's begins. This is a heuristic, not
exact — good enough to find candidates to eyeball, not a ground-truth label.

Runs from any directory (paths are all passed in via --data-dir / --out-dir; the
proteinzen package import resolves via your installed environment, not CWD).

Usage:
    # Find candidate fused systems in a split, print them
    python inspect_fused_pdbs.py find --data-dir /path/to/plinder_pocket_placer_processed/train

    # Find candidates AND convert the first N to PDB in one go
    python inspect_fused_pdbs.py find --data-dir ... --convert --out-dir ./fused_pdbs --limit 5

    # Convert specific system IDs you already know about
    python inspect_fused_pdbs.py convert --data-dir ... \
        --system-ids 1abc__1__A_B__L 2xyz__1__C_D_E__L --out-dir ./fused_pdbs
"""
import argparse
import json
from pathlib import Path

from proteinzen.boltz.data import const
from proteinzen.boltz.data.types import Structure
from proteinzen.data.write.pdb import to_pdb

PROTEIN_ID = const.chain_type_ids["PROTEIN"]


def system_mid(system_id: str) -> str:
    return system_id[1:3]


def find_fused_candidates(data_dir: Path, limit: int = None) -> list:
    auth_dir = data_dir / "auth_maps"
    candidates = []
    for auth_path in sorted(auth_dir.rglob("*.json")):
        system_id = auth_path.stem
        with open(auth_path) as f:
            auth_map = json.load(f)
        if not auth_map:
            continue
        protein_entry = auth_map[0]  # merged protein chain is always index 0 after fusion
        if protein_entry.get("mol_type") != PROTEIN_ID:
            continue
        nums = []
        for x in protein_entry.get("auth_indices", []):
            if x is None:
                nums.append(None)
                continue
            digits = ''.join(c for c in str(x) if c.isdigit() or c == '-')
            nums.append(int(digits) if digits not in ("", "-") else None)

        resets = 0
        prev = None
        for n in nums:
            if n is None:
                continue
            if prev is not None and n < prev - 5:  # allow small numbering noise
                resets += 1
            prev = n

        if resets > 0:
            candidates.append((system_id, resets))
        if limit and len(candidates) >= limit:
            break
    return candidates


def convert_to_pdb(data_dir: Path, system_id: str, out_dir: Path) -> Path:
    mid = system_mid(system_id)
    npz_path = data_dir / "structures" / mid / f"{system_id}.npz"
    structure = Structure.load(npz_path)
    pdb_str = to_pdb(structure, rename_chains=True)
    out_path = out_dir / f"{system_id}.pdb"
    out_path.write_text(pdb_str)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_find = sub.add_parser("find")
    p_find.add_argument("--data-dir", type=Path, required=True)
    p_find.add_argument("--limit", type=int, default=20)
    p_find.add_argument("--convert", action="store_true")
    p_find.add_argument("--out-dir", type=Path, default=Path("./fused_pdbs"))

    p_conv = sub.add_parser("convert")
    p_conv.add_argument("--data-dir", type=Path, required=True)
    p_conv.add_argument("--system-ids", nargs="+", required=True)
    p_conv.add_argument("--out-dir", type=Path, default=Path("./fused_pdbs"))

    args = parser.parse_args()

    if args.cmd == "find":
        candidates = find_fused_candidates(args.data_dir, limit=args.limit)
        print(f"Found {len(candidates)} likely-fused systems (auth numbering resets):")
        for sid, resets in candidates:
            print(f"  {sid}  (resets={resets})")
        if args.convert:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            for sid, _ in candidates:
                path = convert_to_pdb(args.data_dir, sid, args.out_dir)
                print(f"Wrote {path}")
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for sid in args.system_ids:
            path = convert_to_pdb(args.data_dir, sid, args.out_dir)
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
