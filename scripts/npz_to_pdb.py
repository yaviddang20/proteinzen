"""Convert a plinder_processed .npz file to a PDB file."""
import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem
from proteinzen.boltz.data.types import Structure
from proteinzen.boltz.data import const

CHAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def write_pdb(structure, rename_chains=True):
    pt = Chem.GetPeriodicTable()
    lines = []
    serial = 1
    atom_serial = {}  # full atom array index -> pdb serial number

    for ci, chain in enumerate(structure.chains[structure.mask]):
        tag = CHAIN_ALPHABET[ci] if rename_chains else chain["name"]
        is_lig = chain["mol_type"] == const.chain_type_ids["NONPOLYMER"]
        record = "HETATM" if is_lig else "ATOM"

        for res in structure.residues[chain["res_idx"]: chain["res_idx"] + chain["res_num"]]:
            res_name = "LIG" if is_lig else str(res["name"][:3])
            res_seq = res["res_idx"] + 1
            for li, atom in enumerate(structure.atoms[res["atom_idx"]: res["atom_idx"] + res["atom_num"]]):
                gi = res["atom_idx"] + li
                if not atom["is_present"]:
                    continue
                aname = "".join(chr(c + 32) for c in atom["name"] if c != 0)
                aname = aname if len(aname) == 4 else " " + aname
                elem = pt.GetElementSymbol(int(atom["element"])).upper()
                x, y, z = atom["coords"]
                lines.append(
                    f"{record:<6}{serial:>5} {aname:<4} {res_name:>3} {tag}{res_seq:>4}    "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {elem:>2}  "
                )
                atom_serial[gi] = serial
                serial += 1

    for bond in structure.bonds:
        a1, a2 = int(bond["atom_1"]), int(bond["atom_2"])
        if a1 in atom_serial and a2 in atom_serial:
            lines.append(f"CONECT{atom_serial[a1]:>5}{atom_serial[a2]:>5}")

    lines.append("END")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert plinder_processed npz to PDB")
    parser.add_argument("npz", type=Path, help="Path to .npz file")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output .pdb path (default: same stem as input)")
    parser.add_argument("--no-rename-chains", dest="rename_chains", action="store_false",
                        help="Keep original chain names (default: rename to A, B, C, ...)")
    parser.set_defaults(rename_chains=True)
    args = parser.parse_args()

    structure = Structure.load(args.npz)
    pdb_str = write_pdb(structure, rename_chains=args.rename_chains)

    out_path = args.out or args.npz.with_suffix(".pdb")
    out_path.write_text(pdb_str)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
