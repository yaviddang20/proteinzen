"""
For every system_id in a failed-systems list, look up its ligand_ccd_code in
the Plinder annotation table, then write out one SDF file per distinct ligand
code (source: the RDKit mols in ccd.pkl, same dictionary plinder.py itself
uses). Also writes a system_id -> ligand_ccd_code manifest so downstream
consumers can tell which ensembles were generated using a Rosetta residue
type built on the fly from CCD chemistry, rather than Rosetta's own curated
params library.

Usage:
    python build_custom_ligand_sdfs.py \
        --failed-ids sc_diversity_failed_ids.txt \
        --annotation-table plinder/2024-06/v2/index/annotation_table.parquet \
        --ccd-path ccd.pkl \
        --sdf-out-dir sc_diversity_custom_ligand_sdfs \
        --manifest-out sc_diversity_custom_ligand_manifest.json
"""
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from rdkit import Chem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--failed-ids", type=Path, required=True)
    parser.add_argument("--annotation-table", type=Path, required=True)
    parser.add_argument("--ccd-path", type=Path, required=True)
    parser.add_argument("--sdf-out-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args()

    failed_ids = [l.strip() for l in args.failed_ids.read_text().splitlines() if l.strip()]
    print(f"{len(failed_ids)} failed system_ids")

    ann = pd.read_parquet(args.annotation_table, columns=["system_id", "ligand_ccd_code"])
    ann = ann.drop_duplicates("system_id").set_index("system_id")["ligand_ccd_code"]

    with open(args.ccd_path, "rb") as f:
        ccd = pickle.load(f)

    args.sdf_out_dir.mkdir(parents=True, exist_ok=True)

    id_to_code = {}
    unresolved = []
    written_codes = set()
    for sid in failed_ids:
        code = ann.get(sid)
        if code is None or pd.isna(code):
            unresolved.append((sid, None))
            continue
        code = str(code)
        id_to_code[sid] = code
        if code in written_codes:
            continue
        mol = ccd.get(code)
        if mol is None:
            unresolved.append((sid, code))
            continue
        try:
            mol.SetProp("_Name", code)
            w = Chem.SDWriter(str(args.sdf_out_dir / f"{code}.sdf"))
            w.write(mol)
            w.close()
            written_codes.add(code)
        except Exception as e:
            unresolved.append((sid, code))
            print(f"  failed to write SDF for {code} ({sid}): {e}")

    print(f"Wrote {len(written_codes)} distinct ligand SDFs")
    print(f"{len(id_to_code)} system_ids resolved to a ligand code, {len(unresolved)} unresolved")

    args.manifest_out.write_text(json.dumps({
        "id_to_ligand_ccd_code": id_to_code,
        "unresolved": unresolved,
    }, indent=2))
    print(f"Manifest -> {args.manifest_out}")


if __name__ == "__main__":
    main()
