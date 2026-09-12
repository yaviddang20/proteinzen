"""
Build the final flag manifests for the sidechain-diversity ensemble dataset:
- custom_ligand_flagged.json: system_id -> ligand_ccd_code for every system
  whose ensemble was generated using an on-the-fly-registered Rosetta residue
  type (built from CCD/RDKit chemistry via convert_to_ResidueType), rather
  than Rosetta's own curated fa_standard params library. These are less
  vetted than standard entries and worth being able to filter/spot-check.
- known_gaps.json: system_id -> ligand_ccd_code (where resolvable) for
  systems that still failed after the custom-ligand fix, i.e. structures
  sidechain_diversity.py cannot currently produce an ensemble for at all.

Usage:
    python finalize_sc_diversity_manifest.py \
        --out-root plinder_pocket_processed_sc_ensembles \
        --original-failed-manifest sc_diversity_custom_ligand_manifest.json \
        --still-failed-ids sc_diversity_still_failed_ids.txt
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--original-failed-manifest", type=Path, required=True)
    parser.add_argument("--still-failed-ids", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.original_failed_manifest.read_text())
    id_to_code = data["id_to_ligand_ccd_code"]

    still_failed = set(l.strip() for l in args.still_failed_ids.read_text().splitlines() if l.strip())

    flagged = {}
    for sid, code in id_to_code.items():
        if sid in still_failed:
            continue
        if (args.out_root / sid / "ensemble_stats.json").exists():
            flagged[sid] = code

    known_gaps = {sid: id_to_code.get(sid) for sid in still_failed}

    (args.out_root / "_custom_ligand_flagged.json").write_text(json.dumps(flagged, indent=2, sort_keys=True))
    (args.out_root / "_known_gaps.json").write_text(json.dumps(known_gaps, indent=2, sort_keys=True))

    print(f"{len(flagged)} systems flagged as using custom on-the-fly ligand params")
    print(f"{len(known_gaps)} systems still have no ensemble (known gap)")


if __name__ == "__main__":
    main()
