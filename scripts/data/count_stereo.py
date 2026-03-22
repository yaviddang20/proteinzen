"""
Count stereoisomers in the raw GEOM drugs dataset (before filtering),
distinguishing enantiomers from diastereomers.

Reads:
  datadir / summary_{dataset}.json
  datadir / filter_errors_{dataset}.json  (for within-entry inconsistencies)

Writes:
  outdir / stereo_counts_{dataset}.json

Output JSON structure:
  {
    "<no_stereo_canon_smi>": {
      "group_type": "enantiomers_only" | "diastereomers_only" | "mixed",
      "isomers": [
        ["<isomer_canon_smi>", "<entry_smi>", <n_conformers>, "<enantiomer_smi_if_in_group_else_null>"],
        ...
      ]
    },
    ...
  }

Enantiomer detection:
  Two stereoisomers are enantiomers iff inverting ALL tetrahedral chiral centers
  (R<->S) of one yields the other. E/Z double-bond geometry is unchanged under
  mirror reflection, so E/Z-only pairs are always diastereomers.
  Meso compounds (enantiomer == self) are treated as having no enantiomer partner.

Only groups with >=2 distinct stereo SMILES are included.

Stats printed to stdout:
  - Within-entry stereo inconsistency (from filter_errors)
  - Groups with enantiomers only / diastereomers only / mixed
  - Total enantiomeric pairs and diastereomeric pairs
  - Total entries and unique isomers involved
"""

import argparse
import json
import multiprocessing
from collections import defaultdict
from functools import partial
from pathlib import Path
from itertools import combinations

from rdkit import Chem, RDLogger
from p_tqdm import p_umap
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def canonicalize(smi: str, stereo: bool) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=stereo, canonical=True)


def get_enantiomer_smiles(canon_smi: str) -> str | None:
    """
    Return canonical SMILES of the enantiomer by inverting all tetrahedral
    chiral centers. Returns None if:
      - no chiral centers (E/Z-only isomers are diastereomers, not enantiomers)
      - molecule is meso (enantiomer == self)
    """
    mol = Chem.MolFromSmiles(canon_smi)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    has_chiral = False
    for atom in rw.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)
            has_chiral = True
        elif tag == Chem.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
            has_chiral = True
    if not has_chiral:
        return None
    result = Chem.MolToSmiles(rw.GetMol(), isomericSmiles=True, canonical=True)
    return result if result != canon_smi else None  # meso: enantiomer == self


def classify_group(
    unique_isomers: set[str],
) -> tuple[str, dict[str, str | None]]:
    """
    Classify a group of unique isomer SMILES.

    Returns:
      group_type: "enantiomers_only" | "diastereomers_only" | "mixed"
      enant_map: {isomer_smi -> enantiomer_smi_if_in_group_else_None}
    """
    enant_map: dict[str, str | None] = {}
    for smi in unique_isomers:
        enant = get_enantiomer_smiles(smi)
        enant_map[smi] = enant if (enant and enant in unique_isomers) else None

    has_enant_pair = any(v is not None for v in enant_map.values())
    has_diast_pair = any(v is None for v in enant_map.values())

    if has_enant_pair and not has_diast_pair:
        group_type = "enantiomers_only"
    elif has_diast_pair and not has_enant_pair:
        group_type = "diastereomers_only"
    else:
        group_type = "mixed"

    return group_type, enant_map


def process_entry(item: tuple[str, dict]) -> tuple[str, str, str, int] | None:
    """Returns (no_stereo_smi, isomer_smi, entry_smi, n_conformers) or None."""
    entry_smi, data = item
    no_stereo = canonicalize(entry_smi, stereo=False)
    isomer = canonicalize(entry_smi, stereo=True)
    if no_stereo is None or isomer is None:
        return None
    return (no_stereo, isomer, entry_smi, data.get("totalconfs", 0))


def count_stereo(args) -> None:
    datadir = Path(args.datadir)
    if args.outdir is None:
        outdir = Path(datadir)  
    else:
        outdir  = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = datadir / f"summary_{args.dataset}.json"
    print(f"Loading {summary_path} ...")
    with open(summary_path) as f:
        raw = json.load(f)

    metadata = [(smi, data) for smi, data in raw.items() if len(smi) > 1]
    print(f"Total entries: {len(metadata)}")

    max_processes = multiprocessing.cpu_count()
    num_processes = max(1, min(args.num_processes, max_processes, len(metadata)))

    print("Canonicalizing SMILES ...")
    if num_processes > 1:
        results = p_umap(process_entry, metadata, num_cpus=num_processes)
    else:
        results = [process_entry(item) for item in tqdm(metadata)]

    # Group by no-stereo SMILES: {no_stereo -> [(isomer, entry_smi, n_conf), ...]}
    groups: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    n_failed = 0
    for r in results:
        if r is None:
            n_failed += 1
            continue
        no_stereo, isomer, entry_smi, n_conf = r
        groups[no_stereo].append((isomer, entry_smi, n_conf))

    if n_failed:
        print(f"  Failed to parse {n_failed} entries.")

    # --- Build cross-entry groups and classify ---
    out: dict[str, dict] = {}
    n_enant_only = n_diast_only = n_mixed = 0
    n_enant_pairs = n_diast_pairs = 0

    for no_stereo, members in groups.items():
        unique_isomers = {m[0] for m in members}
        if len(unique_isomers) < 2:
            continue

        group_type, enant_map = classify_group(unique_isomers)

        # Count pairs
        for smi_a, smi_b in combinations(unique_isomers, 2):
            if enant_map.get(smi_a) == smi_b:
                n_enant_pairs += 1
            else:
                n_diast_pairs += 1

        if group_type == "enantiomers_only":
            n_enant_only += 1
        elif group_type == "diastereomers_only":
            n_diast_only += 1
        else:
            n_mixed += 1

        out[no_stereo] = {
            "group_type": group_type,
            "isomers": [
                [isomer, entry_smi, n_conf, enant_map.get(isomer)]
                for isomer, entry_smi, n_conf in members
            ],
        }

    # --- Within-entry stereo inconsistency ---
    errors_path = datadir / f"filter_errors_{args.dataset}.json"
    n_within_entry = 0
    if errors_path.exists():
        with open(errors_path) as f:
            errors = json.load(f)
        n_within_entry = len(errors.get("inconsistent_stereo", []))
    else:
        print(f"  Note: {errors_path} not found; within-entry count unavailable.")

    # --- Stats ---
    total_groups = len(out)
    total_entries = sum(len(v["isomers"]) for v in out.values())
    total_unique_isomers = sum(len({t[0] for t in v["isomers"]}) for v in out.values())

    print()
    print("=== Stereoisomer Counts ===")
    print(f"  Within-entry stereo inconsistency (conformers disagree):  {n_within_entry}")
    print()
    print(f"  Cross-entry stereoisomer groups (total):                   {total_groups}")
    print(f"    Enantiomers only (all pairs are mirror images):           {n_enant_only}")
    print(f"    Diastereomers only (no enantiomeric pairs):               {n_diast_only}")
    print(f"    Mixed (both enantiomeric and diastereomeric pairs):       {n_mixed}")
    print()
    print(f"  Unique isomer SMILES across all cross-entry groups:        {total_unique_isomers}")
    print(f"  Total entries involved in cross-entry groups:              {total_entries}")
    print(f"  Enantiomeric pairs (unique isomer pairs):                  {n_enant_pairs}")
    print(f"  Diastereomeric pairs (unique isomer pairs):                {n_diast_pairs}")
    print()

    out_path = outdir / f"stereo_counts_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    
    print(f"Wrote {total_groups} groups to {out_path}")

    txt_out_path = outdir / f"stereo_counts_{args.dataset}.txt"
    with open(txt_out_path, "w") as f:
        f.write(f"Total groups: {total_groups}\n")
        f.write(f"Enantiomers only: {n_enant_only}\n")
        f.write(f"Diastereomers only: {n_diast_only}\n")
        f.write(f"Mixed: {n_mixed}\n")
        f.write(f"Enantiomeric pairs: {n_enant_pairs}\n")
        f.write(f"Diastereomeric pairs: {n_diast_pairs}\n")
        f.write(f"Unique isomer SMILES: {total_unique_isomers}\n")
        f.write(f"Total entries: {total_entries}\n")
        f.write(f"Within-entry stereo inconsistency: {n_within_entry}\n")

    print(f"Wrote {txt_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count and classify stereoisomers in GEOM dataset."
    )
    parser.add_argument("--datadir", type=Path, required=True,
                        help="Directory containing summary_{dataset}.json.")
    parser.add_argument("--dataset", type=str, default="drugs", choices=["qm9", "drugs"])
    parser.add_argument("--outdir", type=Path, help="Directory to write stereo_counts_{dataset}.json.")
    parser.add_argument("--num-processes", type=int, default=multiprocessing.cpu_count() // 2)
    args = parser.parse_args()
    count_stereo(args)
