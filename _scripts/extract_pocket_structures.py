#!/usr/bin/env python
"""Extract pocket-only PDB structures (6A cutoff from any ligand atom -- the
same convention as eval_plinder.py's pocket_fixed_residues, confirmed to also
match Plinder's own `neighboring_residue_threshold=6.0` used to build the
pocket_lddt/pocket_fident clustering columns in the annotation table, see
plinder/data/utils/annotations/ligand_utils.py) for:

  1. Real reference structures: representative systems chosen from Plinder's
     pocket_lddt clustering (one per cluster, see ligand_representatives.json,
     produced separately from the annotation table). These live zipped in
     Plinder's raw systems/<2char_prefix>.zip archives as receptor.pdb +
     ligand_files/<chain>.sdf.
  2. Our own generated samples (already a single PDB with protein ATOM +
     ligand HETATM records, parsed via eval_plinder.parse_pdb_ligand_cond).

Output: one pocket-only PDB per input structure, in per-ligand-code
subdirectories, ready for a GTalign cross-comparison (see pairwise_tmscore.py).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).parent))
from eval_plinder import parse_pdb_ligand_cond  # noqa: E402

CUTOFF = 6.0
SYSTEMS_DIR = Path("/mnt/scratch/user/daviyang/proteinzen/plinder/2024-06/v2/systems")


def _write_pocket_pdb(atom_lines: list[str], keep: set, out_path: Path):
    kept_lines = []
    for line in atom_lines:
        if not line.startswith("ATOM"):
            continue
        chain = line[21]
        resnum = line[22:26].strip()
        if (chain, resnum) in keep:
            kept_lines.append(line)
    if not kept_lines:
        return False
    out_path.write_text("".join(kept_lines) + "END\n")
    return True


def _pocket_residues_from_coords(prot_by_res, prot_res_keys, lig_coords, cutoff=CUTOFF):
    keep = set()
    for res_atoms, (chain, resnum) in zip(prot_by_res, prot_res_keys):
        if not res_atoms or len(lig_coords) == 0:
            continue
        res_coords = np.array(list(res_atoms.values()))
        min_dist = np.linalg.norm(res_coords[:, None, :] - lig_coords[None, :, :], axis=-1).min()
        if min_dist <= cutoff:
            keep.add((chain, resnum))
    return keep


def _parse_receptor_pdb(text: str):
    """Bare ATOM-only parse of Plinder's receptor.pdb (protein only, no ligand)."""
    prot_by_res: list[dict] = []
    prot_res_keys: list[tuple] = []
    cur_key = None
    cur_atoms: dict = {}
    lines = text.splitlines(keepends=True)
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        aname = line[12:16].strip()
        key = (line[21], line[22:26].strip())
        if key != cur_key:
            if cur_atoms:
                prot_by_res.append(cur_atoms)
                prot_res_keys.append(cur_key)
            cur_key = key
            cur_atoms = {}
        cur_atoms[aname] = np.array([x, y, z], dtype=np.float64)
    if cur_atoms:
        prot_by_res.append(cur_atoms)
        prot_res_keys.append(cur_key)
    return prot_by_res, prot_res_keys, lines


def _ligand_coords_from_sdf(sdf_text: str):
    mol = Chem.MolFromMolBlock(sdf_text, sanitize=False, removeHs=False)
    if mol is None or mol.GetNumConformers() == 0:
        return np.zeros((0, 3))
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])


def extract_real_representatives(reps_json: Path, out_dir: Path, full_backbone: bool = False):
    """full_backbone=True: skip the 6A pocket cutoff entirely and write the whole,
    unmodified receptor.pdb -- for comparing full generated backbones against full
    real receptor backbones instead of (much smaller, harder-to-align) pocket-only
    fragments. See module docstring / --full-backbone for why this exists: GTalign's
    own --dev-min-length defaults to 20, silently skipping shorter references, and
    many 6A pockets fall below that."""
    reps = json.loads(reps_json.read_text())
    n_ok, n_fail = 0, 0
    for code, entries in reps.items():
        code_dir = out_dir / code
        code_dir.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            system_id = entry["system_id"]
            pdb_id = entry["pdb_id"]
            prefix = pdb_id[1:3]
            zip_path = SYSTEMS_DIR / f"{prefix}.zip"
            try:
                receptor_text = subprocess.run(
                    ["unzip", "-p", str(zip_path), f"{system_id}/receptor.pdb"],
                    check=True, capture_output=True, text=True,
                ).stdout
            except subprocess.CalledProcessError as e:
                print(f"  FAILED to extract {system_id} from {zip_path}: {e}")
                n_fail += 1
                continue

            out_path = code_dir / f"{system_id.replace('/', '_')}{'_full' if full_backbone else '_pocket'}.pdb"
            if full_backbone:
                out_path.write_text(receptor_text)
                n_ok += 1
                continue

            # The system_id's last field can itself be several underscore-joined ligand
            # chains (e.g. "1.B_1.C" for a multi-chain ligand/cofactor pair) -- confirmed
            # by inspection that ligand_files/ then has ONE SDF PER CHAIN (1.B.sdf,
            # 1.C.sdf), never a single combined "1.B_1.C.sdf". Extract each and pool
            # their coordinates for the pocket-distance cutoff.
            ligand_chains = system_id.split("__")[-1].split("_")
            try:
                lig_coords_list = []
                for lc in ligand_chains:
                    sdf_text = subprocess.run(
                        ["unzip", "-p", str(zip_path), f"{system_id}/ligand_files/{lc}.sdf"],
                        check=True, capture_output=True, text=True,
                    ).stdout
                    lig_coords_list.append(_ligand_coords_from_sdf(sdf_text))
            except subprocess.CalledProcessError as e:
                print(f"  FAILED to extract ligand for {system_id} from {zip_path}: {e}")
                n_fail += 1
                continue

            prot_by_res, prot_res_keys, lines = _parse_receptor_pdb(receptor_text)
            lig_coords = np.concatenate(lig_coords_list) if lig_coords_list else np.zeros((0, 3))
            keep = _pocket_residues_from_coords(prot_by_res, prot_res_keys, lig_coords)
            if _write_pocket_pdb(lines, keep, out_path):
                n_ok += 1
            else:
                print(f"  FAILED (empty pocket) {system_id}")
                n_fail += 1
        print(f"{code}: done")
    print(f"\nTotal: {n_ok} ok, {n_fail} failed")


def extract_generated_samples(samples_dir: Path, out_dir: Path, full_backbone: bool = False):
    """Grouped by ligand code (same _ligand_code_for convention as
    eval_pallatom_ligand_cond.py / pairwise_tmscore.py) so the output layout
    matches real_binder_pockets/<CODE>/ for a direct cross-comparison.
    full_backbone=True: see extract_real_representatives."""
    from eval_pallatom_ligand_cond import _ligand_code_for  # noqa: E402 (local import, heavy chain)
    n_ok, n_fail = 0, 0
    for pdb_path in sorted(samples_dir.glob("*.pdb")):
        code = _ligand_code_for(pdb_path)
        code_dir = out_dir / code
        code_dir.mkdir(parents=True, exist_ok=True)
        if full_backbone:
            out_path = code_dir / f"{pdb_path.stem}_full.pdb"
            out_path.write_text(pdb_path.read_text())
            n_ok += 1
            continue
        (prot_all, prot_ca, resnames, lig_coords, lig_elements, _,
         prot_by_res, prot_res_keys) = parse_pdb_ligand_cond(str(pdb_path))
        keep = _pocket_residues_from_coords(prot_by_res, prot_res_keys, lig_coords)
        lines = pdb_path.read_text().splitlines(keepends=True)
        out_path = code_dir / f"{pdb_path.stem}_pocket.pdb"
        if _write_pocket_pdb(lines, keep, out_path):
            n_ok += 1
        else:
            n_fail += 1
    print(f"generated samples: {n_ok} ok, {n_fail} failed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    p_real = sub.add_parser("real-representatives")
    p_real.add_argument("--reps-json", type=Path, required=True)
    p_real.add_argument("--out-dir", type=Path, required=True)
    p_real.add_argument("--full-backbone", action="store_true",
                        help="Write the whole receptor.pdb instead of a 6A pocket-only "
                             "cutout. Use this if pocket-only GTalign comparisons are "
                             "failing a lot -- pocket fragments are often < GTalign's "
                             "own --dev-min-length=20 default and get silently skipped.")

    p_gen = sub.add_parser("generated-samples")
    p_gen.add_argument("--samples-dir", type=Path, required=True)
    p_gen.add_argument("--out-dir", type=Path, required=True)
    p_gen.add_argument("--full-backbone", action="store_true")

    args = p.parse_args()
    if args.mode == "real-representatives":
        extract_real_representatives(args.reps_json, args.out_dir, full_backbone=args.full_backbone)
    else:
        extract_generated_samples(args.samples_dir, args.out_dir, full_backbone=args.full_backbone)


if __name__ == "__main__":
    main()
