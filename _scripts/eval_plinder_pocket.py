#!/usr/bin/env python
"""Evaluate protein-pocket-conditioned ligand generation on Plinder.

For each system:
  1. Load the GT structure from its npz (protein pocket + ligand).
  2. Load the generated PDB(s) (same pocket template, generated ligand).
  3. Kabsch-align the generated protein pocket onto the GT protein pocket.
     (Because sampling centers coordinates on the protein COM, this alignment
     is theoretically just a translation, but we do Kabsch for robustness.)
  4. Apply the same rigid transform to the generated ligand.
  5. Compute two symmetry-permutation-invariant metrics:
       - pk  : pocket-aligned RMSD — Kabsch on protein pocket, apply same transform
               to gen ligand, best-permutation positional RMSD vs GT ligand.
               Measures docking accuracy (is the ligand in the right place?).
       - lig : ligand-aligned RMSD — best-permutation + best-rotation Kabsch on
               the ligand itself. Measures conformer quality regardless of pocket
               position (same as GetBestRMS but using correct atom types).

Usage
-----
python _scripts/eval_plinder_pocket.py \\
    --samples-dir ./pocket_samples/samples \\
    --data-dir    /path/to/plinder_processed/val \\
    --max-protein-residues 100 \\
    --delta 2.0 5.0
"""

import argparse
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proteinzen.boltz.data import const
from proteinzen.runtime.sampling.protein_pocket import (
    _crop_protein_to_pocket,
    load_structure_from_npz,
)

RDLogger.DisableLog("rdApp.*")


# ============================================================
# Kabsch alignment
# ============================================================

def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation R and translation t that aligns P onto Q (source → target).

    P, Q : (N, 3) float64 arrays.
    Returns (R, t) where  Q_approx = (R @ P.T).T + t.
    """
    cp = P.mean(0)
    cq = Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cq - R @ cp
    return R, t


def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ coords.T).T + t


def pos_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=-1))))


# ============================================================
# PDB parsing
# ============================================================

def parse_pdb_atoms(pdb_path: str):
    """Return (prot_coords, lig_coords, lig_elements, conect) from a PDB file.

    prot_coords  : (N_prot, 3) float64 — ATOM records.
    lig_coords   : (N_lig,  3) float64 — HETATM records.
    lig_elements : list of element symbols (str) for each HETATM atom.
    conect       : set of (local_i, local_j) bond pairs for the ligand (i < j).
    """
    prot, lig, lig_elements = [], [], []
    lig_serial_to_local: dict[int, int] = {}
    raw_conects: list[tuple[int, int]] = []

    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].rstrip()
            if rec == "ATOM":
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    prot.append((x, y, z))
                except ValueError:
                    pass
            elif rec == "HETATM":
                try:
                    serial = int(line[6:11])
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    # element symbol is at cols 77-78 (0-indexed 76:78)
                    element = line[76:78].strip() if len(line) > 76 else ""
                    if not element:
                        # fall back to first non-digit character of atom name (cols 13-16)
                        element = line[12:16].strip().lstrip("0123456789") or "C"
                    lig_serial_to_local[serial] = len(lig)
                    lig.append((x, y, z))
                    lig_elements.append(element.capitalize())
                except ValueError:
                    pass
            elif rec == "CONECT":
                try:
                    nums = [int(line[6 + 5 * i : 11 + 5 * i]) for i in range(4)
                            if line[6 + 5 * i : 11 + 5 * i].strip()]
                    src = nums[0]
                    for dst in nums[1:]:
                        raw_conects.append((src, dst))
                except ValueError:
                    pass

    prot_arr = np.array(prot, dtype=np.float64) if prot else np.zeros((0, 3))
    lig_arr  = np.array(lig,  dtype=np.float64) if lig  else np.zeros((0, 3))

    conect_set: set[tuple[int, int]] = set()
    for s1, s2 in raw_conects:
        if s1 in lig_serial_to_local and s2 in lig_serial_to_local:
            a, b = lig_serial_to_local[s1], lig_serial_to_local[s2]
            conect_set.add((min(a, b), max(a, b)))

    return prot_arr, lig_arr, lig_elements, conect_set


def ligand_mol_from_pdb(pdb_path: str):
    """Build an RDKit mol for the ligand using HETATM records and CONECT bonds.

    Returns (mol, coords) where coords is (N_lig, 3) in PDB-file coordinates,
    or (None, None) on failure.
    """
    _, lig_coords, lig_elements, conect = parse_pdb_atoms(pdb_path)
    n_lig = len(lig_coords)
    if n_lig == 0:
        return None, None

    lines = []
    for i, ((x, y, z), elem) in enumerate(zip(lig_coords, lig_elements)):
        name = f"{elem:<2}"
        lines.append(
            f"HETATM{i+1:5d} {name}   LIG A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2}  "
        )
    for a, b in conect:
        lines.append(f"CONECT{a+1:5d}{b+1:5d}")
    lines.append("END")
    block = "\n".join(lines)

    mol = Chem.MolFromPDBBlock(block, removeHs=True, sanitize=False)
    if mol is None:
        return None, None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    return mol, lig_coords


def _clone_with_coords(mol, coords: np.ndarray):
    """Clone mol and set its conformer to coords (N, 3)."""
    if mol is None or mol.GetNumAtoms() != len(coords):
        return None
    rw = Chem.RWMol(Chem.Mol(mol))
    conf = rw.GetConformer()
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    return rw.GetMol()


def _get_permutations(mol_template, gt_coords, gen_coords):
    """Return list of valid atom permutations via substructure matching.

    Each permutation p is an array where p[i] = gen atom index for gt atom i.
    Returns None on failure.
    """
    gt_mol  = _clone_with_coords(mol_template, gt_coords)
    gen_mol = _clone_with_coords(mol_template, gen_coords)
    if gt_mol is None or gen_mol is None:
        return None
    try:
        matches = gen_mol.GetSubstructMatches(gt_mol, uniquify=False, maxMatches=10000)
        return [np.array(m, dtype=np.intp) for m in matches] if matches else None
    except Exception:
        return None


def pocket_rmsd_sym(mol_template, gt_coords: np.ndarray, gen_aligned_coords: np.ndarray) -> float:
    """Best-permutation positional RMSD after pocket alignment — no re-rotation.

    Measures docking accuracy: is the ligand in the right place in the pocket?
    """
    perms = _get_permutations(mol_template, gt_coords, gen_aligned_coords)
    if perms is None:
        return pos_rmsd(gt_coords, gen_aligned_coords)  # fallback: identity permutation
    return min(pos_rmsd(gt_coords, gen_aligned_coords[p]) for p in perms)


def _lig_rmsd_and_coords(mol_template, gt_coords: np.ndarray, gen_coords: np.ndarray):
    """Best-permutation + best-rotation ligand RMSD, also returning the aligned coords.

    Returns (rmsd, aligned_gen_coords) where aligned_gen_coords is gen_coords
    after applying the best permutation and Kabsch rotation onto gt_coords.
    """
    perms = _get_permutations(mol_template, gt_coords, gen_coords)
    if perms is None:
        perms = [np.arange(len(gt_coords), dtype=np.intp)]
    best_r = float("inf")
    best_coords = gen_coords
    for p in perms:
        permuted = gen_coords[p]
        R, t = kabsch(permuted, gt_coords)
        aligned = apply_transform(permuted, R, t)
        r = pos_rmsd(gt_coords, aligned)
        if r < best_r:
            best_r = r
            best_coords = aligned
    return best_r, best_coords


def lig_rmsd_sym(mol_template, gt_coords: np.ndarray, gen_coords: np.ndarray) -> float:
    r, _ = _lig_rmsd_and_coords(mol_template, gt_coords, gen_coords)
    return r


# ============================================================
# Extract GT coords from cropped Structure
# ============================================================

def extract_gt_coords(struct):
    """Return (prot_coords, lig_coords) as (N, 3) float64 — only is_present atoms."""
    protein_id    = const.chain_type_ids["PROTEIN"]
    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]

    prot_list, lig_list = [], []
    for chain in struct.chains[struct.mask]:
        mol = int(chain["mol_type"])
        a0  = int(chain["atom_idx"])
        atoms = struct.atoms[a0 : a0 + int(chain["atom_num"])]
        present = atoms["is_present"].astype(bool)
        if mol == protein_id:
            prot_list.append(atoms["coords"][present])
        elif mol == nonpolymer_id:
            lig_list.append(atoms["coords"][present])

    prot = np.concatenate(prot_list, 0) if prot_list else np.zeros((0, 3))
    lig  = np.concatenate(lig_list,  0) if lig_list  else np.zeros((0, 3))
    return prot.astype(np.float64), lig.astype(np.float64)


# ============================================================
# PDB output
# ============================================================

def _hetatm_lines(coords: np.ndarray, elements: list) -> list:
    """Minimal HETATM block for a ligand — no residue context needed for vis."""
    lines = []
    for i, ((x, y, z), elem) in enumerate(zip(coords, elements)):
        name = f" {elem:<3}" if len(elem) == 1 else f"{elem:<4}"
        lines.append(
            f"HETATM{i+1:5d} {name} LIG B   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2}  "
        )
    return lines


def _gt_pdb_body(gt_struct) -> str:
    """PDB body for the GT structure (protein + ligand) without the trailing END.

    Bonds are stripped before calling to_pdb: the CONECT writer in to_pdb indexes
    atom_reindex_ter by absolute atom index, which crashes when the cropped struct
    has is_present=False atoms (the list is shorter than the total atom count).
    CONECT records aren't needed for visualisation anyway.
    """
    from dataclasses import replace as dc_replace
    from proteinzen.data.write.pdb import to_pdb
    empty_bonds = gt_struct.bonds[:0]
    empty_conns = gt_struct.connections[:0]
    clean_struct = dc_replace(gt_struct, bonds=empty_bonds, connections=empty_conns)
    return to_pdb(clean_struct).rsplit("END", 1)[0]


def write_pocket_pdb(path: Path, gt_struct, gen_lig_pk: np.ndarray, lig_elements: list):
    """aligned_pocket: MODEL 1 = GT protein + GT ligand, MODEL 2 = pocket-aligned gen ligand."""
    gt_body = _gt_pdb_body(gt_struct)
    lines = ["MODEL        1", gt_body.rstrip(), "ENDMDL",
             "MODEL        2"]
    lines += _hetatm_lines(gen_lig_pk, lig_elements)
    lines += ["ENDMDL", "END"]
    path.write_text("\n".join(lines) + "\n")


def write_lig_pdb(path: Path, gt_struct, gen_lig_lig: np.ndarray, lig_elements: list):
    """aligned_lig: MODEL 1 = GT ligand only, MODEL 2 = lig-aligned gen ligand."""
    # Extract just the HETATM lines from the GT to_pdb output
    gt_hetatm = [ln for ln in _gt_pdb_body(gt_struct).splitlines()
                 if ln.startswith("HETATM")]
    lines = ["MODEL        1"] + gt_hetatm + ["ENDMDL",
             "MODEL        2"]
    lines += _hetatm_lines(gen_lig_lig, lig_elements)
    lines += ["ENDMDL", "END"]
    path.write_text("\n".join(lines) + "\n")


# ============================================================
# Per-system evaluation
# ============================================================

def eval_system(system_id: str, gen_pdb_paths, gt_struct):
    """Evaluate all generated samples for one system.

    Returns list of dicts: system_id, sample_idx, pk, lig, note.
    """
    gt_prot, gt_lig = extract_gt_coords(gt_struct)
    n_gt_prot = len(gt_prot)
    n_gt_lig  = len(gt_lig)

    if n_gt_lig == 0:
        return []

    records = []
    for idx, pdb_path in enumerate(sorted(gen_pdb_paths)):
        gen_prot, gen_lig, lig_elements, _ = parse_pdb_atoms(str(pdb_path))

        if len(gen_prot) != n_gt_prot:
            records.append(dict(
                system_id=system_id, sample_idx=idx,
                pk=np.inf, lig=np.inf,
                note=f"protein atom count mismatch: gen={len(gen_prot)} gt={n_gt_prot}",
            ))
            continue

        if len(gen_lig) != n_gt_lig:
            records.append(dict(
                system_id=system_id, sample_idx=idx,
                pk=np.inf, lig=np.inf,
                note=f"ligand atom count mismatch: gen={len(gen_lig)} gt={n_gt_lig}",
            ))
            continue

        # Kabsch: align generated protein pocket onto GT protein pocket
        R, t = kabsch(gen_prot, gt_prot)
        gen_lig_pk = apply_transform(gen_lig, R, t)   # pocket-aligned gen ligand

        mol_template, _ = ligand_mol_from_pdb(str(pdb_path))

        # pk:  pocket-aligned, best permutation, no ligand re-rotation
        # lig: best permutation + best ligand Kabsch, ignores pocket position
        pk  = pocket_rmsd_sym(mol_template, gt_lig, gen_lig_pk)
        lig, gen_lig_lig = _lig_rmsd_and_coords(mol_template, gt_lig, gen_lig)

        records.append(dict(
            system_id=system_id, sample_idx=idx,
            pk=pk, lig=lig,
            # transformed coords for PDB output
            gen_lig_pk=gen_lig_pk,
            gen_lig_lig=gen_lig_lig,
            lig_elements=lig_elements,
            note="",
        ))

    return records


def _eval_system_job(system_id: str, gen_pdb_paths, npz_path: str, include_h: bool, max_protein_residues: int):
    """Top-level worker for joblib — loads GT and evaluates one system."""
    RDLogger.DisableLog("rdApp.*")
    try:
        gt_struct = load_structure_from_npz(npz_path, include_h=include_h)
    except Exception as e:
        return system_id, [], f"npz load error: {e}"
    gt_struct = _crop_protein_to_pocket(gt_struct, max_protein_residues)
    records = eval_system(system_id, gen_pdb_paths, gt_struct)
    return system_id, records, None


# ============================================================
# Aggregation
# ============================================================

def mean_finite(vals):
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    return float(arr.mean()) if len(arr) else float("nan")


def cov(vals, delta):
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    return float((arr < delta).mean()) if len(arr) else float("nan")


def min_per_system(records_by_system, key):
    mins = []
    for recs in records_by_system.values():
        vals = [r[key] for r in recs if np.isfinite(r[key])]
        if vals:
            mins.append(min(vals))
    return mins


def mean_per_system(records_by_system, key):
    means = []
    for recs in records_by_system.values():
        vals = [r[key] for r in recs if np.isfinite(r[key])]
        if vals:
            means.append(float(np.mean(vals)))
    return means


def print_block(label, pk_vals, lig_vals, deltas):
    print(f"\n--- {label} ---")
    print(f"  n          : {len(pk_vals)}")
    print(f"  pk  (mean) : {mean_finite(pk_vals):.3f} Å")
    for d in deltas:
        print(f"  COV pk  < {d:.1f}Å : {cov(pk_vals, d)*100:.1f}%")
    n_lig = sum(np.isfinite(v) for v in lig_vals)
    if n_lig:
        print(f"  lig (mean) : {mean_finite(lig_vals):.3f} Å  ({n_lig}/{len(lig_vals)} finite)")
        for d in deltas:
            print(f"  COV lig < {d:.1f}Å : {cov(lig_vals, d)*100:.1f}%")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples-dir", type=Path, required=True,
                        help="Directory of generated PDB files ({system_id}_{i}.pdb)")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Plinder processed split directory (manifest.json + structures/)")
    parser.add_argument("--max-protein-residues", type=int, default=100,
                        help="Protein crop radius used during sampling (default: 100)")
    parser.add_argument("--include-h", action="store_true",
                        help="Keep hydrogen atoms (match the --include-h flag used at sample time)")
    parser.add_argument("--delta", type=float, nargs="+", default=[2.0, 5.0],
                        help="RMSD thresholds for coverage reporting (Å; default: 2.0 5.0)")
    parser.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() // 2),
                        help="Parallel workers (default: half of CPU count)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-system details during evaluation")
    args = parser.parse_args()

    # ---- manifest ----
    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"manifest.json not found at {manifest_path}")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    system_ids_in_manifest = {rec["id"] for rec in manifest}
    print(f"Manifest: {len(system_ids_in_manifest)} systems")

    # ---- collect generated PDB files ----
    pdb_files = sorted(args.samples_dir.glob("*.pdb"))
    print(f"Generated PDBs: {len(pdb_files)}")

    groups: dict[str, list[Path]] = defaultdict(list)
    unmatched = []
    for p in pdb_files:
        stem = p.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            groups[parts[0]].append(p)
        else:
            unmatched.append(p.name)

    if unmatched:
        print(f"  Warning: {len(unmatched)} PDB(s) had unrecognised names — skipped")

    common = sorted(set(groups) & system_ids_in_manifest)
    print(f"  Systems with generated samples : {len(groups)}")
    print(f"  Systems evaluated              : {len(common)}")
    extra   = set(groups) - system_ids_in_manifest
    missing = system_ids_in_manifest - set(groups)
    if extra:
        print(f"  Warning: {len(extra)} sampled systems not in manifest")
    if missing:
        print(f"  Note: {len(missing)} manifest systems have no samples")

    # ---- build job list ----
    jobs = []
    npz_by_sid: dict[str, str] = {}
    for sid in common:
        mid = sid[1:3]
        npz_path = args.data_dir / "structures" / mid / f"{sid}.npz"
        if not npz_path.exists():
            print(f"  SKIP {sid}: npz not found")
            continue
        npz_by_sid[sid] = str(npz_path)
        jobs.append((sid, groups[sid], str(npz_path)))

    print(f"  Running {len(jobs)} systems with {args.n_jobs} workers...")

    # ---- parallel evaluation ----
    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(_eval_system_job)(sid, pdbs, npz, args.include_h, args.max_protein_residues)
        for sid, pdbs, npz in tqdm(jobs, desc="evaluating")
    )

    # ---- collect results ----
    all_records: list[dict] = []
    records_by_system: dict[str, list[dict]] = {}
    n_atom_mismatch = 0

    for sid, sys_records, err in results:
        if err:
            print(f"  SKIP {sid}: {err}")
            continue
        if not sys_records:
            if args.verbose:
                print(f"  SKIP {sid}: no valid samples")
            continue

        mismatch_count = sum(1 for r in sys_records if r["note"])
        n_atom_mismatch += mismatch_count
        if args.verbose:
            for r in sys_records:
                if r["note"]:
                    print(f"  [{sid}] sample {r['sample_idx']}: {r['note']}")
            pks = [r["pk"] for r in sys_records if np.isfinite(r["pk"])]
            if pks:
                print(
                    f"  {sid}: {len(pks)} samples, "
                    f"pk min={min(pks):.2f} mean={np.mean(pks):.2f}"
                )

        records_by_system[sid] = sys_records
        all_records.extend(sys_records)

    if n_atom_mismatch:
        print(f"\nWarning: {n_atom_mismatch} samples had atom-count mismatches (skipped)")

    if not records_by_system:
        print("No systems evaluated — check paths.")
        return

    # ---- aggregate ----
    all_pk  = [r["pk"]  for r in all_records]
    all_lig = [r["lig"] for r in all_records]

    sys_min_pk   = min_per_system(records_by_system, "pk")
    sys_min_lig  = min_per_system(records_by_system, "lig")
    sys_mean_pk  = mean_per_system(records_by_system, "pk")
    sys_mean_lig = mean_per_system(records_by_system, "lig")

    deltas = args.delta

    print(f"\n{'='*60}")
    print(f"  PLINDER POCKET EVAL  —  {len(records_by_system)} systems")
    print(f"{'='*60}")

    print_block("Per-sample (all samples pooled)", all_pk, all_lig, deltas)
    print_block(
        "Per-system best sample (min pk per system)",
        sys_min_pk, sys_min_lig, deltas,
    )
    print_block(
        "Per-system mean",
        sys_mean_pk, sys_mean_lig, deltas,
    )

    # ---- per-system table ----
    print("\n--- Per-system summary (sorted by min pk) ---")
    print(f"  {'system_id':<42} {'n':>4} {'min_pk':>7} {'mean_pk':>8} {'min_lig':>8}")
    rows = []
    for sid, recs in records_by_system.items():
        pks  = [r["pk"]  for r in recs if np.isfinite(r["pk"])]
        ligs = [r["lig"] for r in recs if np.isfinite(r["lig"])]
        rows.append((
            sid, len(recs),
            min(pks)  if pks  else np.inf,
            float(np.mean(pks)) if pks else np.inf,
            min(ligs) if ligs else np.inf,
        ))

    def _fmt(v):
        return f"{v:.3f}" if np.isfinite(v) else "  inf"

    rows.sort(key=lambda x: x[2])
    for sid, n, mn_pk, avg_pk, mn_lig in rows:
        print(f"  {sid:<42} {n:>4} {_fmt(mn_pk):>7} {_fmt(avg_pk):>8} {_fmt(mn_lig):>8}")

    # ---- PDB output ----
    pk_dir  = args.samples_dir.parent / "aligned_pocket"
    lig_dir = args.samples_dir.parent / "aligned_lig"
    pk_dir.mkdir(exist_ok=True)
    lig_dir.mkdir(exist_ok=True)

    n_written = 0
    for sid, recs in records_by_system.items():
        valid = [r for r in recs if not r["note"]]
        if not valid:
            continue

        npz_path = npz_by_sid.get(sid)
        if npz_path is None:
            continue
        try:
            gt_struct = load_structure_from_npz(npz_path, include_h=args.include_h)
            gt_struct = _crop_protein_to_pocket(gt_struct, args.max_protein_residues)
        except Exception:
            continue

        best_pk_rec = min(valid, key=lambda r: r["pk"])
        if np.isfinite(best_pk_rec["pk"]):
            write_pocket_pdb(
                pk_dir / f"{sid}_pk{best_pk_rec['pk']:.2f}.pdb",
                gt_struct, best_pk_rec["gen_lig_pk"], best_pk_rec["lig_elements"],
            )

        best_lig_rec = min(valid, key=lambda r: r["lig"])
        if np.isfinite(best_lig_rec["lig"]):
            write_lig_pdb(
                lig_dir / f"{sid}_lig{best_lig_rec['lig']:.2f}.pdb",
                gt_struct, best_lig_rec["gen_lig_lig"], best_lig_rec["lig_elements"],
            )
        n_written += 1

    print(f"\nWrote PDB pairs for {n_written} systems → {pk_dir.parent}")


if __name__ == "__main__":
    main()
