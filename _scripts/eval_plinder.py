#!/usr/bin/env python
"""Evaluate PLINDER pocket tasks.

Two tasks:
  protein_cond  — fixed protein, generated ligand.  Metrics: RMSD (pocket-aligned
                  and ligand-self-aligned), CoM distance, PLIP interaction conservation.

  ligand_cond   — fixed ligand, generated protein.  Metrics: pocket contacts,
                  PoseBusters, and Boltz2 refolding self-consistency (pLDDT/ipTM/scRMSD).

Usage
-----
  # protein_cond (generate ligand)
  python _scripts/eval_plinder.py --task protein_cond \\
      --out-dir  sampling/plinder_pocket_val/protein_cond/<model> \\
      --ref-dir  plinder_pocket_processed/val

  # ligand_cond (generate protein binder)
  python _scripts/eval_plinder.py --task ligand_cond \\
      --out-dir      eval/ligand_cond/<model>/val \\
      --samples-dir  sampling/plinder_pocket_val/ligand_cond/<model>/samples \\
      --ref-dir      plinder_pocket_processed/val
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")

_GPU_SUFFIX = re.compile(r'_gpu\d+_batch\d+_idx\d+')

try:
    from posebusters import PoseBusters
    HAS_POSEBUSTERS = True
except ImportError:
    HAS_POSEBUSTERS = False

KNOWN_SMILES = {
    "SAM": "C[S+](CC[C@@H]([NH3+])C(=O)[O-])[C@@H]1O[C@@H]([C@H](O)[C@@H]1O)n1cnc2c(N)ncnc12",
    "FAD": "Cc1cc2nc3c(=O)[nH]c(=O)nc3n(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)c2cc1C",
    "NAD": "NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(C(O)C3O)n3cnc4c(N)ncnc34)C(O)C2O)c1",
    "ATP": "Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O))[C@@H](O)[C@H]1O",
    "HEM": "CC1=C(CCC(=O)O)C2=CC3=NC(=CC4=NC(=CC5=NC(=CC1=N2)C(=C5CCC(=O)O)C)C(=C4C)C=C)C(=C3C)C=C",
}


# ============================================================
# PLIP interaction analysis
# ============================================================

_PLIP_INTERACTION_TAGS = {
    "hydrophobic_interactions": ("hydrophobic_interaction", "HYDROPHOBIC"),
    "hydrogen_bonds":           ("hydrogen_bond",           "HBOND"),
    "water_bridges":            ("water_bridge",            "WATERBRIDGE"),
    "salt_bridges":             ("salt_bridge",             "SALTBRIDGE"),
    "pi_stacks":                ("pi_stack",                "PISTACK"),
    "pi_cation_interactions":   ("pi_cation_interaction",   "PICATION"),
    "halogen_bonds":            ("halogen_bond",            "HALOGENBOND"),
    "metal_complexes":          ("metal_complex",           "METALCOMPLEX"),
}
PLIP_TYPES = [label for _, label in _PLIP_INTERACTION_TAGS.values()]


def _parse_plip_xml(xml_path: str) -> dict:
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(xml_path)
    except Exception:
        return {label: set() for _, label in _PLIP_INTERACTION_TAGS.values()}

    root = tree.getroot()
    result = {label: set() for _, label in _PLIP_INTERACTION_TAGS.values()}

    for bs in root.iter("bindingsite"):
        interactions_node = bs.find("interactions")
        if interactions_node is None:
            continue
        for container_tag, (child_tag, label) in _PLIP_INTERACTION_TAGS.items():
            container = interactions_node.find(container_tag)
            if container is None:
                continue
            for interaction in container.findall(child_tag):
                resnr_el   = interaction.find("resnr")
                restype_el = interaction.find("restype")
                if resnr_el is None or restype_el is None:
                    continue
                try:
                    resnr = int(resnr_el.text)
                except (ValueError, TypeError):
                    continue
                result[label].add((resnr, restype_el.text.strip()))

    return result


def _run_plip(pdb_path: str, sif_path: str) -> dict:
    empty = {label: set() for _, label in _PLIP_INTERACTION_TAGS.values()}
    try:
        import shutil as _shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            local_pdb = os.path.join(tmpdir, "input.pdb")
            _shutil.copy2(str(pdb_path), local_pdb)
            result = subprocess.run(
                [sif_path, "-f", local_pdb, "-o", tmpdir, "-x"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                print(f"  [PLIP] returncode={result.returncode}\n"
                      f"STDOUT:{result.stdout[-400:]}\nSTDERR:{result.stderr[-800:]}")
                return empty
            xml_files = list(Path(tmpdir).glob("*.xml"))
            if not xml_files:
                print(f"  [PLIP] no XML output for {pdb_path}")
                return empty
            return _parse_plip_xml(str(xml_files[0]))
    except Exception as e:
        print(f"  [PLIP] exception: {e}")
        return empty


def _compare_interactions(gt: dict, gen: dict) -> dict:
    out = {}
    for label in PLIP_TYPES:
        gt_set  = gt.get(label, set())
        gen_set = gen.get(label, set())
        conserved = len(gt_set & gen_set)
        out[label] = {
            "gt":        len(gt_set),
            "gen":       len(gen_set),
            "conserved": conserved,
            "rate":      conserved / len(gt_set) if gt_set else float("nan"),
        }
    return out


# ============================================================
# Kabsch + PDB parsing (protein_cond)
# ============================================================

def kabsch(P: np.ndarray, Q: np.ndarray):
    cp, cq = P.mean(0), Q.mean(0)
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


def parse_pdb_atoms(pdb_path: str):
    prot, lig, lig_elements = [], [], []
    lig_serial_to_local: dict[int, int] = {}
    raw_conects: list[tuple[int, int]] = []
    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].rstrip()
            if rec == "ATOM":
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    elem = (line[76:78].strip() if len(line) > 76 else "").capitalize()
                    if not elem:
                        elem = line[12:16].strip().lstrip("0123456789").capitalize()
                    if elem in ("H", "D"):
                        continue
                    prot.append((x, y, z))
                except ValueError:
                    pass
            elif rec == "HETATM":
                try:
                    serial = int(line[6:11])
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    element = line[76:78].strip() if len(line) > 76 else ""
                    if not element:
                        element = line[12:16].strip().lstrip("0123456789") or "C"
                    element = element.capitalize()
                    if element == "H":
                        continue
                    lig_serial_to_local[serial] = len(lig)
                    lig.append((x, y, z))
                    lig_elements.append(element)
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
    if mol is None or mol.GetNumAtoms() != len(coords):
        return None
    rw = Chem.RWMol(Chem.Mol(mol))
    conf = rw.GetConformer()
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    return rw.GetMol()


def _get_permutations(mol_template, gt_coords, gen_coords):
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
    perms = _get_permutations(mol_template, gt_coords, gen_aligned_coords)
    if perms is None:
        return pos_rmsd(gt_coords, gen_aligned_coords)
    return min(pos_rmsd(gt_coords, gen_aligned_coords[p]) for p in perms)


def _lig_rmsd_and_coords(mol_template, gt_coords: np.ndarray, gen_coords: np.ndarray):
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


def extract_gt_coords(struct):
    from proteinzen.boltz.data import const
    protein_id    = const.chain_type_ids["PROTEIN"]
    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]
    prot_list, lig_list = [], []
    for chain in struct.chains[struct.mask]:
        mol = int(chain["mol_type"])
        a0  = int(chain["atom_idx"])
        atoms = struct.atoms[a0 : a0 + int(chain["atom_num"])]
        present = atoms["is_present"].astype(bool)
        heavy = present & (atoms["element"] != 1)
        if mol == protein_id:
            prot_list.append(atoms["coords"][heavy])
        elif mol == nonpolymer_id:
            lig_list.append(atoms["coords"][heavy])
    prot = np.concatenate(prot_list, 0) if prot_list else np.zeros((0, 3))
    lig  = np.concatenate(lig_list,  0) if lig_list  else np.zeros((0, 3))
    return prot.astype(np.float64), lig.astype(np.float64)


def _hetatm_lines(coords: np.ndarray, elements: list) -> list:
    lines = []
    for i, ((x, y, z), elem) in enumerate(zip(coords, elements)):
        name = f" {elem:<3}" if len(elem) == 1 else f"{elem:<4}"
        lines.append(
            f"HETATM{i+1:5d} {name} LIG B   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2}  "
        )
    return lines


def _gt_pdb_body(gt_struct) -> str:
    from dataclasses import replace as dc_replace
    from proteinzen.data.write.pdb import to_pdb
    empty_bonds = gt_struct.bonds[:0]
    empty_conns = gt_struct.connections[:0]
    clean_struct = dc_replace(gt_struct, bonds=empty_bonds, connections=empty_conns)
    return to_pdb(clean_struct).rsplit("END", 1)[0]


def write_pocket_pdb(path: Path, gt_struct, gen_lig_pk: np.ndarray, lig_elements: list):
    gt_body = _gt_pdb_body(gt_struct)
    prot_lines = [ln for ln in gt_body.splitlines() if ln.startswith("ATOM")]
    lines = ["MODEL        1", gt_body.rstrip(), "ENDMDL", "MODEL        2"]
    lines += prot_lines
    lines += ["TER"]
    lines += _hetatm_lines(gen_lig_pk, lig_elements)
    lines += ["ENDMDL", "END"]
    path.write_text("\n".join(lines) + "\n")


def write_lig_pdb(path: Path, gt_struct, gen_lig_lig: np.ndarray, lig_elements: list):
    gt_hetatm = [ln for ln in _gt_pdb_body(gt_struct).splitlines() if ln.startswith("HETATM")]
    lines = ["MODEL        1"] + gt_hetatm + ["ENDMDL", "MODEL        2"]
    lines += _hetatm_lines(gen_lig_lig, lig_elements)
    lines += ["ENDMDL", "END"]
    path.write_text("\n".join(lines) + "\n")


def eval_system_pocket(system_id: str, gen_pdb_paths, gt_struct,
                       gt_interactions: dict | None = None, plip_sif: str | None = None,
                       run_pb: bool = False):
    gt_prot, gt_lig = extract_gt_coords(gt_struct)
    n_gt_prot = len(gt_prot)
    n_gt_lig  = len(gt_lig)
    if n_gt_lig == 0:
        return []
    records = []
    for idx, pdb_path in enumerate(sorted(gen_pdb_paths)):
        gen_prot, gen_lig, lig_elements, _ = parse_pdb_atoms(str(pdb_path))
        if len(gen_prot) != n_gt_prot:
            n_common = min(len(gen_prot), n_gt_prot)
            if n_common == 0:
                records.append(dict(
                    system_id=system_id, sample_idx=idx, pk=np.inf, lig=np.inf,
                    note=f"protein atom count mismatch: gen={len(gen_prot)} gt={n_gt_prot}",
                ))
                continue
            gen_prot = gen_prot[:n_common]
            gt_prot_aligned = gt_prot[:n_common]
        else:
            gt_prot_aligned = gt_prot
        if len(gen_lig) != n_gt_lig:
            records.append(dict(
                system_id=system_id, sample_idx=idx, pk=np.inf, lig=np.inf,
                note=f"ligand atom count mismatch: gen={len(gen_lig)} gt={n_gt_lig}",
            ))
            continue
        R, t = kabsch(gen_prot, gt_prot_aligned)
        gen_lig_pk = apply_transform(gen_lig, R, t)
        mol_template, _ = ligand_mol_from_pdb(str(pdb_path))
        pk = pocket_rmsd_sym(mol_template, gt_lig, gen_lig_pk)
        lig, gen_lig_lig = _lig_rmsd_and_coords(mol_template, gt_lig, gen_lig)

        com_dist = float(np.linalg.norm(gt_lig.mean(0) - gen_lig_pk.mean(0)))
        record = dict(
            system_id=system_id, sample_idx=idx, pdb_stem=Path(pdb_path).stem,
            pk=pk, lig=lig, com_dist=com_dist,
            gen_lig_pk=gen_lig_pk, gen_lig_lig=gen_lig_lig,
            lig_elements=lig_elements, note="",
        )

        if plip_sif and gt_interactions is not None:
            gen_interactions = _run_plip(str(pdb_path), plip_sif)
            conservation = _compare_interactions(gt_interactions, gen_interactions)
            for itype, counts in conservation.items():
                record[f"plip_{itype}_gt"]       = counts["gt"]
                record[f"plip_{itype}_gen"]       = counts["gen"]
                record[f"plip_{itype}_conserved"] = counts["conserved"]
                record[f"plip_{itype}_rate"]      = counts["rate"]

        if run_pb:
            pb = run_posebusters(str(pdb_path))
            record.update(pb)

        records.append(record)
    return records


def _eval_system_pocket_job(system_id, gen_pdb_paths, npz_path, include_h,
                            max_protein_residues, plip_sif=None, run_pb=False):
    from proteinzen.runtime.sampling.protein_pocket import _crop_protein_to_pocket, load_structure_from_npz
    from proteinzen.data.write.pdb import to_pdb
    from dataclasses import replace as dc_replace
    RDLogger.DisableLog("rdApp.*")
    try:
        gt_struct = load_structure_from_npz(npz_path, include_h=include_h)
    except Exception as e:
        return system_id, [], f"npz load error: {e}"

    if max_protein_residues is not None:
        gt_struct = _crop_protein_to_pocket(gt_struct, max_protein_residues)

    gt_interactions = None
    if plip_sif:
        try:
            clean = dc_replace(gt_struct, bonds=gt_struct.bonds[:0],
                               connections=gt_struct.connections[:0])
            gt_pdb_str = to_pdb(clean)
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tf:
                tf.write(gt_pdb_str)
                gt_pdb_path = tf.name
            gt_interactions = _run_plip(gt_pdb_path, plip_sif)
            Path(gt_pdb_path).unlink(missing_ok=True)
        except Exception:
            gt_interactions = None

    records = eval_system_pocket(system_id, gen_pdb_paths, gt_struct,
                                 gt_interactions=gt_interactions, plip_sif=plip_sif,
                                 run_pb=run_pb)
    return system_id, records, None


def _mean_finite(vals):
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    return float(arr.mean()) if len(arr) else float("nan")


def _cov(vals, delta):
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    return float((arr < delta).mean()) if len(arr) else float("nan")


def _min_per_system(records_by_system, key):
    mins = []
    for recs in records_by_system.values():
        vals = [r[key] for r in recs if key in r and np.isfinite(r[key])]
        if vals:
            mins.append(min(vals))
    return mins


def _mean_per_system(records_by_system, key):
    means = []
    for recs in records_by_system.values():
        vals = [r[key] for r in recs if key in r and np.isfinite(r[key])]
        if vals:
            means.append(float(np.mean(vals)))
    return means


def _print_pocket_block(label, pk_vals, lig_vals, deltas):
    print(f"\n--- {label} ---")
    print(f"  n          : {len(pk_vals)}")
    print(f"  pk  (mean) : {_mean_finite(pk_vals):.3f} Å")
    for d in deltas:
        print(f"  COV pk  < {d:.1f}Å : {_cov(pk_vals, d)*100:.1f}%")
    n_lig = sum(np.isfinite(v) for v in lig_vals)
    if n_lig:
        print(f"  lig (mean) : {_mean_finite(lig_vals):.3f} Å  ({n_lig}/{len(lig_vals)} finite)")
        for d in deltas:
            print(f"  COV lig < {d:.1f}Å : {_cov(lig_vals, d)*100:.1f}%")


# ============================================================
# protein_cond eval — main function
# ============================================================

def run_protein_cond_eval(args):
    from proteinzen.runtime.sampling.protein_pocket import _crop_protein_to_pocket, load_structure_from_npz

    out_dir     = args.out_dir
    data_dir    = args.ref_dir
    samples_dir = out_dir / "samples"
    deltas      = args.pocket_deltas
    n_jobs      = args.n_jobs
    max_prot    = args.max_protein_residues
    include_h   = args.include_h
    verbose     = args.verbose
    plip_sif    = str(args.plip_sif)
    run_pb      = (not args.no_posebusters) and HAS_POSEBUSTERS
    if not args.no_posebusters and not HAS_POSEBUSTERS:
        print("Warning: posebusters not installed — skipping (pip install posebusters to enable).")
    print(f"PLIP: {plip_sif}")

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"manifest.json not found at {manifest_path}")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    system_ids_in_manifest = {rec["id"] for rec in manifest}
    print(f"Manifest: {len(system_ids_in_manifest)} systems")

    pdb_files = sorted(samples_dir.glob("*.pdb"))
    print(f"Generated PDBs: {len(pdb_files)}")

    groups: dict[str, list[Path]] = defaultdict(list)
    unmatched = []
    for p in pdb_files:
        stem = p.stem
        m = _GPU_SUFFIX.search(stem)
        if m:
            groups[stem[:m.start()]].append(p)
        else:
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

    jobs = []
    npz_by_sid: dict[str, str] = {}
    for sid in common:
        mid = sid[1:3]
        npz_path = data_dir / "structures" / mid / f"{sid}.npz"
        if not npz_path.exists():
            print(f"  SKIP {sid}: npz not found")
            continue
        npz_by_sid[sid] = str(npz_path)
        jobs.append((sid, groups[sid], str(npz_path)))
    print(f"  Running {len(jobs)} systems with {n_jobs} workers...")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_eval_system_pocket_job)(sid, pdbs, npz, include_h, max_prot, plip_sif, run_pb)
        for sid, pdbs, npz in tqdm(jobs, desc="evaluating")
    )

    all_records: list[dict] = []
    records_by_system: dict[str, list[dict]] = {}
    n_atom_mismatch = 0
    for sid, sys_records, err in results:
        if err:
            print(f"  SKIP {sid}: {err}")
            continue
        if not sys_records:
            if verbose:
                print(f"  SKIP {sid}: no valid samples")
            continue
        mismatch_count = sum(1 for r in sys_records if r["note"])
        n_atom_mismatch += mismatch_count
        if verbose:
            for r in sys_records:
                if r["note"]:
                    print(f"  [{sid}] sample {r['sample_idx']}: {r['note']}")
            pks = [r["pk"] for r in sys_records if np.isfinite(r["pk"])]
            if pks:
                print(f"  {sid}: {len(pks)} samples, pk min={min(pks):.2f} mean={np.mean(pks):.2f}")
        records_by_system[sid] = sys_records
        all_records.extend(sys_records)

    if n_atom_mismatch:
        print(f"\nWarning: {n_atom_mismatch} samples had atom-count mismatches (skipped)")
    if not records_by_system:
        print("No systems evaluated — check paths.")
        return

    all_pk   = [r["pk"]       for r in all_records]
    all_lig  = [r["lig"]      for r in all_records]
    all_com  = [r["com_dist"] for r in all_records if "com_dist" in r and np.isfinite(r["com_dist"])]
    sys_min_pk   = _min_per_system(records_by_system, "pk")
    sys_min_lig  = _min_per_system(records_by_system, "lig")
    sys_min_com  = _min_per_system(records_by_system, "com_dist")
    sys_mean_pk  = _mean_per_system(records_by_system, "pk")
    sys_mean_lig = _mean_per_system(records_by_system, "lig")
    sys_mean_com = _mean_per_system(records_by_system, "com_dist")

    print(f"\n{'='*60}")
    print(f"  PLINDER POCKET EVAL  —  {len(records_by_system)} systems")
    print(f"{'='*60}")
    _print_pocket_block("Per-sample (pooled)", all_pk, all_lig, deltas)
    _print_pocket_block("Per-system best sample (min pk)", sys_min_pk, sys_min_lig, deltas)
    _print_pocket_block("Per-system mean", sys_mean_pk, sys_mean_lig, deltas)

    print(f"\n--- CoM Distance (Å, pocket-aligned) ---")
    for label, vals in [("Per-sample (pooled)", all_com),
                        ("Per-system best (min)", sys_min_com),
                        ("Per-system mean",       sys_mean_com)]:
        finite = [v for v in vals if np.isfinite(v)]
        if finite:
            print(
                f"  {label}: mean={np.mean(finite):.3f}  median={np.median(finite):.3f}  "
                f"<2Å={np.mean(np.array(finite)<2)*100:.1f}%  <5Å={np.mean(np.array(finite)<5)*100:.1f}%"
            )

    print("\n--- Per-system summary (sorted by min pk) ---")
    print(f"  {'system_id':<42} {'n':>4} {'min_pk':>7} {'mean_pk':>8} {'min_lig':>8}")
    rows = []
    for sid, recs in records_by_system.items():
        pks  = [r["pk"]  for r in recs if np.isfinite(r["pk"])]
        ligs = [r["lig"] for r in recs if np.isfinite(r["lig"])]
        rows.append((sid, len(recs),
                     min(pks) if pks else np.inf,
                     float(np.mean(pks)) if pks else np.inf,
                     min(ligs) if ligs else np.inf))
    rows.sort(key=lambda x: x[2])
    def _fmt(v): return f"{v:.3f}" if np.isfinite(v) else "  inf"
    for sid, n, mn_pk, avg_pk, mn_lig in rows:
        print(f"  {sid:<42} {n:>4} {_fmt(mn_pk):>7} {_fmt(avg_pk):>8} {_fmt(mn_lig):>8}")

    if all_records:
        print(f"\n--- PLIP Interaction Conservation ---")
        print(f"  {'type':<14} {'gt_mean':>8} {'gen_mean':>9} {'conserved':>10} {'rate':>7}")
        for itype in PLIP_TYPES:
            gt_counts  = [r[f"plip_{itype}_gt"]       for r in all_records if f"plip_{itype}_gt" in r]
            gen_counts = [r[f"plip_{itype}_gen"]       for r in all_records if f"plip_{itype}_gen" in r]
            con_counts = [r[f"plip_{itype}_conserved"] for r in all_records if f"plip_{itype}_conserved" in r]
            rates      = [r[f"plip_{itype}_rate"]      for r in all_records
                          if f"plip_{itype}_rate" in r and np.isfinite(r[f"plip_{itype}_rate"])]
            if not gt_counts:
                continue
            print(
                f"  {itype:<14} "
                f"{_mean_finite(gt_counts):>8.2f} "
                f"{_mean_finite(gen_counts):>9.2f} "
                f"{_mean_finite(con_counts):>10.2f} "
                f"{_mean_finite(rates):>7.3f}"
            )

    pk_dir  = out_dir / "aligned_pocket"
    lig_dir = out_dir / "aligned_lig"
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
            gt_struct = load_structure_from_npz(npz_path, include_h=include_h)
            if max_prot is not None:
                gt_struct = _crop_protein_to_pocket(gt_struct, max_prot)
        except Exception:
            continue
        best_pk_rec = min(valid, key=lambda r: r["pk"])
        if np.isfinite(best_pk_rec["pk"]):
            stem = best_pk_rec.get("pdb_stem", sid)
            write_pocket_pdb(
                pk_dir / f"{stem}_pk{best_pk_rec['pk']:.2f}.pdb",
                gt_struct, best_pk_rec["gen_lig_pk"], best_pk_rec["lig_elements"],
            )
        best_lig_rec = min(valid, key=lambda r: r["lig"])
        if np.isfinite(best_lig_rec["lig"]):
            stem = best_lig_rec.get("pdb_stem", sid)
            write_lig_pdb(
                lig_dir / f"{stem}_lig{best_lig_rec['lig']:.2f}.pdb",
                gt_struct, best_lig_rec["gen_lig_lig"], best_lig_rec["lig_elements"],
            )
        n_written += 1
    print(f"\nWrote PDB pairs for {n_written} systems → {out_dir}")


# ============================================================
# ligand_cond helpers — PDB parsing, contacts, refolding
# ============================================================

def parse_pdb_ligand_cond(pdb_path: str):
    """Parse PDB into protein and ligand arrays (for ligand_cond task).

    Returns prot_coords, prot_ca, prot_resnames, lig_coords, lig_elements, conect.
    """
    prot_all, prot_ca, prot_resnames = [], [], []
    lig_coords, lig_elements = [], []
    lig_serial_to_local: dict[int, int] = {}
    raw_conects: list[tuple[int, int]] = []
    seen_ca: set[tuple[str, str]] = set()

    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].rstrip()
            if rec == "ATOM":
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    prot_all.append((x, y, z))
                    if line[12:16] == " CA ":
                        key = (line[21], line[22:26].strip())
                        if key not in seen_ca:
                            seen_ca.add(key)
                            prot_ca.append((x, y, z))
                            prot_resnames.append(line[17:20].strip())
                except ValueError:
                    pass
            elif rec == "HETATM":
                try:
                    serial = int(line[6:11])
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    elem = line[76:78].strip() if len(line) > 76 else ""
                    if not elem:
                        elem = line[12:16].strip().lstrip("0123456789") or "C"
                    lig_serial_to_local[serial] = len(lig_coords)
                    lig_coords.append((x, y, z))
                    lig_elements.append(elem.capitalize())
                except ValueError:
                    pass
            elif rec == "CONECT":
                try:
                    nums = [int(line[6 + 5*i: 11 + 5*i]) for i in range(4)
                            if line[6 + 5*i: 11 + 5*i].strip()]
                    for dst in nums[1:]:
                        raw_conects.append((nums[0], dst))
                except ValueError:
                    pass

    def _arr(lst): return np.array(lst, dtype=np.float64) if lst else np.zeros((0, 3))

    conect: set[tuple[int, int]] = set()
    for s1, s2 in raw_conects:
        if s1 in lig_serial_to_local and s2 in lig_serial_to_local:
            a, b = lig_serial_to_local[s1], lig_serial_to_local[s2]
            conect.add((min(a, b), max(a, b)))

    return _arr(prot_all), _arr(prot_ca), prot_resnames, _arr(lig_coords), lig_elements, conect


_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y",
}


def resnames_to_seq(resnames: list[str]) -> str:
    return "".join(_AA3TO1.get(r, "X") for r in resnames)


def pocket_contacts(prot_coords: np.ndarray, lig_coords: np.ndarray, cutoff: float = 4.0):
    if len(prot_coords) == 0 or len(lig_coords) == 0:
        return float("nan"), 0
    diff = prot_coords[:, None, :] - lig_coords[None, :, :]
    dists = np.sqrt((diff ** 2).sum(-1))
    lig_contacted = (dists < cutoff).any(axis=0).sum()
    frac_lig = float(lig_contacted) / len(lig_coords)
    n_prot_contact = int((dists < cutoff).any(axis=1).sum())
    return frac_lig, n_prot_contact


def run_posebusters(pdb_path: str, smiles: str | None = None) -> dict:
    if not HAS_POSEBUSTERS:
        return {}
    _, _, _, lig_coords, lig_elements, conect = parse_pdb_ligand_cond(pdb_path)
    if len(lig_coords) == 0:
        return {"pb_ligand_parse_failed": True}
    try:
        lines = []
        for i, ((x, y, z), elem) in enumerate(zip(lig_coords, lig_elements)):
            lines.append(
                f"HETATM{i+1:5d}  {elem:<3} LIG A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2}  "
            )
        for a, b in conect:
            lines.append(f"CONECT{a+1:5d}{b+1:5d}")
        lines.append("END")
        mol = Chem.MolFromPDBBlock("\n".join(lines), removeHs=True, sanitize=False)
        if mol is None:
            return {"pb_ligand_mol_failed": True}
        Chem.SanitizeMol(mol)
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tf:
            lig_sdf = tf.name
        writer = Chem.SDWriter(lig_sdf)
        writer.write(mol)
        writer.close()
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tf:
            prot_pdb = tf.name
        with open(pdb_path) as fh, open(prot_pdb, "w") as out:
            for line in fh:
                if line.startswith("ATOM") or line.startswith("TER") or line.startswith("END"):
                    out.write(line)
        pb = PoseBusters(config="dock")
        df = pb.bust(lig_sdf, mol_cond=prot_pdb, full_report=True)
        results = {}
        for col in df.columns:
            val = df[col].iloc[0]
            if isinstance(val, (bool, np.bool_)):
                results[f"pb_{col}"] = bool(val)
            elif isinstance(val, (float, int, np.floating, np.integer)):
                results[f"pb_{col}"] = float(val)
        os.unlink(lig_sdf)
        os.unlink(prot_pdb)
        return results
    except Exception as e:
        return {"pb_error": str(e)}


def _pb_worker(args_tuple):
    """Module-level worker so multiprocessing can pickle it."""
    pdb_path, smiles = args_tuple
    return str(pdb_path), run_posebusters(str(pdb_path), smiles)


def _parse_ca_and_lig_from_cif(cif_path: Path):
    """Return (ca_coords, lig_coords) from a Boltz2 CIF. Protein=chain A, ligand=all other chains (heavy atoms only)."""
    import gemmi
    st = gemmi.read_structure(str(cif_path))
    ca, lig = [], []
    for model in st:
        for chain in model:
            if chain.name == "A":
                for res in chain:
                    for atom in res:
                        if atom.name == "CA":
                            p = atom.pos
                            ca.append([p.x, p.y, p.z])
                            break
            else:
                for res in chain:
                    for atom in res:
                        if atom.element == gemmi.Element("H"):
                            continue
                        p = atom.pos
                        lig.append([p.x, p.y, p.z])
        break
    return np.array(ca, dtype=np.float64), np.array(lig, dtype=np.float64)


def _kabsch_align(P: np.ndarray, Q: np.ndarray):
    """Kabsch: compute rotation R and translation t such that R @ (P - P_mean).T + Q_mean ~= Q. Returns (R, p_mean, q_mean)."""
    p_mean = P.mean(0)
    q_mean = Q.mean(0)
    p = P - p_mean
    q = Q - q_mean
    H = p.T @ q
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, p_mean, q_mean


def _apply_kabsch(coords: np.ndarray, R: np.ndarray, p_mean: np.ndarray, q_mean: np.ndarray) -> np.ndarray:
    return (coords - p_mean) @ R.T + q_mean


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    assert P.shape == Q.shape and P.ndim == 2
    R, p_mean, q_mean = _kabsch_align(P, Q)
    p_rot = _apply_kabsch(P, R, p_mean, q_mean)
    return float(np.sqrt(((p_rot - Q) ** 2).sum(-1).mean()))


def run_refolding(sequence, smiles, gen_ca, refold_input_dir, refold_output_dir,
                  sample_id, boltz_cache, gen_lig=None):
    import yaml as _yaml
    refold_input_dir.mkdir(parents=True, exist_ok=True)
    refold_output_dir.mkdir(parents=True, exist_ok=True)

    boltz_input: dict = {"sequences": [{"protein": {"id": "A", "sequence": sequence, "msa": "empty"}}]}
    if smiles:
        boltz_input["sequences"].append({"ligand": {"id": "B", "smiles": smiles}})

    input_yaml = refold_input_dir / f"{sample_id}.yaml"
    out_dir = refold_output_dir / sample_id
    pred_dir = out_dir / f"boltz_results_{sample_id}" / "predictions" / sample_id

    if not pred_dir.exists():
        input_yaml.write_text(_yaml.dump(boltz_input, default_flow_style=False))
        cmd = ["micromamba", "run", "-n", "boltz",
               "boltz", "predict", str(input_yaml), "--out_dir", str(out_dir),
               "--override"]
        if boltz_cache:
            cmd += ["--cache", str(boltz_cache)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        except subprocess.CalledProcessError as e:
            return {"plddt": float("nan"), "iptm": float("nan"), "sc_rmsd": float("nan"),
                    "lig_rmsd": float("nan"), "boltz_error": e.stderr.decode()[-200:]}
        except subprocess.TimeoutExpired:
            return {"plddt": float("nan"), "iptm": float("nan"), "sc_rmsd": float("nan"),
                    "lig_rmsd": float("nan"), "boltz_error": "timeout"}
    conf_files = sorted(pred_dir.glob("confidence_*_model_0.json"))
    plddt, iptm = float("nan"), float("nan")
    if conf_files:
        try:
            conf = json.loads(conf_files[0].read_text())
            plddt = float(conf.get("complex_plddt", float("nan")))
            iptm = float(conf.get("iptm", float("nan")))
        except Exception:
            pass

    sc_rmsd = float("nan")
    lig_rmsd = float("nan")
    cif_files = sorted(pred_dir.glob("*_model_0.cif"))
    if cif_files and len(gen_ca) > 0:
        try:
            refold_ca, refold_lig = _parse_ca_and_lig_from_cif(cif_files[0])
            if len(refold_ca) == len(gen_ca):
                R, p_mean, q_mean = _kabsch_align(gen_ca, refold_ca)
                gen_ca_rot = _apply_kabsch(gen_ca, R, p_mean, q_mean)
                sc_rmsd = float(np.sqrt(((gen_ca_rot - refold_ca) ** 2).sum(-1).mean()))
                if (gen_lig is not None and len(gen_lig) > 0
                        and len(refold_lig) > 0 and len(refold_lig) == len(gen_lig)):
                    gen_lig_rot = _apply_kabsch(gen_lig, R, p_mean, q_mean)
                    lig_rmsd = float(np.sqrt(((gen_lig_rot - refold_lig) ** 2).sum(-1).mean()))
        except Exception:
            pass

    return {"plddt": plddt, "iptm": iptm, "sc_rmsd": sc_rmsd, "lig_rmsd": lig_rmsd}


def _parse_mpnn_fasta(fasta_path: Path) -> list[str]:
    """Parse sequences from a ProteinMPNN/LigandMPNN FASTA output (skip first/native entry)."""
    entries = []
    if not fasta_path.exists():
        return entries
    with open(fasta_path) as fh:
        header, seq_parts = None, []
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    entries.append("".join(seq_parts))
                header, seq_parts = line, []
            else:
                seq_parts.append(line)
        if header is not None:
            entries.append("".join(seq_parts))
    return entries[1:]  # skip native


_MPNN_ENV = "mpnn"

def _run_ligandmpnn(pdb_path: Path, out_dir: Path, n_seqs: int, script: str, model_type: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "seqs" / f"{pdb_path.stem}.fa"
    if not fasta_path.exists():
        subprocess.run([
            "micromamba", "run", "-n", _MPNN_ENV,
            "python", script,
            "--model_type", model_type,
            "--pdb_path", str(pdb_path),
            "--out_folder", str(out_dir),
            "--number_of_batches", str(n_seqs),
            "--temperature", "0.1",
            "--batch_size", "1",
        ], check=True, timeout=300, cwd=Path(script).parent)
    return _parse_mpnn_fasta(fasta_path)[:n_seqs]


def run_proteinmpnn(pdb_path: Path, out_dir: Path, n_seqs: int, mpnn_script: str) -> list[str]:
    """Run LigandMPNN repo with model_type=protein_mpnn (no ligand context)."""
    return _run_ligandmpnn(pdb_path, out_dir, n_seqs, mpnn_script, "protein_mpnn")


def run_ligandmpnn(pdb_path: Path, out_dir: Path, n_seqs: int, ligandmpnn_script: str) -> list[str]:
    """Run LigandMPNN repo with model_type=ligand_mpnn (ligand-aware)."""
    return _run_ligandmpnn(pdb_path, out_dir, n_seqs, ligandmpnn_script, "ligand_mpnn")


def eval_ligand_cond_sample(pdb_path, smiles, refold_input_dir, refold_output_dir,
                            boltz_cache, run_pb, skip_fold, contact_cutoff=4.0,
                            mpnn_script=None, ligandmpnn_script=None,
                            mpnn_n_seqs=3, mpnn_refold_dir=None, pb_cache=None):
    prot_all, prot_ca, resnames, lig_coords, lig_elements, _ = parse_pdb_ligand_cond(str(pdb_path))
    sequence = resnames_to_seq(resnames)
    frac_lig_contacted, n_prot_contact = pocket_contacts(prot_all, lig_coords, cutoff=contact_cutoff)
    if pb_cache is not None:
        pb = pb_cache.get(str(pdb_path), {})
    else:
        pb = run_posebusters(str(pdb_path), smiles) if run_pb else {}
    gen_lig = lig_coords.astype(np.float64) if len(lig_coords) > 0 else None

    _nan_fold = {"plddt": float("nan"), "iptm": float("nan"), "sc_rmsd": float("nan"), "lig_rmsd": float("nan")}
    if skip_fold or not sequence:
        fold        = _nan_fold.copy()
        fold_nolig  = _nan_fold.copy()
    else:
        fold = run_refolding(
            sequence=sequence, smiles=smiles, gen_ca=prot_ca,
            refold_input_dir=refold_input_dir, refold_output_dir=refold_output_dir,
            sample_id=pdb_path.stem, boltz_cache=boltz_cache,
            gen_lig=gen_lig,
        )
        fold_nolig = run_refolding(
            sequence=sequence, smiles=None, gen_ca=prot_ca,
            refold_input_dir=refold_input_dir, refold_output_dir=refold_output_dir,
            sample_id=f"{pdb_path.stem}_nolig", boltz_cache=boltz_cache,
            gen_lig=None,
        )
    fold_nolig = {f"nolig_{k}": v for k, v in fold_nolig.items()}

    def _run_mpnn_case(seqs, refold_smiles, tag, base_dir):
        results = []
        for i, seq in enumerate(seqs):
            fold_i = run_refolding(
                sequence=seq, smiles=refold_smiles, gen_ca=prot_ca,
                refold_input_dir=base_dir / "refold_inputs",
                refold_output_dir=base_dir / "refold_outputs",
                sample_id=f"{pdb_path.stem}_{tag}{i}",
                boltz_cache=boltz_cache,
                gen_lig=gen_lig,
            )
            results.append(fold_i)
        rmsds    = [r["sc_rmsd"]  for r in results if np.isfinite(r.get("sc_rmsd",  float("nan")))]
        ligrmsds = [r["lig_rmsd"] for r in results if np.isfinite(r.get("lig_rmsd", float("nan")))]
        plddts   = [r["plddt"]    for r in results if np.isfinite(r.get("plddt",    float("nan")))]
        iptms    = [r["iptm"]     for r in results if np.isfinite(r.get("iptm",     float("nan")))]
        return {
            f"{tag}_sc_rmsd_best":  min(rmsds)               if rmsds    else float("nan"),
            f"{tag}_sc_rmsd_mean":  float(np.mean(rmsds))    if rmsds    else float("nan"),
            f"{tag}_lig_rmsd_best": min(ligrmsds)             if ligrmsds else float("nan"),
            f"{tag}_lig_rmsd_mean": float(np.mean(ligrmsds)) if ligrmsds else float("nan"),
            f"{tag}_plddt_best":    max(plddts)               if plddts   else float("nan"),
            f"{tag}_plddt_mean":    float(np.mean(plddts))   if plddts   else float("nan"),
            f"{tag}_iptm_best":     max(iptms)                if iptms    else float("nan"),
            f"{tag}_iptm_mean":     float(np.mean(iptms))    if iptms    else float("nan"),
        }

    pmpnn_metrics, lmpnn_metrics = {}, {}
    if not skip_fold and len(prot_ca) > 0:
        base = mpnn_refold_dir or refold_output_dir.parent / "mpnn_refold"
        if mpnn_script and Path(mpnn_script).exists():
            try:
                seqs = run_proteinmpnn(pdb_path, base / pdb_path.stem / "proteinmpnn", mpnn_n_seqs, mpnn_script)
                pmpnn_metrics = _run_mpnn_case(seqs, None, "pmpnn", base / pdb_path.stem / "pmpnn_refold")
            except Exception as e:
                print(f"  ProteinMPNN error {pdb_path.name}: {e}")
        if ligandmpnn_script and Path(ligandmpnn_script).exists():
            try:
                seqs = run_ligandmpnn(pdb_path, base / pdb_path.stem / "ligandmpnn", mpnn_n_seqs, ligandmpnn_script)
                lmpnn_metrics = _run_mpnn_case(seqs, smiles, "lmpnn", base / pdb_path.stem / "lmpnn_refold")
            except Exception as e:
                print(f"  LigandMPNN error {pdb_path.name}: {e}")

    return {
        "sample_id": pdb_path.stem,
        "pdb_path": str(pdb_path),
        "n_ca": len(prot_ca),
        "n_lig_atoms": len(lig_coords),
        "sequence": sequence,
        "frac_lig_contacted": frac_lig_contacted,
        "n_prot_contact_atoms": n_prot_contact,
        **pb,
        **fold,
        **fold_nolig,
        **pmpnn_metrics,
        **lmpnn_metrics,
    }


def _finite(vals):
    out = []
    for v in vals:
        try:
            f = float(v)
            if np.isfinite(f):
                out.append(f)
        except (TypeError, ValueError):
            pass
    return out


def mean_f(vals):
    arr = _finite(vals)
    return float(np.mean(arr)) if arr else float("nan")


def cov_f(vals, d):
    arr = _finite(vals)
    return float(np.mean([v < d for v in arr])) if arr else float("nan")


def fmt(v):
    return "nan" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.3f}"


# ============================================================
# ligand_cond eval — main function
# ============================================================

def run_ligand_cond_eval(args):
    samples_dir = args.samples_dir or (args.out_dir / "samples")

    smiles_by_sid: dict[str, str] = {}
    if args.ref_dir is not None:
        manifest_path = args.ref_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                for rec in json.load(f):
                    sid = rec.get("id", "")
                    s = rec.get("smiles") or rec.get("ligand_smiles") or rec.get("lig_smiles")
                    if sid and s:
                        smiles_by_sid[sid] = s
            print(f"Loaded SMILES for {len(smiles_by_sid)} systems from manifest")
        else:
            print(f"Warning: manifest.json not found at {manifest_path}")

    global_smiles = args.smiles or KNOWN_SMILES.get(getattr(args, "ligand_name", "").upper(), None)
    if not smiles_by_sid and global_smiles is None:
        print("Warning: no SMILES source. Refolding inputs will omit ligand.")

    run_pb = (not args.no_posebusters) and HAS_POSEBUSTERS
    if not args.no_posebusters and not HAS_POSEBUSTERS:
        print("Warning: posebusters not installed — skipping (pip install posebusters to enable).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    refold_input_dir  = args.out_dir / "refold_inputs"
    refold_output_dir = args.out_dir / "refold_outputs"
    per_sample_dir = args.out_dir / "per_sample"
    per_sample_dir.mkdir(exist_ok=True)

    pdb_files = sorted(samples_dir.glob("*.pdb"))
    if not pdb_files:
        sys.exit(f"No PDB files found in {samples_dir}")
    ligand_name = getattr(args, "ligand_name", "LIG")
    continue_run = getattr(args, "continue_run", False)
    import torch as _torch
    _detected = _torch.cuda.device_count() if _torch.cuda.is_available() else 1
    num_gpus = getattr(args, "num_gpus", None) or _detected
    print(f"Evaluating {len(pdb_files)} generated samples for ligand={ligand_name} on {num_gpus} GPU(s)")

    # Filter out already-cached samples when continuing
    todo_files = []
    cached_results = []
    for pdb_path in pdb_files:
        cache_path = per_sample_dir / f"{pdb_path.stem}.json"
        if continue_run and cache_path.exists():
            cached_results.append(json.loads(cache_path.read_text()))
        else:
            todo_files.append(pdb_path)

    def _ser(v):
        if isinstance(v, (np.floating, np.float32, np.float64)): return float(v)
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, np.bool_): return bool(v)
        return v

    # Pre-run PoseBusters in parallel (CPU-only, doesn't compete with GPU Boltz2)
    pb_cache: dict = {}
    if run_pb and todo_files:
        import multiprocessing as _mp2
        n_pb_workers = _mp2.cpu_count()
        print(f"Running PoseBusters on {len(todo_files)} samples ({n_pb_workers} workers)...")
        def _smiles_for(pdb_path):
            m = _GPU_SUFFIX.search(pdb_path.stem)
            sid = pdb_path.stem[:m.start()] if m else pdb_path.stem.rsplit("_", 1)[0]
            return smiles_by_sid.get(sid, global_smiles)
        pb_args = [(p, _smiles_for(p)) for p in todo_files]
        with _mp2.Pool(n_pb_workers) as pool:
            for pdb_str, pb_result in tqdm(
                pool.imap_unordered(_pb_worker, pb_args, chunksize=4),
                total=len(todo_files), desc="posebusters"
            ):
                pb_cache[pdb_str] = pb_result

    def _eval_one(pdb_path, gpu_id):
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        stem = pdb_path.stem
        m = _GPU_SUFFIX.search(stem)
        sid = stem[:m.start()] if m else stem.rsplit("_", 1)[0]
        smiles = smiles_by_sid.get(sid, global_smiles)
        try:
            r = eval_ligand_cond_sample(
                pdb_path=pdb_path, smiles=smiles,
                refold_input_dir=refold_input_dir, refold_output_dir=refold_output_dir,
                boltz_cache=getattr(args, "boltz_cache", None),
                run_pb=run_pb,
                skip_fold=args.no_fold,
                contact_cutoff=getattr(args, "contact_cutoff", 4.0),
                mpnn_script=getattr(args, "mpnn_script", None),
                ligandmpnn_script=getattr(args, "ligandmpnn_script", None),
                mpnn_n_seqs=getattr(args, "mpnn_n_seqs", 3),
                mpnn_refold_dir=args.out_dir / "mpnn_refold",
                pb_cache=pb_cache,
            )
        except Exception as e:
            r = {
                "sample_id": stem, "pdb_path": str(pdb_path), "error": str(e),
                "frac_lig_contacted": float("nan"),
                "plddt": float("nan"), "iptm": float("nan"), "sc_rmsd": float("nan"),
            }
            print(f"  ERROR {pdb_path.name}: {e}")
        (per_sample_dir / f"{stem}.json").write_text(
            json.dumps({k: _ser(v) for k, v in r.items()}, indent=2)
        )
        return r

    if num_gpus <= 1:
        new_results = []
        for pdb_path in tqdm(todo_files, desc="eval"):
            r = _eval_one(pdb_path, gpu_id=0)
            new_results.append(r)
            if args.verbose:
                print(
                    f"  {pdb_path.name}  n_ca={r.get('n_ca')}  "
                    f"lig_contact={fmt(r.get('frac_lig_contacted'))}  "
                    f"plddt={fmt(r.get('plddt'))}  iptm={fmt(r.get('iptm'))}  "
                    f"sc_rmsd={fmt(r.get('sc_rmsd'))}"
                )
    else:
        import multiprocessing as _mp
        import queue as _queue

        work_queue = _mp.Queue()
        result_queue = _mp.Queue()
        for pdb_path in todo_files:
            work_queue.put(pdb_path)
        # sentinel per worker
        for _ in range(num_gpus):
            work_queue.put(None)

        def _worker(gpu_id, work_q, result_q):
            while True:
                pdb_path = work_q.get()
                if pdb_path is None:
                    break
                r = _eval_one(pdb_path, gpu_id=gpu_id)
                result_q.put(r)

        procs = [_mp.Process(target=_worker, args=(i, work_queue, result_queue), daemon=True)
                 for i in range(num_gpus)]
        for p in procs:
            p.start()
        new_results = []
        for _ in tqdm(range(len(todo_files)), desc="eval"):
            r = result_queue.get()
            new_results.append(r)
            if args.verbose:
                print(
                    f"  n_ca={r.get('n_ca')}  "
                    f"lig_contact={fmt(r.get('frac_lig_contacted'))}  "
                    f"plddt={fmt(r.get('plddt'))}  iptm={fmt(r.get('iptm'))}  "
                    f"sc_rmsd={fmt(r.get('sc_rmsd'))}"
                )
        for p in procs:
            p.join()

    all_results = cached_results + new_results

    serial = [{k: _ser(v) for k, v in r.items()} for r in all_results]
    (args.out_dir / "results.json").write_text(json.dumps(serial, indent=2))

    frac_c        = [r.get("frac_lig_contacted") for r in all_results]
    plddts        = [r.get("plddt")         for r in all_results]
    iptms         = [r.get("iptm")          for r in all_results]
    scrmsds       = [r.get("sc_rmsd")       for r in all_results]
    ligrmsds      = [r.get("lig_rmsd")      for r in all_results]
    nolig_plddts  = [r.get("nolig_plddt")   for r in all_results]
    nolig_iptms   = [r.get("nolig_iptm")    for r in all_results]
    nolig_scrmsds = [r.get("nolig_sc_rmsd") for r in all_results]

    contact_cutoff = getattr(args, "contact_cutoff", 4.0)
    sc_deltas = getattr(args, "delta", [2.0, 5.0])

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  LIGAND-CONDITIONED PROTEIN BINDER EVAL")
    lines.append(f"  ligand={ligand_name}  n={len(all_results)} samples")
    lines.append(f"{'='*60}")
    lines.append(f"\n--- Pocket contacts (cutoff={contact_cutoff:.1f} Å) ---")
    lines.append(f"  frac ligand atoms contacted : {fmt(mean_f(frac_c))}")
    lines.append(f"  samples with >50% lig contact: "
                 f"{sum(1 for v in _finite(frac_c) if v > 0.5)}/{len(all_results)}")
    lines.append(f"\n--- Refolding self-consistency — model sequence (Boltz2) ---")
    if args.no_fold:
        lines.append("  [skipped — remove --no-fold to enable]")
    elif _finite(plddts):
        lines.append(f"  pLDDT mean    : {fmt(mean_f(plddts))}")
        lines.append(f"  ipTM  mean    : {fmt(mean_f(iptms))}")
        lines.append(f"  scRMSD mean   : {fmt(mean_f(scrmsds))} Å")
        for d in sc_deltas:
            lines.append(f"  COV sc_rmsd < {d:.1f} Å : {cov_f(scrmsds, d)*100:.1f}%")
        if _finite(ligrmsds):
            lines.append(f"  ligRMSD mean  : {fmt(mean_f(ligrmsds))} Å")
            for d in sc_deltas:
                lines.append(f"  COV lig_rmsd < {d:.1f} Å : {cov_f(ligrmsds, d)*100:.1f}%")
    else:
        lines.append("  [all NaN — check boltz errors in results.json]")

    lines.append(f"\n--- Refolding self-consistency — model sequence w/o ligand (Boltz2) ---")
    if args.no_fold:
        lines.append("  [skipped — remove --no-fold to enable]")
    elif _finite(nolig_plddts):
        lines.append(f"  pLDDT mean    : {fmt(mean_f(nolig_plddts))}")
        lines.append(f"  ipTM  mean    : {fmt(mean_f(nolig_iptms))}")
        lines.append(f"  scRMSD mean   : {fmt(mean_f(nolig_scrmsds))} Å")
        for d in sc_deltas:
            lines.append(f"  COV sc_rmsd < {d:.1f} Å : {cov_f(nolig_scrmsds, d)*100:.1f}%")
    else:
        lines.append("  [all NaN — check boltz errors in results.json]")

    n_mpnn = getattr(args, "mpnn_n_seqs", 3)
    for tag, label, with_lig in [
        ("pmpnn", f"ProteinMPNN → refold w/o ligand (n={n_mpnn})", False),
        ("lmpnn", f"LigandMPNN  → refold w/  ligand (n={n_mpnn})", True),
    ]:
        best_sc    = [r.get(f"{tag}_sc_rmsd_best")  for r in all_results]
        mean_sc    = [r.get(f"{tag}_sc_rmsd_mean")  for r in all_results]
        best_lig   = [r.get(f"{tag}_lig_rmsd_best") for r in all_results]
        mean_lig   = [r.get(f"{tag}_lig_rmsd_mean") for r in all_results]
        best_plddt = [r.get(f"{tag}_plddt_best")    for r in all_results]
        avg_plddt  = [r.get(f"{tag}_plddt_mean")    for r in all_results]
        best_iptm  = [r.get(f"{tag}_iptm_best")     for r in all_results]
        avg_iptm   = [r.get(f"{tag}_iptm_mean")     for r in all_results]
        if _finite(best_sc):
            lines.append(f"\n--- Refolding self-consistency — {label} ---")
            lines.append(f"  pLDDT best : {fmt(mean_f(best_plddt))}   avg : {fmt(mean_f(avg_plddt))}")
            lines.append(f"  ipTM  best : {fmt(mean_f(best_iptm))}   avg : {fmt(mean_f(avg_iptm))}")
            lines.append(f"  scRMSD best : {fmt(mean_f(best_sc))} Å   avg : {fmt(mean_f(mean_sc))} Å")
            for d in sc_deltas:
                lines.append(f"  COV sc_rmsd < {d:.1f} Å (best) : {cov_f(best_sc, d)*100:.1f}%")
                lines.append(f"  COV sc_rmsd < {d:.1f} Å (avg)  : {cov_f(mean_sc, d)*100:.1f}%")
            if _finite(best_lig):
                lines.append(f"  ligRMSD best : {fmt(mean_f(best_lig))} Å   avg : {fmt(mean_f(mean_lig))} Å")
                for d in sc_deltas:
                    lines.append(f"  COV lig_rmsd < {d:.1f} Å (best) : {cov_f(best_lig, d)*100:.1f}%")
                    lines.append(f"  COV lig_rmsd < {d:.1f} Å (avg)  : {cov_f(mean_lig, d)*100:.1f}%")

    pb_keys = [k for k in (all_results[0] if all_results else {}) if k.startswith("pb_")]
    if pb_keys:
        lines.append(f"\n--- PoseBusters ---")
        for k in pb_keys:
            vals = [r.get(k) for r in all_results if isinstance(r.get(k), bool)]
            if vals:
                pass_rate = sum(vals) / len(vals)
                lines.append(f"  {k:<40}: {pass_rate*100:.1f}% pass ({len(vals)} samples)")
    else:
        lines.append(f"\n--- PoseBusters ---")
        lines.append("  [not run — add --posebusters flag]")

    summary = "\n".join(lines) + "\n"
    print("\n" + summary)
    (args.out_dir / "summary.txt").write_text(summary)
    print(f"Results: {args.out_dir / 'results.json'}")
    print(f"Summary: {args.out_dir / 'summary.txt'}")
    if not args.no_fold:
        print(f"Refold inputs:  {refold_input_dir}/")
        print(f"Refold outputs: {refold_output_dir}/")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task", choices=["protein_cond", "ligand_cond"], required=True,
        help="Evaluation task: protein_cond (fixed protein, gen ligand) or "
             "ligand_cond (fixed ligand, gen protein binder)",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Output directory. For protein_cond: sample.py output dir (contains samples/). "
             "For ligand_cond: eval output directory.",
    )
    parser.add_argument(
        "--ref-dir", type=Path, default=None,
        help="Plinder split directory containing manifest.json + structures/. "
             "For ligand_cond, also used for per-system SMILES lookup.",
    )
    # ligand_cond input
    parser.add_argument(
        "--samples-dir", type=Path, default=None,
        help="[ligand_cond] Directory of generated PDB files. "
             "Defaults to {out_dir}/samples if not set.",
    )
    # protein_cond options
    parser.add_argument(
        "--max-protein-residues", type=int, default=None,
        help="[protein_cond] Crop GT protein to this many residues when writing aligned PDBs "
             "(default: None = no crop).",
    )
    parser.add_argument(
        "--include-h", action="store_true",
        help="[protein_cond] Keep hydrogen atoms (match the flag used at sample time).",
    )
    parser.add_argument(
        "--pocket-deltas", type=float, nargs="+", default=[2.0, 5.0],
        help="[protein_cond] RMSD thresholds for pocket coverage reporting (Å; default: 2.0 5.0).",
    )
    parser.add_argument(
        "--plip-sif", type=Path, default=Path("plip.sif"),
        help="[protein_cond] Path to PLIP Singularity image (default: ./plip.sif).",
    )
    # ligand_cond options
    parser.add_argument(
        "--ligand-name", default="LIG",
        help="[ligand_cond] Short name for the ligand, e.g. SAM, FAD (default: LIG).",
    )
    parser.add_argument(
        "--smiles", default=None,
        help="[ligand_cond] SMILES for the target ligand. Falls back to built-in table "
             "if --ligand-name matches a known cofactor.",
    )
    parser.add_argument(
        "--no-fold", action="store_true",
        help="[ligand_cond] Skip Boltz2 refolding (contact-only eval).",
    )
    parser.add_argument(
        "--boltz-cache", type=Path, default=None,
        help="[ligand_cond] Boltz2 cache directory.",
    )
    parser.add_argument(
        "--no-posebusters", action="store_true",
        help="Skip PoseBusters (runs by default if posebusters is installed).",
    )
    parser.add_argument(
        "--contact-cutoff", type=float, default=4.0,
        help="[ligand_cond] Distance cutoff (Å) for pocket contact analysis (default: 4.0).",
    )
    parser.add_argument(
        "--delta", type=float, nargs="+", default=[2.0, 5.0],
        help="[ligand_cond] scRMSD thresholds for coverage (Å; default: 2.0 5.0).",
    )
    _repo_root = Path(__file__).parent.parent
    _default_ligandmpnn = str(_repo_root / "LigandMPNN" / "run.py")
    parser.add_argument(
        "--mpnn-script", type=str, default=_default_ligandmpnn,
        help="[ligand_cond] Path to LigandMPNN run.py for protein_mpnn mode (default: <repo>/LigandMPNN/run.py).",
    )
    parser.add_argument(
        "--ligandmpnn-script", type=str, default=_default_ligandmpnn,
        help="[ligand_cond] Path to LigandMPNN run.py for ligand_mpnn mode (default: <repo>/LigandMPNN/run.py).",
    )
    parser.add_argument(
        "--mpnn-n-seqs", type=int, default=3,
        help="[ligand_cond] Sequences per structure for ProteinMPNN/LigandMPNN (default: 3).",
    )
    # common
    parser.add_argument(
        "--n-jobs", type=int, default=max(1, mp.cpu_count() // 2),
        help="[protein_cond] Parallel workers (default: half of CPU count).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-system / per-sample details.",
    )
    parser.add_argument(
        "--continue-run", action="store_true", default=False,
        help="Skip already-evaluated samples (cached in out_dir/per_sample/). "
             "Default: False (wipe out_dir and rerun everything).",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None,
        help="[ligand_cond] Number of GPUs to use for parallel Boltz refolding. "
             "Each GPU runs as a separate worker process. Default: 1.",
    )
    args = parser.parse_args()

    if not args.out_dir.exists() and args.task == "protein_cond":
        sys.exit(f"--out-dir not found: {args.out_dir}")

    if args.ref_dir is None:
        sys.exit("--ref-dir is required")

    continue_run = getattr(args, "continue_run", False)
    if not continue_run and args.out_dir.exists():
        import shutil
        # Only remove eval artifacts, never the samples directory
        samples_dir_default = args.out_dir / "samples"
        for subdir in ["refold_inputs", "refold_outputs", "per_sample",
                       "aligned_pocket", "aligned_lig"]:
            p = args.out_dir / subdir
            if p.exists():
                shutil.rmtree(p)
        for f in ["results.json", "summary.txt"]:
            p = args.out_dir / f
            if p.exists():
                p.unlink()
        print(f"Cleared eval artifacts in: {args.out_dir}")

    if args.task == "protein_cond":
        run_protein_cond_eval(args)
    else:
        run_ligand_cond_eval(args)


if __name__ == "__main__":
    main()
