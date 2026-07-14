"""
Filter Plinder systems by pocket quality, producing a text file of passing system IDs.

Two-phase pipeline:
  Phase 1 (fast): annotation table pre-filter — crystal contacts, pocket residue count,
                  electron density quality.
  Phase 2 (structure): per-system structure-level filters:
      - fpocket: druggability score, alpha sphere density (concavity proxy), volume
      - freesasa: buried fraction of pocket residues

Output: a text file with one passing system_id per line, loadable by plinder.py via
--system-ids-file.

Usage:
    python filter_plinder_pocket.py \\
        --plinder-dir /path/to/plinder/2024-06/v2 \\
        --outfile pocket_filtered_ids.txt \\
        [--splits train val] \\
        [--num-processes 16]
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Optional
import multiprocessing

import numpy as np
import pyarrow.parquet as pq
import gemmi
import freesasa
from tqdm import tqdm


# ── annotation table ─────────────────────────────────────────────────────────

def load_annotation_table(plinder_dir: Path, columns: list[str]) -> dict:
    """Return {system_id: {col: val}} for requested columns."""
    path = plinder_dir / "index" / "annotation_table.parquet"
    all_cols = ["system_id"] + [c for c in columns if c != "system_id"]
    t = pq.ParquetFile(path).read(columns=all_cols)
    ids = t["system_id"].to_pylist()
    result = {}
    for i, sid in enumerate(ids):
        result[sid] = {c: t[c][i].as_py() for c in all_cols}
    return result


def annotation_prefilter(row: dict, args) -> bool:
    """Return True if the system passes annotation-table thresholds."""
    # Must be a proper ligand (not artifact/buffer/cofactor)
    if not row.get("ligand_is_proper"):
        return False

    # Exclude covalently bonded ligands
    if row.get("ligand_is_covalent"):
        return False

    # Ligand must be buried enough — crystal contacts indicate surface exposure
    frac_cc = row.get("ligand_fraction_atoms_with_crystal_contacts")
    if frac_cc is not None and frac_cc > args.max_crystal_contacts:
        return False

    # Pocket must have enough residues
    n_pocket = row.get("ligand_num_pocket_residues")
    if n_pocket is not None and n_pocket < args.min_pocket_res:
        return False

    # Pocket electron density quality
    rscc = row.get("system_pocket_validation_average_rscc")
    if rscc is not None and rscc < args.min_rscc:
        return False

    return True


# ── fpocket ───────────────────────────────────────────────────────────────────

_CHAIN_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _remap_chains(st) -> dict[str, str]:
    """Remap multi-char chain names to single-char in-place. Returns old→new mapping."""
    mapping: dict[str, str] = {}
    idx = 0
    for model in st:
        for chain in model:
            if chain.name not in mapping:
                if len(chain.name) > 1:
                    while idx < len(_CHAIN_CHARS) and _CHAIN_CHARS[idx] in mapping.values():
                        idx += 1
                    mapping[chain.name] = _CHAIN_CHARS[idx] if idx < len(_CHAIN_CHARS) else chain.name[0]
                    idx += 1
                else:
                    mapping[chain.name] = chain.name
            chain.name = mapping[chain.name]
    return mapping


def _cif_to_pdb(cif_path: Path, pdb_path: Path) -> bool:
    """Convert mmCIF to PDB using gemmi. Returns False on failure."""
    try:
        st = gemmi.read_structure(str(cif_path))
        st.remove_hydrogens()
        _remap_chains(st)
        st.write_pdb(str(pdb_path))
        return True
    except Exception:
        return False


def _parse_fpocket_info(info_path: Path) -> list[dict]:
    """Parse fpocket *_info.txt into a list of pocket dicts."""
    pockets = []
    current: Optional[dict] = None
    with open(info_path) as f:
        for line in f:
            m = re.match(r"^Pocket\s+(\d+)\s+:", line)
            if m:
                current = {"pocket_id": int(m.group(1))}
                pockets.append(current)
                continue
            if current is None:
                continue
            # Parse "Key : value" lines
            kv = re.match(r"^([^:]+):\s*([\d.eE+-]+)", line)
            if kv:
                key = kv.group(1).strip()
                try:
                    val = float(kv.group(2))
                except ValueError:
                    val = None
                current[key] = val
    return pockets


def _parse_pocket_pqr(pqr_path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Parse fpocket pockets.pqr.

    Returns (centers, raw_coords) where:
      centers[pocket_id]    = mean alpha-sphere center (3,)
      raw_coords[pocket_id] = all alpha-sphere positions (N, 3)

    fpocket encodes pocket membership in the residue number field (no chain column).
    """
    # fpocket pqr columns: record serial name resname resseq x    y    z    charge radius
    # indices:             0      1      2    3       4      5    6    7    8      9
    coords_by_pocket: dict[int, list] = {}
    with open(pqr_path) as f:
        for line in f:
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                pocket_id = int(parts[4])
                x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
            except (ValueError, IndexError):
                continue
            coords_by_pocket.setdefault(pocket_id, []).append([x, y, z])
    centers = {pid: np.mean(pts, axis=0) for pid, pts in coords_by_pocket.items()}
    raw = {pid: np.array(pts, dtype=np.float32) for pid, pts in coords_by_pocket.items()}
    return centers, raw


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _fpocket_cmd(args) -> list[str]:
    sif = Path(args.fpocket_sif)
    runner = shutil.which("singularity") or shutil.which("apptainer") or "singularity"
    return [runner, "exec", str(sif), "fpocket"]


def run_fpocket(
    receptor_cif: Path, ligand_centroid: np.ndarray, args
) -> tuple[Optional[dict], Optional[np.ndarray]]:
    """Run fpocket on receptor_cif, return (pocket_metrics, alpha_sphere_coords).

    alpha_sphere_coords is shape (N, 3) for the matched pocket, or None on failure.
    """
    cmd_prefix = _fpocket_cmd(args)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pdb_path = tmpdir / "receptor.pdb"
        if not _cif_to_pdb(receptor_cif, pdb_path):
            return None, None

        try:
            result = subprocess.run(
                cmd_prefix + ["-f", str(pdb_path)],
                capture_output=True,
                cwd=tmpdir,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return None, None
        if result.returncode != 0:
            return None, None

        out_dir = tmpdir / "receptor_out"
        info_path = out_dir / "receptor_info.txt"
        pqr_path = out_dir / "receptor_pockets.pqr"

        if not info_path.exists():
            return None, None

        pockets = _parse_fpocket_info(info_path)
        if not pockets:
            return None, None

        # Get pocket centers and raw alpha-sphere coords
        if pqr_path.exists():
            pocket_centers, pocket_raw_coords = _parse_pocket_pqr(pqr_path)
        else:
            pocket_centers, pocket_raw_coords = {}, {}

        best_pocket = None
        best_dist = float("inf")
        for p in pockets:
            pid = p["pocket_id"]
            center = pocket_centers.get(pid)
            if center is None:
                continue
            dist = np.linalg.norm(center - ligand_centroid)
            if dist < best_dist:
                best_dist = dist
                best_pocket = p

        if best_pocket is None or best_dist > args.max_pocket_center_dist:
            return None, None

        alpha_coords = pocket_raw_coords.get(best_pocket["pocket_id"])
        return best_pocket, alpha_coords


def fpocket_filter(pocket: dict, args) -> bool:
    """Return True if fpocket metrics for this pocket pass thresholds."""
    drugg = pocket.get("Druggability Score")
    density = pocket.get("Alpha sphere density")
    volume = pocket.get("Volume")

    if drugg is not None and drugg < args.min_druggability:
        return False
    if density is not None and density < args.min_concavity:
        return False
    if volume is not None and volume < args.min_volume:
        return False

    return True


# ── freesasa: buried fraction ─────────────────────────────────────────────────

def compute_buried_fraction(
    receptor_cif: Path,
    ligand_centroid: np.ndarray,
    pocket_radius: float,
) -> Optional[float]:
    """Compute buried fraction of pocket residues using freesasa.

    buried_fraction = 1 - (pocket_sasa_in_protein / pocket_sasa_isolated)
    Higher = more buried = more concave pocket.
    """
    try:
        st = gemmi.read_structure(str(receptor_cif))
        if not st or not st[0]:
            return None
        model = st[0]

        # Remap chains before writing PDB (PDB requires single-char chain IDs)
        chain_map = _remap_chains(st)

        # Find pocket residues: all protein residues within pocket_radius of ligand centroid
        # Store as (remapped_chain_name, resseq_str) pairs for freesasa selection
        pocket_res: list[tuple[str, str]] = []
        for chain in model:
            for res in chain:
                coords = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res if not a.is_hydrogen()])
                if len(coords) == 0:
                    continue
                if np.linalg.norm(coords - ligand_centroid[None], axis=-1).min() < pocket_radius:
                    pocket_res.append((chain.name, str(res.seqid.num)))

        if not pocket_res:
            return None

        # Convert to PDB for freesasa (freesasa doesn't support mmCIF)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tf:
            full_pdb = tf.name
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tf:
            iso_pdb = tf.name

        try:
            st.write_pdb(full_pdb)

            # Full-protein SASA, accumulate pocket residue contribution via residueAreas()
            # (avoids selectArea string-matching issues with insertion codes / auth numbering)
            freesasa.setVerbosity(freesasa.silent)
            structure_full = freesasa.Structure(full_pdb)
            result_full = freesasa.calc(structure_full)
            pocket_set = {(c, r) for c, r in pocket_res}
            per_res = result_full.residueAreas()  # {chain: {resi_str: ResidueArea}}
            sasa_in_protein = sum(
                area.total
                for chain_id, chain_areas in per_res.items()
                for resi_str, area in chain_areas.items()
                if (chain_id, resi_str.strip()) in pocket_set
            )
            if sasa_in_protein == 0.0:
                return None

            # Isolated pocket residues SASA
            mini_st = gemmi.Structure()
            mini_model = gemmi.Model("1")
            for chain in model:
                mini_chain = gemmi.Chain(chain.name)
                for res in chain:
                    if (chain.name, str(res.seqid.num)) in pocket_set:
                        mini_chain.add_residue(res.clone())
                if len(mini_chain) > 0:
                    mini_model.add_chain(mini_chain)
            mini_st.add_model(mini_model)
            mini_st.write_pdb(iso_pdb)

            structure_iso = freesasa.Structure(iso_pdb)
            result_iso = freesasa.calc(structure_iso)
            sasa_isolated = result_iso.totalArea()
        finally:
            os.unlink(full_pdb)
            os.unlink(iso_pdb)

        if sasa_isolated < 1.0:
            return None

        return float(1.0 - sasa_in_protein / sasa_isolated)

    except Exception:
        return None


# ── per-system check ──────────────────────────────────────────────────────────

def check_system(
    system_id: str, plinder_dir: Path, args, verbose: bool = False
) -> tuple[bool, Optional[np.ndarray]]:
    """Return (passed, alpha_sphere_coords) for the system.

    alpha_sphere_coords is shape (N, 3) for the matched fpocket pocket, or None.
    """
    def log(msg):
        if verbose:
            print(f"  [{system_id}] {msg}")

    system_dir = plinder_dir / "systems" / system_id
    receptor_cif = system_dir / "receptor.cif"
    system_cif = system_dir / "system.cif"

    if not receptor_cif.exists() or not system_cif.exists():
        log(f"missing files (receptor={receptor_cif.exists()}, system={system_cif.exists()})")
        return False, None

    # Compute ligand centroid from system.cif
    try:
        st = gemmi.read_structure(str(system_cif))
        ligand_coords = []
        for chain in st[0]:
            for res in chain:
                if res.entity_type == gemmi.EntityType.NonPolymer:
                    for atom in res:
                        if not atom.is_hydrogen():
                            ligand_coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
        if not ligand_coords:
            log("no ligand atoms found (entity_type check)")
            return False, None
        ligand_centroid = np.mean(ligand_coords, axis=0)
        log(f"ligand centroid={ligand_centroid.round(2)}, n_atoms={len(ligand_coords)}")
    except Exception as e:
        log(f"ligand centroid failed: {e}")
        return False, None

    # fpocket filter
    alpha_coords = None
    if args.min_druggability > 0 or args.min_concavity > 0 or args.min_volume > 0:
        pocket, alpha_coords = run_fpocket(receptor_cif, ligand_centroid, args)
        if pocket is None:
            log("fpocket returned None (no matching pocket or fpocket failed)")
            return False, None
        log(f"fpocket pocket: drugg={pocket.get('Druggability Score')}, density={pocket.get('Alpha sphere density')}, vol={pocket.get('Volume')}")
        if not fpocket_filter(pocket, args):
            log("failed fpocket thresholds")
            return False, None

    # freesasa filter
    if args.min_buried_fraction > 0:
        buried = compute_buried_fraction(receptor_cif, ligand_centroid, args.pocket_radius)
        log(f"buried_fraction={buried}")
        if buried is None or buried < args.min_buried_fraction:
            log("failed buried fraction")
            return False, None

    return True, alpha_coords


# ── worker ────────────────────────────────────────────────────────────────────

_state: dict = {}


def _worker_init(plinder_dir, args, pocket_data_dir=None):
    _state["plinder_dir"] = plinder_dir
    _state["args"] = args
    _state["pocket_data_dir"] = pocket_data_dir


def _system_mid(system_id: str) -> str:
    return system_id[1:3]


def _worker(system_id: str) -> Optional[str]:
    try:
        passed, alpha_coords = check_system(system_id, _state["plinder_dir"], _state["args"])
        if passed:
            if _state.get("pocket_data_dir") is not None and alpha_coords is not None:
                mid = _system_mid(system_id)
                out_dir = Path(_state["pocket_data_dir"]) / mid
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / f"{system_id}.npy", alpha_coords)
            return system_id
    except Exception:
        traceback.print_exc()
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Filter Plinder systems by pocket quality.")
    parser.add_argument("--plinder-dir", type=Path, required=True,
                        help="Plinder data root (e.g. /path/to/plinder/2024-06/v2)")
    parser.add_argument("--outfile", type=Path, required=True,
                        help="Output text file: one passing system_id per line")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Which splits to consider (default: train val test)")
    parser.add_argument("--num-processes", type=int, default=min(8, multiprocessing.cpu_count()))

    # Phase 1: annotation table thresholds
    parser.add_argument("--max-crystal-contacts", type=float, default=0.3,
                        help="Max fraction of ligand atoms with crystal contacts (0=skip)")
    parser.add_argument("--min-pocket-res", type=int, default=6,
                        help="Min number of pocket residues (0=skip)")
    parser.add_argument("--min-rscc", type=float, default=0.7,
                        help="Min average RSCC for pocket residues (0=skip)")

    # Phase 2: fpocket thresholds (set to 0 to disable)
    parser.add_argument("--min-druggability", type=float, default=0.3,
                        help="Min fpocket druggability score")
    parser.add_argument("--min-concavity", type=float, default=0.1,
                        help="Min fpocket alpha sphere density (concavity proxy)")
    parser.add_argument("--min-volume", type=float, default=100.0,
                        help="Min fpocket pocket volume (Å³)")
    parser.add_argument("--max-pocket-center-dist", type=float, default=8.0,
                        help="Max distance (Å) from ligand centroid to fpocket pocket center")

    # Phase 3: freesasa thresholds (set to 0 to disable)
    parser.add_argument("--min-buried-fraction", type=float, default=0.2,
                        help="Min buried fraction of pocket residues (0=skip freesasa)")
    parser.add_argument("--pocket-radius", type=float, default=8.0,
                        help="Radius (Å) around ligand centroid defining pocket residues for SASA")
    parser.add_argument("--fpocket-sif", type=str,
                        default=str(_REPO_ROOT / "fpocket.sif"),
                        help="Path to fpocket Singularity image (default: {repo_root}/fpocket.sif)")
    parser.add_argument("--debug-n", type=int, default=0,
                        help="Run verbosely on the first N candidates and exit (for debugging)")
    parser.add_argument("--pocket-data-dir", type=Path,
                        default=_REPO_ROOT / "plinder_pocket_alpha_spheres",
                        help="Directory to save per-system alpha-sphere coords (*.npy) for pocket-aware cropping")

    args = parser.parse_args()

    # Load annotation table
    print("Loading annotation table...")
    ann_cols = [
        "system_id",
        "ligand_is_proper",
        "ligand_is_covalent",
        "ligand_fraction_atoms_with_crystal_contacts",
        "ligand_num_pocket_residues",
        "system_pocket_validation_average_rscc",
    ]
    annotations = load_annotation_table(args.plinder_dir, ann_cols)
    print(f"  {len(annotations)} systems in annotation table")

    # Load split
    split_path = args.plinder_dir / "splits" / "split.parquet"
    t = pq.ParquetFile(split_path).read(columns=["system_id", "split"])
    split_map = dict(zip(t["system_id"].to_pylist(), t["split"].to_pylist()))

    # Phase 1: annotation pre-filter
    candidate_ids = []
    for sid, row in annotations.items():
        if split_map.get(sid) not in args.splits:
            continue
        if annotation_prefilter(row, args):
            candidate_ids.append(sid)

    print(f"  {len(candidate_ids)} pass annotation pre-filter")

    # Phase 2+3: structure-level filters
    skip_structure = (
        args.min_druggability <= 0
        and args.min_concavity <= 0
        and args.min_volume <= 0
        and args.min_buried_fraction <= 0
    )

    pocket_data_dir = args.pocket_data_dir
    if pocket_data_dir is not None:
        pocket_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Alpha-sphere sidecars will be saved to: {pocket_data_dir}")

    if args.debug_n > 0:
        print(f"\nDebug mode: checking first {args.debug_n} candidates verbosely")
        for sid in candidate_ids[:args.debug_n]:
            print(f"\n--- {sid} ---")
            passed, alpha_coords = check_system(sid, args.plinder_dir, args, verbose=True)
            print(f"  => {'PASS' if passed else 'FAIL'}")
        return

    if skip_structure:
        passing_ids = candidate_ids
        print("  (Structure-level filters disabled — using annotation result directly)")
    else:
        print("Running structure-level filters (fpocket + freesasa)...")
        n = min(args.num_processes, len(candidate_ids))
        if n > 1:
            with multiprocessing.Pool(
                processes=n,
                initializer=_worker_init,
                initargs=(args.plinder_dir, args, pocket_data_dir),
            ) as pool:
                results = list(tqdm(
                    pool.imap_unordered(_worker, candidate_ids, chunksize=4),
                    total=len(candidate_ids),
                ))
        else:
            _worker_init(args.plinder_dir, args, pocket_data_dir)
            results = [_worker(sid) for sid in tqdm(candidate_ids)]

        passing_ids = [sid for sid in results if sid is not None]

    print(f"\n{len(passing_ids)} systems pass all filters")

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(args.outfile, "w") as f:
        for sid in sorted(passing_ids):
            f.write(sid + "\n")
    print(f"Written to {args.outfile}")


if __name__ == "__main__":
    main()
