"""
Process Plinder protein-ligand dataset into proteinzen npz format.

Each Plinder system directory contains:
  receptor.cif          - protein receptor (mmCIF)
  system.cif            - full complex (mmCIF, protein + ligand)
  ligand_files/*.sdf    - one SDF per ligand chain
  chain_mapping.json    - chain metadata

Output per system:
  structures/{mid}/{system_id}.npz   - Structure arrays + rot_bond data
  records/{mid}/{system_id}.json     - Record metadata

Finalize step merges records into manifest.json.
"""

import argparse
import json
import multiprocessing
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from p_tqdm import p_umap
from tqdm import tqdm

from mmcif import parse_mmcif

from proteinzen.boltz.data import const
from proteinzen.boltz.data.types import (
    AffinityInfo,
    ChainInfo,
    InterfaceInfo,
    Record,
    StructureInfo,
)
from proteinzen.data.featurize.mol.sampling import (
    compute_ring_atom_masks,
    compute_rot_bond_fragments,
    compute_sym_groups,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def system_mid(system_id: str) -> str:
    """Two-char subdirectory key derived from PDB ID (chars 1-2 of system_id)."""
    return system_id[1:3]


def load_clusters(plinder_dir: Path, metric: str = "pocket_fident_qcov", threshold: int = 30) -> dict:
    """Load cluster parquet and return {system_id: cluster_label} dict.

    Cluster parquet has columns: system_id, metric, cluster, directed, threshold, label
    We filter to the requested metric/threshold and use 'cluster' as the cluster ID.
    """
    cluster_dir = plinder_dir / "clusters"
    parquet_files = list(cluster_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"Warning: no cluster parquet files found under {cluster_dir}")
        return {}

    frames = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f, columns=["system_id", "metric", "cluster", "threshold"])
            frames.append(df)
        except Exception:
            continue

    if not frames:
        print("Warning: could not load any cluster parquet files")
        return {}

    df = pd.concat(frames, ignore_index=True)
    df = df[(df["metric"] == metric) & (df["threshold"] == threshold)]
    return dict(zip(df["system_id"], df["cluster"]))


def load_annotation_table(plinder_dir: Path) -> pd.DataFrame:
    """Load the main annotation table."""
    path = plinder_dir / "index" / "annotation_table.parquet"
    return pd.read_parquet(path)


def load_split(plinder_dir: Path) -> pd.DataFrame:
    """Load the split definitions."""
    path = plinder_dir / "splits" / "split.parquet"
    return pd.read_parquet(path)


def get_ligand_sdfs(system_dir: Path) -> dict:
    """Return {ligand_name: Path} for all SDF files in ligand_files/."""
    ligand_dir = system_dir / "ligand_files"
    if not ligand_dir.exists():
        return {}
    return {f.stem: f for f in ligand_dir.glob("*.sdf")}


def compute_rot_bond_data(mol: Chem.Mol, atom_offset: int, n_total_atoms: int) -> dict:
    """Compute rot_bond arrays for a ligand mol, offsetting into the global atom array."""
    rot_bonds, frag_a = compute_rot_bond_fragments(mol)  # (B,2), (B, n_lig)
    ring_masks = compute_ring_atom_masks(mol)              # (R, n_lig)
    sym_groups, sym_group_sizes = compute_sym_groups(mol)  # (G, max_sz), (G,)

    n_lig = mol.GetNumAtoms()
    B = rot_bonds.shape[0]
    R = ring_masks.shape[0]

    # Offset rot_bonds atom indices into global atom array
    if B > 0:
        rot_bonds_global = rot_bonds + atom_offset
        frag_a_global = np.zeros((B, n_total_atoms), dtype=bool)
        frag_a_global[:, atom_offset:atom_offset + n_lig] = frag_a
    else:
        rot_bonds_global = np.zeros((0, 2), dtype=np.int32)
        frag_a_global = np.zeros((0, n_total_atoms), dtype=bool)

    if R > 0:
        ring_masks_global = np.zeros((R, n_total_atoms), dtype=bool)
        ring_masks_global[:, atom_offset:atom_offset + n_lig] = ring_masks
    else:
        ring_masks_global = np.zeros((0, n_total_atoms), dtype=bool)

    return {
        "rot_bonds": rot_bonds_global,
        "rot_frag_a": frag_a_global,
        "ring_masks": ring_masks_global,
        "sym_groups": sym_groups,
        "sym_group_sizes": sym_group_sizes,
    }


def merge_rot_bond_data(all_data: list, n_total_atoms: int) -> dict:
    """Concatenate rot_bond data from multiple ligands."""
    if not all_data:
        return {
            "rot_bonds": np.zeros((0, 2), dtype=np.int32),
            "rot_frag_a": np.zeros((0, n_total_atoms), dtype=bool),
            "ring_masks": np.zeros((0, n_total_atoms), dtype=bool),
            "sym_groups": np.zeros((0, 1), dtype=np.int32),
            "sym_group_sizes": np.zeros(0, dtype=np.int32),
        }

    rot_bonds = np.concatenate([d["rot_bonds"] for d in all_data], axis=0)
    rot_frag_a = np.concatenate([d["rot_frag_a"] for d in all_data], axis=0)
    ring_masks = np.concatenate([d["ring_masks"] for d in all_data], axis=0)

    # sym_groups: take from first ligand with any groups, or empty
    sym_groups = np.zeros((0, 1), dtype=np.int32)
    sym_group_sizes = np.zeros(0, dtype=np.int32)
    for d in all_data:
        if d["sym_group_sizes"].shape[0] > 0:
            sym_groups = d["sym_groups"]
            sym_group_sizes = d["sym_group_sizes"]
            break

    return {
        "rot_bonds": rot_bonds,
        "rot_frag_a": rot_frag_a,
        "ring_masks": ring_masks,
        "sym_groups": sym_groups,
        "sym_group_sizes": sym_group_sizes,
    }


# ── per-system processing ─────────────────────────────────────────────────────

def process_system(
    system_id: str,
    plinder_dir: Path,
    outdir: Path,
    clusters: dict,
    annotation_row: dict,
) -> None:
    mid = system_mid(system_id)
    struct_path = outdir / "structures" / mid / f"{system_id}.npz"
    record_path = outdir / "records" / mid / f"{system_id}.json"

    if struct_path.exists() and record_path.exists():
        return

    system_dir = plinder_dir / "systems" / mid / system_id
    cif_path = system_dir / "system.cif"
    if not cif_path.exists():
        return

    try:
        # Parse full complex (protein + ligand chains via CCD)
        parsed = parse_mmcif(str(cif_path), components={}, ignore_connections=False)
        structure = parsed.data
    except Exception:
        traceback.print_exc()
        print(f"Failed to parse {system_id}")
        return

    # Load ligand SDF files
    ligand_sdfs = get_ligand_sdfs(system_dir)
    n_total_atoms = len(structure.atoms)
    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]
    protein_id = const.chain_type_ids["PROTEIN"]

    # Compute rot_bond data for each NONPOLYMER chain
    all_rot_bond_data = []
    for chain in structure.chains:
        if int(chain["mol_type"]) != nonpolymer_id:
            continue

        chain_name = chain["name"].strip()
        atom_offset = int(chain["atom_idx"])

        # Find matching SDF — try by chain name, then take first available
        sdf_path = ligand_sdfs.get(chain_name)
        if sdf_path is None and ligand_sdfs:
            sdf_path = next(iter(ligand_sdfs.values()))
        if sdf_path is None:
            continue

        mol = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)[0]
        if mol is None:
            continue

        rot_data = compute_rot_bond_data(mol, atom_offset, n_total_atoms)
        all_rot_bond_data.append(rot_data)

    rot_bond_data = merge_rot_bond_data(all_rot_bond_data, n_total_atoms)

    # Build ChainInfo list
    cluster_id = clusters.get(system_id, -1)
    affinity_chain_id = None
    chain_infos = []
    for i, chain in enumerate(structure.chains):
        mol_type = int(chain["mol_type"])
        c_cluster_id = cluster_id if mol_type == protein_id else -1
        chain_infos.append(ChainInfo(
            chain_id=i,
            chain_name=chain["name"].strip(),
            mol_type=mol_type,
            cluster_id=c_cluster_id,
            msa_id="",
            num_residues=int(chain["res_num"]),
            entity_id=int(chain["entity_id"]),
        ))
        if mol_type == nonpolymer_id and affinity_chain_id is None:
            affinity_chain_id = i

    # Affinity
    affinity = None
    if annotation_row.get("system_has_binding_affinity") and affinity_chain_id is not None:
        mw = annotation_row.get("ligand_molecular_weight") or annotation_row.get("system_proper_ligand_max_molecular_weight")
        if mw is not None:
            affinity = AffinityInfo(chain_id=affinity_chain_id, mw=float(mw))

    # Interface info
    interface_infos = [
        InterfaceInfo(chain_1=int(iface["chain_1"]), chain_2=int(iface["chain_2"]))
        for iface in structure.interfaces
    ]

    record = Record(
        id=system_id,
        structure=parsed.info,
        chains=chain_infos,
        interfaces=interface_infos,
        affinity=affinity,
    )

    # Save
    (outdir / "structures" / mid).mkdir(parents=True, exist_ok=True)
    (outdir / "records" / mid).mkdir(parents=True, exist_ok=True)

    save_dict = asdict(structure)
    save_dict.update(rot_bond_data)
    np.savez_compressed(struct_path, **save_dict)

    with open(record_path, "w") as f:
        json.dump(asdict(record), f)


def process_system_worker(
    system_id: str,
    plinder_dir: Path,
    outdir: Path,
    clusters: dict,
    annotations: dict,
) -> None:
    annotation_row = annotations.get(system_id, {})
    try:
        process_system(system_id, plinder_dir, outdir, clusters, annotation_row)
    except Exception:
        traceback.print_exc()
        print(f"Unhandled error processing {system_id}")


# ── finalize ──────────────────────────────────────────────────────────────────

def finalize(outdir: Path) -> None:
    records = []
    failed = 0
    for record_file in (outdir / "records").rglob("*.json"):
        try:
            with open(record_file) as f:
                records.append(json.load(f))
        except Exception:
            failed += 1
    if failed:
        print(f"Failed to parse {failed} record files")
    with open(outdir / "manifest.json", "w") as f:
        json.dump(records, f)
    print(f"Wrote manifest with {len(records)} entries")


# ── main ──────────────────────────────────────────────────────────────────────

def process(args) -> None:
    plinder_dir = args.plinder_dir
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Set rdkit pickle options
    pickle_option = rdkit.Chem.PropertyPickleOptions.AllProps
    rdkit.Chem.SetDefaultPickleProperties(pickle_option)

    print("Loading clusters...")
    clusters = load_clusters(plinder_dir, metric=args.cluster_metric, threshold=args.cluster_threshold)
    print(f"Loaded {len(clusters)} cluster assignments")

    print("Loading annotation table...")
    ann_df = load_annotation_table(plinder_dir)
    annotations = {row["system_id"]: row for row in ann_df.to_dict(orient="records")}

    print("Loading split...")
    split_df = load_split(plinder_dir)

    # Filter to requested splits
    if args.splits:
        split_df = split_df[split_df["split"].isin(args.splits)]

    system_ids = split_df["system_id"].tolist()

    if args.max_systems is not None:
        system_ids = system_ids[:args.max_systems]

    print(f"Processing {len(system_ids)} systems...")

    num_processes = min(args.num_processes, multiprocessing.cpu_count(), len(system_ids))

    fn = partial(
        process_system_worker,
        plinder_dir=plinder_dir,
        outdir=outdir,
        clusters=clusters,
        annotations=annotations,
    )

    if num_processes > 1:
        p_umap(fn, system_ids, num_cpus=num_processes)
    else:
        for sid in tqdm(system_ids):
            fn(sid)

    finalize(outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Plinder dataset into proteinzen npz format.")
    parser.add_argument("--plinder-dir", type=Path, required=True,
                        help="Path to plinder data root (e.g. /datastor1/dy4652/plinder/2024-06/v2)")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Output directory for npz + manifest")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Which splits to include (default: train val test)")
    parser.add_argument("--cluster-metric", type=str, default="pocket_fident_qcov",
                        help="Cluster metric to use for sampling weights")
    parser.add_argument("--cluster-threshold", type=int, default=30,
                        help="Cluster threshold to use for sampling weights")
    parser.add_argument("--num-processes", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--max-systems", type=int, default=None,
                        help="Cap number of systems (for debugging)")
    args = parser.parse_args()
    process(args)
