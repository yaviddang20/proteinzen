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
import os
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from functools import partial

import pickle

import numpy as np
import pyarrow.parquet as pq
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
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


def load_clusters(
    plinder_dir: Path,
    algorithm: str = "communities",
    directed: bool = False,
    metric: str = "pli_qcov",
    threshold: int = 50,
) -> dict:
    """Load cluster parquet and return {system_id: cluster_label} dict."""
    cluster_path = (
        plinder_dir
        / "clusters"
        / f"cluster={algorithm}"
        / f"directed={directed}"
        / f"metric={metric}"
        / f"threshold={threshold}"
        / "data.parquet"
    )
    if not cluster_path.exists():
        print(f"Warning: cluster parquet not found at {cluster_path}")
        return {}
    t = pq.ParquetFile(cluster_path).read(columns=["system_id", "label"])
    return dict(zip(t["system_id"].to_pylist(), t["label"].to_pylist()))


def load_annotation_table(plinder_dir: Path) -> dict:
    """Load annotation table, return {system_id: row_dict} with only affinity columns."""
    path = plinder_dir / "index" / "annotation_table.parquet"
    cols = [
        "system_id",
        "system_has_binding_affinity",
        "ligand_molecular_weight",
        "system_proper_ligand_max_molecular_weight",
    ]
    t = pq.ParquetFile(path).read(columns=cols)
    result = {}
    ids = t["system_id"].to_pylist()
    for i, sid in enumerate(ids):
        result[sid] = {c: t[c][i].as_py() for c in cols}
    return result


def load_split(plinder_dir: Path) -> dict:
    """Load split definitions, return {system_id: split_name} dict."""
    path = plinder_dir / "splits" / "split.parquet"
    t = pq.ParquetFile(path).read(columns=["system_id", "split"])
    return dict(zip(t["system_id"].to_pylist(), t["split"].to_pylist()))


def get_ligand_sdfs(system_dir: Path) -> dict:
    """Return {ligand_name: Path} for all SDF files in ligand_files/."""
    ligand_dir = system_dir / "ligand_files"
    if not ligand_dir.exists():
        return {}
    return {f.stem: f for f in ligand_dir.glob("*.sdf")}


# Canonical SMILES for common crystallography buffer/cryo-protectant artifacts.
# Built at import time so comparison uses RDKit canonical form.
_BUFFER_SMILES_RAW = [
    "OCC(O)CO",           # glycerol (GOL)
    "OCCO",               # ethylene glycol (EDO)
    "OCC(C)O",            # 1,2-propanediol
    "CC(O)CC(C)CO",       # 2-methyl-2,4-pentanediol (MPD)
    "CS(C)=O",            # DMSO (DMS)
    "CCO",                # ethanol (EOH)
    "CO",                 # methanol (MOH)
    "CC(C)O",             # isopropanol (IPA)
    "OCCS",               # beta-mercaptoethanol (BME)
    "OCC(S)C(S)CO",       # DTT
    "OCC(N)(CO)CO",       # Tris (TRS)
    "c1cnc[nH]1",         # imidazole (IMD)
    "CC#N",               # acetonitrile (ACN)
    "OC(CC(=O)O)(CC(=O)O)C(=O)O",    # citric acid (CIT)
    "[O-]C(=O)C(O)C(O)C([O-])=O",    # tartrate (TLA)
    "OC(C(O)C(=O)O)C(=O)O",           # tartaric acid
    "OCC[NH+]1CCOCC1",               # morpholine-ethanol (MES-like)
    "OCCNCCS(=O)(=O)O",              # HEPES-like fragment
    "OCC[NH+]1CCN(CCS([O-])(=O)=O)CC1",  # HEPES
    "O=C1CCCCC1",         # cyclohexanone
    "OCCOCCO",            # diethylene glycol (PEG2)
    "OCCOCCOCCO",         # triethylene glycol (PEG3)
    "OCCOCCOCCOCCO",      # PEG4
    "C(CO)O",             # 1,3-propanediol
    "OCC(O)C(O)CO",       # erythritol
    "OCC(O)C(O)C(O)CO",   # xylitol / ribitol
]
_BUFFER_SMILES = set()
for _s in _BUFFER_SMILES_RAW:
    _m = Chem.MolFromSmiles(_s)
    if _m is not None:
        _BUFFER_SMILES.add(Chem.MolToSmiles(_m))


def _longest_unbranched_hydrocarbon_chain(mol: Chem.Mol) -> int:
    """Length of the longest unbranched chain of non-ring carbons bonded only to C/H."""
    pure_c = {
        a.GetIdx() for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6
        and not a.IsInRing()
        and all(n.GetAtomicNum() in (1, 6) for n in a.GetNeighbors())
    }
    adj: dict[int, list[int]] = {i: [] for i in pure_c}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in pure_c and j in pure_c:
            adj[i].append(j)
            adj[j].append(i)

    visited: set[int] = set()
    max_chain = 0
    for start in pure_c:
        if start in visited:
            continue
        component: list[int] = []
        stack = [start]
        comp_vis: set[int] = set()
        while stack:
            node = stack.pop()
            if node in comp_vis:
                continue
            comp_vis.add(node)
            component.append(node)
            for nb in adj[node]:
                if nb not in comp_vis:
                    stack.append(nb)
        visited.update(comp_vis)
        # Linear (unbranched) iff every node has at most 2 neighbors within the component
        if all(len(adj[n]) <= 2 for n in component):
            max_chain = max(max_chain, len(component))
    return max_chain


def is_valid_ligand(mol: Chem.Mol) -> bool:
    """Return False for ligands that fail biological/therapeutic relevance filters."""
    # Minimum size
    n_heavy = mol.GetNumAtoms()
    n_carbon = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    if n_heavy < 5 or n_carbon < 2:
        return False

    # Unspecified atoms (wildcard / unknown element)
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        return False

    # Highly charged
    if abs(Chem.GetFormalCharge(mol)) > 2:
        return False

    # Long unbranched hydrocarbon linker (lipid/detergent contamination)
    if _longest_unbranched_hydrocarbon_chain(mol) > 12:
        return False

    # Common buffer/cryoprotectant artifacts
    if Chem.MolToSmiles(mol) in _BUFFER_SMILES:
        return False

    return True


def compute_rot_bond_data(mol: Chem.Mol) -> dict:
    """Compute rot_bond arrays for a ligand mol in local (0-indexed) ligand atom space.

    All arrays use local indices 0..n_lig-1.  The datamodule remaps them to global
    cropped rigid space at load time using rigids_is_atom_mask, which correctly
    handles protein cropping without requiring any global offset here.
    """
    rot_bonds, frag_a = compute_rot_bond_fragments(mol)  # (B,2), (B, n_lig)
    ring_masks = compute_ring_atom_masks(mol)              # (R, n_lig)
    sym_groups, sym_group_sizes = compute_sym_groups(mol)  # (G, max_sz), (G,)
    n_lig = mol.GetNumAtoms()
    return {
        "rot_bonds": rot_bonds,    # (B, 2) local
        "rot_frag_a": frag_a,      # (B, n_lig) local
        "ring_masks": ring_masks,  # (R, n_lig) local
        "n_lig": n_lig,
        "sym_groups": sym_groups,
        "sym_group_sizes": sym_group_sizes,
    }


def merge_rot_bond_data(all_data: list) -> dict:
    """Merge rot_bond data from multiple ligands using cumulative n_lig offsets.

    Produces arrays in merged local space: ligand-1 atoms 0..n1-1,
    ligand-2 atoms n1..n1+n2-1, etc.  The datamodule remaps to global rigid space.
    """
    if not all_data:
        return {
            "rot_bonds": np.zeros((0, 2), dtype=np.int32),
            "rot_frag_a": np.zeros((0, 0), dtype=bool),
            "ring_masks": np.zeros((0, 0), dtype=bool),
            "sym_groups": np.zeros((0, 1), dtype=np.int32),
            "sym_group_sizes": np.zeros(0, dtype=np.int32),
        }

    total_n_lig = sum(d["n_lig"] for d in all_data)

    rot_bonds_list, rot_frag_a_list, ring_masks_list = [], [], []
    n_cumulative = 0
    for d in all_data:
        n = d["n_lig"]
        B_i = d["rot_bonds"].shape[0]
        R_i = d["ring_masks"].shape[0]

        if B_i > 0:
            rot_bonds_list.append(d["rot_bonds"] + n_cumulative)
            fa = np.zeros((B_i, total_n_lig), dtype=bool)
            fa[:, n_cumulative:n_cumulative + n] = d["rot_frag_a"]
            rot_frag_a_list.append(fa)

        if R_i > 0:
            rm = np.zeros((R_i, total_n_lig), dtype=bool)
            rm[:, n_cumulative:n_cumulative + n] = d["ring_masks"]
            ring_masks_list.append(rm)

        n_cumulative += n

    rot_bonds = np.concatenate(rot_bonds_list, axis=0) if rot_bonds_list else np.zeros((0, 2), dtype=np.int32)
    rot_frag_a = np.concatenate(rot_frag_a_list, axis=0) if rot_frag_a_list else np.zeros((0, total_n_lig), dtype=bool)
    ring_masks = np.concatenate(ring_masks_list, axis=0) if ring_masks_list else np.zeros((0, total_n_lig), dtype=bool)

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
    ccd: dict,
) -> None:
    mid = system_mid(system_id)
    struct_path = outdir / "structures" / mid / f"{system_id}.npz"
    record_path = outdir / "records" / mid / f"{system_id}.json"

    if struct_path.exists() and record_path.exists():
        return

    system_dir = plinder_dir / "systems" / system_id
    cif_path = system_dir / "system.cif"
    if not cif_path.exists():
        return

    try:
        # Parse full complex (protein + ligand chains via CCD)
        parsed = parse_mmcif(str(cif_path), components=ccd, ignore_connections=False, use_assembly=False)
        structure = parsed.data
    except Exception:
        traceback.print_exc()
        print(f"Failed to parse {system_id}")
        return

    # Skip systems with nucleic acid content (DNA/RNA — not supported by tokenizer)
    # Check both mol_type (correctly parsed chains) and residue names (gemmi misclassification)
    _nuc_residues = {'A', 'G', 'C', 'U', 'DA', 'DG', 'DC', 'DT'}
    dna_id = const.chain_type_ids["DNA"]
    rna_id = const.chain_type_ids["RNA"]
    for chain in structure.chains:
        if int(chain["mol_type"]) in (dna_id, rna_id):
            return
        # Catch gemmi misclassifying DNA chains as PeptideL
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        for res in structure.residues[res_start:res_end]:
            if str(res["name"]) in _nuc_residues:
                return

    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]
    protein_id = const.chain_type_ids["PROTEIN"]

    # Only process systems with exactly one protein chain and one ligand chain
    n_protein = sum(1 for c in structure.chains if int(c["mol_type"]) == protein_id)
    n_ligand = sum(1 for c in structure.chains if int(c["mol_type"]) == nonpolymer_id)
    if n_protein != 1 or n_ligand != 1:
        return

    # Load ligand SDF files and validate each ligand mol
    ligand_sdfs = get_ligand_sdfs(system_dir)

    all_rot_bond_data = []
    for chain in structure.chains:
        if int(chain["mol_type"]) != nonpolymer_id:
            continue

        chain_name = chain["name"].strip()

        # Find matching SDF — try by chain name, then take first available
        sdf_path = ligand_sdfs.get(chain_name)
        if sdf_path is None and ligand_sdfs:
            sdf_path = next(iter(ligand_sdfs.values()))
        if sdf_path is None:
            return  # no SDF for this ligand chain → skip system

        mol = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)[0]
        if mol is None:
            return

        if not is_valid_ligand(mol):
            return

        rot_data = compute_rot_bond_data(mol)
        all_rot_bond_data.append(rot_data)

    rot_bond_data = merge_rot_bond_data(all_rot_bond_data)

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


_worker_state = {}

def _worker_init(plinder_dir, outdir, clusters, annotations, ccd_path):
    """Load large shared data once per worker process."""
    global _worker_state
    with open(ccd_path, "rb") as f:
        ccd = pickle.load(f)
    _worker_state = {
        "plinder_dir": plinder_dir,
        "outdir": outdir,
        "clusters": clusters,
        "annotations": annotations,
        "ccd": ccd,
    }


def process_system_worker(system_id: str) -> None:
    s = _worker_state
    annotation_row = s["annotations"].get(system_id, {})
    try:
        process_system(system_id, s["plinder_dir"], s["outdir"], s["clusters"], annotation_row, s["ccd"])
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

def process(args, clusters: dict, annotations: dict, split: dict) -> None:
    plinder_dir = args.plinder_dir
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    system_ids = [sid for sid, s in split.items() if s in args.splits]

    if args.max_systems is not None:
        system_ids = system_ids[:args.max_systems]

    print(f"Processing {len(system_ids)} systems...")

    num_processes = min(args.num_processes, multiprocessing.cpu_count(), len(system_ids))

    if num_processes > 1:
        initargs = (plinder_dir, outdir, clusters, annotations, args.ccd_path)
        with multiprocessing.Pool(
            processes=num_processes,
            initializer=_worker_init,
            initargs=initargs,
        ) as pool:
            list(tqdm(pool.imap_unordered(process_system_worker, system_ids, chunksize=4), total=len(system_ids)))
    else:
        _worker_init(plinder_dir, outdir, clusters, annotations, args.ccd_path)
        for sid in tqdm(system_ids):
            process_system_worker(sid)

    finalize(outdir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Plinder dataset into proteinzen npz format.")
    parser.add_argument("--ccd-path", type=Path, default=Path(os.environ.get("REPO_ROOT", ".")) / "ccd.pkl",
                        help="Path to ccd.pkl (default: $REPO_ROOT/ccd.pkl)")
    parser.add_argument("--plinder-dir", type=Path, required=True,
                        help="Path to plinder data root (e.g. /mnt/scratch/.../plinder/2024-06/v2)")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Output root — one subdirectory per split (train/val/test)")
    parser.add_argument("--cluster-algorithm", type=str, default="communities")
    parser.add_argument("--cluster-directed", action="store_true", default=False)
    parser.add_argument("--cluster-metric", type=str, default="pli_qcov")
    parser.add_argument("--cluster-threshold", type=int, default=50)
    parser.add_argument("--num-processes", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--max-systems", type=int, default=None,
                        help="Cap number of systems per split (for debugging)")
    args = parser.parse_args()

    # Set rdkit pickle options
    pickle_option = rdkit.Chem.PropertyPickleOptions.AllProps
    rdkit.Chem.SetDefaultPickleProperties(pickle_option)

    # Load shared data once
    print("Loading clusters...")
    clusters = load_clusters(
        args.plinder_dir,
        algorithm=args.cluster_algorithm,
        directed=args.cluster_directed,
        metric=args.cluster_metric,
        threshold=args.cluster_threshold,
    )
    print(f"Loaded {len(clusters)} cluster assignments")

    print("Loading annotation table...")
    annotations = load_annotation_table(args.plinder_dir)
    print(f"Loaded {len(annotations)} annotation rows")

    print("Loading split...")
    split = load_split(args.plinder_dir)
    print(f"Loaded {len(split)} split assignments")

    for split_name in ["train", "val", "test"]:
        print(f"\n=== Processing split: {split_name} ===")
        split_args = argparse.Namespace(**{**vars(args), "splits": [split_name], "outdir": args.outdir / split_name})
        process(split_args, clusters, annotations, split)
