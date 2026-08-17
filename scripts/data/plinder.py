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
from typing import Optional

import pickle

import numpy as np
import pyarrow.parquet as pq
import rdkit
import yaml
from scipy.spatial.distance import cdist
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

from mmcif import parse_mmcif

from proteinzen.boltz.data import const
from rdkit.Chem import AllChem
from proteinzen.boltz.data.types import (
    AffinityInfo,
    Atom,
    ChainInfo,
    InterfaceInfo,
    Record,
    Structure,
    StructureInfo,
)
from proteinzen.data.featurize.mol.sampling import (
    compute_ring_atom_masks,
    compute_rot_bond_fragments,
    compute_sym_groups,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _atom_name_bytes(name: str) -> tuple:
    name = name.strip()[:4]
    chars = [ord(c) - 32 for c in name]
    chars += [0] * (4 - len(chars))
    return tuple(chars)


def _insert_ligand_h(structure: "Structure", mol_with_h) -> "Structure":
    """Append RDKit-estimated H atoms to the ligand (nonpolymer) chain in structure."""
    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]

    if mol_with_h is None or mol_with_h.GetNumConformers() == 0:
        return structure

    conf = mol_with_h.GetConformer()
    h_entries = []
    h_idx = 1
    for atom in mol_with_h.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords = (float(pos.x), float(pos.y), float(pos.z))
        name = f"H{h_idx}" if h_idx > 1 else "H"
        h_entries.append((
            _atom_name_bytes(name),
            1,       # element: H
            0,       # charge
            coords,  # coords
            coords,  # conformer (same)
            True,    # is_present
            0,       # chirality
        ))
        h_idx += 1

    if not h_entries:
        return structure

    n_h = len(h_entries)

    # Find the ligand chain
    lig_chain_idx = next(
        (i for i, c in enumerate(structure.chains)
         if int(c["mol_type"]) == nonpolymer_id),
        None
    )
    if lig_chain_idx is None:
        return structure

    lig_chain = structure.chains[lig_chain_idx]
    insert_at = int(lig_chain["atom_idx"]) + int(lig_chain["atom_num"])

    h_atoms = np.array(h_entries, dtype=np.dtype(Atom))
    new_atoms = np.concatenate([
        structure.atoms[:insert_at],
        h_atoms,
        structure.atoms[insert_at:],
    ])

    new_residues = structure.residues.copy()
    lig_res_start = int(lig_chain["res_idx"])
    lig_res_end = lig_res_start + int(lig_chain["res_num"])
    for r in range(lig_res_start, lig_res_end):
        new_residues[r]["atom_num"] += n_h

    for r in range(len(new_residues)):
        if new_residues[r]["atom_idx"] >= insert_at:
            new_residues[r]["atom_idx"] += n_h
        if new_residues[r]["atom_center"] >= insert_at:
            new_residues[r]["atom_center"] += n_h
        if new_residues[r]["atom_disto"] >= insert_at:
            new_residues[r]["atom_disto"] += n_h

    new_chains = structure.chains.copy()
    new_chains[lig_chain_idx]["atom_num"] += n_h
    for c in range(lig_chain_idx + 1, len(new_chains)):
        new_chains[c]["atom_idx"] += n_h

    return Structure(
        atoms=new_atoms,
        bonds=structure.bonds,
        residues=new_residues,
        chains=new_chains,
        connections=structure.connections,
        interfaces=structure.interfaces,
        mask=structure.mask,
    )


def fuse_protein_chains(
    structure: "Structure",
    auth_seq_map: Optional[list] = None,
) -> tuple:
    """Merge all PROTEIN chains in `structure` into a single chain/entity.

    Residues from each original protein chain are concatenated in their original
    chain order into one merged chain, with a fresh, continuous local `res_idx`
    sequence (no gap between the original chains — they're treated as one
    contiguous protein). Non-protein chains (e.g. the ligand) are kept as-is,
    renumbered to sit after the merged chain. `structure.interfaces` is dropped
    (mirrors `Structure.remove_invalid_chains`, which does the same whenever chain
    indices are restructured) since it isn't used downstream of this script.

    If `structure` has <= 1 protein chain, returns the inputs unchanged.
    """
    protein_id = const.chain_type_ids["PROTEIN"]
    protein_idxs = [i for i, c in enumerate(structure.chains) if int(c["mol_type"]) == protein_id]
    if len(protein_idxs) <= 1:
        return structure, auth_seq_map

    protein_idx_set = set(protein_idxs)
    other_idxs = [i for i in range(len(structure.chains)) if i not in protein_idx_set]

    n_atoms_total = len(structure.atoms)
    n_res_total = len(structure.residues)
    atom_remap = np.empty(n_atoms_total, dtype=np.int32)
    res_remap = np.empty(n_res_total, dtype=np.int32)

    new_atom_chunks: list = []
    new_res_chunks: list = []
    atom_cursor = 0
    res_cursor = 0

    def append_block(old_chain_idx: int):
        nonlocal atom_cursor, res_cursor
        chain = structure.chains[old_chain_idx]
        a0, an = int(chain["atom_idx"]), int(chain["atom_num"])
        r0, rn = int(chain["res_idx"]), int(chain["res_num"])

        atoms = structure.atoms[a0:a0 + an].copy()
        residues = structure.residues[r0:r0 + rn].copy()

        atom_shift = atom_cursor - a0
        if rn:
            residues["atom_idx"] += atom_shift
            residues["atom_center"] += atom_shift
            residues["atom_disto"] += atom_shift

        atom_remap[a0:a0 + an] = np.arange(atom_cursor, atom_cursor + an)
        res_remap[r0:r0 + rn] = np.arange(res_cursor, res_cursor + rn)

        new_atom_chunks.append(atoms)
        new_res_chunks.append(residues)

        new_atom_idx, new_res_idx = atom_cursor, res_cursor
        atom_cursor += an
        res_cursor += rn
        return new_atom_idx, an, new_res_idx, rn, residues

    # Merged protein chain block goes first; residues get a fresh, contiguous
    # local res_idx sequence spanning all constituent chains (no gap).
    merged_atom_idx = atom_cursor
    merged_res_idx = res_cursor
    local_res_cursor = 0
    merged_auth_indices = []
    for pidx in protein_idxs:
        _, _, _, rn, residues = append_block(pidx)
        if rn:
            local_min = int(residues["res_idx"].min())
            residues["res_idx"] += local_res_cursor - local_min
            local_res_cursor = int(residues["res_idx"].max()) + 1
        if auth_seq_map is not None:
            merged_auth_indices.extend(auth_seq_map[pidx]["auth_indices"])
    merged_atom_num = atom_cursor - merged_atom_idx
    merged_res_num = res_cursor - merged_res_idx

    other_blocks = [(oidx, *append_block(oidx)) for oidx in other_idxs]

    new_atoms = np.concatenate(new_atom_chunks, dtype=structure.atoms.dtype) if new_atom_chunks else structure.atoms[:0]
    new_residues = np.concatenate(new_res_chunks, dtype=structure.residues.dtype) if new_res_chunks else structure.residues[:0]

    new_chains = np.zeros(1 + len(other_idxs), dtype=structure.chains.dtype)
    new_chains[0] = structure.chains[protein_idxs[0]]
    new_chains[0]["atom_idx"] = merged_atom_idx
    new_chains[0]["atom_num"] = merged_atom_num
    new_chains[0]["res_idx"] = merged_res_idx
    new_chains[0]["res_num"] = merged_res_num
    new_chains[0]["asym_id"] = 0
    new_chains[0]["sym_id"] = 0

    chain_map = {pidx: 0 for pidx in protein_idxs}
    for k, (oidx, new_a_idx, an, new_r_idx, rn, _) in enumerate(other_blocks, start=1):
        new_chains[k] = structure.chains[oidx]
        new_chains[k]["atom_idx"] = new_a_idx
        new_chains[k]["atom_num"] = an
        new_chains[k]["res_idx"] = new_r_idx
        new_chains[k]["res_num"] = rn
        new_chains[k]["asym_id"] = k
        chain_map[oidx] = k

    new_bonds = structure.bonds.copy()
    if len(new_bonds):
        new_bonds["atom_1"] = atom_remap[new_bonds["atom_1"]]
        new_bonds["atom_2"] = atom_remap[new_bonds["atom_2"]]

    new_connections = structure.connections.copy()
    if len(new_connections):
        new_connections["chain_1"] = [chain_map[c] for c in structure.connections["chain_1"]]
        new_connections["chain_2"] = [chain_map[c] for c in structure.connections["chain_2"]]
        new_connections["res_1"] = res_remap[structure.connections["res_1"]]
        new_connections["res_2"] = res_remap[structure.connections["res_2"]]
        new_connections["atom_1"] = atom_remap[structure.connections["atom_1"]]
        new_connections["atom_2"] = atom_remap[structure.connections["atom_2"]]

    new_structure = replace(
        structure,
        atoms=new_atoms,
        bonds=new_bonds,
        residues=new_residues,
        chains=new_chains,
        connections=new_connections,
        interfaces=np.zeros(0, dtype=structure.interfaces.dtype),
        mask=np.ones(len(new_chains), dtype=bool),
    )

    new_auth_seq_map = None
    if auth_seq_map is not None:
        merged_entry = dict(auth_seq_map[protein_idxs[0]])
        merged_entry["auth_indices"] = merged_auth_indices
        new_auth_seq_map = [merged_entry] + [auth_seq_map[oidx] for oidx in other_idxs]

    return new_structure, new_auth_seq_map


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
        "ligand_is_covalent",
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

    # Only allow drug-like organic elements; excludes metals and wildcard atoms
    _ORGANIC_ATOMS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}  # H,B,C,N,O,F,Si,P,S,Cl,Se,Br,I
    if any(a.GetAtomicNum() not in _ORGANIC_ATOMS for a in mol.GetAtoms()):
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

def _load_pocket_alpha_spheres(system_id: str, pocket_data_dir: Optional[Path]) -> Optional[np.ndarray]:
    """Load pre-computed fpocket alpha-sphere coords for system_id, or None."""
    if pocket_data_dir is None:
        return None
    mid = system_id[1:3]
    path = pocket_data_dir / mid / f"{system_id}.npy"
    if not path.exists():
        return None
    return np.load(path)


def process_system(
    system_id: str,
    plinder_dir: Path,
    outdir: Path,
    clusters: dict,
    annotation_row: dict,
    ccd: dict,
    pocket_data_dir: Optional[Path] = None,
    allowed_protein_chain_counts: tuple = (1,),
    fuse_multi_chain: bool = False,
) -> None:
    mid = system_mid(system_id)
    struct_path = outdir / "structures" / mid / f"{system_id}.npz"
    record_path = outdir / "records" / mid / f"{system_id}.json"
    auth_map_path = outdir / "auth_maps" / mid / f"{system_id}.json"

    if struct_path.exists() and record_path.exists() and auth_map_path.exists():
        return None

    system_dir = plinder_dir / "systems" / system_id
    cif_path = system_dir / "system.cif"
    if not cif_path.exists():
        return "no_cif"

    if annotation_row.get("ligand_is_covalent"):
        return "covalent"

    try:
        # Parse full complex (protein + ligand chains via CCD)
        parsed = parse_mmcif(str(cif_path), components=ccd, ignore_connections=False, use_assembly=False)
        structure = parsed.data
        auth_seq_map = parsed.auth_seq_map
    except Exception:
        traceback.print_exc()
        print(f"Failed to parse {system_id}")
        return "parse_error"

    # Skip systems with nucleic acid content (DNA/RNA — not supported by tokenizer)
    # Check both mol_type (correctly parsed chains) and residue names (gemmi misclassification)
    _nuc_residues = {'A', 'G', 'C', 'U', 'DA', 'DG', 'DC', 'DT'}
    dna_id = const.chain_type_ids["DNA"]
    rna_id = const.chain_type_ids["RNA"]
    for chain in structure.chains:
        if int(chain["mol_type"]) in (dna_id, rna_id):
            return "nucleic_acid"
        # Catch gemmi misclassifying DNA chains as PeptideL
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        for res in structure.residues[res_start:res_end]:
            if str(res["name"]) in _nuc_residues:
                return "nucleic_acid"

    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]
    protein_id = const.chain_type_ids["PROTEIN"]

    # Only process systems with an allowed protein chain count and exactly one ligand chain
    n_protein = sum(1 for c in structure.chains if int(c["mol_type"]) == protein_id)
    n_ligand = sum(1 for c in structure.chains if int(c["mol_type"]) == nonpolymer_id)
    if n_protein not in allowed_protein_chain_counts or n_ligand != 1:
        return f"chain_filter_protein{n_protein}_ligand{n_ligand}"

    if fuse_multi_chain and n_protein > 1:
        structure, auth_seq_map = fuse_protein_chains(structure, auth_seq_map)

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
            return "no_sdf"

        mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)[0]
        if mol is None:
            return "mol_parse_error"

        mol_no_h = AllChem.RemoveHs(mol)
        if not is_valid_ligand(mol_no_h):
            return "invalid_ligand"

        mol_with_h = AllChem.AddHs(mol, addCoords=True) if mol.GetNumConformers() > 0 else None
        ligand_smiles = Chem.MolToSmiles(mol_no_h)
        rot_data = compute_rot_bond_data(mol_no_h)
        # compute_rot_bond_fragments/compute_ring_atom_masks each redo their own AddHs/RemoveHs
        # round-trip internally and can (rarely, for degenerate ligands) land on a different heavy
        # atom count than n_lig here; guard against the resulting shape mismatch rather than crash.
        n_lig = rot_data["n_lig"]
        if rot_data["rot_frag_a"].shape[1] != n_lig or rot_data["ring_masks"].shape[1] != n_lig:
            return "rot_bond_shape_mismatch"
        all_rot_bond_data.append(rot_data)

    rot_bond_data = merge_rot_bond_data(all_rot_bond_data)

    # Add H to ligand atoms (mmcif parser strips H; re-add from RDKit geometry)
    structure = _insert_ligand_h(structure, mol_with_h)

    # Compute interaction_residue_mask: protein residues within atom_interface_cutoff of any ligand atom
    interaction_residue_mask = np.zeros(len(structure.residues), dtype=bool)
    ligand_coords_list = []
    for chain in structure.chains:
        if int(chain["mol_type"]) != nonpolymer_id:
            continue
        a_start = int(chain["atom_idx"])
        a_end = a_start + int(chain["atom_num"])
        present = structure.atoms[a_start:a_end]["is_present"].astype(bool)
        if present.any():
            ligand_coords_list.append(structure.atoms[a_start:a_end]["coords"][present])
    if ligand_coords_list:
        lig_coords = np.concatenate(ligand_coords_list)
        for chain in structure.chains:
            if int(chain["mol_type"]) != protein_id:
                continue
            res_start = int(chain["res_idx"])
            res_end = res_start + int(chain["res_num"])
            for r in range(res_start, res_end):
                res = structure.residues[r]
                if not res["is_present"]:
                    continue
                a_start = int(res["atom_idx"])
                a_end = a_start + int(res["atom_num"])
                res_atoms = structure.atoms[a_start:a_end]
                present_coords = res_atoms["coords"][res_atoms["is_present"].astype(bool)]
                if len(present_coords) == 0:
                    continue
                if cdist(present_coords, lig_coords).min() < const.atom_interface_cutoff:
                    interaction_residue_mask[r] = True

    # Build ChainInfo list
    cluster_id = clusters.get(system_id, -1)
    affinity_chain_id = None
    chain_infos = []
    for i, chain in enumerate(structure.chains):
        mol_type = int(chain["mol_type"])
        c_cluster_id = cluster_id if mol_type == protein_id else -1
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        num_resolved = int(structure.residues[res_start:res_end]["is_present"].sum())
        chain_infos.append(ChainInfo(
            chain_id=i,
            chain_name=chain["name"].strip(),
            mol_type=mol_type,
            cluster_id=c_cluster_id,
            msa_id="",
            num_residues=int(chain["res_num"]),
            num_resolved_residues=num_resolved,
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
        InterfaceInfo(
            chain_1=int(iface["chain_1"]),
            chain_2=int(iface["chain_2"]),
            chain_1_num_res=int(iface["chain_1_num_res"]),
            chain_2_num_res=int(iface["chain_2_num_res"]),
        )
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
    (outdir / "auth_maps" / mid).mkdir(parents=True, exist_ok=True)

    # Compute pocket_residue_mask from fpocket alpha-sphere coords if available.
    # A protein residue is in the pocket if any of its atoms is within 4Å of any alpha sphere.
    alpha_spheres = _load_pocket_alpha_spheres(system_id, pocket_data_dir)
    pocket_residue_mask = np.zeros(len(structure.residues), dtype=bool)
    if alpha_spheres is not None and len(alpha_spheres) > 0:
        for chain in structure.chains:
            if int(chain["mol_type"]) != protein_id:
                continue
            res_start = int(chain["res_idx"])
            res_end = res_start + int(chain["res_num"])
            for r in range(res_start, res_end):
                res = structure.residues[r]
                if not res["is_present"]:
                    continue
                a_start = int(res["atom_idx"])
                a_end = a_start + int(res["atom_num"])
                res_atoms = structure.atoms[a_start:a_end]
                present_coords = res_atoms["coords"][res_atoms["is_present"].astype(bool)]
                if len(present_coords) == 0:
                    continue
                if cdist(present_coords, alpha_spheres).min() < 4.0:
                    pocket_residue_mask[r] = True

    save_dict = asdict(structure)
    save_dict.update(rot_bond_data)
    save_dict['interaction_residue_mask'] = interaction_residue_mask
    save_dict['pocket_residue_mask'] = pocket_residue_mask
    np.savez_compressed(struct_path, **save_dict)

    record_dict = asdict(record)
    record_dict['smiles'] = ligand_smiles
    with open(record_path, "w") as f:
        json.dump(record_dict, f)

    auth_map_path = outdir / "auth_maps" / mid / f"{system_id}.json"
    with open(auth_map_path, "w") as f:
        json.dump(auth_seq_map, f)

    return None


_worker_state = {}

def _worker_init(plinder_dir, outdir, clusters, annotations, ccd_path, pocket_data_dir=None,
                  allowed_protein_chain_counts=(1,), fuse_multi_chain=False):
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
        "pocket_data_dir": pocket_data_dir,
        "allowed_protein_chain_counts": allowed_protein_chain_counts,
        "fuse_multi_chain": fuse_multi_chain,
    }


def process_system_worker(system_id: str) -> tuple[str, Optional[str]]:
    s = _worker_state
    annotation_row = s["annotations"].get(system_id, {})
    try:
        reason = process_system(
            system_id, s["plinder_dir"], s["outdir"], s["clusters"],
            annotation_row, s["ccd"], pocket_data_dir=s.get("pocket_data_dir"),
            allowed_protein_chain_counts=s.get("allowed_protein_chain_counts", (1,)),
            fuse_multi_chain=s.get("fuse_multi_chain", False),
        )
        return system_id, reason
    except Exception:
        traceback.print_exc()
        print(f"Unhandled error processing {system_id}")
        return system_id, "unhandled_error"


# ── finalize ──────────────────────────────────────────────────────────────────

def finalize(outdir: Path) -> int:
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
        json.dump(records, f, indent=2)
    print(f"Wrote manifest with {len(records)} entries")
    return len(records)


# ── main ──────────────────────────────────────────────────────────────────────

def process(args, clusters: dict, annotations: dict, split: dict) -> int:
    plinder_dir = args.plinder_dir
    outdir = args.outdir
    if args.overwrite and outdir.exists():
        import shutil
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    system_ids = [sid for sid, s in split.items() if s in args.splits]

    if hasattr(args, "system_ids_file") and args.system_ids_file is not None:
        allowed = set(Path(args.system_ids_file).read_text().split())
        before = len(system_ids)
        system_ids = [sid for sid in system_ids if sid in allowed]
        print(f"system_ids_file: {before} → {len(system_ids)} after pocket filter")

    if args.max_systems is not None:
        system_ids = system_ids[:args.max_systems]

    print(f"Processing {len(system_ids)} systems...")

    num_processes = min(args.num_processes, multiprocessing.cpu_count(), len(system_ids))

    pocket_data_dir = getattr(args, "pocket_data_dir", None)
    allowed_protein_chain_counts = tuple(getattr(args, "allowed_protein_chain_counts", (1,)))
    fuse_multi_chain = getattr(args, "fuse_multi_chain", False)

    if num_processes > 1:
        initargs = (plinder_dir, outdir, clusters, annotations, args.ccd_path, pocket_data_dir,
                    allowed_protein_chain_counts, fuse_multi_chain)
        with multiprocessing.Pool(
            processes=num_processes,
            initializer=_worker_init,
            initargs=initargs,
        ) as pool:
            results = list(tqdm(pool.imap_unordered(process_system_worker, system_ids, chunksize=4), total=len(system_ids)))
    else:
        _worker_init(plinder_dir, outdir, clusters, annotations, args.ccd_path, pocket_data_dir,
                      allowed_protein_chain_counts, fuse_multi_chain)
        results = [process_system_worker(sid) for sid in tqdm(system_ids)]

    # Tally filter reasons
    filtered_ids: dict[str, list[str]] = {}
    for sid, reason in results:
        if reason is not None:
            filtered_ids.setdefault(reason, []).append(sid)

    filter_counts = {reason: len(ids) for reason, ids in filtered_ids.items()}
    n_ok = sum(1 for _, r in results if r is None)
    print(f"Processed: {n_ok} written, {len(system_ids) - n_ok} filtered")
    for reason, count in sorted(filter_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    with open(outdir / "filter_stats.json", "w") as f:
        json.dump(filter_counts, f, indent=2)
    with open(outdir / "filtered_ids.json", "w") as f:
        json.dump(filtered_ids, f, indent=2)

    return finalize(outdir)



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
    parser.add_argument("--system-ids-file", type=Path, default=None,
                        help="Optional text file of allowed system IDs (one per line); from filter_plinder_pocket.py")
    parser.add_argument("--max-systems", type=int, default=None,
                        help="Cap number of systems per split (for debugging)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Delete and recreate the output directory before processing")
    parser.add_argument("--pocket-data-dir", type=Path,
                        default=Path(os.environ.get("REPO_ROOT", ".")) / "plinder_pocket_alpha_spheres",
                        help="Directory of per-system alpha-sphere .npy files from filter_plinder_pocket.py")
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

    split_counts = {}
    for split_name in ["train", "val", "test"]:
        print(f"\n=== Processing split: {split_name} ===")
        split_args = argparse.Namespace(**{**vars(args), "splits": [split_name], "outdir": args.outdir / split_name})
        split_counts[split_name] = process(split_args, clusters, annotations, split)

    stats = {**split_counts, "total": sum(split_counts.values())}
    args.outdir.mkdir(parents=True, exist_ok=True)
    with open(args.outdir / "dataset_stats.yaml", "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote dataset stats to {args.outdir / 'dataset_stats.yaml'}")
