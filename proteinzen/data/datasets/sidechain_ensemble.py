"""
Splice alternate sidechain conformations, generated offline via PyRosetta
(see _scripts/sidechain_diversity.py), into a loaded training Structure --
for non-pocket-critical ("packable") residues only. Purely optional data
augmentation: falls back to a no-op whenever a system has no generated
ensemble, or any required metadata is missing.

Alignment chain (why this needs three pieces of metadata, not one):
  packable_resnums (Rosetta pose index, no relation to author numbering)
    -> pose_resnum_map: pose index -> [gemmi single-char chain id, auth_resnum]
    -> chain_map.json (reversed): gemmi chain id -> original Plinder chain_name
    -> auth_seq_map: (chain_name, auth_resnum) -> position in struct.residues
"""
import json
import random as _random
from dataclasses import replace
from pathlib import Path
from typing import Optional

import gemmi

_BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O"}


def _decode_atom_name(name_arr) -> str:
    return "".join(chr(int(c) + 32) for c in name_arr if c != 0).strip()


def _auth_mid(system_id: str) -> str:
    if "AF-" in system_id or "af-" in system_id:
        return system_id[6:8]
    return system_id[1:3]


def load_auth_seq_map(data_dir, mode: str, system_id: str) -> Optional[list]:
    """Per-chain list of author residue numbers, parallel to struct.chains
    order; each inner list has length chain['res_num']."""
    path = Path(data_dir) / mode / "auth_maps" / _auth_mid(system_id) / f"{system_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_ensemble_substitution(ensemble_root, system_id: str, rng=_random) -> Optional[dict]:
    """Pick one random ensemble sample for system_id and return
    {(orig_chain_name, auth_resnum): {atom_name: (x, y, z)}}, restricted to
    packable residues, or None if unavailable/incomplete."""
    sys_dir = Path(ensemble_root) / system_id
    stats_path = sys_dir / "ensemble_stats.json"
    chain_map_path = sys_dir / "chain_map.json"
    if not stats_path.exists() or not chain_map_path.exists():
        return None

    with open(stats_path) as f:
        stats = json.load(f)
    samples = stats.get("samples")
    pose_resnum_map = stats.get("pose_resnum_map")
    if not samples or not pose_resnum_map:
        return None

    with open(chain_map_path) as f:
        chain_map = json.load(f)  # orig_chain_name -> gemmi_chain_id
    gemmi_to_orig = {v: k for k, v in chain_map.items()}

    packable = set(stats.get("packable_resnums", []))
    # (gemmi_chain_id, resnum) -> orig_chain_name, restricted to packable pose indices
    packable_targets = {}
    for pose_idx in packable:
        entry = pose_resnum_map.get(str(pose_idx))
        if entry is None:
            continue
        gemmi_chain, resnum = entry
        orig_chain = gemmi_to_orig.get(gemmi_chain)
        if orig_chain is None:
            continue
        packable_targets[(gemmi_chain, resnum)] = orig_chain
    if not packable_targets:
        return None

    # rng.random() rather than rng.choice(samples) -- np.random.choice on a
    # list of dicts risks numpy trying to coerce it into an ndarray; a plain
    # float index works identically for both `random` and `np.random`.
    idx = min(int(rng.random() * len(samples)), len(samples) - 1)
    sample = samples[idx]
    sample_pdb = sys_dir / f"sample_{sample['sample_index']:03d}.pdb"
    if not sample_pdb.exists():
        return None

    st = gemmi.read_structure(str(sample_pdb))
    if len(st) == 0:
        return None
    model = st[0]

    result = {}
    for chain in model:
        for res in chain:
            key = (chain.name, res.seqid.num)
            orig_chain = packable_targets.get(key)
            if orig_chain is None:
                continue
            result[(orig_chain, res.seqid.num)] = {
                atom.name: (atom.pos.x, atom.pos.y, atom.pos.z) for atom in res
            }
    return result if result else None


def apply_sidechain_ensemble(struct, auth_seq_map, substitution):
    """Overwrite non-backbone atom coordinates in `struct` for residues
    matched (via auth_seq_map) against substitution's
    (orig_chain_name, auth_resnum) -> {atom_name: xyz} entries. Returns a
    new Structure (atoms array copied on first real write); returns
    `struct` unchanged if there's nothing to substitute or metadata is
    missing/inconsistent -- this must never raise on mismatched data, since
    it's opportunistic augmentation, not a required path.
    """
    if not substitution or not auth_seq_map:
        return struct

    new_atoms = None
    n_chains = min(len(struct.chains), len(auth_seq_map))
    for chain_idx in range(n_chains):
        chain = struct.chains[chain_idx]
        chain_entry = auth_seq_map[chain_idx]
        chain_name = chain_entry.get("chain_name")
        auth_indices = chain_entry.get("auth_indices") or []
        res_start = int(chain["res_idx"])
        res_num = int(chain["res_num"])
        if len(auth_indices) != res_num:
            # auth_seq_map out of sync with this chain -- skip rather than misalign
            continue
        for local_idx, auth_id in enumerate(auth_indices):
            if auth_id is None:
                continue
            try:
                auth_num = int(auth_id)
            except (TypeError, ValueError):
                continue
            atom_coords = substitution.get((chain_name, auth_num))
            if atom_coords is None:
                continue
            res = struct.residues[res_start + local_idx]
            a_start, a_num = int(res["atom_idx"]), int(res["atom_num"])
            for a in range(a_start, a_start + a_num):
                name = _decode_atom_name(struct.atoms[a]["name"])
                if name in _BACKBONE_ATOM_NAMES:
                    continue
                xyz = atom_coords.get(name)
                if xyz is None:
                    continue
                if new_atoms is None:
                    new_atoms = struct.atoms.copy()
                new_atoms[a]["coords"] = xyz

    if new_atoms is None:
        return struct
    return replace(struct, atoms=new_atoms)
