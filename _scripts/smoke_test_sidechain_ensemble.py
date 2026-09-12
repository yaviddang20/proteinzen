"""
Standalone smoke test for proteinzen/data/datasets/sidechain_ensemble.py
against real processed data + a real generated ensemble. Not part of any
automated test suite -- run manually to sanity-check the alignment logic
before trusting it in actual training.

Usage:
    python smoke_test_sidechain_ensemble.py <system_id>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/mnt/scratch/user/daviyang/proteinzen"))
sys.path.insert(0, str(REPO_ROOT))

from proteinzen.data.datasets.datamodule import load_input
from proteinzen.boltz.data.types import Record
from proteinzen.data.datasets.sidechain_ensemble import (
    load_auth_seq_map,
    load_ensemble_substitution,
    apply_sidechain_ensemble,
    _decode_atom_name,
    _BACKBONE_ATOM_NAMES,
)

DATA_DIR = REPO_ROOT / "plinder_pocket_processed"
ENSEMBLE_ROOT = REPO_ROOT / "plinder_pocket_processed_sc_ensembles"

system_id = sys.argv[1]

manifest = json.loads((DATA_DIR / "train" / "manifest.json").read_text())
record_dict = next(r for r in manifest if r["id"] == system_id)
record = Record.from_dict(record_dict)

struct, *_ = load_input(record, DATA_DIR / "train")
auth_seq_map = load_auth_seq_map(DATA_DIR, "train", system_id)
print(f"auth_seq_map: {len(auth_seq_map) if auth_seq_map else None} chains")

substitution = load_ensemble_substitution(ENSEMBLE_ROOT, system_id, rng=np.random)
if substitution is None:
    print("No substitution available for this system -- pick a different system_id")
    sys.exit(1)
print(f"substitution covers {len(substitution)} residues")

new_struct = apply_sidechain_ensemble(struct, auth_seq_map, substitution)

old_coords = struct.atoms["coords"]
new_coords = new_struct.atoms["coords"]
changed_atoms = np.where(np.any(old_coords != new_coords, axis=1))[0]
print(f"{len(changed_atoms)} / {len(old_coords)} atoms changed")

n_backbone_changed = 0
n_ligand_changed = 0
protein_id = 0
for a in changed_atoms:
    name = _decode_atom_name(struct.atoms[a]["name"])
    if name in _BACKBONE_ATOM_NAMES:
        n_backbone_changed += 1

for chain in struct.chains:
    if int(chain["mol_type"]) == protein_id:
        continue
    a_start, a_num = int(chain["atom_idx"]), int(chain["atom_num"])
    lig_changed = set(range(a_start, a_start + a_num)) & set(changed_atoms.tolist())
    n_ligand_changed += len(lig_changed)

print(f"backbone atoms changed (should be 0): {n_backbone_changed}")
print(f"ligand atoms changed (should be 0): {n_ligand_changed}")

if len(changed_atoms) > 0 and n_backbone_changed == 0 and n_ligand_changed == 0:
    print("PASS: substitution touched only non-backbone, non-ligand atoms")
else:
    print("FAIL: check output above")
