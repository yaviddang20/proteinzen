"""
Pi-stacking analysis on generated conformers from samples/ directory.

Uses ring topology (ring_masks + bond types) from the corresponding reference
NPZ in the training data, applied to the generated PDB coordinates.
"""

import numpy as np
import os
import glob
from pathlib import Path
from multiprocessing import Pool
from collections import defaultdict
import json

SAMPLES_DIR = Path("sampling/geom_conformer_train/geom_identityRot_256_conformer_6std_stereo_norm_scale/samples")
NPZ_BASE = Path("data/geom_drugs_conformers/train/structures")
N_WORKERS = 16

BOND_TYPE_AROMATIC = 4

PARALLEL_ANGLE_MAX = 30.0
VERTICAL_DIST_MIN = 3.3
VERTICAL_DIST_MAX = 4.5
FF_LATERAL_MAX = 1.5
DP_LATERAL_MAX = 3.5

EF_DIST_MAX = 6.0
EF_DIST_MIN = 4.0
EF_ANGLE_MIN = 60.0


def get_ring_normal(coords):
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]
    return centroid, normal / np.linalg.norm(normal)


def angle_between_normals(n1, n2):
    cos = abs(np.dot(n1, n2))
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))


def find_reference_npz(mol_hash):
    """Find any reference npz for this molecule hash."""
    # Subdir is first 2 chars of hash (matching records layout)
    subdir = mol_hash[1:3]
    # Try _0 conformer first
    candidate = NPZ_BASE / subdir / f"{mol_hash}_0.npz"
    if candidate.exists():
        return candidate
    # Fall back to glob
    matches = list((NPZ_BASE / subdir).glob(f"{mol_hash}_*.npz"))
    return matches[0] if matches else None


def parse_pdb_coords(pdb_path):
    """Extract heavy atom coordinates from a single-model PDB."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords, dtype=np.float32)


def analyze_generated(pdb_path):
    pdb_path = Path(pdb_path)
    try:
        # Parse molecule hash from filename: {hash}_{idx}.pdb
        stem = pdb_path.stem
        mol_hash = '_'.join(stem.split('_')[:-1])

        # Load reference topology
        ref_npz = find_reference_npz(mol_hash)
        if ref_npz is None:
            return None

        d = np.load(ref_npz, allow_pickle=True)
        atoms = d['atoms']
        bonds = d['bonds']
        ring_masks = d['ring_masks']
        heavy_mask = atoms['element'] != 1

        # ring_masks shape: (n_rings, n_heavy_atoms)
        if ring_masks.shape[0] < 2:
            return {'n_aromatic_rings': 0, 'has_pi_stack': False,
                    'all_stack_types': [], 'n_pairs': 0}

        # Get generated coordinates
        gen_coords = parse_pdb_coords(pdb_path)
        if len(gen_coords) != heavy_mask.sum():
            return None  # atom count mismatch

        # Identify aromatic atoms from reference bond types
        aromatic_atom_set = set()
        for b in bonds:
            if b['type'] == BOND_TYPE_AROMATIC:
                # Bond indices are over ALL atoms; map to heavy-atom indices
                a1, a2 = int(b['atom_1']), int(b['atom_2'])
                aromatic_atom_set.add(a1)
                aromatic_atom_set.add(a2)

        # Map all-atom indices → heavy-atom indices
        all_to_heavy = {}
        heavy_idx = 0
        for i, is_heavy in enumerate(heavy_mask):
            if is_heavy:
                all_to_heavy[i] = heavy_idx
                heavy_idx += 1

        aromatic_heavy_set = {all_to_heavy[a] for a in aromatic_atom_set if a in all_to_heavy}

        # Identify aromatic rings
        aromatic_rings = []
        for i in range(ring_masks.shape[0]):
            ring_atom_idx = np.where(ring_masks[i])[0]  # heavy-atom indices
            if len(ring_atom_idx) < 5:
                continue
            n_arom = sum(1 for a in ring_atom_idx if a in aromatic_heavy_set)
            if n_arom >= len(ring_atom_idx) * 0.6:
                ring_coords = gen_coords[ring_atom_idx]
                centroid, normal = get_ring_normal(ring_coords)
                aromatic_rings.append((centroid, normal))

        n_aromatic = len(aromatic_rings)
        if n_aromatic < 2:
            return {'n_aromatic_rings': n_aromatic, 'has_pi_stack': False,
                    'all_stack_types': [], 'n_pairs': 0}

        stack_types = []
        for i in range(n_aromatic):
            for j in range(i + 1, n_aromatic):
                c1, n1 = aromatic_rings[i]
                c2, n2 = aromatic_rings[j]
                vec = c2 - c1
                dist = np.linalg.norm(vec)
                angle = angle_between_normals(n1, n2)

                if angle <= PARALLEL_ANGLE_MAX:
                    vertical = abs(np.dot(vec, n1))
                    lateral = np.sqrt(max(dist**2 - vertical**2, 0.0))
                    if VERTICAL_DIST_MIN <= vertical <= VERTICAL_DIST_MAX:
                        if lateral <= FF_LATERAL_MAX:
                            stack_types.append('face_to_face')
                        elif lateral <= DP_LATERAL_MAX:
                            stack_types.append('displaced_parallel')
                        else:
                            stack_types.append('failed_parallel_over_shifted')
                    else:
                        stack_types.append('failed_parallel_too_far')
                elif angle >= EF_ANGLE_MIN:
                    if EF_DIST_MIN <= dist <= EF_DIST_MAX:
                        stack_types.append('t_shaped')
                    else:
                        stack_types.append('failed_t_shaped')

        return {
            'n_aromatic_rings': n_aromatic,
            'has_pi_stack': len(stack_types) > 0,
            'all_stack_types': stack_types,
            'n_pairs': n_aromatic * (n_aromatic - 1) // 2,
        }
    except Exception:
        return None


def main():
    pdb_files = list(SAMPLES_DIR.glob("*.pdb"))
    print(f"Found {len(pdb_files):,} generated PDB files")

    with Pool(N_WORKERS) as pool:
        results = pool.map(analyze_generated, pdb_files)

    results = [r for r in results if r is not None]
    print(f"Valid results: {len(results):,}")

    total = len(results)
    n_multiring = sum(1 for r in results if r['n_aromatic_rings'] >= 2)
    n_with_stack = sum(1 for r in results if r['has_pi_stack'])

    type_counts = defaultdict(int)
    ring_counts = defaultdict(int)
    for r in results:
        for t in r['all_stack_types']:
            type_counts[t] += 1
        ring_counts[r['n_aromatic_rings']] += 1

    print("\n===== PI STACKING ANALYSIS (GENERATED) =====")
    print(f"Total conformers: {total:,}")
    print(f"With >=2 aromatic rings: {n_multiring:,} ({n_multiring/total:.1%})")
    print(f"With pi stacking: {n_with_stack:,} ({n_with_stack/total:.1%})")
    print(f"  Among multi-ring: {n_with_stack/n_multiring:.1%}" if n_multiring else "")
    print()
    print("Stacking type counts:")
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,}")
    print()
    print("Aromatic ring count distribution:")
    for n in sorted(ring_counts.keys()):
        print(f"  {n} aromatic rings: {ring_counts[n]:,} ({ring_counts[n]/total:.1%})")

    summary = {
        'total_analyzed': total,
        'n_multiring': n_multiring,
        'n_with_any_stack': n_with_stack,
        'frac_any_stack': n_with_stack / total,
        'frac_multiring_with_stack': n_with_stack / n_multiring if n_multiring else 0,
        'type_counts': dict(type_counts),
        'ring_count_distribution': {str(k): v for k, v in ring_counts.items()},
    }
    out = Path("analysis/pi_stacking_results_generated.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
