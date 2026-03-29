"""
Pi-stacking analysis on geom_drugs_conformers training data.

Pi stacking criteria (using vertical + lateral decomposition for parallel cases):
  - Face-to-face (FF): normal angle 0-30°, vertical dist 3.3-4.5 Å, lateral offset <1.5 Å
  - Displaced parallel (DP): normal angle 0-30°, vertical dist 3.3-4.5 Å, lateral offset 1.5-3.5 Å
  - T-shaped / edge-to-face (EF): centroid dist 4.0-6.0 Å, normal angle 60-90°
"""

import numpy as np
import os
import random
from pathlib import Path
from multiprocessing import Pool
from collections import defaultdict
import json

import sys
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"
DATA_DIR = Path(f"data/geom_drugs_conformers/{SPLIT}/structures")
SAMPLE_SIZE = 50_000   # conformers to sample
N_WORKERS = 16
SEED = 42

BOND_TYPE_AROMATIC = 4

# Pi stacking thresholds
PARALLEL_ANGLE_MAX = 30.0   # degrees between ring normals
VERTICAL_DIST_MIN = 3.3     # Å — component along ring normal
VERTICAL_DIST_MAX = 4.5
FF_LATERAL_MAX = 1.5        # Å — lateral offset for face-to-face
DP_LATERAL_MAX = 3.5        # Å — lateral offset for displaced parallel

EF_DIST_MAX = 6.0           # centroid-centroid dist for T-shaped
EF_DIST_MIN = 4.0
EF_ANGLE_MIN = 60.0
EF_ANGLE_MAX = 90.0


def get_ring_normal(coords):
    """Fit a plane to ring atom coords, return unit normal."""
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # least variance direction
    return centroid, normal / np.linalg.norm(normal)


def angle_between_normals(n1, n2):
    """Angle in degrees between two unit normals (0-90°)."""
    cos = abs(np.dot(n1, n2))
    cos = np.clip(cos, 0, 1)
    return np.degrees(np.arccos(cos))


def analyze_file(npz_path):
    """
    Returns a dict of results for one conformer, or None on error.
    """
    try:
        d = np.load(npz_path, allow_pickle=True)
        atoms = d['atoms']
        bonds = d['bonds']
        ring_masks = d['ring_masks']

        # ring_masks is indexed over heavy atoms only; build heavy-atom coords
        # and a mapping from all-atom index → heavy-atom index for bond lookup
        heavy_mask = atoms['element'] != 1
        heavy_coords = atoms['coords'][heavy_mask]  # (n_heavy, 3)
        all_to_heavy = {}
        for heavy_idx, all_idx in enumerate(np.where(heavy_mask)[0]):
            all_to_heavy[int(all_idx)] = heavy_idx

        if ring_masks.shape[0] < 2:
            return {'n_rings': ring_masks.shape[0], 'n_aromatic_rings': 0,
                    'has_pi_stack': False, 'stack_type': None,
                    'n_pairs': 0}

        # Aromatic atoms in heavy-atom index space
        aromatic_atom_set = set()
        for b in bonds:
            if b['type'] == BOND_TYPE_AROMATIC:
                for a in (int(b['atom_1']), int(b['atom_2'])):
                    if a in all_to_heavy:
                        aromatic_atom_set.add(all_to_heavy[a])

        aromatic_rings = []
        for i in range(ring_masks.shape[0]):
            ring_atom_idx = np.where(ring_masks[i])[0]  # heavy-atom indices
            if len(ring_atom_idx) < 5:
                continue
            n_aromatic = sum(1 for a in ring_atom_idx if a in aromatic_atom_set)
            if n_aromatic >= len(ring_atom_idx) * 0.6:
                ring_coords = heavy_coords[ring_atom_idx]
                centroid, normal = get_ring_normal(ring_coords)
                aromatic_rings.append((centroid, normal, len(ring_atom_idx)))

        n_aromatic = len(aromatic_rings)
        if n_aromatic < 2:
            return {'n_rings': ring_masks.shape[0], 'n_aromatic_rings': n_aromatic,
                    'has_pi_stack': False, 'stack_type': None,
                    'n_pairs': 0}

        # Check all ring pairs for pi stacking
        stack_types = []
        for i in range(n_aromatic):
            for j in range(i + 1, n_aromatic):
                c1, n1, _ = aromatic_rings[i]
                c2, n2, _ = aromatic_rings[j]
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

        has_stack = len(stack_types) > 0
        primary_type = stack_types[0] if stack_types else None

        return {
            'n_rings': ring_masks.shape[0],
            'n_aromatic_rings': n_aromatic,
            'has_pi_stack': has_stack,
            'stack_type': primary_type,
            'all_stack_types': stack_types,
            'n_pairs': n_aromatic * (n_aromatic - 1) // 2,
        }
    except Exception as e:
        return None


def main():
    random.seed(SEED)

    # Collect all npz paths
    print("Collecting file paths...")
    all_files = list(DATA_DIR.rglob("*.npz"))
    print(f"Total conformer files: {len(all_files):,}")

    if len(all_files) > SAMPLE_SIZE:
        sampled = random.sample(all_files, SAMPLE_SIZE)
    else:
        sampled = all_files

    print(f"Analyzing {len(sampled):,} conformers with {N_WORKERS} workers...")

    with Pool(N_WORKERS) as pool:
        results = pool.map(analyze_file, sampled)

    # Filter out errors
    results = [r for r in results if r is not None]
    print(f"Valid results: {len(results):,}")

    # Aggregate stats
    total = len(results)
    n_with_stack = sum(1 for r in results if r['has_pi_stack'])
    n_multiring = sum(1 for r in results if r['n_aromatic_rings'] >= 2)

    type_counts = defaultdict(int)
    for r in results:
        for t in r.get('all_stack_types', []):
            type_counts[t] += 1

    # Molecules with >=1 pi stack
    frac_any_stack = n_with_stack / total
    frac_multiring_with_stack = n_with_stack / n_multiring if n_multiring > 0 else 0

    # Distribution of n_aromatic_rings
    ring_counts = defaultdict(int)
    for r in results:
        ring_counts[r['n_aromatic_rings']] += 1

    print("\n===== PI STACKING ANALYSIS =====")
    print(f"Total conformers analyzed: {total:,}")
    print(f"Conformers with >=2 aromatic rings: {n_multiring:,} ({n_multiring/total:.1%})")
    print(f"Conformers with pi stacking: {n_with_stack:,} ({frac_any_stack:.1%})")
    print(f"  Among multi-ring mols: {frac_multiring_with_stack:.1%}")
    print()
    print("Stacking type counts (can overlap if multiple pairs):")
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,}")
    print()
    print("Aromatic ring count distribution:")
    for n in sorted(ring_counts.keys()):
        print(f"  {n} aromatic rings: {ring_counts[n]:,} ({ring_counts[n]/total:.1%})")

    # Save results
    summary = {
        'total_analyzed': total,
        'n_multiring': n_multiring,
        'n_with_any_stack': n_with_stack,
        'frac_any_stack': frac_any_stack,
        'frac_multiring_with_stack': frac_multiring_with_stack,
        'type_counts': dict(type_counts),
        'ring_count_distribution': {str(k): v for k, v in ring_counts.items()},
    }
    out_path = Path(f"analysis/pi_stacking_results_{SPLIT}.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
