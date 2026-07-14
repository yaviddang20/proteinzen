"""
Trajectory CoM evaluation for protein-conditioned ligand generation.

For each sampled trajectory (*_traj_clean.pdb), computes the ligand
center-of-mass distance to the GT crystal ligand CoM at each ODE step.

Usage
-----
python _scripts/traj_com_eval.py \
    --traj-dir sampling/plinder/protein_cond/plinder_protein_cond/traj \
    --ref-dir  plinder_processed/val
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

_GPU_SUFFIX = re.compile(r'_gpu\d+_batch\d+_idx\d+')


def _parse_xyz(line):
    return float(line[30:38]), float(line[38:46]), float(line[46:54])

def _parse_elem(line):
    elem = (line[76:78].strip() if len(line) > 76 else "").capitalize()
    if not elem:
        elem = line[12:16].strip().lstrip("0123456789").capitalize()
    return elem

def parse_traj_pdb(path: Path):
    """Parse a multi-MODEL PDB. Returns list of (prot_coords, lig_coords) per step."""
    steps = []
    cur_prot, cur_lig = [], []
    in_model = False
    with open(path) as fh:
        for line in fh:
            rec = line[:6].rstrip()
            if rec == "MODEL":
                in_model = True
                cur_prot, cur_lig = [], []
            elif rec == "ENDMDL":
                prot = np.array(cur_prot, dtype=np.float64) if cur_prot else np.zeros((0, 3))
                lig  = np.array(cur_lig,  dtype=np.float64) if cur_lig  else np.zeros((0, 3))
                steps.append((prot, lig))
                in_model = False
            elif in_model:
                if rec in ("ATOM", "HETATM"):
                    try:
                        elem = _parse_elem(line)
                        if elem in ("H", "D"):
                            continue
                        xyz = _parse_xyz(line)
                        if rec == "ATOM":
                            cur_prot.append(xyz)
                        else:
                            cur_lig.append(xyz)
                    except ValueError:
                        pass
    return steps


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


def load_gt(npz_path: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Returns (gt_prot_coords, gt_lig_coords) heavy atoms only."""
    from proteinzen.runtime.sampling.protein_pocket import load_structure_from_npz, _crop_protein_to_pocket
    from proteinzen.boltz.data import const
    protein_id    = const.chain_type_ids["PROTEIN"]
    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]
    try:
        struct = load_structure_from_npz(npz_path, include_h=False)
        struct = _crop_protein_to_pocket(struct, max_protein_residues=100)
    except Exception as e:
        print(f"  [GT load error] {npz_path}: {e}")
        return None, None
    prot_list, lig_list = [], []
    for chain in struct.chains[struct.mask]:
        mol = int(chain["mol_type"])
        a0  = int(chain["atom_idx"])
        atoms = struct.atoms[a0: a0 + int(chain["atom_num"])]
        present = atoms["is_present"].astype(bool)
        heavy = present & (atoms["element"] != 1)
        if mol == protein_id:
            prot_list.append(atoms["coords"][heavy])
        elif mol == nonpolymer_id:
            lig_list.append(atoms["coords"][heavy])
    if not lig_list:
        return None, None
    prot = np.concatenate(prot_list, 0).astype(np.float64) if prot_list else np.zeros((0, 3))
    lig  = np.concatenate(lig_list,  0).astype(np.float64)
    return prot, lig


def system_id_from_stem(stem: str) -> str | None:
    m = _GPU_SUFFIX.search(stem)
    if m:
        return stem[:m.start()]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=Path, required=True,
                    help="Directory containing *_traj_clean.pdb files")
    ap.add_argument("--ref-dir", type=Path, required=True,
                    help="Plinder split dir with manifest.json + structures/")
    ap.add_argument("--n-steps-report", type=int, default=50,
                    help="Number of evenly-spaced steps to report (default 50)")
    args = ap.parse_args()

    manifest_path = args.ref_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"manifest.json not found at {manifest_path}")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    npz_by_sid = {}
    for rec in manifest:
        sid = rec["id"]
        mid = sid[1:3]
        npz = args.ref_dir / "structures" / mid / f"{sid}.npz"
        if npz.exists():
            npz_by_sid[sid] = str(npz)

    traj_files = sorted(args.traj_dir.glob("*_clean_traj.pdb"))
    if not traj_files:
        sys.exit(f"No *_traj_clean.pdb files found in {args.traj_dir}")
    print(f"Found {len(traj_files)} trajectory files")

    # group by system id
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in traj_files:
        sid = system_id_from_stem(p.stem.replace("_clean_traj", ""))
        if sid:
            groups[sid].append(p)

    common = sorted(set(groups) & set(npz_by_sid))
    print(f"Systems with trajectories and GT: {len(common)}")

    all_com_dists: list[np.ndarray] = []  # per trajectory, array of CoM dist per step

    for sid in tqdm(common, desc="evaluating"):
        gt_prot, gt_lig = load_gt(npz_by_sid[sid])
        if gt_lig is None:
            continue
        gt_com = gt_lig.mean(0)
        n_gt_prot = len(gt_prot)
        for traj_path in sorted(groups[sid]):
            steps = parse_traj_pdb(traj_path)
            if not steps:
                continue
            dists = []
            for gen_prot, gen_lig in steps:
                if len(gen_lig) == 0:
                    dists.append(np.nan)
                    continue
                # Kabsch-align generated protein onto GT protein, apply to ligand
                if len(gen_prot) > 0 and n_gt_prot > 0:
                    n_common = min(len(gen_prot), n_gt_prot)
                    R, t = kabsch(gen_prot[:n_common], gt_prot[:n_common])
                    aligned_lig = apply_transform(gen_lig, R, t)
                else:
                    aligned_lig = gen_lig
                com = aligned_lig.mean(0)
                dists.append(float(np.linalg.norm(com - gt_com)))
            all_com_dists.append(np.array(dists))

    if not all_com_dists:
        print("No valid trajectories found.")
        return

    indices = np.linspace(0, 1, args.n_steps_report)

    print(f"\n{'='*60}")
    print(f"  TRAJECTORY CoM DISTANCE  ({len(all_com_dists)} trajectories)")
    print(f"{'='*60}")
    print(f"  {'step':>6}  {'mean':>8}  {'median':>8}  {'<2Å':>7}  {'<5Å':>7}")

    for frac in indices:
        col = []
        for d in all_com_dists:
            idx = int(round(frac * (len(d) - 1)))
            v = d[idx]
            if np.isfinite(v):
                col.append(v)
        if not col:
            continue
        col = np.array(col)
        label = f"{frac:.2f}"
        print(f"  {label:>6}  {col.mean():>8.3f}  {np.median(col):>8.3f}  "
              f"{np.mean(col < 2)*100:>6.1f}%  {np.mean(col < 5)*100:>6.1f}%")

    # final step summary
    finals = [d[-1] for d in all_com_dists if np.isfinite(d[-1])]
    if finals:
        finals = np.array(finals)
        print(f"\n  Final step: mean={finals.mean():.3f}  median={np.median(finals):.3f}  "
              f"<2Å={np.mean(finals<2)*100:.1f}%  <5Å={np.mean(finals<5)*100:.1f}%")

    # build per-fraction arrays for plotting
    means, medians, p25, p75 = [], [], [], []
    for frac in indices:
        col = []
        for d in all_com_dists:
            idx = int(round(frac * (len(d) - 1)))
            v = d[idx]
            if np.isfinite(v):
                col.append(v)
        col = np.array(col) if col else np.array([np.nan])
        means.append(np.nanmean(col))
        medians.append(np.nanmedian(col))
        p25.append(np.nanpercentile(col, 25))
        p75.append(np.nanpercentile(col, 75))

    means, medians = np.array(means), np.array(medians)
    p25, p75 = np.array(p25), np.array(p75)

    out_dir = args.traj_dir.parent / "traj_com"
    out_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(indices, means,   label="mean",   color="steelblue", lw=2)
    ax.plot(indices, medians, label="median", color="orange",    lw=2, ls="--")
    ax.fill_between(indices, p25, p75, alpha=0.2, color="steelblue", label="25–75%")
    ax.set_xlabel("ODE step (fraction of trajectory)")
    ax.set_ylabel("CoM distance to GT (Å)")
    ax.set_title(f"Ligand CoM distance over trajectory  (n={len(all_com_dists)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = out_dir / "com_dist_traj.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
