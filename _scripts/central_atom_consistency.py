"""
For each molecule in geom_drugs_conformers (processed NPZ), measure how
consistently the same atom is the geometric centre (closest to unweighted
centroid) across:
  (a) all processed GEOM conformers (one NPZ per conformer)
  (b) 20 RDKit ETKDG-generated conformers

Reports:
  - top-1 consistency  : fraction of conformers where rank-1 central atom == majority atom
  - top-3 consistency  : fraction of conformers where {top-3 atoms} == majority set (unordered)
                         and a secondary "rank-ordered" version
"""

import argparse
import json
import random
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

PROJECT_DIR  = Path(__file__).parent.parent
GEOM_DIR     = PROJECT_DIR / "data" / "geom_drugs_conformers" / "train"
MANIFEST_PATH = GEOM_DIR / "manifest.json"


# ── helpers ──────────────────────────────────────────────────────────────────

def central_atom_ranks(coords):
    """coords: (N,3) numpy array. Returns atom indices sorted closest-to-centroid first."""
    centroid = coords.mean(axis=0)
    dists = np.linalg.norm(coords - centroid, axis=1)
    return np.argsort(dists).tolist()


def consistency(items):
    if not items:
        return 0.0, 0, 0
    cnt = Counter(items)
    top_count = cnt.most_common(1)[0][1]
    return top_count / len(items), top_count, len(items)


def _compute_metrics(ranks_list):
    n = len(ranks_list)
    frac1, _, _ = consistency([r[0] for r in ranks_list])
    frac3u, _, _ = consistency([frozenset(r[:3]) for r in ranks_list])
    frac3o, _, _ = consistency([tuple(r[:3]) for r in ranks_list])
    return dict(n_conf=n, top1=frac1, top3_unordered=frac3u, top3_ordered=frac3o,
                n_atoms=len(ranks_list[0]))


# ── GEOM (processed NPZ) analysis ────────────────────────────────────────────

def _load_npz_coords(npz_path):
    """Load heavy-atom coords from a processed NPZ; returns (N,3) or None."""
    try:
        data = np.load(npz_path)
        atoms = data['atoms']
        coords = atoms['coords'].astype(np.float32)   # (N, 3)
        return coords
    except Exception:
        return None


def analyse_geom(entry, structures_dir):
    ids = entry.get('ids', [])
    if len(ids) < 2:
        return None

    ranks_list = []
    for mol_id in ids:
        subdir = mol_id[1:3]
        npz_path = structures_dir / subdir / f"{mol_id}.npz"
        coords = _load_npz_coords(npz_path)
        if coords is None or len(coords) < 3:
            continue
        ranks_list.append(central_atom_ranks(coords))

    if len(ranks_list) < 2:
        return None
    return _compute_metrics(ranks_list)


# ── RDKit conformer analysis ──────────────────────────────────────────────────

def analyse_rdkit(entry, n_rdkit=20, seed=42):
    try:
        method = entry['structures'][0]['method']
        if not method.startswith('QM9:'):
            return None
        smiles = method[4:]
    except (KeyError, IndexError):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1

    mol_multi = Chem.RWMol(mol)
    ids = AllChem.EmbedMultipleConfs(mol_multi, numConfs=n_rdkit, params=params)
    if len(ids) < 2:
        return None

    mol_multi = mol_multi.GetMol()
    AllChem.MMFFOptimizeMoleculeConfs(mol_multi, numThreads=1, mmffVariant="MMFF94s")
    mol_noh = AllChem.RemoveHs(mol_multi)
    if mol_noh.GetNumAtoms() < 3:
        return None

    ranks_list = []
    for cid in mol_noh.GetConformers():
        conf = cid
        coords = np.array([conf.GetAtomPosition(i) for i in range(mol_noh.GetNumAtoms())])
        ranks_list.append(central_atom_ranks(coords))

    if len(ranks_list) < 2:
        return None
    return _compute_metrics(ranks_list)


# ── worker ───────────────────────────────────────────────────────────────────

_structures_dir = None
_n_rdkit = None

def _init_worker(structures_dir, n_rdkit):
    global _structures_dir, _n_rdkit
    _structures_dir = structures_dir
    _n_rdkit = n_rdkit


def _worker(entry):
    g = analyse_geom(entry, _structures_dir)
    r = analyse_rdkit(entry, n_rdkit=_n_rdkit)
    return g, r


# ── summary stats ────────────────────────────────────────────────────────────

def summarise(results, label):
    vals = {k: [] for k in ("top1", "top3_unordered", "top3_ordered", "n_conf", "n_atoms")}
    for r in results:
        if r is None:
            continue
        for k in vals:
            vals[k].append(r[k])

    n = len(vals["top1"])
    if n == 0:
        print(f"\n{label}: no valid molecules")
        return

    def pct(arr, lbl):
        a = np.array(arr)
        print(f"  {lbl:25s}  mean={a.mean():.3f}  median={np.median(a):.3f}"
              f"  >=0.9: {(a>=0.9).mean()*100:.1f}%  >=0.5: {(a>=0.5).mean()*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"{label}  (n={n} molecules)")
    print(f"  conformers/mol: mean={np.mean(vals['n_conf']):.1f}  median={np.median(vals['n_conf']):.0f}")
    print(f"  heavy atoms:    mean={np.mean(vals['n_atoms']):.1f}  median={np.median(vals['n_atoms']):.0f}")
    pct(vals["top1"],          "top-1 consistency")
    pct(vals["top3_unordered"],"top-3 set consistency")
    pct(vals["top3_ordered"],  "top-3 order consistency")

    edges = [0.0, 0.5, 0.7, 0.9, 1.0]
    bucket_labels = ["<0.5", "0.5-0.7", "0.7-0.9", "1.0"]
    arr = np.array(vals["top1"])
    counts, _ = np.histogram(arr, bins=edges)
    print(f"\n  top-1 distribution:  " + "  ".join(f"{l}:{c}" for l, c in zip(bucket_labels, counts)))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--n-mols", type=int, default=500,
                    help="max molecules to sample (0 = all)")
    ap.add_argument("--n-rdkit-confs", type=int, default=20,
                    help="RDKit conformers to generate per molecule")
    ap.add_argument("--workers", type=int, default=cpu_count())
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    structures_dir = manifest_path.parent / "structures"

    print(f"Loading manifest from {manifest_path} ...")
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"  {len(manifest)} entries")

    # filter to multi-conformer molecules
    multi = [e for e in manifest if len(e.get('ids', [])) >= 2]
    print(f"  {len(multi)} with >=2 conformers")

    if args.n_mols > 0 and args.n_mols < len(multi):
        random.seed(args.seed)
        multi = random.sample(multi, args.n_mols)
        print(f"  sampled {len(multi)}")

    geom_results = []
    rdkit_results = []

    print(f"\nProcessing with {args.workers} workers ...")
    with Pool(processes=args.workers,
              initializer=_init_worker,
              initargs=(structures_dir, args.n_rdkit_confs)) as pool:
        for g, r in tqdm(pool.imap_unordered(_worker, multi), total=len(multi)):
            geom_results.append(g)
            rdkit_results.append(r)

    summarise(geom_results, "GEOM conformers (processed NPZ)")
    summarise(rdkit_results, f"RDKit ETKDG ({args.n_rdkit_confs} confs)")


if __name__ == "__main__":
    main()
