"""
Scaffold-based train/val/test split of GEOM summary data.

Reads:  datadir / filtered_summary_{dataset}.json
Writes: outdir  / train_filtered_summary_{dataset}.json
        outdir  / val_filtered_summary_{dataset}.json
        outdir  / test_filtered_summary_{dataset}.json
        img_dir / {train,val,test}_molecules.png
        img_dir / {train,val,test}_scaffolds.png
        img_dir / scaffold_example_{train,val,test}.png
"""
import argparse
import json
import multiprocessing
import random
from collections import defaultdict
from functools import partial
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from p_tqdm import p_umap


def compute_scaffold(smi: str) -> tuple[str, str]:
    """Returns (smiles, scaffold_smiles). scaffold is '' on failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return (smi, "")
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return (smi, sc)
    except Exception:
        return (smi, "")


def murcko_split(
    smiles: list[str],
    n_val: int,
    n_test: int,
    num_processes: int,
) -> tuple[list[int], list[int], list[int], dict[str, list[str]]]:
    parallel = num_processes > 1
    if parallel:
        results = p_umap(compute_scaffold, smiles, num_cpus=num_processes)
    else:
        results = [compute_scaffold(smi) for smi in tqdm(smiles, desc="Computing scaffolds")]

    scaffold_to_ids: dict[str, list[int]] = defaultdict(list)
    scaffold_to_smiles: dict[str, list[str]] = defaultdict(list)
    for i, (smi, sc) in enumerate(results):
        scaffold_to_ids[sc].append(i)
        scaffold_to_smiles[sc].append(smi)

    groups = list(scaffold_to_ids.values())

    train, val, test = [], [], []
    for g in groups:
        if len(test) < n_test:
            test += g
        elif len(val) < n_val:
            val += g
        else:
            train += g

    return train, val, test, scaffold_to_smiles


def draw_grid(smiles_list: list[str], n_cols: int = 6, img_size: int = 300) -> object:
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    return Draw.MolsToGridImage(mols, molsPerRow=n_cols, subImgSize=(img_size, img_size))


def draw_scaffold_with_members(
    scaffold_smi: str,
    member_smis: list[str],
    n_cols: int = 5,
    img_size: int = 300,
) -> object:
    top_mol = Chem.MolFromSmiles(scaffold_smi)
    members = [Chem.MolFromSmiles(s) for s in member_smis]

    first_row = [None] * n_cols
    first_row[n_cols // 2] = top_mol
    cells = first_row + members
    while len(cells) % n_cols != 0:
        cells.append(None)

    return Draw.MolsToGridImage(cells, molsPerRow=n_cols, subImgSize=(img_size, img_size))


def largest_scaffold(scaffold_sample: list[str], scaffold_to_smiles: dict) -> tuple[str, list[str]]:
    best = max(scaffold_sample, key=lambda sc: len(scaffold_to_smiles[sc]))
    return best, scaffold_to_smiles[best]


def save_images(
    split_smiles: dict[str, list[str]],
    scaffold_to_smiles: dict[str, list[str]],
    img_dir: Path,
    sample_size: int,
    seed: int,
) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for mode, smiles_list in split_smiles.items():
        # Collect scaffolds for this split
        scaffolds = list({MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
                          for s in smiles_list
                          if Chem.MolFromSmiles(s) is not None})

        mol_sample = rng.sample(smiles_list, min(sample_size, len(smiles_list)))
        sc_sample = rng.sample(scaffolds, min(sample_size, len(scaffolds)))

        # Molecules grid
        img = draw_grid(mol_sample)
        img.save(img_dir / f"{mode}_molecules.png")
        print(f"Saved {mode}_molecules.png")

        # Scaffolds grid
        img = draw_grid(sc_sample)
        img.save(img_dir / f"{mode}_scaffolds.png")
        print(f"Saved {mode}_scaffolds.png")

        # Largest scaffold + its members
        sc_example, sc_members = largest_scaffold(sc_sample, scaffold_to_smiles)
        img = draw_scaffold_with_members(sc_example, sc_members[:20])
        img.save(img_dir / f"scaffold_example_{mode}.png")
        print(f"Saved scaffold_example_{mode}.png  (scaffold has {len(sc_members)} members)")


def main(args) -> None:
    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    img_dir = Path(args.img_dir) if args.img_dir else outdir / "scaffold_images"
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = datadir / f"filtered_summary_{args.dataset}.json"
    print(f"Loading {summary_path} ...")
    with open(summary_path) as fp:
        summary = json.load(fp)

    smiles = list(summary.keys())
    print(f"Total molecules: {len(smiles)}")

    max_processes = multiprocessing.cpu_count()
    num_processes = max(1, min(args.num_processes, max_processes))

    print("Computing scaffold split ...")
    train_idx, val_idx, test_idx, scaffold_to_smiles = murcko_split(
        smiles, args.n_val, args.n_test, num_processes
    )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    idx_sets = {mode: set(idxs) for mode, idxs in splits.items()}

    split_summaries = {mode: {} for mode in splits}
    split_smiles_lists: dict[str, list[str]] = {mode: [] for mode in splits}
    for i, (smi, data) in enumerate(summary.items()):
        for mode, idx_set in idx_sets.items():
            if i in idx_set:
                split_summaries[mode][smi] = data
                split_smiles_lists[mode].append(smi)
                break

    for mode, split_summary in split_summaries.items():
        assert len(split_summary) == len(splits[mode]), \
            f"Size mismatch for {mode}: {len(split_summary)} vs {len(splits[mode])}"
        out_path = outdir / f"{mode}_filtered_summary_{args.dataset}.json"
        with out_path.open("w") as fp:
            json.dump(split_summary, fp, indent=4)
        print(f"Wrote {len(split_summary):>6} entries → {out_path.name}")

    print("Saving images ...")
    save_images(split_smiles_lists, scaffold_to_smiles, img_dir, args.sample_size, args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scaffold train/val/test split.")
    parser.add_argument("--datadir", type=Path, required=True,
                        help="Directory containing filtered_summary_{dataset}.json.")
    parser.add_argument("--dataset", type=str, required=True, choices=["qm9", "drugs"])
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Directory to write split summary JSONs.")
    parser.add_argument("--img-dir", type=Path, default=None,
                        help="Directory to write molecule/scaffold images (default: outdir/scaffold_images).")
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--sample-size", type=int, default=30,
                        help="Number of molecules/scaffolds to sample for images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-processes", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    main(args)
