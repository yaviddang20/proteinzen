"""Generate sample.py task YAML for ligand-conditioned protein generation.

Builds a minimal Structure NPZ from a ligand SMILES (3D conformer via RDKit),
adds a dummy protein scaffold, and writes LigandPocketConditionedSampling tasks.
No Plinder data required — just a target ligand name/SMILES.

Target ligands (built-in):
  SAM  — S-adenosyl methionine
  FAD  — flavin adenine dinucleotide
  IAI  — iodoacetamide / as present in Plinder
  NAD  — nicotinamide adenine dinucleotide
  ATP  — adenosine triphosphate
  HEM  — heme

Usage
-----
  # Built-in ligands (default: SAM FAD IAI):
  python _scripts/make_ligand_cond_yaml.py \\
      --out-yaml sampling/ligand_cond/val

  # Custom ligand:
  python _scripts/make_ligand_cond_yaml.py \\
      --ligand-codes NAD \\
      --smiles "NC(=O)c1ccc..." \\
      --out-yaml sampling/ligand_cond/val

  # Then run sampling:
  python sample.py \\
      sampler.tasks_yaml=sampling/ligand_cond/val_ligand_cond.yaml \\
      model_dir=/path/to/run \\
      out_dir=sampling/plinder/ligand_cond/<model>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# ── built-in SMILES ──────────────────────────────────────────────────────────

KNOWN_SMILES = {
    "SAM": "C[S+](CC[C@@H]([NH3+])C(=O)[O-])[C@@H]1O[C@@H]([C@H](O)[C@@H]1O)n1cnc2c(N)ncnc12",
    "FAD": "Cc1cc2nc3c(=O)[nH]c(=O)nc3n(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)c2cc1C",
    "IAI": "ICC(=O)N",
    "NAD": "NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(C(O)C3O)n3cnc4c(N)ncnc34)C(O)C2O)c1",
    "ATP": "Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O))[C@@H](O)[C@H]1O",
    "HEM": "CC1=C(CCC(=O)O)C2=CC3=NC(=CC4=NC(=CC5=NC(=CC1=N2)C(=C5CCC(=O)O)C)C(=C4C)C=C)C(=C3C)C=C",
}

TARGET_LIGANDS = ["SAM", "FAD", "IAI"]

# ── NPZ construction ─────────────────────────────────────────────────────────

# atom name encoding: [ord(c) - 32 for c in name.ljust(4)]
def _encode_name(name: str) -> list[int]:
    name = name.strip().ljust(4)[:4]
    return [ord(c) - 32 for c in name]


_RDKIT_BOND_TO_TYPE = {
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
}

_ATOM_DTYPE = np.dtype([
    ("name", "i1", (4,)),
    ("element", "i1"),
    ("charge", "i1"),
    ("coords", "f4", (3,)),
    ("conformer", "f4", (3,)),
    ("is_present", "?"),
    ("chirality", "i1"),
])

_BOND_DTYPE = np.dtype([
    ("atom_1", "i4"),
    ("atom_2", "i4"),
    ("type", "i1"),
])

_RESIDUE_DTYPE = np.dtype([
    ("name", "<U5"),
    ("res_type", "i1"),
    ("res_idx", "i4"),
    ("atom_idx", "i4"),
    ("atom_num", "i4"),
    ("atom_center", "i4"),
    ("atom_disto", "i4"),
    ("is_standard", "?"),
    ("is_present", "?"),
    ("is_copy", "?"),
])

_CHAIN_DTYPE = np.dtype([
    ("name", "<U5"),
    ("mol_type", "i1"),
    ("entity_id", "i4"),
    ("sym_id", "i4"),
    ("asym_id", "i4"),
    ("atom_idx", "i4"),
    ("atom_num", "i4"),
    ("res_idx", "i4"),
    ("res_num", "i4"),
    ("cyclic_period", "i4"),
])

_CONNECTION_DTYPE = np.dtype([
    ("chain_1", "i4"),
    ("chain_2", "i4"),
    ("res_1", "i4"),
    ("res_2", "i4"),
    ("atom_1", "i4"),
    ("atom_2", "i4"),
])

_INTERFACE_DTYPE = np.dtype([
    ("chain_1", "i4"),
    ("chain_2", "i4"),
    ("chain_1_num_res", "i4"),
    ("chain_2_num_res", "i4"),
])

# chain type ids: PROTEIN=0, NONPOLYMER=3
_PROTEIN_CHAIN_TYPE = 0
_NONPOLYMER_CHAIN_TYPE = 3
_UNK_RES_TYPE = 22  # const.token_ids["UNK"]


def _build_npz(smiles: str, n_prot_res: int, seed: int) -> dict:
    """Build a minimal Structure NPZ dict from a ligand SMILES.

    Layout:
      atoms[0 : N_lig]            — ligand heavy atoms (conformer coords)
      atoms[N_lig : N_lig+N_prot] — dummy protein CA atoms (near ligand centroid)
      residues[0]                 — one ligand residue
      residues[1 : 1+N_prot]     — one UNK residue per protein position
      chains[0]                   — NONPOLYMER (ligand)
      chains[1]                   — PROTEIN (dummy scaffold)
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles[:60]}")
    mol = AllChem.RemoveHs(mol)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    mol = AllChem.RemoveHs(mol)

    conf = mol.GetConformer()
    n_lig = mol.GetNumAtoms()

    # ── ligand atoms ────────────────────────────────────────────────────────
    lig_atoms = np.zeros(n_lig, dtype=_ATOM_DTYPE)
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        xyz = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        sym = atom.GetSymbol()
        # pad atom name to 4 chars (e.g. "C1  ")
        aname = f"{sym}{i+1}"[:4]
        lig_atoms[i]["name"] = _encode_name(aname)
        lig_atoms[i]["element"] = atom.GetAtomicNum()
        lig_atoms[i]["charge"] = atom.GetFormalCharge()
        lig_atoms[i]["coords"] = xyz
        lig_atoms[i]["conformer"] = xyz
        lig_atoms[i]["is_present"] = True
        lig_atoms[i]["chirality"] = 0

    # ── ligand bonds ────────────────────────────────────────────────────────
    lig_bonds = []
    for bond in mol.GetBonds():
        btype_str = bond.GetBondTypeAsDouble()
        if btype_str == 1.0:
            btype = 1
        elif btype_str == 2.0:
            btype = 2
        elif btype_str == 3.0:
            btype = 3
        elif btype_str == 1.5:
            btype = 4  # aromatic
        else:
            btype = 0  # other
        lig_bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), btype))
    bonds_arr = np.array(lig_bonds, dtype=_BOND_DTYPE) if lig_bonds else np.zeros(0, dtype=_BOND_DTYPE)

    # ── dummy protein atoms (CA only, clustered near ligand centroid) ────────
    rng = np.random.default_rng(seed)
    centroid = lig_atoms["coords"].mean(axis=0)
    # place CA atoms in a small cloud around the centroid
    prot_offsets = rng.normal(0, 3.0, size=(n_prot_res, 3)).astype(np.float32)
    prot_coords = centroid[None] + prot_offsets

    prot_atoms = np.zeros(n_prot_res, dtype=_ATOM_DTYPE)
    ca_name = np.array(_encode_name("CA"), dtype=np.int8)
    for i in range(n_prot_res):
        prot_atoms[i]["name"] = ca_name
        prot_atoms[i]["element"] = 6  # carbon
        prot_atoms[i]["charge"] = 0
        prot_atoms[i]["coords"] = prot_coords[i]
        prot_atoms[i]["conformer"] = prot_coords[i]
        prot_atoms[i]["is_present"] = True
        prot_atoms[i]["chirality"] = 0

    # ── combine atoms ────────────────────────────────────────────────────────
    all_atoms = np.concatenate([lig_atoms, prot_atoms])
    n_total_atoms = len(all_atoms)

    # ── residues ─────────────────────────────────────────────────────────────
    n_res_total = 1 + n_prot_res
    residues = np.zeros(n_res_total, dtype=_RESIDUE_DTYPE)

    # residue 0: ligand
    residues[0]["name"] = "LIG"
    residues[0]["res_type"] = _UNK_RES_TYPE
    residues[0]["res_idx"] = 0
    residues[0]["atom_idx"] = 0
    residues[0]["atom_num"] = n_lig
    residues[0]["atom_center"] = 0  # absolute atom index of center atom
    residues[0]["atom_disto"] = 0
    residues[0]["is_standard"] = False
    residues[0]["is_present"] = True
    residues[0]["is_copy"] = False

    # residues 1..n_prot_res: dummy protein UNK residues
    for i in range(n_prot_res):
        abs_atom_idx = n_lig + i
        residues[1 + i]["name"] = "UNK"
        residues[1 + i]["res_type"] = _UNK_RES_TYPE
        residues[1 + i]["res_idx"] = i
        residues[1 + i]["atom_idx"] = abs_atom_idx
        residues[1 + i]["atom_num"] = 1
        residues[1 + i]["atom_center"] = abs_atom_idx
        residues[1 + i]["atom_disto"] = abs_atom_idx
        residues[1 + i]["is_standard"] = False
        residues[1 + i]["is_present"] = True
        residues[1 + i]["is_copy"] = False

    # ── chains ────────────────────────────────────────────────────────────────
    chains = np.zeros(2, dtype=_CHAIN_DTYPE)

    # chain 0: NONPOLYMER (ligand)
    chains[0]["name"] = "B"
    chains[0]["mol_type"] = _NONPOLYMER_CHAIN_TYPE
    chains[0]["entity_id"] = 1
    chains[0]["sym_id"] = 0
    chains[0]["asym_id"] = 0
    chains[0]["atom_idx"] = 0
    chains[0]["atom_num"] = n_lig
    chains[0]["res_idx"] = 0
    chains[0]["res_num"] = 1
    chains[0]["cyclic_period"] = 0

    # chain 1: PROTEIN (dummy)
    chains[1]["name"] = "A"
    chains[1]["mol_type"] = _PROTEIN_CHAIN_TYPE
    chains[1]["entity_id"] = 2
    chains[1]["sym_id"] = 0
    chains[1]["asym_id"] = 1
    chains[1]["atom_idx"] = n_lig
    chains[1]["atom_num"] = n_prot_res
    chains[1]["res_idx"] = 1
    chains[1]["res_num"] = n_prot_res
    chains[1]["cyclic_period"] = 0

    mask = np.array([True, True], dtype=bool)
    connections = np.zeros(0, dtype=_CONNECTION_DTYPE)
    interfaces = np.array(
        [(0, 1, 1, n_prot_res)],
        dtype=_INTERFACE_DTYPE,
    )

    return {
        "atoms": all_atoms,
        "bonds": bonds_arr,
        "residues": residues,
        "chains": chains,
        "connections": connections,
        "interfaces": interfaces,
        "mask": mask,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-yaml", type=Path, required=True,
                        help="Output path stem; _ligand_cond.yaml is appended")
    parser.add_argument("--ligand-codes", nargs="+", default=None,
                        help=f"Ligand names to use (default: {TARGET_LIGANDS})")
    parser.add_argument("--smiles", default=None,
                        help="SMILES override — used only when a single --ligand-codes is given")
    parser.add_argument("--n-conformers", type=int, default=1,
                        help="Number of RDKit conformers per ligand (= number of YAML tasks per ligand, default: 1)")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Protein samples to generate per conformer (default: 10)")
    parser.add_argument("--n-prot-res", type=int, default=64,
                        help="Number of dummy protein residues in the scaffold (default: 64)")
    parser.add_argument("--trans-std", type=float, default=16.0,
                        help="Translation noise std for protein backbone (default: 16.0)")
    parser.add_argument("--include-h", action="store_true",
                        help="Include hydrogen atoms")
    parser.add_argument("--npz-dir", type=Path, default=None,
                        help="Directory to save ligand NPZ files (default: <out-yaml parent>/ligand_npz)")
    args = parser.parse_args()

    codes = args.ligand_codes or TARGET_LIGANDS
    if args.smiles and len(codes) > 1:
        sys.exit("--smiles can only be used with a single --ligand-codes value")

    npz_dir = args.npz_dir or (args.out_yaml.parent / "ligand_npz")
    npz_dir.mkdir(parents=True, exist_ok=True)
    args.out_yaml.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    for code in codes:
        smiles = (args.smiles if args.smiles else None) or KNOWN_SMILES.get(code.upper())
        if smiles is None:
            print(f"  {code}: no SMILES — skipping. Pass --smiles or add to KNOWN_SMILES.")
            continue

        for conf_idx in range(args.n_conformers):
            seed = hash(code) % (2**31) + conf_idx
            npz_path = npz_dir / f"{code}_conf{conf_idx}.npz"
            try:
                npz_data = _build_npz(smiles, n_prot_res=args.n_prot_res, seed=seed)
                np.savez_compressed(str(npz_path), **npz_data)
                print(f"  {code} conf{conf_idx}: {npz_data['atoms'].shape[0]} atoms → {npz_path.name}")
            except Exception as e:
                print(f"  {code} conf{conf_idx}: FAILED — {e}")
                continue

            tasks.append({
                "_target_": "proteinzen.runtime.sampling.protein_pocket.LigandPocketConditionedSampling",
                "name": f"{code}_conf{conf_idx}",
                "npz_path": str(npz_path.resolve()),
                "num_samples": args.num_samples,
                "trans_std": args.trans_std,
                "include_h": args.include_h,
                "max_protein_residues": args.n_prot_res,
            })

    out_path = args.out_yaml.parent / (args.out_yaml.name + "_ligand_cond.yaml")
    with open(out_path, "w") as f:
        yaml.dump(tasks, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote {len(tasks)} tasks → {out_path}")


if __name__ == "__main__":
    main()
