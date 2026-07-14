#!/usr/bin/env python3
"""Compare interaction-only vs pocket-priority cropping on a processed system NPZ.

Usage:
    python _scripts/compare_crops.py <npz_path> [--max-tokens 512] [--n-trials 10] [--out-pdb crops.pdb]

Example:
    python _scripts/compare_crops.py plinder_pocket_processed/structures/3g/3grt__1__1.A_1.B.npz

PDB output (--out-pdb):
    MODEL 1 — interaction-only crop (crop A)
    MODEL 2 — pocket+interaction crop (crop B)
    B-factor: 0.00 = shared between both crops, 1.00 = unique to this model.
    In PyMOL: split_states <obj>, then color by b-factor to highlight diffs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proteinzen.boltz.data import const
from proteinzen.boltz.data.types import Connection, Interface, Structure
from proteinzen.data.datasets.datamodule import (
    _build_priority_token_mask,
    strip_h_from_structure,
)
from proteinzen.data.featurize.cropper import Cropper
from proteinzen.data.featurize.tokenize import Tokenized, tokenize_structure

# single-char chain ID pool for PDB format
_CHAIN_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# atomic number -> element symbol (common subset)
_ELEMENT = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    12: 'MG', 14: 'SI', 15: 'P', 16: 'S', 17: 'CL',
    20: 'CA', 25: 'MN', 26: 'FE', 27: 'CO', 28: 'NI',
    29: 'CU', 30: 'ZN', 34: 'SE', 35: 'BR', 53: 'I',
}

def _decode_atom_name(raw):
    """Decode 4-byte encoded atom name (ord(c)-32 encoding) back to string."""
    return ''.join(chr(int(b) + 32) for b in raw if b != 0).strip()


def _load_struct(npz_path: Path):
    raw = np.load(npz_path, allow_pickle=False)

    chains = raw["chains"]
    if "cyclic_period" not in chains.dtype.names:
        new_dtype = chains.dtype.descr + [("cyclic_period", "i4")]
        new_chains = np.empty(chains.shape, dtype=new_dtype)
        for name in chains.dtype.names:
            new_chains[name] = chains[name]
        new_chains["cyclic_period"] = 0
        chains = new_chains

    atoms = raw["atoms"]
    if "chirality" not in atoms.dtype.names:
        new_dtype = atoms.dtype.descr + [("chirality", "i1")]
        new_atoms = np.empty(atoms.shape, dtype=new_dtype)
        for name in atoms.dtype.names:
            new_atoms[name] = atoms[name]
        new_atoms["chirality"] = 0
        atoms = new_atoms

    interfaces = raw["interfaces"]
    if interfaces.dtype.names is None or "chain_1" not in interfaces.dtype.names:
        interfaces = np.array([], dtype=np.dtype(Interface))
    elif "chain_1_num_res" not in interfaces.dtype.names:
        new_interfaces = np.zeros(len(interfaces), dtype=np.dtype(Interface))
        for name in interfaces.dtype.names:
            new_interfaces[name] = interfaces[name]
        interfaces = new_interfaces

    struct = Structure(
        atoms=atoms,
        bonds=raw["bonds"],
        residues=raw["residues"],
        chains=chains,
        connections=raw["connections"].astype(Connection),
        interfaces=interfaces,
        mask=raw["mask"],
    )
    struct = strip_h_from_structure(struct)

    interaction_mask = raw["interaction_residue_mask"] if "interaction_residue_mask" in raw else None
    pocket_mask = raw["pocket_residue_mask"] if "pocket_residue_mask" in raw else None

    return struct, interaction_mask, pocket_mask


def _dummy_task_data(struct):
    n_atoms = len(struct.atoms)
    n_res = len(struct.residues)
    return {
        "t": 0.5,
        "atom_noising_mask": np.ones(n_atoms, dtype=bool),
        "res_type_noising_mask": np.ones(n_res, dtype=bool),
        "copy_indexed_residue_mask": np.zeros(n_res, dtype=bool),
        "copy_unindexed_residue_mask": np.zeros(n_res, dtype=bool),
        "copy_atomized_residue_mask": np.zeros(n_res, dtype=bool),
    }


def _chain_label(struct, asym_id):
    for chain in struct.chains:
        if int(chain["asym_id"]) == asym_id:
            return chain["name"]
    return str(asym_id)


def _token_residue_set(tokens, struct):
    """Return set of (chain_name, res_idx) for every token."""
    result = set()
    for tok in tokens:
        chain_name = _chain_label(struct, int(tok["asym_id"]))
        result.add((chain_name, int(tok["res_idx"])))
    return result


def _build_res_table(struct):
    """Build list of residue dicts with all atom info for PDB writing."""
    protein_type = const.chain_type_ids["PROTEIN"]
    rows = []
    for chain in struct.chains[struct.mask]:
        chain_name = chain["name"]
        is_protein = int(chain["mol_type"]) == protein_type
        res_start = int(chain["res_idx"])
        for k in range(int(chain["res_num"])):
            ri = res_start + k
            res = struct.residues[ri]
            atom_start = int(res["atom_idx"])
            atom_end = atom_start + int(res["atom_num"])
            atoms = []
            for ai in range(atom_start, atom_end):
                atom = struct.atoms[ai]
                atoms.append({
                    "name": _decode_atom_name(atom["name"]),
                    "element": _ELEMENT.get(int(atom["element"]), 'X'),
                    "coords": atom["coords"],
                })
            rows.append({
                "chain": chain_name,
                "res_idx": int(res["res_idx"]),
                "res_array_idx": ri,
                "res_name": str(res["name"])[:3].upper(),
                "is_protein": is_protein,
                "atoms": atoms,
            })
    return rows


def _pdb_chain_id(name: str, asym_to_pdb: dict) -> str:
    """Map a chain name to a single-char PDB chain ID."""
    if name not in asym_to_pdb:
        idx = len(asym_to_pdb)
        asym_to_pdb[name] = _CHAIN_POOL[idx % len(_CHAIN_POOL)]
    return asym_to_pdb[name]


def write_crop_pdb(out_path: Path, struct, set_a: set, set_b: set, seed_mask_a, seed_mask_b):
    """Write a 2-model all-atom PDB (protein heavy atoms + ligand all atoms).

    MODEL 1 = interaction-only crop (crop A)
    MODEL 2 = pocket+interaction crop (crop B)
    B-factor: 0.00 shared, 1.00 unique to this model, 0.50 seeded (priority) residue.

    set_a / set_b: sets of (chain_name, res_idx) already computed by _token_residue_set.
    seed_mask_a/b: bool arrays indexed by residue array index.
    """
    res_table = _build_res_table(struct)

    seeded_a = set(seed_mask_a.nonzero()[0]) if seed_mask_a is not None else set()
    seeded_b = set(seed_mask_b.nonzero()[0]) if seed_mask_b is not None else set()

    lines = []
    lines.append("REMARK  MODEL 1: interaction-only crop (crop A)")
    lines.append("REMARK  MODEL 2: pocket+interaction crop (crop B)")
    lines.append("REMARK  B-factor: 0.00=shared, 1.00=unique to model, 0.50=seeded (priority) residue")

    for model_num, (crop_set, seeded_set) in enumerate([(set_a, seeded_a), (set_b, seeded_b)], start=1):
        other = set_b if model_num == 1 else set_a
        lines.append(f"MODEL        {model_num}")
        asym_to_pdb: dict = {}
        serial = 1
        for r in res_table:
            key = (r["chain"], r["res_idx"])
            if key not in crop_set:
                continue
            pdb_chain = _pdb_chain_id(r["chain"], asym_to_pdb)
            res_seq = r["res_idx"] % 9999
            record = "ATOM  " if r["is_protein"] else "HETATM"
            if key not in other:
                bfac = 1.00
            elif r["res_array_idx"] in seeded_set:
                bfac = 0.50
            else:
                bfac = 0.00
            res_name = r["res_name"].ljust(3)
            for atom in r["atoms"]:
                aname = atom["name"]
                # PDB atom name: 4 chars, right-padded; 1-char elements start at col 14
                if len(aname) < 4:
                    aname_fmt = f" {aname:<3s}"
                else:
                    aname_fmt = f"{aname:<4s}"
                x, y, z = float(atom["coords"][0]), float(atom["coords"][1]), float(atom["coords"][2])
                elem = atom["element"]
                lines.append(
                    f"{record}{serial:5d} {aname_fmt} {res_name} {pdb_chain}{res_seq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}          {elem:>2s}"
                )
                serial += 1
        lines.append("ENDMDL")

    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")
    name = out_path.stem
    print(f"\nWrote {out_path}")
    print(f"  PyMOL:    load {out_path}; split_states {name}")
    print(f"            spectrum b, blue_red, {name}_0001  # blue=shared, red=unique")
    print(f"  ChimeraX: open {out_path}")
    print(f"            color bfactor #1.1 palette blue:white:red range 0,1")
    print(f"            color bfactor #1.2 palette blue:white:red range 0,1")




def _calc_ligand_sasa(struct, crop_set, res_table):
    """Compute total ligand SASA (Å²) within the cropped protein context.

    Builds a freesasa Structure from all atoms in the crop, then sums
    per-residue SASA for non-protein chains.
    Returns (ligand_sasa, total_sasa) or (None, None) if freesasa unavailable.
    """
    try:
        import freesasa
        freesasa.setVerbosity(freesasa.silent)
    except ImportError:
        return None, None

    protein_type = const.chain_type_ids["PROTEIN"]
    fs = freesasa.Structure()
    lig_res_keys = set()

    for r in res_table:
        key = (r["chain"], r["res_idx"])
        if key not in crop_set:
            continue
        if not r["is_protein"]:
            lig_res_keys.add(key)
        res_name = r["res_name"][:3]
        chain_char = r["chain"][0] if r["chain"] else "A"
        res_num = str(r["res_idx"] % 9999)
        for atom in r["atoms"]:
            x, y, z = [float(v) for v in atom["coords"]]
            try:
                fs.addAtom(atom["name"], res_name, res_num, chain_char, x, y, z)
            except Exception:
                pass

    if fs.nAtoms() == 0:
        return None, None

    result = freesasa.calc(fs)
    areas = result.residueAreas()

    lig_sasa = 0.0
    total_sasa = 0.0
    for chain_id, residues in areas.items():
        for res_num_str, area in residues.items():
            total_sasa += area.total
            # check if this residue belongs to a ligand chain
            # we keyed lig_res_keys by (chain_name, res_idx) — match by res_num
            for key in lig_res_keys:
                if str(key[1] % 9999) == res_num_str:
                    lig_sasa += area.total
                    break

    return lig_sasa, total_sasa


def _mask_summary(mask, struct, label):
    if mask is None:
        print(f"  {label}: None")
        return
    n_set = int(mask.sum())
    # Map residue indices to (chain, res_idx) for display
    residue_chains = []
    for chain in struct.chains[struct.mask]:
        res_start = int(chain["res_idx"])
        for k in range(int(chain["res_num"])):
            residue_chains.append((chain["name"], int(struct.residues[res_start + k]["res_idx"])))

    marked = [(residue_chains[i] if i < len(residue_chains) else ("?", i))
              for i in range(len(mask)) if mask[i]]
    chains_hit = sorted({c for c, _ in marked})
    print(f"  {label}: {n_set} / {len(mask)} residues seeded, chains: {chains_hit}")


def main():
    parser = argparse.ArgumentParser(description="Compare interaction vs pocket-priority crops")
    parser.add_argument("npz", type=Path, help="Path to processed .npz file")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--pdb-seed", type=int, default=0,
                        help="Random seed used for the single crop written to the PDB file")
    parser.add_argument("--out-pdb", type=Path, default=None,
                        help="Output PDB path (default: <stem>_crops.pdb in cwd)")
    args = parser.parse_args()

    struct, interaction_mask, pocket_mask = _load_struct(args.npz)

    print(f"System: {args.npz.stem}")
    print(f"Residues: {len(struct.residues)}  Atoms (heavy): {len(struct.atoms)}")
    print(f"Chains: {[c['name'] for c in struct.chains[struct.mask]]}")
    print()

    _mask_summary(interaction_mask, struct, "interaction_residue_mask")

    pocket_or_interaction = None
    if pocket_mask is not None and interaction_mask is not None:
        pocket_or_interaction = interaction_mask | pocket_mask
    elif pocket_mask is not None:
        pocket_or_interaction = pocket_mask

    if pocket_mask is not None:
        _mask_summary(pocket_mask, struct, "pocket_residue_mask")
        if interaction_mask is not None:
            pocket_only = pocket_mask & ~interaction_mask
            _mask_summary(pocket_only, struct, "pocket-only (new additions)")
    else:
        print("  pocket_residue_mask: not present in this NPZ (run plinder.py with --pocket-data-dir)")

    task_data = _dummy_task_data(struct)
    token_data, rigid_data, token_bonds = tokenize_structure(struct, task_data, use_identity_rot=True)
    tokenized = Tokenized(tokens=token_data, rigids=rigid_data, bonds=token_bonds, structure=struct)
    n_tokens_total = len(token_data)

    print(f"\nTotal tokens pre-crop: {n_tokens_total}  (max_tokens={args.max_tokens})")

    if n_tokens_total <= args.max_tokens:
        print("  Structure fits within max_tokens — cropping is a no-op for this system.")
        print("  Try a larger system or reduce --max-tokens to see crop differences.")
        return

    cropper = Cropper(min_neighborhood=0, max_neighborhood=40)

    pri_a = _build_priority_token_mask(token_data, struct, interaction_mask)
    pri_b = _build_priority_token_mask(token_data, struct, pocket_or_interaction)

    n_seeded_a = int(pri_a.sum()) if pri_a is not None else 0
    n_seeded_b = int(pri_b.sum()) if pri_b is not None else 0
    print(f"\nSeeded tokens — interaction: {n_seeded_a}  pocket+interaction: {n_seeded_b}  (delta: +{n_seeded_b - n_seeded_a})")

    stats = {"tokens_a": [], "tokens_b": [], "overlap": [], "pocket_gained": [], "interaction_lost": []}

    print(f"\n{'Trial':>5}  {'A tokens':>8}  {'B tokens':>8}  {'Overlap':>7}  {'Pock gained':>11}  {'Interac lost':>12}")
    print("-" * 65)

    for trial in range(args.n_trials):
        rng_a = np.random.RandomState(trial)
        crop_a = cropper.crop(tokenized, max_tokens=args.max_tokens, random=rng_a, priority_token_mask=pri_a)

        rng_b = np.random.RandomState(trial)
        crop_b = cropper.crop(tokenized, max_tokens=args.max_tokens, random=rng_b, priority_token_mask=pri_b)

        set_a = _token_residue_set(crop_a.tokens, struct)
        set_b = _token_residue_set(crop_b.tokens, struct)

        overlap = len(set_a & set_b)
        gained = len(set_b - set_a)   # in B but not A (pocket additions)
        lost = len(set_a - set_b)     # in A but not B (displaced by pocket)

        stats["tokens_a"].append(len(set_a))
        stats["tokens_b"].append(len(set_b))
        stats["overlap"].append(overlap)
        stats["pocket_gained"].append(gained)
        stats["interaction_lost"].append(lost)

        print(f"{trial+1:>5}  {len(set_a):>8}  {len(set_b):>8}  {overlap:>7}  {gained:>11}  {lost:>12}")

    print("-" * 65)
    print(f"{'mean':>5}  "
          f"{np.mean(stats['tokens_a']):>8.1f}  "
          f"{np.mean(stats['tokens_b']):>8.1f}  "
          f"{np.mean(stats['overlap']):>7.1f}  "
          f"{np.mean(stats['pocket_gained']):>11.1f}  "
          f"{np.mean(stats['interaction_lost']):>12.1f}")

    print()
    print("Interpretation:")
    print("  'Pock gained'  = residues in B but not A (pocket seeds pulled these in)")
    print("  'Interac lost' = residues in A but not B (displaced to fit pocket additions)")

    # --- PDB output ---
    out_pdb = args.out_pdb or Path(f"{args.npz.stem}_crops.pdb")

    # build residue-level priority masks (by residue array index) for B-factor coloring
    res_table = _build_res_table(struct)
    res_key_to_array_idx = {(r["chain"], r["res_idx"]): r["res_array_idx"] for r in res_table}

    def _pri_to_res_set(pri_mask):
        if pri_mask is None:
            return set()
        seeded_tok_keys = set()
        for i, tok in enumerate(token_data):
            if pri_mask[i]:
                seeded_tok_keys.add((_chain_label(struct, int(tok["asym_id"])), int(tok["res_idx"])))
        return {res_key_to_array_idx[k] for k in seeded_tok_keys if k in res_key_to_array_idx}

    seeded_a_res = _pri_to_res_set(pri_a)
    seeded_b_res = _pri_to_res_set(pri_b)

    rng_a = np.random.RandomState(args.pdb_seed)
    pdb_crop_a = cropper.crop(tokenized, max_tokens=args.max_tokens, random=rng_a, priority_token_mask=pri_a)
    rng_b = np.random.RandomState(args.pdb_seed)
    pdb_crop_b = cropper.crop(tokenized, max_tokens=args.max_tokens, random=rng_b, priority_token_mask=pri_b)

    pdb_set_a = _token_residue_set(pdb_crop_a.tokens, struct)
    pdb_set_b = _token_residue_set(pdb_crop_b.tokens, struct)

    # pass seeded sets as numpy bool arrays indexed by res_array_idx
    n_res = len(struct.residues)
    seed_arr_a = np.zeros(n_res, dtype=bool)
    seed_arr_b = np.zeros(n_res, dtype=bool)
    for i in seeded_a_res:
        if i < n_res:
            seed_arr_a[i] = True
    for i in seeded_b_res:
        if i < n_res:
            seed_arr_b[i] = True

    write_crop_pdb(out_pdb, struct, pdb_set_a, pdb_set_b, seed_arr_a, seed_arr_b)

    res_table = _build_res_table(struct)
    sasa_a, total_a = _calc_ligand_sasa(struct, pdb_set_a, res_table)
    sasa_b, total_b = _calc_ligand_sasa(struct, pdb_set_b, res_table)
    if sasa_a is not None:
        print(f"\nLigand SASA (seed={args.pdb_seed}):")
        print(f"  Crop A (interaction):         {sasa_a:7.1f} Å²  (total context: {total_a:.1f} Å²)")
        print(f"  Crop B (pocket+interaction):  {sasa_b:7.1f} Å²  (total context: {total_b:.1f} Å²)")
        delta = sasa_b - sasa_a
        print(f"  Delta B-A: {delta:+.1f} Å²  ({'more exposed' if delta > 0 else 'more buried'} in pocket crop)")


if __name__ == "__main__":
    main()
