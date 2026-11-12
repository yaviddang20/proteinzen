#!/usr/bin/env python
"""Standalone PottsMPNN-adapted-LigandMPNN generation worker, invoked as a subprocess
by run_pottsmpnn_batch() in eval_plinder.py (in its own "pottsmpnn" conda env -- a
different torch/omegaconf pin than whatever env runs the main eval script, the same
reason LigandMPNN/LASErMPNN each get their own subprocess+env rather than running
in-process).

Generation follows the PottsMPNN paper's actual two-stage protocol (Birnbaum & Keating,
"Beyond native sequence recovery: Improved modeling of the sequence-energy landscape of
protein structures"), ported here rather than reinvented -- validated against an
80-structure test run on CHPC before this file existed
(pottsmpnn_ligand_check/potts_generate_v2.py, the throwaway script this one
generalizes into a reusable batch worker):
  1. Autoregressive decoding (ProteinMPNN-style model.sample(), random decode order,
     temperature=0.1 by default) for --n-seqs diverse candidates per structure.
  2. Deterministic-ish Potts-energy local optimization (iterated conditional modes,
     ICM): sweep designable positions in random order, resample each via
     softmax(-energy/refine_temp) given neighbors' CURRENT identities, repeat until
     converged (energy delta == 0) or --max-sweeps.
Pocket residues (passed in per-item via --items-json, same (chain, resnum) convention
as eval_plinder.py's _pocket_fixed_keys) are held fixed throughout both stages.

Input --items-json: a JSON list of {"sid": ..., "pdb_path": ...,
"fixed_residues": [[chain, resnum_str], ...]} objects.
Output --out-json: {sid: [seq, ...]}, MERGED with whatever that file already contains
on disk -- a narrower later call (e.g. retrying only failed sids) doesn't discard other
sids' previously-generated sequences, same merge-before-overwrite philosophy
run_lasermpnn_batch uses for its own shared designs.fasta.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def positional_potts_energy(etab_dense, E_idx, seq, pos):
    self_etab = etab_dense[0, pos, 0:1]
    pair_etab = etab_dense[0, pos, 1:]
    self_nrgs = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1).squeeze()
    k = etab_dense.shape[2]
    E_idx_jn = E_idx[0, pos, 1:].unsqueeze(-1)
    E_aa = torch.gather(seq[0].unsqueeze(-1).expand(-1, k - 1), 0, E_idx_jn)
    E_aa = E_aa.view(list(E_idx_jn.shape) + [1]).expand(-1, pair_etab.shape[-2], -1)
    pair_nrgs = torch.gather(pair_etab, -1, E_aa).squeeze(-1).sum(0)
    return self_nrgs + pair_nrgs


def optimize_sequence_icm(seq_str, etab_dense, E_idx, designable_mask, alphabet,
                           optimization_temp, max_sweeps, rng):
    L = len(seq_str)
    seq_ints = torch.tensor([alphabet.index(c) if c in alphabet else alphabet.index("X")
                              for c in seq_str], dtype=torch.int64, device=E_idx.device).unsqueeze(0)
    designable_positions = [i for i in range(L) if designable_mask[i]]
    n_sweeps = 0
    ener_delta = 1.0
    while abs(ener_delta) > 1e-6 and n_sweeps < max_sweeps:
        ener_delta = 0.0
        order = designable_positions[:]
        rng.shuffle(order)
        for pos in order:
            predicted_E = positional_potts_energy(etab_dense, E_idx, seq_ints, pos)[:20]
            cur_aa = seq_ints[0, pos].item()
            cur_E = predicted_E[cur_aa].item() if cur_aa < 20 else predicted_E.mean().item()
            logits = -predicted_E / optimization_temp
            probs = F.softmax(logits, dim=-1)
            mut_res = torch.multinomial(probs, num_samples=1).item()
            new_E = predicted_E[mut_res].item()
            ener_delta += (new_E - cur_E)
            seq_ints[0, pos] = mut_res
        n_sweeps += 1
    return "".join(alphabet[seq_ints[0, i].item()] for i in range(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fork-dir", required=True,
                     help="Directory containing model_utils.py/ligandmpnn_utils.py/"
                          "etab_utils.py (the PottsMPNN-adapted-LigandMPNN fork) and "
                          "model_params/ligandmpnn_v_32_010_25.pt (the base checkpoint "
                          "k_neighbors/atom_context_num are read from).")
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--refine-temp", type=float, default=0.0001)
    ap.add_argument("--max-sweeps", type=int, default=30)
    ap.add_argument("--mpnn-cutoff-for-score", type=float, default=8.0)
    args = ap.parse_args()

    sys.path.insert(0, args.fork_dir)
    from model_utils import ProteinMPNN
    from ligandmpnn_utils import parse_PDB, featurize, alphabet
    import etab_utils
    from Bio.PDB import PDBParser

    ALPHABET = "".join(alphabet)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base_checkpoint = torch.load(Path(args.fork_dir) / "model_params" / "ligandmpnn_v_32_010_25.pt",
                                  map_location=dev)
    checkpoint = torch.load(args.checkpoint, map_location=dev)
    k_neighbors = base_checkpoint["num_edges"]
    atom_context_num = base_checkpoint["atom_context_num"]

    model = ProteinMPNN(
        node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3,
        k_neighbors=k_neighbors, device=dev,
        atom_context_num=atom_context_num,
        model_type="ligand_mpnn",
        ligand_mpnn_use_side_chain_context=False,
        use_potts=True, clone=True, node_self_sub="",
        for_energy=True, updated_alist=True,
        potts_context="simple",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(dev).eval()
    print(f"Model loaded OK (device={dev}).", flush=True)

    items = json.loads(Path(args.items_json).read_text())
    out_path = Path(args.out_json)
    results = {}
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            results = {}

    bioparser = PDBParser(QUIET=True)
    n_done, n_err = 0, 0
    for item_idx, item in enumerate(items):
        sid = item["sid"]
        pdb_path = item["pdb_path"]
        fixed_keys = {(str(c), str(r)) for c, r in item.get("fixed_residues", [])}
        try:
            struct = bioparser.get_structure(sid, pdb_path)
            chain_ids = [c.id for c in struct[0] if any(True for _ in c)]
            chain_ids = [c for c in chain_ids if c.strip() != ""]

            protein_dict, *_rest = parse_PDB(
                pdb_path, device=dev, chains=chain_ids,
                parse_all_atoms=False, parse_atoms_with_zero_occupancy=1,
                updated_alist=True,
            )
            R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
            chain_letters_list = list(protein_dict["chain_letters"])
            chain_mask_vals = [
                0 if (str(chain_letters_list[i]), str(R_idx_list[i])) in fixed_keys else 1
                for i in range(len(R_idx_list))
            ]
            protein_dict["chain_mask"] = torch.tensor(chain_mask_vals, device=dev, dtype=torch.int32)

            feature_dict = featurize(
                protein_dict, cutoff_for_score=args.mpnn_cutoff_for_score,
                use_atom_context=1, number_of_ligand_atoms=atom_context_num,
                model_type="ligand_mpnn",
            )
            L = feature_dict["X"].shape[1]
            feature_dict["batch_size"] = args.n_seqs
            feature_dict["bias"] = torch.zeros([1, L, len(ALPHABET)], device=dev, dtype=torch.float32)
            feature_dict["symmetry_residues"] = [[]]
            feature_dict["symmetry_weights"] = [[]]
            # batch=1 here -- forward()'s own encode() runs at batch=1, and for_energy=True
            # makes the actual decode-order VALUES irrelevant to the energy table, but shapes
            # must still match h_V/E_idx/chain_mask (all computed once at batch=1 below).
            feature_dict["randn"] = torch.randn([1, feature_dict["mask"].shape[1]], device=dev)
            feature_dict["temperature"] = args.temperature

            with torch.no_grad():
                _, etab_raw, E_idx = model(feature_dict)
                etab_dense = etab_utils.functionalize_etab(etab_raw, E_idx)
                etab_dense = torch.nn.functional.pad(etab_dense, (0, 2, 0, 2), "constant", 0)

                # Re-batch to n_seqs only now, right before sample() -- it internally
                # .repeat(B_decoder, 1, ...)'s E_idx/h_V/etc to match this.
                feature_dict["randn"] = torch.randn([args.n_seqs, L], device=dev)
                out = model.sample(feature_dict)

            S_sampled = out["S"]
            sampled_seqs = [
                "".join(ALPHABET[S_sampled[n, i].item()] for i in range(L))
                for n in range(args.n_seqs)
            ]

            designable_mask = [chain_mask_vals[i] == 1 for i in range(L)]
            refined_seqs = []
            for n, seq_str in enumerate(sampled_seqs):
                rng = random.Random(hash((sid, n)) & 0xFFFFFFFF)
                refined_seqs.append(optimize_sequence_icm(
                    seq_str, etab_dense, E_idx, designable_mask, ALPHABET,
                    optimization_temp=args.refine_temp, max_sweeps=args.max_sweeps, rng=rng,
                ))

            results[sid] = refined_seqs
            n_done += 1
            print(f"[{item_idx + 1}/{len(items)}] {sid}: n_seqs={len(refined_seqs)} "
                  f"n_distinct={len(set(refined_seqs))}", flush=True)
        except Exception as e:
            print(f"  ERROR {sid}: {e}", flush=True)
            n_err += 1
            continue

        if n_done % 10 == 0:
            out_path.write_text(json.dumps(results))

    out_path.write_text(json.dumps(results))
    print(f"POTTS_GEN_BATCH_DONE: {n_done} ok, {n_err} errors, wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
