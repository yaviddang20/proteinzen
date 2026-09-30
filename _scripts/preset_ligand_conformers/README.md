# Preset ligand conformers

Real, crystal-observed poses for `--use-preset-conformer` in `make_ligand_cond_yaml.py`.
Each file is a minimal HETATM+CONECT block (one ligand instance, heavy atoms only, no
protein/solvent) extracted directly from a real RCSB deposit -- not RDKit-generated,
not idealized CCD coordinates.

| Code | Source PDB | Chain/resnum | Context | Why this one |
|---|---|---|---|---|
| OQO | 7V11 | A 701 | Factor XIa in complex with compound 2E | Verified atom-for-atom (38 heavy atoms) against RFD3's own bundled `docs/input_pdbs/7v11.pdb` example |
| IAI | 5SDV | A 803 | PDE10 complex | Verified atom-for-atom (33 heavy atoms) against RFD3's own bundled `docs/input_pdbs/IAI.pdb` example |
| FAD | 1CJ3 | A 395 | p-hydroxybenzoate hydroxylase (Tyr38Glu mutant) -- the archetypal FAD-dependent flavoprotein aromatic hydroxylase | Medoid of the largest real conformational cluster (10/20 sampled real FAD-containing depositions fell within 2.5A tail RMSD of this one, aligned on the rigid isoalloxazine ring) |
| SAM | 1QAO | A 245 | rRNA methyltransferase ErmC' | Medoid of the largest real conformational cluster (9/15 sampled real SAM-containing depositions fell within 2.5A tail RMSD of this one, aligned on the rigid adenine ring) |

## Why FAD/SAM needed a different selection method than OQO/IAI

OQO and IAI are rare, essentially single-context ligand codes (each appears in only a
handful of PDB depositions), so a single real bound pose is unambiguously "the"
reference conformer, and RFD3's own paper happens to bundle exactly those two as
example inputs, letting us verify against their choice directly.

FAD and SAM are common cofactors (~3300 and ~800 PDB depositions respectively) with a
real, substantial rigid-core/flexible-tail split: the aromatic ring system
(isoalloxazine for FAD, adenine for SAM) is essentially invariant across different
bound structures (<0.3A RMSD self-aligned), but the flexible tail (ribityl/pyrophosphate
for FAD, methionine/sulfonium for SAM -- both have several freely-rotatable single
bonds with no ring constraint) swings into very different overall poses depending on
the protein pocket (5-13A RMSD when aligned on the ring). There is no single
dominant "the" bound conformation the way there is for OQO/IAI.

Given that, the representative for each was chosen by real clustering: a diverse
sample of real depositions (20 for FAD, 15 for SAM, spread across the PDB's
alphabetical ID range) were pairwise-compared by ring-aligned tail RMSD, and the
medoid (lowest average distance to all others) of the largest cluster at a 2.5A
threshold was picked. This is a real majority conformation, not an arbitrary pick,
but -- unlike OQO/IAI -- it is not uniquely "the" correct answer; a different sample
or threshold could plausibly pick a different (though still real, still plausible)
representative from the same or an adjacent cluster.

RFD3's own paper does not specify a source conformer for FAD/SAM either -- its
Figure 3 caption explicitly describes them as "common in the PDB" (vs. IAI/OQO as
"uncommon"), and does not bundle an example input file for either the way it does
for IAI/OQO.
