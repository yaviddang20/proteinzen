# Small-molecule binder design benchmark numbers, as reported in the papers

Reference numbers pulled directly from the papers themselves (quoted/transcribed, not our
own results -- our own results live in `summary.txt` next to each eval run). Saved here so
neither of us has to re-open the PDFs to remember what each paper actually reported.

All three papers benchmark overlapping-but-not-identical small molecules, with different
success-criteria strictness and different structure-prediction engines (AF3 / RF3 / our
own Boltz2 substitution) -- **the raw percentages below are not directly comparable across
papers**. See the "why these aren't comparable" section at the bottom.

---

## Pallatom-Ligand (Wang et al. 2026, ICLR 2026)

Success formulas (verbatim, Sec 4.1-4.2):
- **Fold Success**: Cα-RMSD < 2Å & protein-pLDDT > 80
- **Pocket Success**: Fold + ligand-D_center < 4Å & ligand-pLDDT > 80
- **Pose Success**: Fold + ligand-RMSD < 2Å

Folding engine: AlphaFold3 (no MSA). LigandMPNN redesigns non-pocket residues (6Å cutoff)
as an explicit, headline-number-producing step of their own protocol.

**w/out SA + MPNN** (Appendix A.11.1-A.11.6 -- the variant most comparable to our own
pipeline, since we have no structure-adapter/SA-conditioning equivalent):

| Ligand | Fold | Pocket | Pose |
|---|---|---|---|
| FMN | 93.0% | 15.0% | 9.0% |
| DOG | 94.0% | 17.0% | 1.0% |
| LDP | 91.0% | 4.0% | 0.0% |
| SRO | 90.0% | 14.0% | 1.0% |
| IAI | 91.0% | 4.0% | 2.0% |
| FAD | 86.0% | 13.0% | 1.0% |
| SAM | 93.0% | 19.0% | 2.0% |
| OQO | 90.0% | 16.0% | 2.0% |
| **Avg.** | **89.6%** | **13.8%** | **3.0%** |

**w/ SA + MPNN** (their headline variant, uses structure-adapter conditioning we don't have):

| Ligand | Fold | Pocket | Pose |
|---|---|---|---|
| FMN | 92% | 17% | 10% |
| DOG | 85% | 22% | 1% |
| LDP | 91% | 38% | 9% |
| SRO | 91% | 26% | 6% |
| IAI | 87% | 17% | 8% |
| FAD | 78% | 6% | 2% |
| SAM | 87% | 22% | 7% |
| OQO | 94% | 14% | 8% |
| **Avg.** | **88.0%** | **18.6%** | **6.4%** |

Pattern to expect: pose success is the hardest metric everywhere (mostly 0-10%), fold
success is comparatively easy once MPNN redesign is applied (mostly 85-94%).

---

## RFdiffusion3 / RFD3 (Butcher et al. 2025, bioRxiv, Sec 3.3 + Fig 3c)

Success criterion (single combined condition, verbatim, Fig 3c caption): backbone-aligned
ligand RMSD < 5Å & backbone-aligned backbone RMSD < 1.5Å & min chain-pair PAE < 1.5 &
iPTM > 0.8. "Fraction of designs indicates how many of the 400 generated backbones had at
least one [LigandMPNN-designed] sequence that passed the criteria" -- i.e. best-of-several
sequences per backbone, not a single-shot rate.

Folding engine: AlphaFold3. Ligands: FAD, SAM, IAI, OQO (4 of Pallatom-Ligand's 8; 2 common
in the PDB -- FAD, SAM -- and 2 uncommon -- IAI, OQO).

**No numeric table is published** -- results are bar-chart-only (Fig 3c). The numbers below
are our own approximate visual read of that chart, not exact reported values:

| Target | RFdiffusion-AllAtom (baseline) | RFD3, fixed ligand | RFD3, diffused ligand + RASA |
|---|---|---|---|
| FAD | ~1% | ~9% | ~16% |
| SAM | ~10% | ~26% | ~29% |
| IAI | ~5% | ~33% | ~24% |
| OQO | ~2% | ~42% | ~21% |

Note the two RFD3 variants trade off differently per ligand (diffused+RASA wins FAD/SAM,
fixed-ligand wins IAI/OQO) -- there isn't one single "RFD3 number" per ligand.

---

## Proteína-Complexa (Didi et al. 2026, ICLR 2026, Table 1, p.8)

Success criterion (single combined condition, verbatim, Sec 3.3/Appendix F, "following Cho
et al. 2025"): min-ipAE(s) < 2 & Binder-RMSD(s) < 2Å & Ligand-RMSD(s) < 5Å. "min ipAE" =
"the minimum entry of the cross-chain elements in the pAE matrix" (Appendix F) -- the same
underlying statistic RFD3 calls "min chain-pair PAE," just a different name/threshold/engine.

Folding engine: RosettaFold-3 (RF3). Ligands: SAM, OQO, FAD, IAI (same 4 as RFD3).

**Table 1, exact, "# Unique Successes" out of 200 generated binders** (self-generated
sequences, no MPNN redesign needed):

| Model | SAM | OQO | FAD | IAI |
|---|---|---|---|---|
| RFDiffusion-AllAtom (uses LigandMPNN) | 2 | 3 | 5 | 8 |
| **Complexa (ours)** | **10** | **6** | **17** | **19** |

As a fraction of 200 (for comparing against our own per-ligand rates): SAM 5%, OQO 3%,
FAD 8.5%, IAI 9.5% (Complexa); SAM 1%, OQO 1.5%, FAD 2.5%, IAI 4% (RFDiffusion-AllAtom).

Complexa also reports ~6.5x faster sampling (13.5s vs 87.4s) and comparable novelty
(~0.71 vs 0.72) against that baseline.

**"RFDiffusion-AllAtom" (Krishna et al. 2024) is a different, older model than
"RFdiffusion3"/RFD3 (Butcher et al. 2025) above** -- despite the similar name, RFD3 never
appears in this table, and Complexa never benchmarks directly against RFD3.

---

## Why these numbers aren't directly comparable across papers

| | Pallatom-Ligand | RFD3 | Complexa |
|---|---|---|---|
| Fold engine | AlphaFold3 | AlphaFold3 | RosettaFold-3 |
| Ligand-RMSD threshold | < 2Å (pose tier) | < 5Å | < 5Å |
| Confidence metric | pLDDT | PAE / iPTM | ipAE (= min cross-chain PAE) |
| Sampling protocol | fixed N per ligand | best-of-several seqs / backbone | dedup'd unique successes / 200 |

Pallatom-Ligand's ligand-RMSD bar (<2Å) is 2.5x stricter than RFD3/Complexa's (<5Å) --
the single biggest reason its "pose success" numbers look so much lower than RFD3/Complexa's
overlapping-ligand rates. RFD3 and Complexa are methodologically closer to each other
(both PAE-family confidence metrics, same 4-ligand overlap, Complexa explicitly follows
RFD3's benchmark lineage) than either is to Pallatom-Ligand's from-scratch pLDDT-based
3-tier framework.

Our own eval (`eval_pallatom_ligand_cond.py`) computes all three papers' formulas against
Boltz2 refolds (not AF3 or RF3) -- so even our own numbers are only *structurally*
analogous to any of the above, not numerically comparable to them either.
