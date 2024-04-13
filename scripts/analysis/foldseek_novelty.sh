#!/bin/bash
conda activate foldseek

DESIGNABLE_DIR=$1
OUTPUTDIR=$2
# i think i also got this from jason yim, need to double check
foldseek easy-search $DESIGNABLE_DIR \
    /wynton/group/kortemme/foldseek_db/PDB \
    $OUTPUTDIR/aln_PDB.tsv \
    $OUTPUTDIR/tmpFolder \
    --alignment-type 1 \
    --format-output query,target,alntmscore,lddt \
    --tmscore-threshold 0.0 \
    --exhaustive-search \
    --max-seqs 10000000000  | tee -a foldseek_novelty.out