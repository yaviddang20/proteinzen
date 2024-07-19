#!/bin/bash
#$ -S /bin/bash
#$ -o /dev/null
#$ -j y
#$ -r y
#$ -cwd
#$ -l mem_free=8G
#$ -l h_rt=6:00:00

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source $SCRIPT_DIR/../../env_vars.sh

conda activate foldseek

DESIGNABLE_DIR=$1
OUTPUTDIR=$2
mkdir $OUTPUTDIR

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

conda activate ${ENV_NAME}
python ${REPO_ROOT}/scripts/analysis/summarize_foldseek.py