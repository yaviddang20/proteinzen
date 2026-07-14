dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/../../env_vars.sh

PLINDER_DIR=${REPO_ROOT}/plinder/2024-06/v2
OUTDIR=${REPO_ROOT}/plinder_pocket_processed
FILTERED_IDS=${REPO_ROOT}/plinder_pocket_filtered_ids.txt
POCKET_DATA_DIR=${REPO_ROOT}/plinder_pocket_alpha_spheres
FPOCKET_SIF=${REPO_ROOT}/fpocket.sif
NUM_PROCESSES=16

echo "=== Phase 1: pocket quality filter ==="
python ${REPO_ROOT}/scripts/data/filter_plinder_pocket.py \
    --plinder-dir ${PLINDER_DIR} \
    --outfile ${FILTERED_IDS} \
    --splits train val test \
    --fpocket-sif ${FPOCKET_SIF} \
    --num-processes ${NUM_PROCESSES} \
    --max-crystal-contacts 0.3 \
    --min-pocket-res 6 \
    --min-rscc 0.7 \
    --min-druggability 0.3 \
    --min-concavity 0.1 \
    --min-volume 100.0 \
    --min-buried-fraction 0.2 \
    --pocket-data-dir ${POCKET_DATA_DIR}

echo "=== Phase 2: process filtered systems ==="
python ${REPO_ROOT}/scripts/data/plinder.py \
    --plinder-dir ${PLINDER_DIR} \
    --outdir ${OUTDIR} \
    --system-ids-file ${FILTERED_IDS} \
    --ccd-path ${REPO_ROOT}/ccd.pkl \
    --num-processes ${NUM_PROCESSES} \
    --pocket-data-dir ${POCKET_DATA_DIR}
