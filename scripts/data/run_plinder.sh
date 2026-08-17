dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/../../env_vars.sh

PLINDER_DIR=${REPO_ROOT}/plinder/2024-06/v2
OUTDIR=${REPO_ROOT}/plinder_processed
NUM_PROCESSES=16

echo "=== Processing all systems (no pocket-quality filter) ==="
python ${REPO_ROOT}/scripts/data/plinder.py \
    --plinder-dir ${PLINDER_DIR} \
    --outdir ${OUTDIR} \
    --ccd-path ${REPO_ROOT}/ccd.pkl \
    --num-processes ${NUM_PROCESSES}
