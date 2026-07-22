
#!/usr/bin/env bash
# Usage:
#   bash _scripts/run_eval.sh conformer <model_name> [extra eval_conformer.py args...]
#   bash _scripts/run_eval.sh pocket    <model_name> [split] [extra eval_plinder.py args...]
#
# Examples:
#   bash _scripts/run_eval.sh conformer geom_identityRot_256_conformer_3std
#   bash _scripts/run_eval.sh pocket    binder_zen_plinder_pocket_cond_no_prealign val
#   bash _scripts/run_eval.sh conformer my_model --geom-datadir ./data/geom_drugs_conformers

set -euo pipefail

TASK="${1:?Usage: $0 <conformer|pocket> <model_name> [split] [extra args...]}"
MODEL="${2:?Usage: $0 <conformer|pocket> <model_name> [split] [extra args...]}"
shift 2

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${TASK}" == "conformer" ]]; then
    for SPLIT in train test; do
        echo "=== ${TASK} / ${SPLIT} ==="
        python "${DIR}/_scripts/eval_conformer.py" \
            --out-dir "${DIR}/sampling/geom_conformer_${SPLIT}/${MODEL}" \
            --ref-dir "${DIR}/sampling/geom_conformer_${SPLIT}/conformer_mols" \
            "$@"
    done
elif [[ "${TASK}" == "pocket" ]]; then
    SPLIT="${1:-val}"
    shift 1 || true
    echo "=== ${TASK} / protein_cond / ${SPLIT} ==="
    python "${DIR}/_scripts/eval_plinder.py" \
        --task protein_cond \
        --out-dir "${DIR}/sampling/plinder_pocket_${SPLIT}/protein_cond/${MODEL}" \
        --ref-dir "${DIR}/plinder_pocket_processed/${SPLIT}" \
        --plip-sif "${DIR}/plip.sif" \
        --include-h \
        "$@"
else
    echo "Unknown task: ${TASK} (expected conformer or pocket)" >&2
    exit 1
fi
