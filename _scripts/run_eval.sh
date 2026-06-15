
#!/usr/bin/env bash
# Usage:
#   bash _scripts/run_eval.sh conformer <model_name> [extra eval.py args...]
#   bash _scripts/run_eval.sh pocket    <model_name> [extra eval.py args...]
#
# Examples:
#   bash _scripts/run_eval.sh conformer geom_identityRot_256_conformer_3std
#   bash _scripts/run_eval.sh pocket    plinder_protein_cond
#   bash _scripts/run_eval.sh conformer my_model --geom-datadir ./data/geom_drugs_conformers

set -euo pipefail

TASK="${1:?Usage: $0 <conformer|pocket> <model_name> [extra args...]}"
MODEL="${2:?Usage: $0 <conformer|pocket> <model_name> [extra args...]}"
shift 2

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for SPLIT in train test; do
    echo "=== ${TASK} / ${SPLIT} ==="
    if [[ "${TASK}" == "conformer" ]]; then
        python "${DIR}/_scripts/eval.py" \
            --out-dir "${DIR}/sampling/geom_conformer_${SPLIT}/${MODEL}" \
            --ref-dir "${DIR}/sampling/geom_conformer_${SPLIT}/conformer_mols" \
            "$@"
    elif [[ "${TASK}" == "pocket" ]]; then
        python "${DIR}/_scripts/eval.py" \
            --out-dir "${DIR}/sampling/plinder_${SPLIT}/${MODEL}" \
            --ref-dir "${DIR}/plinder_processed/${SPLIT}" \
            "$@"
    else
        echo "Unknown task: ${TASK} (expected conformer or pocket)" >&2
        exit 1
    fi
done
