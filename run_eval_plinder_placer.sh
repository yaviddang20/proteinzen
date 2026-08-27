#!/usr/bin/env bash
# Usage:
#   bash run_eval_plinder_placer.sh <model_name> [extra eval_plinder_placer.py args...]

set -euo pipefail

MODEL="${1:?Usage: $0 <model_name> [extra args...]}"
shift

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${DIR}/env_vars.sh"
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

for split in train test; do
    SAMPLES_DIR="${DIR}/sampling/plinder_pocket_${split}/placer/${MODEL}/samples"
    if [ ! -d "${SAMPLES_DIR}" ]; then
        echo "=== plinder placer / ${split} — skipping (no samples at ${SAMPLES_DIR}) ==="
        continue
    fi
    echo "=== plinder placer / ${split} ==="
    python "${DIR}/_scripts/eval_plinder_placer.py" \
        --samples-dir "${SAMPLES_DIR}" \
        --data-dir    "${DIR}/plinder_pocket_processed/${split}" \
        --out         "${DIR}/sampling/plinder_pocket_${split}/placer/${MODEL}/eval_results.json" \
        "$@"
done
