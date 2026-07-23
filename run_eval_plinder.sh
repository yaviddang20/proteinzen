#!/usr/bin/env bash
# Usage:
#   bash run_eval_plinder.sh <model_name> [split=val] [extra eval_plinder.py args...]

set -euo pipefail

MODEL="${1:?Usage: $0 <model_name> [split] [extra args...]}"
SPLIT="${2:-train}"
shift 2 || true

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${DIR}/env_vars.sh"
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

for direction in protein_cond ligand_cond; do
    OUT_DIR="${DIR}/sampling/plinder_pocket_${SPLIT}/${direction}/${MODEL}"
    if [ ! -d "${OUT_DIR}" ]; then
        echo "=== plinder / ${direction} / ${SPLIT} — skipping (no samples at ${OUT_DIR}) ==="
        continue
    fi
    echo "=== plinder / ${direction} / ${SPLIT} ==="
    python "${DIR}/_scripts/eval_plinder.py" \
        --task        "${direction}" \
        --out-dir     "${OUT_DIR}" \
        --ref-dir     "${DIR}/plinder_pocket_processed/${SPLIT}" \
        --plip-sif    "${DIR}/plip.sif" \
        --include-h \
        "$@"
done
