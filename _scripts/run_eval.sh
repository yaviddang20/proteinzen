
#!/usr/bin/env bash
# Usage:
#   bash _scripts/run_eval.sh <model_name> [extra eval_conformer.py args...]

set -euo pipefail

MODEL="${1:?Usage: $0 <model_name> [extra args...]}"
shift

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for SPLIT in train test; do
    echo "=== conformer / ${SPLIT} ==="
    python "${DIR}/_scripts/eval_conformer.py" \
        --out-dir "${DIR}/sampling/geom_conformer_${SPLIT}/${MODEL}" \
        --ref-dir "${DIR}/sampling/geom_conformer_${SPLIT}/conformer_mols" \
        "$@"
done
