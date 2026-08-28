#!/usr/bin/env bash
# Run this on CHPC to set up the mpnn micromamba env and download LigandMPNN weights.
# Usage: bash setup_mpnn_env.sh

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pull LigandMPNN submodule if not already done
if [ ! -f "${DIR}/LigandMPNN/run.py" ]; then
    echo "=== Initialising LigandMPNN submodule ==="
    git -C "${DIR}" submodule update --init LigandMPNN
fi

if micromamba env list | grep -q "^mpnn "; then
    echo "=== mpnn env already exists, skipping create ==="
else
    echo "=== Creating mpnn micromamba env ==="
    micromamba create -n mpnn -y \
        python=3.10 \
        pytorch pytorch-cuda=11.8 \
        "numpy<1.24" \
        -c pytorch -c nvidia -c conda-forge
fi

echo "=== Installing Python deps (idempotent) ==="
micromamba run -n mpnn pip install prody ml-collections dm-tree

if [ -d "${DIR}/LigandMPNN/model_params" ] && [ "$(ls -A "${DIR}/LigandMPNN/model_params"/*.pt 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "=== LigandMPNN weights already present, skipping download ==="
else
    echo "=== Downloading LigandMPNN model weights ==="
    bash "${DIR}/LigandMPNN/get_model_params.sh" "${DIR}/LigandMPNN/model_params"
fi

echo "=== Done. Test with: ==="
echo "  micromamba run -n mpnn python ${DIR}/LigandMPNN/run.py --help"
