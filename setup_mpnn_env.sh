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

echo "=== Creating mpnn micromamba env ==="
micromamba create -n mpnn -y \
    python=3.10 \
    pytorch pytorch-cuda=11.8 \
    numpy \
    -c pytorch -c nvidia -c conda-forge

echo "=== Installing Python deps ==="
micromamba run -n mpnn pip install prody ml-collections

echo "=== Downloading LigandMPNN model weights ==="
bash "${DIR}/LigandMPNN/get_model_params.sh" "${DIR}/LigandMPNN/model_params"

echo "=== Done. Test with: ==="
echo "  micromamba run -n mpnn python ${DIR}/LigandMPNN/run.py --help"
