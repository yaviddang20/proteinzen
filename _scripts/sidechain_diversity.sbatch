#!/usr/bin/env bash
#SBATCH --job-name=sc_diversity
#SBATCH --partition=community
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/scratch/user/daviyang/sc_diversity_logs/%x_%A_%a.out

# Usage:
#   mkdir -p /mnt/scratch/user/daviyang/sc_diversity_logs
#   sbatch sidechain_diversity.sbatch <pdb_list.txt> <out_root>
#
# <pdb_list.txt>: one input PDB path per line (e.g. every structures/*.pdb
#   or samples/*.pdb you want ensembles for).
# <out_root>: one subdirectory per input PDB gets created here.
#
# 20 array tasks x 96 cores/node = 1,920 cores total. Each array task claims
# one whole community node and fans its slice of the PDB list out across
# that node's cores via sidechain_diversity_batch.py's multiprocessing.Pool.

set -euo pipefail

PDB_LIST="$1"
OUT_ROOT="$2"

# NOTE: sbatch copies this script to a per-job spool dir before running it,
# so ${BASH_SOURCE[0]} does NOT point at the real script location on the
# compute node. Use SLURM_SUBMIT_DIR instead (requires submitting sbatch
# from the same directory this script and sidechain_diversity_batch.py live in).
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"

# Call the env's python binary directly rather than "micromamba run -n ..." --
# with 20 array tasks starting at once, micromamba run's shared lockfile under
# ~/.cache/mamba/proc races and randomly fails a handful of tasks.
/mnt/scratch/user/daviyang/micromamba/envs/pyrosetta_test/bin/python3 \
    "${SCRIPT_DIR}/sidechain_diversity_batch.py" \
    --pdb-list "${PDB_LIST}" \
    --out-root "${OUT_ROOT}" \
    --chunk-index "${SLURM_ARRAY_TASK_ID}" \
    --num-chunks "$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))" \
    --num-workers "${SLURM_CPUS_PER_TASK}"
