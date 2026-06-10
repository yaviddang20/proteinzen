#!/bin/bash
#SBATCH --ntasks-per-node=8						   # How many instances of the script will run, if using --nodes, set --ntasks to the same value.
#SBATCH --cpus-per-task=4				   # How many CPUs per instance of the script will be needed.
#SBATCH --mem-per-cpu=16g                           # Job memory request.
#SBATCH --partition=pod                    # Run on partition "dgx" (e.g. not the default partition called "long").
#SBATCH --gres=gpu:nvidia_h200:8                       # Allocate 1 GPU resource for this job.
#SBATCH --time=TRAINHOURS:00:00
#SBATCH -o OUTFILE.log
### OPTIONAL
#SBATCH --nodes=2						   # Run on 2 nodes (if resources available).
										   #The number of ntasks should be set to the same number, e.g. --ntasks=13.
pwd
source REPOROOT/env_vars.sh
# Load all of your necessary modules.
# module load nvidia/nvhpc/nvhpc-hpcx/25.5
# module load openmpi5/5.0.7
source activate ${ENV_NAME}
# magic memory flag
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

## make every task get its own triton cache dir

# choose a fast local path on each node
export LOCAL_SCRATCH=/tmp/$USER
# make the Inductor cache unique per node & rank (prevents cross-rank collisions)
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_SCRATCH/inductor/${SLURM_JOB_ID}/${SLURM_NODEID}/${LOCAL_RANK}"
# Triton will, by default, place its cache under TORCHINDUCTOR_CACHE_DIR/triton,
# but setting explicitly is fine too:
export TRITON_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}/triton"
mkdir -p "$TRITON_CACHE_DIR"

ulimit -n 2048
cd ${REPO_ROOT}
srun python train.py "$@" +job_id=${JOB_ID} +num_days=TRAINDAYS
# args = ARGS
