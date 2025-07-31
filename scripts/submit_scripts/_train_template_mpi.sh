#!/bin/bash
#$ -S /bin/bash
#$ -o OUTFILE.log
#$ -j y
#$ -r y
#$ -R y
#$ -cwd
#$ -q gpu.q
#$ -pe mpi NUMGPU
#$ -l mem_free=32G
#$ -l excl=true
#$ -l h_rt=TRAINHOURS:00:00
#$ -l compute_cap=61,gpu_mem=40G
#$ -l hostname=(qb3-atgpu31)|(qb3-atgpu32)|(qb3-atgpu33)|(qb3-idgpu18)
##$ -l h=!(qb3-idgpu18)

source REPOROOT/env_vars.sh

echo $JOB_ID
conda activate ${ENV_NAME}

# export CUDA_VISIBLE_DEVICES=$SGE_GPU
# export GEOMSTATS_BACKEND=pytorch
# echo $SGE_GPU
# echo $CUDA_VISIBLE_DEVICES
cat $PE_HOSTFILE

module load Sali cuda
module load mpi/openmpi-x86_64

BASE_PORT=7777
INCREMENT=1

port=$BASE_PORT
isfree=$(netstat -taln | grep $port)

while [[ -n "$isfree" ]]; do
    port=$((port+INCREMENT))
    isfree=$(netstat -taln | grep $port)
done

echo "Usable Port: $port"

master_addr=$(cat $PE_HOSTFILE | head -n 1 | awk '{print $1;}')
export MASTER_PORT=$port
export MASTER_ADDR=$master_addr
export WORLD_SIZE=NUMGPU
cat $PE_HOSTFILE | awk '{print $1 " slots=" $2 " max_slots=" $2}' > REPOROOT/scripts/submit_scripts/autogen_scripts/${JOB_ID}_mpi_hostfile.txt

gpuprof=$(dcgmi group -c mygpus -a $SGE_GPU | awk '{print $10}')
dcgmi stats -g $gpuprof -e
dcgmi stats -g $gpuprof -s $JOB_ID

# magic memory flag
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 2048
cd ${REPO_ROOT}

mpirun -np NUMGPU env \
    MPI_HOSTFILE=REPOROOT/scripts/submit_scripts/autogen_scripts/${JOB_ID}_mpi_hostfile.txt \
    MASTER_PORT=$port \
    MASTER_ADDR=$master_addr \
    NUM_NODES=$(wc -l < $PE_HOSTFILE) \
    WORLD_SIZE=NUMGPU \
    python train_mpi.py "$@" +job_id=${JOB_ID} +num_days=TRAINDAYS
# args = ARGS

dcgmi stats -g $gpuprof -x $JOB_ID
dcgmi stats -g $gpuprof -v -j $JOB_ID
dcgmi group -d $gpuprof

## End-of-job summary, if running as a job
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
