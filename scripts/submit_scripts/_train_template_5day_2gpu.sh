#!/bin/bash
#$ -S /bin/bash
#$ -o OUTFILE.log
#$ -j y
#$ -r y
#$ -cwd
#$ -q gpu.q
#$ -pe smp 2
#$ -l mem_free=32G
#$ -l h_rt=120:00:00
#$ -l compute_cap=61,gpu_mem=40G

echo $JOB_ID
conda activate proteinzen

export CUDA_VISIBLE_DEVICES=$SGE_GPU
export GEOMSTATS_BACKEND=pytorch
echo $SGE_GPU
echo $CUDA_VISIBLE_DEVICES

module load Sali cuda

gpuprof=$(dcgmi group -c mygpus -a $SGE_GPU | awk '{print $10}')
dcgmi stats -g $gpuprof -e
dcgmi stats -g $gpuprof -s $JOB_ID

ulimit -n 2048
cd ~/projects/ligbinddiff
python train.py "$@" +job_id=${JOB_ID} +num_days=5
# args = ARGS

dcgmi stats -g $gpuprof -x $JOB_ID
dcgmi stats -g $gpuprof -v -j $JOB_ID
dcgmi group -d $gpuprof
