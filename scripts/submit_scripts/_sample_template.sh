#!/bin/bash
#$ -S /bin/bash
#$ -o OUTFILE.log
#$ -j y
#$ -r y
#$ -cwd
#$ -q gpu.q
#$ -pe smp 1
#$ -l mem_free=32G
#$ -l h_rt=24:00:00
#$ -l compute_cap=61,gpu_mem=40G

echo "Job ID ${JOB_ID}"
conda activate binderdiff

export CUDA_VISIBLE_DEVICES=$SGE_GPU
echo $SGE_GPU
echo $CUDA_VISIBLE_DEVICES

module load Sali cuda

gpuprof=$(dcgmi group -c mygpus -a $SGE_GPU | awk '{print $10}')
dcgmi stats -g $gpuprof -e
dcgmi stats -g $gpuprof -s $JOB_ID

cd ~/projects/ligbinddiff
python sample.py "$@" job_id=${JOB_ID}
# args = ARGS

dcgmi stats -g $gpuprof -x $JOB_ID
dcgmi stats -g $gpuprof -v -j $JOB_ID
dcgmi group -d $gpuprof
