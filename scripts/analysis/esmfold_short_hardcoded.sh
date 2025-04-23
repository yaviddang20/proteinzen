#!/bin/bash
#$ -S /bin/bash
#$ -o esmfold.log
#$ -j y
#$ -r y
#$ -cwd
#$ -q gpu.q
#$ -pe smp 1
#$ -l mem_free=32G
#$ -l h_rt=2:00:00
#$ -l compute_cap=61,gpu_mem=40G

export CUDA_VISIBLE_DEVICES=$SGE_GPU
echo $SGE_GPU
echo $CUDA_VISIBLE_DEVICES

module load Sali cuda

fasta=$PWD/../seqs/
outdir=$PWD
cd /wynton
apptainer exec --nv /wynton/group/kortemme/esmfold/esmfold_1128.sif python /wynton/home/kortemme/alexjli/projects/proteinzen-clone/scripts/analysis/batched_esmfold_inference.py -i $fasta -o $outdir
cd $PWD
qsub ~/projects/proteinzen-clone/scripts/analysis/finish_eval_pipeline.sh &> /dev/null
