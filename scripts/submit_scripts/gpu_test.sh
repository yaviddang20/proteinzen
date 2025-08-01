#!/bin/bash
#$ -S /bin/bash
#$ -o tmp.log
#$ -j y
#$ -r y
#$ -cwd
#$ -q gpu.q
#$ -pe smp 1
#$ -l mem_free=32G
#$ -l h_rt=12:00:00
#$ -l compute_cap=61,gpu_mem=40G
#$ -l h=(qb3-atgpu31)|(qb3-atgpu32)|(qb3-atgpu33)|(qb3-atgpu34)

# this is a dummy script to reserve lab gpus
# -l h=(qb3-atgpu31)|(qb3-atgpu32)|(qb3-atgpu33)|(qb3-atgpu34)
nvidia-smi
sleep 5d

