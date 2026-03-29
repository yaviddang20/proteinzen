#!/bin/bash
#SBATCH --job-name=gen_geom_conformer
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --partition=pod
#SBATCH --gres=gpu:nvidia_h200:8

eval "$(micromamba shell hook --shell bash)"
micromamba activate proteinzen


source ../../env_vars.sh

export HYDRA_FULL_ERROR=1


eval "${1:?Usage: sbatch submit_gpu_pod.slurm <cmd> <script>}"