#!/bin/bash
#SBATCH --job-name=gen_geom_conformer
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

eval "$(micromamba shell hook --shell bash)"
micromamba activate proteinzen


source ../../env_vars.sh

eval "${1:?Usage: sbatch submit_gpu.slurm <cmd> <script>}"