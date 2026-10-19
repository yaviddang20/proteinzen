#!/bin/bash
#SBATCH --job-name=gen_geom_conformer
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=ucsf_kortemme
#SBATCH --partition=large_gpu
#SBATCH --qos=kortemme_lab_qos


eval "$(micromamba shell hook --shell bash)"
micromamba activate proteinzen


source ../../env_vars.sh

export HYDRA_FULL_ERROR=1

eval "${1:?Usage: sbatch submit_gpu.slurm <cmd> <script>}"
