#!/bin/bash
#SBATCH --job-name=gen_geom_conformer
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=cpu

eval "$(micromamba shell hook --shell bash)"
micromamba activate proteinzen

bash gen_all.sh
