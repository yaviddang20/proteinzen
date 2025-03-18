#!/bin/bash
#$ -S /bin/bash
#$ -o /wynton/home/kortemme/alexjli/projects/proteinzen/tmp.log
#$ -j y
#$ -r y
#$ -cwd
#$ -l mem_free=32G
#$ -l h_rt=0:30:00

conda activate proteinzen-refactor
echo $PWD
python ~/projects/proteinzen/scripts/analysis/esm_analysis.py --esmlog esmfold.log --folded_folder $PWD --samples ../samples
cd ..
bash ~/projects/proteinzen/scripts/analysis/analysis_suite.sh