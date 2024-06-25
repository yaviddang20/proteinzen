#!/bin/bash
#$ -S /bin/bash
#$ -o /dev/null
#$ -j y
#$ -r y
#$ -cwd
#$ -l mem_free=2G
#$ -l h_rt=2:00:00

cd $1/samples/
ls *.pdb > usalign.list
cd -
~/software/bin/USalign -dir $1/samples/ $1/samples/usalign.list -outfmt 2 > $1/usalign.tsv
python ~/projects/ligbinddiff/scripts/analysis/summarize_tmalign.py