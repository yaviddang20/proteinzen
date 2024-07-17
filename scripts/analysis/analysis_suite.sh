#!/bin/bash

python /wynton/home/kortemme/alexjli/projects/ligbinddiff/scripts/analysis/extract_designable_samples.py --csv esmfold/best_sc_rmsd.csv --folded_folders esmfold/ --samples samples/
qsub ~/projects/ligbinddiff/scripts/analysis/foldseek_novelty.sh designable_samples_folded/ novelty/
python ~/projects/ligbinddiff/scripts/analysis/plot_sample_dssp.py
python ~/projects/ligbinddiff/scripts/analysis/esmfold_over_len.py

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/wynton/home/kortemme/alexjli/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/wynton/home/kortemme/alexjli/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/wynton/home/kortemme/alexjli/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/wynton/home/kortemme/alexjli/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate foldseek
python ~/projects/ligbinddiff/scripts/analysis/foldseek_diversity.py
conda activate proteinzen
bash ~/projects/ligbinddiff/scripts/analysis/tmalign.sh $PWD