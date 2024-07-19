#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source $SCRIPT_DIR/../../env_vars.sh

python ${REPO_ROOT}/scripts/analysis/extract_designable_samples.py --csv esmfold/best_sc_rmsd.csv --folded_folders esmfold/ --samples samples/
qsub ${REPO_ROOT}/scripts/analysis/foldseek_novelty.sh designable_samples_folded/ novelty/
python ${REPO_ROOT}/scripts/analysis/plot_sample_dssp.py
python ${REPO_ROOT}/scripts/analysis/esmfold_over_len.py

# TODO: there's gotta be a better way than this
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
python ${REPO_ROOT}/scripts/analysis/foldseek_diversity.py
conda activate ${ENV_NAME}
bash ${REPO_ROOT}/scripts/analysis/tmalign.sh $PWD