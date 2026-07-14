dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/env_vars.sh
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME

python _scripts/train_com_predictor.py \
    --model-dir outputs/plinder_protein_cond/train \
    --dataset-config configs/train/data/plinder_protein_cond.yaml \
    --val-dataset-config configs/train/data/plinder_val.yaml \
    --out-dir outputs/com_predictor