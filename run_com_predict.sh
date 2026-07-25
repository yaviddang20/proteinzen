dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/env_vars.sh
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME

python _scripts/train_com_predictor.py \
    --model-dir proteinzen_weights/binder_design_phase2_6 \
    --dataset-config configs/train/data/com_pocket.yaml \
    --val-dataset-config configs/train/data/com_pocket_val.yaml \
    --out-dir outputs/com_predictor