dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/env_vars.sh
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME

 python _scripts/train_com_predictor.py \
      --manifest /path/to/plinder_pocket_processed/train/manifest.json \
      --val-manifest /path/to/plinder_pocket_processed/val/manifest.json