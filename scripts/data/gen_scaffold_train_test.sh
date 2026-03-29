source "$(dirname "$0")/../../env_vars.sh"

python scaffold_train_test.py \
  --datadir "$REPO_ROOT/data/rdkit/" \
  --dataset drugs \
  --outdir "$REPO_ROOT/data/rdkit/" \
  --img-dir "$REPO_ROOT/data/scaffold_images/" \
  --n-val 500 \
  --n-test 1000 \
  --num-processes 40
