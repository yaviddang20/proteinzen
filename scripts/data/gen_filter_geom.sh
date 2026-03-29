source "$(dirname "$0")/../../env_vars.sh"

python filter_geom.py \
  --datadir "$REPO_ROOT/data/rdkit/" \
  --dataset drugs \
  --outdir "$REPO_ROOT/data/rdkit/" \
  --num-processes 40
