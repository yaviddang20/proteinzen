source "$(dirname "$0")/../../env_vars.sh"

python geom_conformer.py \
  --datadir "$REPO_ROOT/data/rdkit/" \
  --dataset drugs \
  --outdir "$REPO_ROOT/data/geom_drugs_conformers/" \
  --num-processes 180
