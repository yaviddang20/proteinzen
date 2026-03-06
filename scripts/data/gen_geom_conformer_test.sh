source "$(dirname "$0")/../../env_vars.sh"

python geom_conformer_test.py \
  --datadir "$REPO_ROOT/scripts/data/rdkit/rdkit_folder" \
  --dataset drugs \
  --outdir "$REPO_ROOT/geom_drugs_conformers_test/" \
  --num-processes 10
