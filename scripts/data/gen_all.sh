source "$(dirname "$0")/../../env_vars.sh"

echo "=== Step 1: Filtering GEOM ==="
python filter_geom.py \
  --datadir "$REPO_ROOT/data/rdkit/" \
  --dataset drugs \
  --outdir "$REPO_ROOT/data/rdkit/" \
  --num-processes 40

echo "=== Step 2: Train/Val/Test Split ==="
python scaffold_train_test.py \
  --datadir "$REPO_ROOT/data/rdkit/" \
  --dataset drugs \
  --outdir "$REPO_ROOT/data/rdkit/" \
  --img-dir "$REPO_ROOT/data/scaffold_images/" \
  --n-val 500 \
  --n-test 1000 \
  --num-processes 40

echo "=== Step 3: GEOM Conformers ==="
python geom_conformer.py \
  --datadir "$REPO_ROOT/data/rdkit/" \
  --dataset drugs \
  --outdir "$REPO_ROOT/data/geom_drugs_conformers/" \
  --num-processes 40

echo "=== Done ==="
