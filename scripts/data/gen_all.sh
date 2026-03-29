source "$(dirname "$0")/../../env_vars.sh"

NUM_PROCESSES=40
DATASET="drugs"
DATADIR="$REPO_ROOT/data/rdkit/"
OUTDIR="$REPO_ROOT/data/rdkit/"

echo "=== Step 1: Filtering GEOM ==="
python filter_geom.py \
  --datadir "$DATADIR" \
  --dataset "$DATASET" \
  --outdir "$OUTDIR" \
  --num-processes $NUM_PROCESSES

echo "=== Step 2: Train/Val/Test Split ==="
python scaffold_train_test.py \
  --datadir "$DATADIR" \
  --dataset "$DATASET" \
  --outdir "$OUTDIR" \
  --img-dir "$REPO_ROOT/data/scaffold_images/" \
  --n-val 500 \
  --n-test 1000 \
  --num-processes $NUM_PROCESSES

echo "=== Step 3: GEOM Conformers ==="
python geom_conformer.py \
  --datadir "$DATADIR" \
  --dataset "$DATASET" \
  --outdir "$REPO_ROOT/data/geom_drugs_conformers/" \
  --num-processes $NUM_PROCESSES

echo "=== Done ==="
