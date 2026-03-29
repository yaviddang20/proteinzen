dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/../../env_vars.sh 

python ${REPO_ROOT}/scripts/data/geom_conformer.py \
  --datadir "${REPO_ROOT}/data/rdkit/" \
  --dataset drugs \
  --outdir "${REPO_ROOT}/data/geom_drugs_conformers/" \
  --num-processes 180
