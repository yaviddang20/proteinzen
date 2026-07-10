dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

CKPT="/{dir}/../proteinzen/outputs/geom_identityRot_256_conformer_3std_stereo_hydrogen/train/checkpoints/last.ckpt"
HYDRA_CFG="/{dir}/../outputs/geom_identityRot_256_conformer_3std_stereo_hydrogen/train/.hydra/config.yaml"
DATA_CFG="${dir}/../configs/train/data/geom_conformer_val.yaml"

python ${dir}/test_equivariance.py \
    --ckpt        "${CKPT}" \
    --hydra_config "${HYDRA_CFG}" \
    --data_config  "${DATA_CFG}" \
    --t 0.5 \
    --seed 42 \
    --no_amp    
