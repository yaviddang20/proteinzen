dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/env_vars.sh
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME
# CUDA_VISIBLE_DEVICES=0

python $REPO_ROOT/sample.py \
    model_dir=${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_bondlength/train \
    out_dir=${REPO_ROOT}/sampling/geom_conformer_train/geom_identityRot_256_conformer_3std_bondlength \
    sampler.tasks_yaml=${REPO_ROOT}/sampling/geom_conformer_train/smiles.yaml \
    sampler.batch_size=32 \
    save_traj=true \
    +version_num=29 \
    corrupter.sampling_noise_mode=null