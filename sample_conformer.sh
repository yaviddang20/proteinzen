dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/env_vars.sh
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME
model_name=geom_identityRot_256_conformer_6std_stereo_norm_scale
split="train"
trans_std=3
# CUDA_VISIBLE_DEVICES=0

python $REPO_ROOT/sample.py \
    model_dir=${REPO_ROOT}/outputs/${model_name}/${split} \
    out_dir=${REPO_ROOT}/sampling/geom_conformer_${split}/${model_name} \
    sampler.tasks_yaml=${REPO_ROOT}/sampling/geom_conformer_${split}/smiles.yaml \
    sampler.batch_size=32 \
    sampler.trans_std=${trans_std} \
    save_traj=true \
    +version_num=48835 \
    corrupter.sampling_noise_mode=null