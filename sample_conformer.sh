dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/env_vars.sh
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME
model_name=geom_identityRot_256_conformer_6std_stereo_norm_scale
trans_std=1
version_num=48835
# CUDA_VISIBLE_DEVICES=0

for split in train test; do
    python $REPO_ROOT/sample.py \
        model_dir=${REPO_ROOT}/outputs/${model_name}/${split} \
        out_dir=${REPO_ROOT}/sampling/geom_conformer_${split}/${model_name} \
        sampler.tasks_yaml=${REPO_ROOT}/sampling/geom_conformer_${split}/smiles.yaml \
        sampler.batch_size=32 \
        sampler.trans_std=${trans_std} \
        save_traj=true \
        +version_num=${version_num} \
        corrupter.sampling_noise_mode=null \
        sampler.include_h=true \
        identity_rot_noise=true \
        no_rot_sampling=true
done

python $REPO_ROOT/sample.py \
    model_dir=${REPO_ROOT}/outputs/${model_name}/test \
    out_dir=${REPO_ROOT}/sampling/xl_processed/${model_name} \
    sampler.tasks_yaml=${REPO_ROOT}/sampling/xl_processed/smiles.yaml \
    sampler.batch_size=32 \
    sampler.trans_std=${trans_std} \
    save_traj=true \
    +version_num=${version_num} \
    corrupter.sampling_noise_mode=null \
    sampler.include_h=true \
    identity_rot_noise=true \
    no_rot_sampling=true