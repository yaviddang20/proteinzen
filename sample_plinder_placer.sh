dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/env_vars.sh
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME

model_name=binder_zen_plinder_pocket_cond_no_prealign
version_num=1001941

trans_std=3.0

for split in train test; do
    yaml_dir=${REPO_ROOT}/sampling/plinder_pocket_${split}/placer
    mkdir -p ${yaml_dir}

    python ${REPO_ROOT}/_scripts/make_plinder_pocket_yaml.py \
        --data-dir ${REPO_ROOT}/plinder_pocket_processed/${split} \
        --out-yaml ${yaml_dir} \
        --num-samples 20 \
        --trans-std ${trans_std} \
        --include-h

    python ${REPO_ROOT}/sample.py \
        model_dir=${REPO_ROOT}/outputs/${model_name}/train \
        out_dir=${REPO_ROOT}/sampling/plinder_pocket_${split}/placer/${model_name} \
        sampler.tasks_yaml=${yaml_dir}/placer.yaml \
        sampler.batch_size=32 \
        sampler.trans_std=${trans_std} \
        sampler.include_h=true \
        +version_num=${version_num} \
        identity_rot_noise=false \
        integrator=euler \
        diffeq=base_euler_ode \
        save_traj=true
done
