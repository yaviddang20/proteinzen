dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/env_vars.sh
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME

model_name=binder_zen_plinder_cond_no_rotvf_no_prealign
version_num=834099
trans_std=16.0
data_dir=${REPO_ROOT}/plinder_processed/val

yaml_dir=${REPO_ROOT}/sampling/plinder/ligand_cond
mkdir -p ${yaml_dir}

python ${REPO_ROOT}/_scripts/make_ligand_cond_yaml.py \
    --data-dir ${data_dir} \
    --out-yaml ${yaml_dir}/val \
    --num-samples 10 \
    --trans-std ${trans_std} \
    --include-h

python ${REPO_ROOT}/sample.py \
    model_dir=${REPO_ROOT}/outputs/${model_name}/train \
    out_dir=${REPO_ROOT}/sampling/plinder/ligand_cond/${model_name} \
    sampler.tasks_yaml=${yaml_dir}/val_ligand_cond.yaml \
    sampler.batch_size=32 \
    sampler.trans_std=${trans_std} \
    sampler.include_h=true \
    +version_num=${version_num} \
    identity_rot_noise=true \
    integrator=euler_no_rot \
    diffeq=base_euler_ode \
    save_traj=true
