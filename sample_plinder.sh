dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/env_vars.sh
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME

model_name=binder_zen_plinder_pocket_cond_no_prealign
version_num=918508

trans_std_protein_cond=3.0
trans_std_ligand_cond=16.0

declare -A TRANS_STD
TRANS_STD[protein_cond]=${trans_std_protein_cond}
TRANS_STD[ligand_cond]=${trans_std_ligand_cond}

for split in train test; do
    for direction in protein_cond ligand_cond; do
        yaml_dir=${REPO_ROOT}/sampling/plinder_pocket_${split}/${direction}
        mkdir -p ${yaml_dir}

        python ${REPO_ROOT}/_scripts/make_plinder_pocket_yaml.py \
            --data-dir ${REPO_ROOT}/plinder_pocket_processed/${split} \
            --out-yaml ${yaml_dir} \
            --num-samples 10 \
            --trans-std ${TRANS_STD[$direction]} \
            --include-h

        python ${REPO_ROOT}/sample.py \
            model_dir=${REPO_ROOT}/outputs/${model_name}/train \
            out_dir=${REPO_ROOT}/sampling/plinder_pocket_${split}/${direction}/${model_name} \
            sampler.tasks_yaml=${yaml_dir}/${direction}.yaml \
            sampler.batch_size=32 \
            sampler.trans_std=${TRANS_STD[$direction]} \
            sampler.include_h=true \
            +version_num=${version_num} \
            identity_rot_noise=false \
            integrator=euler \
            diffeq=base_euler_ode \
            save_traj=true
    done
done
