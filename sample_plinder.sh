dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/env_vars.sh
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME

model_name=plinder_protein_cond
trans_std=3.0
version_num=0

tasks_yaml_stem=${REPO_ROOT}/sampling/plinder/val

# Generate tasks yamls (protein_cond + ligand_cond)
python ${REPO_ROOT}/_scripts/make_plinder_pocket_yaml.py \
    --data-dir ${REPO_ROOT}/plinder_processed/val \
    --out-yaml ${tasks_yaml_stem} \
    --num-samples 10 \
    --trans-std ${trans_std} \
    --include-h

for direction in protein_cond ligand_cond; do
    python ${REPO_ROOT}/sample.py \
        model_dir=${REPO_ROOT}/outputs/${model_name}/train \
        out_dir=${REPO_ROOT}/sampling/plinder/${model_name}/${direction} \
        sampler.tasks_yaml=${tasks_yaml_stem}_${direction}.yaml \
        sampler.batch_size=4 \
        sampler.trans_std=${trans_std} \
        sampler.include_h=true \
        +version_num=${version_num} \
        corrupter.sampling_noise_mode=null \
        save_traj=true
done
