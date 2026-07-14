dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/env_vars.sh
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME

model_name=plinder_protein_cond
version_num=0

# trans_std differs by direction:
#   protein_cond (generate ligand): 3.0  — ligand fits in a ~3 Å pocket
#   ligand_cond  (generate protein): 16.0 — protein backbone spreads ~16 Å
trans_std_protein_cond=3.0
trans_std_ligand_cond=16.0

declare -A TRANS_STD
TRANS_STD[protein_cond]=${trans_std_protein_cond}
TRANS_STD[ligand_cond]=${trans_std_ligand_cond}

for direction in protein_cond ligand_cond; do
    yaml_dir=${REPO_ROOT}/sampling/plinder/${direction}
    mkdir -p ${yaml_dir}

    python ${REPO_ROOT}/_scripts/make_plinder_pocket_yaml.py \
        --data-dir ${REPO_ROOT}/plinder_processed/val \
        --out-yaml ${yaml_dir}/val \
        --num-samples 10 \
        --trans-std ${TRANS_STD[$direction]} \
        --include-h

    python ${REPO_ROOT}/sample.py \
        model_dir=${REPO_ROOT}/outputs/${model_name}/train \
        out_dir=${REPO_ROOT}/sampling/plinder/${direction}/${model_name} \
        sampler.tasks_yaml=${yaml_dir}/val_${direction}.yaml \
        sampler.batch_size=32 \
        sampler.trans_std=${TRANS_STD[$direction]} \
        sampler.include_h=true \
        +version_num=${version_num} \
        identity_rot_noise=true \
        integrator=euler_no_rot \
        diffeq=base_euler_ode \
        save_traj=true
done
