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

tasks_yaml_stem=${REPO_ROOT}/sampling/plinder/val

# Generate YAMLs separately so each has the correct trans_std baked in.
# make_plinder_pocket_yaml.py always writes both _protein_cond.yaml and
# _ligand_cond.yaml; we call it twice with different stems and pick the
# relevant file from each call.
python ${REPO_ROOT}/_scripts/make_plinder_pocket_yaml.py \
    --data-dir ${REPO_ROOT}/plinder_processed/val \
    --out-yaml ${tasks_yaml_stem}_pc_gen \
    --num-samples 10 \
    --trans-std ${trans_std_protein_cond} \
    --include-h

python ${REPO_ROOT}/_scripts/make_plinder_pocket_yaml.py \
    --data-dir ${REPO_ROOT}/plinder_processed/val \
    --out-yaml ${tasks_yaml_stem}_lc_gen \
    --num-samples 10 \
    --trans-std ${trans_std_ligand_cond} \
    --include-h

declare -A TASKS_YAML
TASKS_YAML[protein_cond]=${tasks_yaml_stem}_pc_gen_protein_cond.yaml
TASKS_YAML[ligand_cond]=${tasks_yaml_stem}_lc_gen_ligand_cond.yaml

declare -A TRANS_STD
TRANS_STD[protein_cond]=${trans_std_protein_cond}
TRANS_STD[ligand_cond]=${trans_std_ligand_cond}

for direction in protein_cond ligand_cond; do
    python ${REPO_ROOT}/sample.py \
        model_dir=${REPO_ROOT}/outputs/${model_name}/train \
        out_dir=${REPO_ROOT}/sampling/plinder/${model_name}/${direction} \
        sampler.tasks_yaml=${TASKS_YAML[$direction]} \
        sampler.batch_size=32 \
        sampler.trans_std=${TRANS_STD[$direction]} \
        sampler.include_h=true \
        +version_num=${version_num} \
        identity_rot_noise=true \
        integrator=euler_no_rot \
        diffeq=base_euler_ode \
        save_traj=true
done
