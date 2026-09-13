dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/env_vars.sh
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME

model_name=binder_zen_geom_plinder_pocket_ligand_cond_no_prealign_seq_noise
version_num=1639285
checkpoint_stem='epoch=1880-step=248500'
trans_std=16.0

ligand_codes="FAD FMN SAM DOG SRO LDP IAI OQO"
num_samples=100

# One directory per model, reused for everything (tasks yaml, samples/, and eval
# artifacts alongside it) -- same convention as run_eval_plinder.sh / run_eval_plinder_placer.sh.
out_dir=${REPO_ROOT}/sampling/plinder_pallatom/${model_name}
mkdir -p ${out_dir}

# 1) Build sampling tasks: 100 designs x each of the 8 Pallatom-Ligand benchmark ligands
python ${REPO_ROOT}/_scripts/make_ligand_cond_yaml.py \
    --ligand-codes ${ligand_codes} \
    --out-yaml ${out_dir}/pallatom \
    --num-samples ${num_samples} \
    --trans-std ${trans_std} \
    --include-h

# 2) Generate designs
python ${REPO_ROOT}/sample.py \
    model_dir=${REPO_ROOT}/outputs/${model_name}/train \
    out_dir=${out_dir} \
    sampler.tasks_yaml=${out_dir}/pallatom_ligand_cond.yaml \
    sampler.batch_size=16 \
    sampler.trans_std=${trans_std} \
    sampler.include_h=true \
    +version_num=${version_num} \
    identity_rot_noise=false \
    integrator=euler \
    diffeq=base_euler_ode \
    save_traj=true \
    +lmodule.seq_noise_schedule=true \
    overwrite=false \
    "checkpoint_stem='${checkpoint_stem}'"

# 3) Pallatom-style eval: Boltz2 single-sequence refold, chemically-valid-correspondence
#    ligand RMSD, ligand centroid displacement, protein/ligand pLDDT split, success
#    fractions overall and per ligand code. Writes into the SAME out_dir as samples/.
python ${REPO_ROOT}/_scripts/eval_pallatom_ligand_cond.py \
    --out-dir ${out_dir}
