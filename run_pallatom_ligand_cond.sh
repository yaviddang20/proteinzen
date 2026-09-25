dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/env_vars.sh
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME

model_name=binder_zen_geom_plinder_pocket_ligand_cond_no_prealign_seq_noise
version_num=1639285
checkpoint_stem='epoch=1880-step=248500'
trans_std=16.0

# use_preset_conformers: toggle between the original RDKit-generated-conformer
# run (all 8 Pallatom-Ligand ligands) and a real-crystal-sourced-conformer run
# restricted to the 2 ligands we've actually verified a real PDB source for
# (OQO=PDB 7v11, IAI=PDB 5sdv -- see _scripts/preset_ligand_conformers/). Do NOT
# add more codes here without a verified preset file first -- make_ligand_cond_yaml.py
# will error out rather than silently fabricate one.
use_preset_conformers=true

if [ "$use_preset_conformers" = "true" ]; then
    ligand_codes="OQO IAI"
    preset_flag="--use-preset-conformer"
    run_name=pallatom_preset
else
    ligand_codes="FAD FMN SAM DOG SRO LDP IAI OQO"
    preset_flag=""
    run_name=pallatom
fi
num_samples=100

# yaml_dir: shared, model-independent task definition (same convention as
# sample_ligand_cond.sh) -- the ligand conformers/dummy scaffold don't depend on
# which checkpoint you sample against, so it lives one level above any given
# model's output and can be reused across models without regenerating it.
yaml_dir=${REPO_ROOT}/sampling/${run_name}
mkdir -p ${yaml_dir}

# out_dir: per-model, one level under yaml_dir. Used for BOTH sample.py's own
# output AND the eval step below -- one directory per model, not a separate
# eval/ tree (same convention as run_eval_plinder.sh / run_eval_plinder_placer.sh).
out_dir=${yaml_dir}/${model_name}

# 1) Build sampling tasks: 100 designs x each ligand
python ${REPO_ROOT}/_scripts/make_ligand_cond_yaml.py \
    --ligand-codes ${ligand_codes} \
    --out-yaml ${yaml_dir}/pallatom \
    --num-samples ${num_samples} \
    --trans-std ${trans_std} \
    --include-h \
    ${preset_flag}

# 2) Generate designs
python ${REPO_ROOT}/sample.py \
    model_dir=${REPO_ROOT}/outputs/${model_name}/train \
    out_dir=${out_dir} \
    sampler.tasks_yaml=${yaml_dir}/pallatom_ligand_cond.yaml \
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
