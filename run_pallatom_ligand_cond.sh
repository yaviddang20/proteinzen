dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/env_vars.sh
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME

# Same trained ligand_cond checkpoint as sample_ligand_cond.sh
model_name=binder_zen_plinder_cond_no_rotvf_no_prealign
version_num=834099
trans_std=16.0

ligand_codes="FAD FMN SAM DOG SRO LDP IAI OQO"
num_samples=100

yaml_dir=${REPO_ROOT}/sampling/plinder/pallatom
out_dir=${REPO_ROOT}/sampling/plinder/pallatom/${model_name}
eval_out_dir=${REPO_ROOT}/eval/pallatom_ligand_cond/${model_name}
mkdir -p ${yaml_dir}

# 1) Build sampling tasks: 100 designs x each of the 8 Pallatom-Ligand benchmark ligands
python ${REPO_ROOT}/_scripts/make_ligand_cond_yaml.py \
    --ligand-codes ${ligand_codes} \
    --out-yaml ${yaml_dir}/pallatom \
    --num-samples ${num_samples} \
    --trans-std ${trans_std} \
    --include-h

# 2) Generate designs
python ${REPO_ROOT}/sample.py \
    model_dir=${REPO_ROOT}/outputs/${model_name}/train \
    out_dir=${out_dir} \
    sampler.tasks_yaml=${yaml_dir}/pallatom_ligand_cond.yaml \
    sampler.batch_size=32 \
    sampler.trans_std=${trans_std} \
    sampler.include_h=true \
    +version_num=${version_num} \
    identity_rot_noise=true \
    integrator=euler_no_rot \
    diffeq=base_euler_ode \
    save_traj=true

# 3) Pallatom-style eval: Boltz2 single-sequence refold, chemically-valid-correspondence
#    ligand RMSD, ligand centroid displacement, protein/ligand pLDDT split, success
#    fractions overall and per ligand code.
python ${REPO_ROOT}/_scripts/eval_pallatom_ligand_cond.py \
    --samples-dir ${out_dir}/samples \
    --out-dir ${eval_out_dir} \
    --ligand-codes ${ligand_codes}
