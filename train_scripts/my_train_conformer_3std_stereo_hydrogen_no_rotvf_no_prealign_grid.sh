#!/usr/bin/env bash

dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/../env_vars.sh

STDS=(0.5 1.0 1.5 2.0 2.5 3.5 4.0 4.5)

run_one() {
    local IDX=$1
    local STD=${STDS[$IDX]}
    local STD_TAG=$(printf '%g' $STD)
    local RUN_DIR="${REPO_ROOT}/outputs/geom_identityRot_256_conformer_${STD_TAG}std_stereo_hydrogen_no_rotvf_no_prealign/train"


    echo "Run $IDX (GPU $IDX): trans_prior_std=$STD at run_dir $RUN_DIR"

    CUDA_VISIBLE_DEVICES=$IDX python ${REPO_ROOT}/train.py \
        domain=protein \
        paradigm=multiframefm \
        datamodule.batch_size=12 \
        datamodule.num_workers=4 \
        model.c_s=256 \
        model.c_cond=256 \
        model.c_frame=256 \
        model.c_framepair=64 \
        model.z_broadcast=true \
        model.rigid_transformer_num_blocks=1 \
        model.rigid_transformer_rigid_updates=true \
        model.use_embedder_sc_rigid_transformer=true \
        model.use_ipa_gating=true \
        model.use_qk_norm=true \
        model.use_amp=true \
        model.rot_preconditioning=true \
        model.num_blocks=8 \
        lmodule.use_ema=true \
        lmodule.strict_weight_loading=false \
        corrupter.use_stochastic_centering=false \
        corrupter.center_on_motif=false \
        corrupter.trans_prior_std=${STD} \
        dataset.config="'${REPO_ROOT}/configs/train/data/geom_conformer.yaml'" \
        +dataset.val_config="'${REPO_ROOT}/configs/train/data/geom_conformer_val.yaml'" \
        experiment.lightning.devices=1 \
        experiment.lightning.strategy=auto \
        experiment.checkpointer.train_time_interval=null \
        experiment.checkpointer.every_n_train_steps=500 \
        hydra.run.dir="'${RUN_DIR}'" \
        experiment.lightning.max_epochs=-1 \
        model.use_bond_rotation=false \
        experiment.lightning.accumulate_grad_batches=1 \
        lmodule.bond_rotation_head_only=false \
        lmodule.scale_bond_length_loss=false \
        lmodule.scale_bond_angle_loss=false \
        lmodule.scale_ring_planarity_loss=false \
        dataset.include_h=true \
        model.patch_unit_vec_bug=true \
        model.patch_rel_quat_bug=true \
        lmodule.identity_rot_noise=true \
        lmodule.use_rot_vf_loss=false \
        lmodule.use_cosine_annealing=true \
        lmodule.cosine_annealing_T_max=500 \
        corrupter.prealign_noise=false
}

for i in 0 1 2 3 4 5 6 7; do
    run_one $i &
done
wait
