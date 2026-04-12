#!/bin/bash
dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/../env_vars.sh

run() {
    local gpu=$1
    local name=$2
    shift 2
    CUDA_VISIBLE_DEVICES=$gpu python ${REPO_ROOT}/train.py \
        domain=protein \
        paradigm=multiframefm \
        datamodule.batch_size=20 \
        datamodule.num_workers=8 \
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
        lmodule.use_ema=true \
        lmodule.strict_weight_loading=false \
        corrupter.use_stochastic_centering=false \
        corrupter.center_on_motif=false \
        corrupter.trans_prior_std=3 \
        "dataset.config='${REPO_ROOT}/configs/train/data/geom_conformer_scaffold.yaml'" \
        "+dataset.val_config='${REPO_ROOT}/configs/train/data/geom_conformer_val.yaml'" \
        experiment.lightning.devices=1 \
        experiment.lightning.strategy=auto \
        experiment.checkpointer.train_time_interval=null \
        experiment.checkpointer.every_n_train_steps=500 \
        experiment.lightning.max_epochs=-1 \
        model.use_bond_rotation=false \
        experiment.lightning.accumulate_grad_batches=2 \
        lmodule.bond_rotation_head_only=false \
        lmodule.scale_bond_length_loss=false \
        lmodule.scale_bond_angle_loss=false \
        lmodule.scale_ring_planarity_loss=false \
        model.patch_unit_vec_bug=true \
        lmodule.use_cosine_annealing=true \
        lmodule.cosine_annealing_T_max=500 \
        "$@" &
    echo "Launched GPU $gpu: $name (PID $!)"
}

# --- hydrogen=true ---
run 0 hydrogen_no_rotvf_4blocks \
    dataset.include_h=true \
    model.num_blocks=4 \
    lmodule.identity_rot_noise=true \
    lmodule.use_rot_vf_loss=false \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_hydrogen_molscaffold_no_rotvf_4blocks/train'"

run 1 hydrogen_no_rotvf_energy \
    dataset.include_h=true \
    model.num_blocks=8 \
    lmodule.identity_rot_noise=true \
    lmodule.use_rot_vf_loss=false \
    model.predict_energy=true \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_hydrogen_molscaffold_no_rotvf_energy/train'"

run 2 hydrogen_no_rotvf_mse \
    dataset.include_h=true \
    model.num_blocks=8 \
    lmodule.identity_rot_noise=true \
    lmodule.use_rot_vf_loss=false \
    lmodule.use_trans_mse_loss=true \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_hydrogen_molscaffold_no_rotvf_mse/train'"

run 3 hydrogen_noIdentityRot \
    dataset.include_h=true \
    model.num_blocks=8 \
    lmodule.identity_rot_noise=false \
    lmodule.use_rot_vf_loss=true \
    +dataset.use_identity_rot=false \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_hydrogen_molscaffold_noIdentityRot/train'"

# --- hydrogen=false ---
run 4 no_rotvf_4blocks \
    dataset.include_h=false \
    model.num_blocks=4 \
    lmodule.identity_rot_noise=true \
    lmodule.use_rot_vf_loss=false \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_molscaffold_no_rotvf_4blocks/train'"

run 5 no_rotvf_energy \
    dataset.include_h=false \
    model.num_blocks=8 \
    lmodule.identity_rot_noise=true \
    lmodule.use_rot_vf_loss=false \
    model.predict_energy=true \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_molscaffold_no_rotvf_energy/train'"

run 6 no_rotvf_mse \
    dataset.include_h=false \
    model.num_blocks=8 \
    lmodule.identity_rot_noise=true \
    lmodule.use_rot_vf_loss=false \
    lmodule.use_trans_mse_loss=true \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_molscaffold_no_rotvf_mse/train'"

run 7 noIdentityRot \
    dataset.include_h=false \
    model.num_blocks=8 \
    lmodule.identity_rot_noise=false \
    lmodule.use_rot_vf_loss=true \
    +dataset.use_identity_rot=false \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_molscaffold_noIdentityRot/train'"

wait
echo "All ablations2 finished."
