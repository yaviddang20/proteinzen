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
        model.num_blocks=8 \
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
run 0 hydrogen_IdentityNoise \
    dataset.include_h=true \
    lmodule.identity_rot_noise=true \
    lmodule.use_rot_vf_loss=false \
    "experiment.warm_start='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_hydrogen_molscaffold_IdentityNoise/train/lightning_logs/version_62672/checkpoints/best.ckpt'" \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_hydrogen_molscaffold_IdentityNoise/train'"

run 1 hydrogen_no_fafe \
    dataset.include_h=true \
    lmodule.use_fafe_loss=false \
    "experiment.warm_start='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_hydrogen_molscaffold_no_fafe/train/lightning_logs/version_62672/checkpoints/best.ckpt'" \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_hydrogen_molscaffold_no_fafe/train'"

run 2 hydrogen_no_rotvf \
    dataset.include_h=true \
    lmodule.identity_rot_noise=true \
    lmodule.use_rot_vf_loss=false \
    "experiment.warm_start='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_hydrogen_molscaffold_no_rotvf/train/lightning_logs/version_62672/checkpoints/best.ckpt'" \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_hydrogen_molscaffold_no_rotvf/train'"

run 3 hydrogen_predict_time \
    dataset.include_h=true \
    model.predict_time=true \
    "experiment.warm_start='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_hydrogen_molscaffold_predict_time/train/lightning_logs/version_62672/checkpoints/best.ckpt'" \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_hydrogen_molscaffold_predict_time/train'"

# --- hydrogen=false ---
run 4 IdentityNoise \
    dataset.include_h=false \
    lmodule.identity_rot_noise=true \
    lmodule.use_rot_vf_loss=false \
    "experiment.warm_start='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_molscaffold_IdentityNoise/train/lightning_logs/version_62672/checkpoints/best.ckpt'" \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_molscaffold_IdentityNoise/train'"

run 5 no_fafe \
    dataset.include_h=false \
    lmodule.use_fafe_loss=false \
    "experiment.warm_start='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_molscaffold_no_fafe/train/lightning_logs/version_62672/checkpoints/best.ckpt'" \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_molscaffold_no_fafe/train'"

run 6 no_rotvf \
    dataset.include_h=false \
    lmodule.identity_rot_noise=true \
    lmodule.use_rot_vf_loss=false \
    "experiment.warm_start='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_molscaffold_no_rotvf/train/lightning_logs/version_62672/checkpoints/best.ckpt'" \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_molscaffold_no_rotvf/train'"

run 7 predict_time \
    dataset.include_h=false \
    model.predict_time=true \
    "hydra.run.dir='${REPO_ROOT}/outputs/geom_identityRot_256_conformer_3std_stereo_molscaffold_predict_time/train'"

wait
echo "All ablations finished."
