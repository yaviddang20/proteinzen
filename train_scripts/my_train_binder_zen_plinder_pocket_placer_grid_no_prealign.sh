#!/usr/bin/env bash
# Usage: bash my_train_plinder_pocket_placer_grid.sh <0-7>
# Grid search over side_chain_trans_prior_std and lig_trans_prior_std.
#
# Index | sc_std | lig_std
#   0   |  1.0   |  1.0
#   1   |  2.0   |  2.0
#   2   |  3.0   |  3.0
#   3   |  0.5   |  1.0
#   4   |  0.5   |  3.0
#   5   |  1.0   |  2.0
#   6   |  1.0   |  3.0
#   7   |  1.5   |  3.0

dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/../env_vars.sh
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1

SC_STDS=(1.0 2.0 3.0 0.5 0.5 1.0 1.0 1.5)
LIG_STDS=(1.0 2.0 3.0 1.0 3.0 2.0 3.0 3.0)

IDX=${1:?Usage: $0 <0-7>}
SC_STD=${SC_STDS[$IDX]}
LIG_STD=${LIG_STDS[$IDX]}

# format for directory name (replace . with p)
SC_TAG=$(echo $SC_STD | tr '.' 'p')
LIG_TAG=$(echo $LIG_STD | tr '.' 'p')
RUN_DIR="${REPO_ROOT}/outputs/binder_zen_plinder_pocket_placer_sc${SC_TAG}_lig${LIG_TAG}/train"

export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=${REPO_ROOT}/outputs/binder_zen_plinder_pocket_placer_sc${SC_TAG}_lig${LIG_TAG}/debug/nccl_trace_rank_
mkdir -p ${REPO_ROOT}/outputs/binder_zen_plinder_pocket_placer_sc${SC_TAG}_lig${LIG_TAG}/debug

echo "Run $IDX: side_chain_trans_prior_std=$SC_STD  lig_trans_prior_std=$LIG_STD"
echo "Output: $RUN_DIR"

python ${REPO_ROOT}/train.py \
    domain=protein \
    paradigm=multiframefm \
    datamodule.batch_size=3 \
    datamodule.num_workers=8 \
    model.c_s=768 \
    model.c_cond=768 \
    model.c_frame=256 \
    model.c_framepair=64 \
    model.z_broadcast=true \
    model.rigid_transformer_num_blocks=1 \
    model.rigid_transformer_rigid_updates=true \
    model.use_embedder_sc_rigid_transformer=true \
    model.use_ipa_gating=true \
    model.use_qk_norm=true \
    model.use_amp=true \
    model.rot_preconditioning=false \
    model.num_blocks=12 \
    model.patch_unit_vec_bug=true \
    model.use_bond_rotation=false \
    model.add_same_chain_feature=true \
    model.disable_absolute_res_idx=true \
    model.embed_hotspot_type=true \
    model.embed_token_is_copy_mask=true \
    model.embed_rigids_noising_mask=true \
    model.use_entity_id_unmasking=false \
    lmodule.use_ema=true \
    lmodule.strict_weight_loading=false \
    lmodule.use_interchain_fafe_loss=true \
    lmodule.use_brownian_rot_path_loss=true \
    lmodule.bond_rotation_head_only=false \
    lmodule.scale_bond_length_loss=false \
    lmodule.scale_bond_angle_loss=false \
    lmodule.scale_ring_planarity_loss=false \
    lmodule.identity_rot_noise=false \
    lmodule.use_rot_vf_loss=true \
    lmodule.use_cosine_annealing=true \
    lmodule.cosine_annealing_T_max=1000 \
    corrupter.prealign_noise=false \
    corrupter.use_stochastic_centering=true \
    corrupter.center_on_motif_then_hotspots=true \
    corrupter.trans_prior_std=3 \
    corrupter.sig_perturb=2 \
    corrupter.use_uniform_rot_noise=true \
    corrupter.rots_use_brownian_path=true \
    dataset.config="'${REPO_ROOT}/configs/train/data/plinder_pocket_placer.yaml'" \
    +dataset.val_config="'${REPO_ROOT}/configs/train/data/plinder_pocket_placer_val.yaml'" \
    dataset.include_h=true \
    dataset.placer_side_chain_trans_prior_std=${SC_STD} \
    dataset.placer_lig_trans_prior_std=${LIG_STD} \
    experiment.optim.lr=0.0001 \
    experiment.lightning.devices=4 \
    experiment.lightning.strategy=ddp_find_unused_parameters_true \
    experiment.lightning.max_epochs=-1 \
    experiment.lightning.accumulate_grad_batches=2 \
    experiment.checkpointer.train_time_interval=null \
    experiment.checkpointer.every_n_train_steps=500 \
    hydra.run.dir="'${RUN_DIR}'" \
    "experiment.warm_start='${REPO_ROOT}/proteinzen_weights/binder_design_phase2_6/lightning_logs/version_0/checkpoints/last_no_opt.ckpt'"
