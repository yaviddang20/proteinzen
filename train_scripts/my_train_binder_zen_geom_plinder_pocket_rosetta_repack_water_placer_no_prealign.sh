dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
source $dir/../env_vars.sh

python ${REPO_ROOT}/train.py \
    domain=protein \
    paradigm=multiframefm \
    datamodule.batch_size=8 \
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
    dataset.config="'${REPO_ROOT}/configs/train/data/plinder_pocket_rosetta_repack_water_placer.yaml'" \
    +dataset.val_config="'${REPO_ROOT}/configs/train/data/plinder_pocket_rosetta_repack_water_placer_val.yaml'" \
    dataset.include_h=false \
    experiment.optim.lr=0.0001 \
    experiment.lightning.devices=4 \
    experiment.lightning.strategy=ddp_find_unused_parameters_true \
    experiment.lightning.max_epochs=-1 \
    experiment.lightning.accumulate_grad_batches=1 \
    experiment.checkpointer.train_time_interval=null \
    experiment.checkpointer.every_n_train_steps=500 \
    hydra.run.dir="'${REPO_ROOT}/outputs/binder_zen_geom_plinder_pocket_rosetta_repack_water_placer_no_prealign/train'"
    # NOTE: reference script (my_train_binder_zen_geom_plinder_pocket_water_placer_no_prealign.sh)
    # warm-started from a specific prior checkpoint (version_1564150/checkpoints/last.ckpt).
    # Deliberately NOT copied here -- starting this sidechain-ensemble-augmented run fresh by
    # default. Add an experiment.warm_start=... line yourself if you want to continue that lineage.
