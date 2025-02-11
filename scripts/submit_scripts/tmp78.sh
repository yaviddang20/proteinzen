BATCH_SIZE=200704

bash submit_train_h100.sh 4 14 \
    domain=protein \
    paradigm=multiframefm \
    datamodule.batch_size=$BATCH_SIZE \
    datamodule=afdb_512 \
    datamodule.max_len=448 \
    datamodule.sample_from_clusters=true \
    datamodule.length_batch=true \
    datamodule.max_num_per_batch=32 \
    model.c_frame=256 \
    model.c_framepair=64 \
    model.transformer_pair_bias=true \
    model.use_traj_predictions=true \
    model.use_embedder_v2=true \
    model.z_broadcast=true \
    model.propagate_framepair_embed=true \
    model.rigid_transformer_num_blocks=1 \
    model.rigid_transformer_rigid_updates=true \
    model.use_embedder_sc_rigid_transformer=true \
    model.use_traj_framepair_predictor=true \
    model.use_traj_seqpair_predictor=true \
    model.use_ipa_gating=true \
    tasks.use_fafe=true \
    tasks.scale_fafe=true \
    tasks.fafe_weight=0.5 \
    tasks.polar_upweight=true \
    tasks.dummy_rigid_to_sidechain_rigid=true \
    model.cg_version=8 \
    tasks.cg_version=8 \
    lmodule.use_ema=true \
    lmodule.use_posthoc_ema=true \
    corrupter.lognorm_t_sched=true \
    experiment.lightning.devices=4 \
    experiment.lightning.strategy=ddp_find_unused_parameters_true \

sleep 1s
