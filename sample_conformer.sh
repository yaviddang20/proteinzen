MODE=train
MODEL_NAME=geom_identityRot_256_conformer_3std_bondlength

python sample.py \
    model_dir=/datastor1/dy4652/proteinzen/outputs/${MODEL_NAME}/train \
    out_dir=sampling/geom_conformer_${MODE}/${MODEL_NAME}_0_5_trans_vf_scale \
    sampler.tasks_yaml=sampling/geom_conformer_${MODE}/smiles.yaml \
    sampler.batch_size=32 \
    save_traj=true \
    +version_num=29 \
    corrupter.sampling_noise_mode=null