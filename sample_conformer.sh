python sample.py \
    model_dir=/datastor1/dy4652/proteinzen/outputs/geom_identityRot_256_conformer_3std_bondlength/train \
    out_dir=sampling/geom_conformer_train/geom_identityRot_256_conformer_train_3std_bondlength \
    sampler.tasks_yaml=sampling/geom_conformer_train_mol.yaml \
    sampler.batch_size=32 \
    save_traj=true \
    +version_num=29 \
    corrupter.sampling_noise_mode=null