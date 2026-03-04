python sample.py \
    model_dir=/datastor1/dy4652/proteinzen/outputs/geom_identityRot_256_conformer_3std_10000/train \
    out_dir=sampling/geom_conformer_train/ \
    sampler.tasks_yaml=sampling/geom_conformer_sampling_test_mol.yaml \
    sampler.batch_size=32 \
    save_traj=true \
    +version_num=7