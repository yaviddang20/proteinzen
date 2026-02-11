python sample.py \
    model_dir=/datastor1/dy4652/proteinzen/outputs/geom_og/train \
    out_dir=sampling/geom_og_test \
    sampler.tasks_yaml=sampling/geom_sampling_qmugs_test_mol.yaml \
    sampler.batch_size=32 \
    save_traj=true \
    +version_num=1


# corrupter.trans_step_size=1
# corrupter.rot_step_size=1
# corrupter.sampling_noise_mode=null