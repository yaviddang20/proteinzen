python sample.py \
    model_dir=/datastor1/dy4652/proteinzen/outputs/geom_og/train \
    out_dir=sampling/geom_og_train \
    sampler.tasks_yaml=sampling/test.yaml \
    sampler.batch_size=8 \
    save_traj=true \
    +version_num=1