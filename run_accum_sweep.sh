#!/bin/bash
dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

for i in $(seq 0 7); do
    accum=$(( (i + 1) * 2 ))
    echo "Launching CUDA_VISIBLE_DEVICES=${i} accum=${accum}"
    CUDA_VISIBLE_DEVICES=${i} bash ${dir}/my_train_conformer_3std_stereo_hydrogen_molscaffold.sh ${accum} &
done

wait
