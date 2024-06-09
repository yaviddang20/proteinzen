#!/bin/bash

ROOT_DIR=${HOME}/projects/ligbinddiff
SUBMIT_TIME=$(date +"%y-%m-%d_%H%M%S")

sed -e "s|OUTFILE|${ROOT_DIR}/outputs/sge_outs/train_${SUBMIT_TIME}|g" \
    < "${ROOT_DIR}/scripts/submit_scripts/_train_template.sh" \
    > "${ROOT_DIR}/scripts/submit_scripts/autogen_scripts/train_${SUBMIT_TIME}.sh"
echo "#ARGS = $@" >> "${ROOT_DIR}/scripts/submit_scripts/autogen_scripts/train_${SUBMIT_TIME}.sh"

qsub -terse "${ROOT_DIR}/scripts/submit_scripts/autogen_scripts/train_${SUBMIT_TIME}.sh" "$@"
