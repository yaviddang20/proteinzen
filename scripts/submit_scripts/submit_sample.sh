#!/bin/bash

ROOT_DIR=${HOME}/projects/ligbinddiff
SUBMIT_TIME=$(date +"%y-%m-%d_%H%M%S")

sed -e "s|OUTFILE|${ROOT_DIR}/outputs/sge_outs/sample_${SUBMIT_TIME}|g" \
    -e "s|ARGS|$@" \
    < "${ROOT_DIR}/scripts/submit_scripts/_sample_template.sh" \
    > "${ROOT_DIR}/scripts/submit_scripts/autogen_scripts/sample_${SUBMIT_TIME}.sh"

qsub "${ROOT_DIR}/scripts/submit_scripts/autogen_scripts/sample_${SUBMIT_TIME}.sh" "$@"
