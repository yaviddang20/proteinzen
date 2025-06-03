#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source $SCRIPT_DIR/../../env_vars.sh

SUBMIT_TIME=$(date +"%y-%m-%d_%H%M%S")

sed -e "s|OUTFILE|${REPO_ROOT}/outputs/sge_outs/train_${SUBMIT_TIME}|g" \
    -e "s|REPOROOT|${REPO_ROOT}|g" \
    < "${REPO_ROOT}/scripts/submit_scripts/_train_template_5day.sh" \
    > "${REPO_ROOT}/scripts/submit_scripts/autogen_scripts/train_${SUBMIT_TIME}.sh"
echo "#ARGS = $@" >> "${REPO_ROOT}/scripts/submit_scripts/autogen_scripts/train_${SUBMIT_TIME}.sh"

qsub -terse "${REPO_ROOT}/scripts/submit_scripts/autogen_scripts/train_${SUBMIT_TIME}.sh" "$@"
