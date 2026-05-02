#!/bin/bash
# Daint launcher: submit a (model, lr) train job on Daint via train_daint.sbatch.
#
# Usage:
#   scripts/submit_train_daint.sh                       # sfno @ 5e-4 on galewsky
#   MODEL=fno LR=1e-4 BS=8 NAME=galewsky-fno-1e-4 \
#       scripts/submit_train_daint.sh
#   TEST_MODE=true scripts/submit_train_daint.sh        # test the just-trained checkpoint
#
# A 24h train job is submitted, followed by a 2h afterok test job.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-sfno}"
DATASET="${DATASET:-galewsky}"
TRAIN_CFG="${TRAIN_CFG:-configs/train_4-to-1.yaml}"
LR="${LR:-5e-4}"
# Conservative batch size for GH200 (96 GB HBM3, vs the H200 baseline at 141 GB).
# Override in the environment when probing the OOM cliff.
BS="${BS:-8}"
NAME="${NAME:-${DATASET}}"

MODEL_CFG="configs/models/${MODEL}.yaml"
DATA_CFG="configs/data/${DATASET}.yaml"

echo "Submitting: model=${MODEL} dataset=${DATASET} lr=${LR} bs=${BS} name=${NAME}"

TRAIN_JOB=$(sbatch --parsable --time=24:00:00 \
    --job-name="fots-${MODEL}-${LR}" \
    --export="MODEL_CONFIG=${MODEL_CFG},DATA_CONFIG=${DATA_CFG},TRAIN_CONFIG=${TRAIN_CFG},LR=${LR},BATCH_SIZE=${BS},NAME=${NAME}" \
    scripts/train_daint.sbatch)
echo "submitted train -> ${TRAIN_JOB}"

if [ "${SKIP_TEST:-false}" != "true" ]; then
    TEST_JOB=$(sbatch --parsable --time=02:00:00 \
        --job-name="fots-test-${MODEL}-${LR}" \
        --dependency=afterok:${TRAIN_JOB} \
        --export="MODEL_CONFIG=${MODEL_CFG},DATA_CONFIG=${DATA_CFG},TRAIN_CONFIG=${TRAIN_CFG},LR=${LR},BATCH_SIZE=${BS},NAME=${NAME},TEST_MODE=true" \
        scripts/train_daint.sbatch)
    echo "submitted test  -> ${TEST_JOB} (afterok ${TRAIN_JOB})"
fi
