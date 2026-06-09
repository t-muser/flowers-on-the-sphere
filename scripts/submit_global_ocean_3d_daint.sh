#!/bin/bash
# Daint sweep on the 3-D global-ocean cs32x15 dataset (64x128 lat/lon, 6 levels).
#   models: {fno, sfno, flower, zinnia_v5, local_r_transformer, local_s2_transformer}
#   lrs:    {1e-4, 5e-4, 1e-3}
#   = 18 train+test job pairs. Each cell: 24h DDP train on 4 GH200 +
#   afterok 2h test, mirroring submit_global_ocean_daint.sh.
#
# SKIP_COMBOS lets you pass already-submitted (model,lr) pairs that the
# script should leave alone (so re-runs are idempotent).
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

MODELS=(fno sfno flower zinnia_v5 local_r_transformer local_s2_transformer)
DATASET=global_ocean_3d
LRS=(1e-4 5e-4 1e-3)
SKIP_COMBOS="${SKIP_COMBOS:-}"   # space-separated list, e.g. "flower:5e-4 sfno:1e-3"

NAME="$DATASET"
for model in "${MODELS[@]}"; do
    for lr in "${LRS[@]}"; do
        if [[ " $SKIP_COMBOS " == *" ${model}:${lr} "* ]]; then
            echo "skip            ${model} ${DATASET} lr=${lr} (in SKIP_COMBOS)"
            continue
        fi
        bs=$(fots_batch_size "$model" "$DATASET")

        TRAIN_JOB=$(sbatch --parsable --time=24:00:00 \
            --job-name="fots-${model}-${DATASET}-${lr}" \
            --export="MODEL_CONFIG=configs/models/${model}.yaml,DATA_CONFIG=configs/data/${DATASET}.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${NAME}" \
            scripts/train_daint.sbatch)
        echo "submitted train ${model} ${DATASET} lr=${lr} bs=${bs} -> ${TRAIN_JOB}"

        TEST_JOB=$(sbatch --parsable --time=02:00:00 \
            --job-name="fots-test-${model}-${DATASET}-${lr}" \
            --dependency=afterok:${TRAIN_JOB} \
            --export="MODEL_CONFIG=configs/models/${model}.yaml,DATA_CONFIG=configs/data/${DATASET}.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${NAME},TEST_MODE=true" \
            scripts/train_daint.sbatch)
        echo "submitted test  ${model} ${DATASET} lr=${lr} -> ${TEST_JOB} (afterok ${TRAIN_JOB})"
    done
done
