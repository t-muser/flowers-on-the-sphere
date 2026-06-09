#!/bin/bash
# Daint sweep on the pre-regridded 3-D global-ocean dataset (latlon variant).
#   models: {fno, sfno, flower, zinnia_v5, local_r_transformer, local_s2_transformer}
#   lrs:    {1e-4, 5e-4, 1e-3}
#   = 18 train+test job pairs. Mirrors submit_global_ocean_3d_daint.sh, but
#   points at configs/data/global_ocean_3d_latlon.yaml and uses the
#   global_ocean_3d_latlon batch-size bucket.
#
# SKIP_COMBOS skips already-submitted (model,lr) pairs so re-runs are
# idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

MODELS=(fno sfno flower zinnia_v5 local_r_transformer local_s2_transformer)
DATASET=global_ocean_3d_latlon
LRS=(1e-4 5e-4 1e-3)
SKIP_COMBOS="${SKIP_COMBOS:-}"

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
