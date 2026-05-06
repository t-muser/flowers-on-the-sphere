#!/bin/bash
# Daint sweep on the MITgcm Held-Suarez 3-D ensemble (64x128 lat/lon, 25
# stacked channels: u/v/T over 8 ERA5 pressure levels + ps).
#   models: {fno, sfno, flower, zinnia_v5, local_r_transformer, local_s2_transformer}
#   lrs:    {1e-4, 5e-4, 1e-3}
#   = 18 train+test job pairs. Mirrors submit_global_ocean_daint.sh.
#
# Optional: pass SMOKE_JOB to chain all train jobs as afterok:$SMOKE_JOB so
# they only run after the in-container DataModule smoke succeeds.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

MODELS=(fno sfno flower zinnia_v5 local_r_transformer local_s2_transformer)
DATASET=held_suarez
LRS=(1e-4 5e-4 1e-3)

SMOKE_JOB="${SMOKE_JOB:-}"
TRAIN_DEP=""
if [ -n "$SMOKE_JOB" ]; then
    TRAIN_DEP="--dependency=afterok:${SMOKE_JOB}"
    echo "chaining all train jobs to depend on smoke job ${SMOKE_JOB}"
fi

NAME="$DATASET"
for model in "${MODELS[@]}"; do
    for lr in "${LRS[@]}"; do
        bs=$(fots_batch_size "$model" "$DATASET")

        TRAIN_JOB=$(sbatch --parsable --time=24:00:00 \
            --job-name="fots-${model}-${DATASET}-${lr}" \
            ${TRAIN_DEP} \
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
