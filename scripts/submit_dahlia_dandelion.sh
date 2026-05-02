#!/bin/bash
# PlanetSWE sweep for the two new spherical Flower variants:
# {dahlia, dandelion} x {1e-4, 5e-4, 1e-3} = 6 train+test job pairs.
# Mirrors submit_train_and_test.sh: 24h train followed by an `afterok` 2h test.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

MODELS=(dahlia dandelion)
LRS=(1e-4 5e-4 1e-3)

DATASET="planetswe"
NAME="$DATASET"
for model in "${MODELS[@]}"; do
    for lr in "${LRS[@]}"; do
        bs=$(fots_batch_size "$model" "$DATASET")

        TRAIN_JOB=$(sbatch --parsable --time=24:00:00 \
            --job-name="fots-${model}-${lr}" \
            --export="MODEL_CONFIG=configs/models/${model}.yaml,DATA_CONFIG=configs/data/${DATASET}.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${NAME}" \
            scripts/train.sbatch)
        echo "submitted train ${model} lr=${lr} bs=${bs} -> ${TRAIN_JOB}"

        TEST_JOB=$(sbatch --parsable --time=02:00:00 \
            --job-name="fots-test-${model}-${lr}" \
            --dependency=afterok:${TRAIN_JOB} \
            --export="MODEL_CONFIG=configs/models/${model}.yaml,DATA_CONFIG=configs/data/${DATASET}.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${NAME},TEST_MODE=true" \
            scripts/train.sbatch)
        echo "submitted test ${model} lr=${lr} -> ${TEST_JOB} (afterok ${TRAIN_JOB})"
    done
done
