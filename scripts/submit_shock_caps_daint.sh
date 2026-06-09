#!/bin/bash
# Daint sweep on shock_caps for the six models the user wants benchmarked:
#   {fno, sfno, local_r_transformer, local_s2_transformer, flower, zinnia_v5}
#   x {1e-4, 5e-4, 1e-3} = 18 train+test job pairs.
# Each cell: 24h DDP train on 4 GH200 + afterok 2h test.
#
# Capstor-independent: train_daint.sbatch defaults to the iopsstor toml
# (image on /iopsstor) and EXPERIMENT_DIR=/iopsstor/scratch/cscs/$USER/fots-runs.
#
# shock_caps data lives at /iopsstor/.../shock-caps-v2 (zarr v2, converted
# from the source v3 store via scripts/convert_zarr_v3_to_v2.py).
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

MODELS=(fno sfno local_r_transformer local_s2_transformer flower zinnia_v5)
LRS=(1e-4 5e-4 1e-3)

DATASET="shock_caps"
NAME="$DATASET"
for model in "${MODELS[@]}"; do
    for lr in "${LRS[@]}"; do
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
