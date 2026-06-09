#!/bin/bash
# Daint sweep for ZinniaV5 on the three non-PlanetSWE, non-shock datasets:
#   {zinnia_v5} x {galewsky, cahn_hilliard, mickelin} x {1e-4, 5e-4, 1e-3}
#   = 9 train+test job pairs.
# Each cell: 24h DDP train on 4 GH200 + afterok 2h test.
#
# Capstor-independent: train_daint.sbatch defaults to the iopsstor toml
# (image on /iopsstor) and EXPERIMENT_DIR=/iopsstor/scratch/cscs/$USER/fots-runs.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

MODELS=(zinnia_v5)
DATASETS=(galewsky cahn_hilliard mickelin)
LRS=(1e-4 5e-4 1e-3)

for dataset in "${DATASETS[@]}"; do
    NAME="$dataset"
    for model in "${MODELS[@]}"; do
        for lr in "${LRS[@]}"; do
            bs=$(fots_batch_size "$model" "$dataset")

            TRAIN_JOB=$(sbatch --parsable --time=24:00:00 \
                --job-name="fots-${model}-${dataset}-${lr}" \
                --export="MODEL_CONFIG=configs/models/${model}.yaml,DATA_CONFIG=configs/data/${dataset}.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${NAME}" \
                scripts/train_daint.sbatch)
            echo "submitted train ${model} ${dataset} lr=${lr} bs=${bs} -> ${TRAIN_JOB}"

            TEST_JOB=$(sbatch --parsable --time=02:00:00 \
                --job-name="fots-test-${model}-${dataset}-${lr}" \
                --dependency=afterok:${TRAIN_JOB} \
                --export="MODEL_CONFIG=configs/models/${model}.yaml,DATA_CONFIG=configs/data/${dataset}.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${NAME},TEST_MODE=true" \
                scripts/train_daint.sbatch)
            echo "submitted test  ${model} ${dataset} lr=${lr} -> ${TEST_JOB} (afterok ${TRAIN_JOB})"
        done
    done
done
