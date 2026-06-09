#!/bin/bash
# Daint sweep for the two new spherical Flower variants on the four
# non-PlanetSWE datasets:
#   {dahlia, dandelion} x {shock_caps, galewsky, mickelin, cahn_hilliard}
#   x {1e-4, 5e-4, 1e-3} = 24 train+test job pairs.
# Each cell: 24h DDP train on 4 GH200 + afterok 2h test.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

MODELS=(dahlia dandelion)
DATASETS=(shock_caps galewsky mickelin cahn_hilliard)
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
