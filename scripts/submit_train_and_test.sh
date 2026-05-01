#!/bin/bash
# PlanetSWE baseline sweep: {fno, sfno, flower, zinnia, local_r_transformer,
# local_s2_transformer} x {1e-4, 5e-4, 1e-3} = 18 train+test job pairs.
# Submits a 24h train job followed by an `afterok` 2h test job for each cell.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(fno sfno flower zinnia local_r_transformer local_s2_transformer)
LRS=(1e-4 5e-4 1e-3)

# Per-model batch size on 1xH200 (141 GB) at 256x512x12 input channels.
# Derived from scripts/probe_memory.py AR(x2) sweep — chosen one step
# below the empirical OOM cliff to leave 5-8 GiB headroom for Adam state,
# allocator fragmentation, and dataloader prefetch.
declare -A BATCH_SIZE=(
    [fno]=40                    # confirmed 126 GiB / 90%; OOM at B=44
    [sfno]=38                   # est ~129 GiB / 92%; B=40 ok at 135 GiB, OOM at B=44
    [flower]=33                 # est ~131 GiB / 93%; B=34 ok at 137 GiB, OOM at B=36
    [zinnia]=14                 # est ~125 GiB / 89%; B=16 ok at 138 GiB, OOM at B=20
    [local_r_transformer]=11    # est ~125 GiB / 89%; B=12 ok at 136 GiB, OOM at B=14
    [local_s2_transformer]=6    # confirmed 128 GiB / 91%; OOM at B=8
)

NAME="planetswe"
for model in "${MODELS[@]}"; do
    for lr in "${LRS[@]}"; do
        bs=${BATCH_SIZE[$model]}

        TRAIN_JOB=$(sbatch --parsable --time=24:00:00 \
            --job-name="fots-${model}-${lr}" \
            --export="MODEL_CONFIG=configs/models/${model}.yaml,DATA_CONFIG=configs/data/planetswe.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${NAME}" \
            scripts/train.sbatch)
        echo "submitted train ${model} lr=${lr} bs=${bs} -> ${TRAIN_JOB}"

        TEST_JOB=$(sbatch --parsable --time=02:00:00 \
            --job-name="fots-test-${model}-${lr}" \
            --dependency=afterok:${TRAIN_JOB} \
            --export="MODEL_CONFIG=configs/models/${model}.yaml,DATA_CONFIG=configs/data/planetswe.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${NAME},TEST_MODE=true" \
            scripts/train.sbatch)
        echo "submitted test ${model} lr=${lr} -> ${TEST_JOB} (afterok ${TRAIN_JOB})"
    done
done
