#!/bin/bash
# PlanetSWE baseline sweep: {fno, sfno, flower, zinnia, local_r_transformer,
# local_s2_transformer} x {1e-4, 5e-4, 1e-3} = 18 train+test job pairs.
# Submits a 24h train job followed by an `afterok` 2h test job for each cell.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(fno sfno flower zinnia local_r_transformer local_s2_transformer)
LRS=(1e-4 5e-4 1e-3)

# Per-model batch size on 1xH200 (141 GB) at 256x512x12 input channels.
# Empirically derived from scripts/probe_memory.py — peak_reserved at the
# AR(x2) train step + headroom for validation/rollout.
declare -A BATCH_SIZE=(
    [fno]=32                    # 4L, 19.5M, peak 102 GiB (39 GiB headroom)
    [sfno]=32                   # 4L, 19.5M, peak 108 GiB (33 GiB headroom)
    [flower]=32                 # 17.3M, peak 124 GiB (cuDNN-leaner at B=32 vs B=28)
    [zinnia]=10                 # lifting_dim=150, 19.0M, peak 98 GiB
    [local_r_transformer]=10    # 18.97M, peak 113 GiB (28 GiB headroom)
    [local_s2_transformer]=4    # ~19M, peak 85 GiB (Triton autotune anomalous; B=4 is leanest)
)

NAME="planetswe"
for model in "${MODELS[@]}"; do
    for lr in "${LRS[@]}"; do
        bs=${BATCH_SIZE[$model]}
        export MODEL_CONFIG="configs/models/${model}.yaml"
        export DATA_CONFIG="configs/data/planetswe.yaml"
        export TRAIN_CONFIG="configs/train_4-to-1.yaml"
        export LR="$lr" BATCH_SIZE="$bs" NAME="$NAME"

        TRAIN_JOB=$(sbatch --parsable --time=24:00:00 \
            --job-name="fots-${model}-${lr}" \
            --export=ALL \
            scripts/train.sbatch)
        echo "submitted train ${model} lr=${lr} bs=${bs} -> ${TRAIN_JOB}"

        TEST_JOB=$(sbatch --parsable --time=02:00:00 \
            --job-name="fots-test-${model}-${lr}" \
            --dependency=afterok:${TRAIN_JOB} \
            --export=ALL,TEST_MODE=true \
            scripts/train.sbatch)
        echo "submitted test ${model} lr=${lr} -> ${TEST_JOB} (afterok ${TRAIN_JOB})"
    done
done
