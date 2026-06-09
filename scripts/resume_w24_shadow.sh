#!/bin/bash
# Snapshot the latest checkpoint of every currently-running latlon train job
# and submit a parallel "shadow" run with num_workers=24. Once a shadow shows
# faster throughput than its original, the original gets cancelled.
#
# Shadow runs use NAME=global_ocean_3d_latlon_w24 so wandb / experiment_dir
# don't collide with the originals. Each shadow loads weights via
# checkpoint_override pointing at the snapshot.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

SNAP_ROOT=/iopsstor/scratch/cscs/tmuser/fots-resume-snapshots
mkdir -p "$SNAP_ROOT"

EXP_ROOT=/iopsstor/scratch/cscs/tmuser/fots-runs

# model token  ->  class-name suffix used in the experiment-dir name.
declare -A CLASSNAME=(
    [fno]=Fno
    [sfno]=Sfno
    [flower]=Flower2D
    [zinnia_v5]=ZinniaV5
    [local_r_transformer]=LocalRTransformer
    [local_s2_transformer]=LocalS2Transformer
)

# Map LR strings to the float form used by the experiment dir name.
declare -A DIRLR=(
    [1e-4]=0.0001
    [5e-4]=0.0005
    [1e-3]=0.001
)

MODELS=(fno sfno flower zinnia_v5 local_r_transformer local_s2_transformer)
LRS=(1e-4 5e-4 1e-3)
DATASET=global_ocean_3d_latlon
NEW_NAME=global_ocean_3d_latlon_w24

for model in "${MODELS[@]}"; do
    for lr in "${LRS[@]}"; do
        cls=${CLASSNAME[$model]}
        dirlr=${DIRLR[$lr]}
        orig_dir="$EXP_ROOT/global_ocean_3d_latlon-global_ocean_3d_latlon-${cls}-${dirlr}/0/checkpoints"
        recent="$orig_dir/recent.pt"
        if [ ! -f "$recent" ]; then
            echo "skip ${model} ${lr}: no checkpoint at $recent"
            continue
        fi
        snap="$SNAP_ROOT/${model}-${lr}.pt"
        # cp to .tmp + mv → atomic destination (source may briefly be mid-write
        # but that's rare; the shadow will torch.load() which errors on partial,
        # caught by sbatch retry below if needed).
        cp -f "$recent" "${snap}.tmp" && mv "${snap}.tmp" "$snap"
        echo "snapshot ${model} ${lr} -> ${snap} ($(stat -c %s "$snap") bytes)"

        bs=$(fots_batch_size "$model" "$DATASET")

        # data.num_workers override + checkpoint_override → load snapshot.
        OVR="data.num_workers=24 checkpoint_override=${snap}"

        TRAIN_JOB=$(sbatch --parsable --time=24:00:00 \
            --job-name="fots-${model}-${NEW_NAME}-${lr}" \
            --export="MODEL_CONFIG=configs/models/${model}.yaml,DATA_CONFIG=configs/data/${DATASET}.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${NEW_NAME},DATA_OVERRIDE=${OVR}" \
            scripts/train_daint.sbatch)
        echo "shadow train ${model} ${lr} bs=${bs} -> ${TRAIN_JOB}"

        TEST_JOB=$(sbatch --parsable --time=02:00:00 \
            --job-name="fots-test-${model}-${NEW_NAME}-${lr}" \
            --dependency=afterok:${TRAIN_JOB} \
            --export="MODEL_CONFIG=configs/models/${model}.yaml,DATA_CONFIG=configs/data/${DATASET}.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${NEW_NAME},TEST_MODE=true" \
            scripts/train_daint.sbatch)
        echo "shadow test  ${model} ${lr} -> ${TEST_JOB} (afterok ${TRAIN_JOB})"
    done
done
