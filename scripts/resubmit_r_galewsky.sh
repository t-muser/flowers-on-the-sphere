#!/bin/bash
# Stage + resume the two local_r_transformer/galewsky runs that didn't
# finish:
#   lr=1e-4: last completed epoch 21 (AR x2), recent.pt on /capstor only
#   lr=5e-4: last completed epoch 22 (AR x2), recent.pt on /capstor only
# Total epochs = 25 (20 base + 5 AR), so 3-4 epochs left per run.
#
# These were missed by copy_runs_capstor_to_iopsstor.sh (only the 1e-3
# cell was in its Bucket B), so we rsync them ourselves first, then submit
# with auto_resume=true via DATA_OVERRIDE.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

MODEL=local_r_transformer
DATASET=galewsky
LRS=(1e-4 5e-4)
BS=$(fots_batch_size "$MODEL" "$DATASET")

declare -A LR_DEC=([1e-4]=0.0001 [5e-4]=0.0005 [1e-3]=0.001)
SRC=/capstor/scratch/cscs/${USER}/fots-runs
DST=/iopsstor/scratch/cscs/${USER}/fots-runs
mkdir -p "$DST"

for lr in "${LRS[@]}"; do
    name="${DATASET}-${DATASET}-LocalRTransformer-${LR_DEC[$lr]}"
    if [ ! -d "$SRC/$name" ]; then
        echo "MISSING on capstor: $name -- skipping" >&2
        continue
    fi
    echo "rsync  $name (capstor -> iopsstor)"
    rsync -a --info=stats1 "$SRC/$name/" "$DST/$name/"
done

for lr in "${LRS[@]}"; do
    jobname="fots-${MODEL}-${DATASET}-${lr}"
    testname="fots-test-${MODEL}-${DATASET}-${lr}"
    exports="MODEL_CONFIG=configs/models/${MODEL}.yaml,DATA_CONFIG=configs/data/${DATASET}.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${BS},NAME=${DATASET},DATA_OVERRIDE=auto_resume=true"

    train_job=$(sbatch --parsable --time=24:00:00 \
        --job-name="$jobname" \
        --export="$exports" \
        scripts/train_daint.sbatch)
    echo "submitted train ${MODEL} ${DATASET} lr=${lr} bs=${BS} -> ${train_job} (resume)"

    test_job=$(sbatch --parsable --time=02:00:00 \
        --job-name="$testname" \
        --dependency=afterok:${train_job} \
        --export="${exports},TEST_MODE=true" \
        scripts/train_daint.sbatch)
    echo "submitted test  ${MODEL} ${DATASET} lr=${lr}              -> ${test_job} (afterok ${train_job})"
done
