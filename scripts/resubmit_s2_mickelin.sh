#!/bin/bash
# Resume the three local_s2_transformer/mickelin runs that didn't finish:
#   lr=1e-4 (job 3332336): SLURM TIMEOUT mid-epoch 23, recent.pt at epoch 22
#   lr=5e-4 (job 3332338): SLURM TIMEOUT mid-epoch 23, recent.pt at epoch 22
#   lr=1e-3 (job 3332340): NCCL watchdog hang during epoch 19 rollout_val,
#                          recent.pt at epoch 18
# Total epochs = 25 (20 base + 5 AR), so 3-7 epochs left per run.
#
# Trainer supports full state restore (model, optimizer, lr_scheduler,
# epoch, best_val_loss). auto_resume=true picks the highest-numbered
# subdir under runs/.../<NAME>/ and loads <ckpt_dir>/recent.pt.
# Passed via DATA_OVERRIDE env (the only slot train_daint.sbatch currently
# forwards as a Hydra override in the train branch).
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

MODEL=local_s2_transformer
DATASET=mickelin
LRS=(1e-4 5e-4 1e-3)
BS=$(fots_batch_size "$MODEL" "$DATASET")

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
