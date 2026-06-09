#!/bin/bash
# Resume the three local_s2_transformer/cahn_hilliard runs that didn't finish:
#   lr=1e-4 (job 3332329): SLURM TIMEOUT mid-epoch 23 (AR), recent.pt May 5 16:22
#   lr=5e-4 (job 3332331): SLURM TIMEOUT mid-epoch 23 (AR), recent.pt May 5 16:24
#   lr=1e-3 (job 3332333): SLURM TIMEOUT after epoch 23 valid,  recent.pt May 5 11:38
# Total epochs = 25 (20 base + 5 AR), so 2-3 epochs left per run.
#
# Same recipe as resubmit_s2_mickelin.sh: auto_resume=true makes
# fots.utils.configure_experiment reuse the existing run-0 folder and
# load checkpoints/recent.pt; the trainer restores model + optimizer +
# lr_scheduler + starting_epoch and continues to max_epoch=25.
# wandb.init has no resume= flag, so each resume opens a fresh wandb run.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

MODEL=local_s2_transformer
DATASET=cahn_hilliard
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
