#!/bin/bash
# Resubmit Bucket B (crashed train+test) and Bucket C (test-only) cells
# from the original sweep, filtered to galewsky / cahn_hilliard / mickelin
# and dropping zinnia / dahlia / dandelion (rerun fresh as zinnia_v5 or
# out of scope).
#
# Runs against /iopsstor since train_daint.sbatch now defaults
# EXPERIMENT_DIR=/iopsstor/scratch/cscs/$USER/fots-runs and uses the
# iopsstor sqsh — capstor-independent.
#
# REQUIRES: scripts/copy_runs_capstor_to_iopsstor.sh has been run so the
# checkpoints exist under /iopsstor/.../fots-runs/. Bucket C jobs that
# can't find their checkpoint will fail; Bucket B jobs with auto_resume
# will silently start from scratch (24h instead of finishing the partial).

set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

submit_pair() {
    local model="$1" dataset="$2" lr="$3"
    local bs jobname testname exports
    bs=$(fots_batch_size "$model" "$dataset")
    jobname="fots-${model}-${dataset}-${lr}"
    testname="fots-test-${model}-${dataset}-${lr}"
    exports="MODEL_CONFIG=configs/models/${model}.yaml,DATA_CONFIG=configs/data/${dataset}.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${dataset}"

    local train_job test_job
    train_job=$(sbatch --parsable --time=24:00:00 \
        --job-name="$jobname" \
        --export="$exports" \
        scripts/train_daint.sbatch)
    echo "submitted train ${model} ${dataset} lr=${lr} bs=${bs} -> ${train_job}"

    test_job=$(sbatch --parsable --time=02:00:00 \
        --job-name="$testname" \
        --dependency=afterok:${train_job} \
        --export="${exports},TEST_MODE=true" \
        scripts/train_daint.sbatch)
    echo "submitted test  ${model} ${dataset} lr=${lr}              -> ${test_job} (afterok ${train_job})"
}

submit_test() {
    local model="$1" dataset="$2" lr="$3"
    local bs testname exports
    bs=$(fots_batch_size "$model" "$dataset")
    testname="fots-test-${model}-${dataset}-${lr}"
    exports="MODEL_CONFIG=configs/models/${model}.yaml,DATA_CONFIG=configs/data/${dataset}.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${dataset},TEST_MODE=true"
    local test_job
    test_job=$(sbatch --parsable --time=02:00:00 \
        --job-name="$testname" \
        --export="$exports" \
        scripts/train_daint.sbatch)
    echo "submitted test  ${model} ${dataset} lr=${lr} -> ${test_job}"
}

# Bucket B — crashed train+test (auto_resume picks up surviving checkpoint).
CELLS_B=(
    "flower    cahn_hilliard 5e-4"
    "fno       cahn_hilliard 1e-3"
    "local_r_transformer  galewsky 1e-3"
    "local_s2_transformer galewsky 1e-4"
    "local_s2_transformer galewsky 5e-4"
    "local_s2_transformer galewsky 1e-3"
    "sfno                 galewsky 5e-4"
    "sfno                 galewsky 1e-3"
    "flower              mickelin 5e-4"
    "fno                 mickelin 1e-4"
    "local_r_transformer mickelin 1e-3"
)

# Bucket C — train finished, only test missing.
CELLS_C=(
    "flower              cahn_hilliard 1e-4"
    "flower              cahn_hilliard 1e-3"
    "fno                 cahn_hilliard 1e-4"
    "fno                 cahn_hilliard 5e-4"
    "local_r_transformer cahn_hilliard 5e-4"
    "sfno                cahn_hilliard 1e-4"
    "sfno                cahn_hilliard 5e-4"
    "sfno                cahn_hilliard 1e-3"
    "flower    galewsky 1e-4"
    "flower    galewsky 5e-4"
    "flower    galewsky 1e-3"
    "fno       galewsky 1e-4"
    "fno       galewsky 5e-4"
    "fno       galewsky 1e-3"
    "sfno      galewsky 1e-4"
    "flower              mickelin 1e-4"
    "flower              mickelin 1e-3"
    "fno                 mickelin 5e-4"
    "fno                 mickelin 1e-3"
    "local_r_transformer mickelin 1e-4"
    "local_r_transformer mickelin 5e-4"
    "sfno                mickelin 1e-4"
    "sfno                mickelin 5e-4"
    "sfno                mickelin 1e-3"
)

echo "=== Bucket B: ${#CELLS_B[@]} crashed train+test pairs ==="
for cell in "${CELLS_B[@]}"; do
    # shellcheck disable=SC2086
    submit_pair $cell
done

echo
echo "=== Bucket C: ${#CELLS_C[@]} test-only jobs ==="
for cell in "${CELLS_C[@]}"; do
    # shellcheck disable=SC2086
    submit_test $cell
done

echo
echo "submitted ${#CELLS_B[@]} train+test pairs + ${#CELLS_C[@]} test jobs"
