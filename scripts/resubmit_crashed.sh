#!/bin/bash
# Resubmit the train+test cells that crashed/failed in the previous sweep.
# Excludes "still running" cells. PlanetSWE is excluded (different cluster).
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

submit_cell() {
    local model="$1" dataset="$2" lr="$3"
    local bs jobname testname
    bs=$(fots_batch_size "$model" "$dataset")
    jobname="fots-${model}-${dataset}-${lr}"
    testname="fots-test-${model}-${dataset}-${lr}"

    local exports="MODEL_CONFIG=configs/models/${model}.yaml,DATA_CONFIG=configs/data/${dataset}.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${dataset}"

    local train_job
    train_job=$(sbatch --parsable --time=24:00:00 \
        --job-name="$jobname" \
        --export="$exports" \
        scripts/train_daint.sbatch)
    echo "submitted train ${model} ${dataset} lr=${lr} bs=${bs} -> ${train_job}"

    local test_job
    test_job=$(sbatch --parsable --time=02:00:00 \
        --job-name="$testname" \
        --dependency=afterok:${train_job} \
        --export="${exports},TEST_MODE=true" \
        scripts/train_daint.sbatch)
    echo "submitted test  ${model} ${dataset} lr=${lr}              -> ${test_job} (afterok ${train_job})"
}

# (model, dataset, lr) triples to resubmit.
CELLS=(
    # cahn_hilliard
    "dandelion cahn_hilliard 5e-4"
    "flower    cahn_hilliard 5e-4"
    "fno       cahn_hilliard 1e-3"
    "zinnia    cahn_hilliard 1e-4"
    "zinnia    cahn_hilliard 1e-3"

    # galewsky
    "dandelion            galewsky 1e-3"
    "local_r_transformer  galewsky 1e-3"
    "local_s2_transformer galewsky 1e-4"
    "local_s2_transformer galewsky 5e-4"
    "local_s2_transformer galewsky 1e-3"
    "sfno                 galewsky 5e-4"
    "sfno                 galewsky 1e-3"
    "zinnia               galewsky 1e-4"
    "zinnia               galewsky 5e-4"

    # mickelin
    "dahlia              mickelin 5e-4"
    "dandelion           mickelin 1e-4"
    "dandelion           mickelin 5e-4"
    "dandelion           mickelin 1e-3"
    "flower              mickelin 5e-4"
    "fno                 mickelin 1e-4"
    "local_r_transformer mickelin 1e-3"
    "zinnia              mickelin 1e-4"

    # shock_caps
    "dandelion            shock_caps 1e-3"
    "flower               shock_caps 5e-4"
    "local_s2_transformer shock_caps 1e-4"
    "local_s2_transformer shock_caps 5e-4"
    "local_s2_transformer shock_caps 1e-3"
    "zinnia               shock_caps 5e-4"
    "zinnia               shock_caps 1e-3"
)

for cell in "${CELLS[@]}"; do
    # shellcheck disable=SC2086
    submit_cell $cell
done

echo
echo "submitted ${#CELLS[@]} train+test pairs"
