#!/bin/bash
# Resubmit test-only jobs for finished-train cells whose test run never landed.
# 53 cells across cahn_hilliard, galewsky, mickelin, shock_caps.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

submit_test() {
    local model="$1" dataset="$2" lr="$3"
    local bs testname
    bs=$(fots_batch_size "$model" "$dataset")
    testname="fots-test-${model}-${dataset}-${lr}"
    local exports="MODEL_CONFIG=configs/models/${model}.yaml,DATA_CONFIG=configs/data/${dataset}.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${dataset},TEST_MODE=true"
    local test_job
    test_job=$(sbatch --parsable --time=02:00:00 \
        --job-name="$testname" \
        --export="$exports" \
        scripts/train_daint.sbatch)
    echo "submitted test ${model} ${dataset} lr=${lr} -> ${test_job}"
}

CELLS=(
    # cahn_hilliard (14)
    "dahlia              cahn_hilliard 1e-4"
    "dahlia              cahn_hilliard 5e-4"
    "dahlia              cahn_hilliard 1e-3"
    "dandelion           cahn_hilliard 1e-4"
    "dandelion           cahn_hilliard 1e-3"
    "flower              cahn_hilliard 1e-4"
    "flower              cahn_hilliard 1e-3"
    "fno                 cahn_hilliard 1e-4"
    "fno                 cahn_hilliard 5e-4"
    "local_r_transformer cahn_hilliard 5e-4"
    "sfno                cahn_hilliard 1e-4"
    "sfno                cahn_hilliard 5e-4"
    "sfno                cahn_hilliard 1e-3"
    "zinnia              cahn_hilliard 5e-4"

    # galewsky (9)
    "dandelion galewsky 1e-4"
    "flower    galewsky 1e-4"
    "flower    galewsky 5e-4"
    "flower    galewsky 1e-3"
    "fno       galewsky 1e-4"
    "fno       galewsky 5e-4"
    "fno       galewsky 1e-3"
    "sfno      galewsky 1e-4"
    "zinnia    galewsky 1e-3"

    # mickelin (13)
    "dahlia              mickelin 1e-4"
    "dahlia              mickelin 1e-3"
    "flower              mickelin 1e-4"
    "flower              mickelin 1e-3"
    "fno                 mickelin 5e-4"
    "fno                 mickelin 1e-3"
    "local_r_transformer mickelin 1e-4"
    "local_r_transformer mickelin 5e-4"
    "sfno                mickelin 1e-4"
    "sfno                mickelin 5e-4"
    "sfno                mickelin 1e-3"
    "zinnia              mickelin 5e-4"
    "zinnia              mickelin 1e-3"

    # shock_caps (17)
    "dahlia              shock_caps 1e-4"
    "dahlia              shock_caps 5e-4"
    "dahlia              shock_caps 1e-3"
    "dandelion           shock_caps 1e-4"
    "dandelion           shock_caps 5e-4"
    "flower              shock_caps 1e-4"
    "flower              shock_caps 1e-3"
    "fno                 shock_caps 1e-4"
    "fno                 shock_caps 5e-4"
    "fno                 shock_caps 1e-3"
    "local_r_transformer shock_caps 1e-4"
    "local_r_transformer shock_caps 5e-4"
    "local_r_transformer shock_caps 1e-3"
    "sfno                shock_caps 1e-4"
    "sfno                shock_caps 5e-4"
    "sfno                shock_caps 1e-3"
    "zinnia              shock_caps 1e-4"
)

for cell in "${CELLS[@]}"; do
    # shellcheck disable=SC2086
    submit_test $cell
done

echo
echo "submitted ${#CELLS[@]} test jobs"
