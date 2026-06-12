#!/bin/bash
# Daint sweep on the Held-Suarez (ClimaAtmos) HDF5 dataset (144x288 lat/lon,
# level axis folded into channels -> 25 channels: T x8 levels, ps, velocity
# u/v x8 levels). See configs/data/held_suarez_hdf5.yaml.
#   models: {flower, sfno}
#   lrs:    {1e-4, 5e-4, 1e-3}
#   = 6 train (24h) + 6 test (2h, afterok) job pairs.
#
# Differs from submit_held_suarez_daint.sh (the MITgcm 64x128 zarr sweep):
#   * --account=uba04 on the CLI overrides the uba03 default baked into
#     train_daint.sbatch (uba03 is over compute quota). Storage paths are
#     unchanged (tmuser scratch is fine).
#   * TOMLPATH -> torchcontainer_daint.toml (the /capstor container image),
#     because the capstor-independent /iopsstor copy has been deleted; the
#     default _iopsstor.toml would fail at container start.
#
# Optional: pass SMOKE_JOB=<jobid> to chain all train jobs as afterok so they
# only start after an in-container smoke succeeds.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

MODELS=(flower sfno)
DATASET=held_suarez_hdf5
LRS=(1e-4 5e-4 1e-3)

ACCOUNT=uba04
TOMLPATH="${TOMLPATH:-$(pwd)/scripts/torchcontainer_daint.toml}"

SMOKE_JOB="${SMOKE_JOB:-}"
TRAIN_DEP=""
if [ -n "$SMOKE_JOB" ]; then
    TRAIN_DEP="--dependency=afterok:${SMOKE_JOB}"
    echo "chaining all train jobs to depend on smoke job ${SMOKE_JOB}"
fi

NAME="$DATASET"
COMMON_EXPORT="DATA_CONFIG=configs/data/${DATASET}.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,NAME=${NAME},TOMLPATH=${TOMLPATH}"

for model in "${MODELS[@]}"; do
    for lr in "${LRS[@]}"; do
        bs=$(fots_batch_size "$model" "$DATASET")

        TRAIN_JOB=$(sbatch --parsable --account="$ACCOUNT" --time=24:00:00 \
            --job-name="fots-${model}-${DATASET}-${lr}" \
            ${TRAIN_DEP} \
            --export="MODEL_CONFIG=configs/models/${model}.yaml,${COMMON_EXPORT},LR=${lr},BATCH_SIZE=${bs}" \
            scripts/train_daint.sbatch)
        echo "submitted train ${model} ${DATASET} lr=${lr} bs=${bs} -> ${TRAIN_JOB}"

        TEST_JOB=$(sbatch --parsable --account="$ACCOUNT" --time=02:00:00 \
            --job-name="fots-test-${model}-${DATASET}-${lr}" \
            --dependency=afterok:${TRAIN_JOB} \
            --export="MODEL_CONFIG=configs/models/${model}.yaml,${COMMON_EXPORT},LR=${lr},BATCH_SIZE=${bs},TEST_MODE=true" \
            scripts/train_daint.sbatch)
        echo "submitted test  ${model} ${DATASET} lr=${lr} -> ${TEST_JOB} (afterok ${TRAIN_JOB})"
    done
done
