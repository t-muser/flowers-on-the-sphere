#!/bin/bash
# Daint sweep of the ViT (ClimaX) baseline on the two datasets the user
# wants benchmarked:
#   vit x {mickelin, shock_caps} x {1e-4, 5e-4, 1e-3} = 6 train+test pairs.
# Each cell: 24h DDP train on 4 GH200 + afterok 2h test.
#
# Data: the in-house HDF5 spec (configs/data/<dataset>_hdf5.yaml,
# NotWellDataModule + SphericalHDF5Dataset). Runs in the modulus container;
# train_daint.sbatch installs the_well/h5py/timm into ~/.local on first use.
#
# Batch sizes come from scripts/batch_sizes.sh (vit:128x256, vit:256x512);
# keyed by dataset name -> resolution, unchanged by the zarr->HDF5 move.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/batch_sizes.sh

# Force the capstor container toml — the iopsstor sqsh
# (/iopsstor/scratch/cscs/$USER/container_images/aurora_arm64_modulus2412.sqsh)
# is currently absent on this system, so train_daint.sbatch's default
# TOMLPATH fails with pyxis "Invalid argument".
TOMLPATH_OVERRIDE="${PWD}/scripts/torchcontainer_daint.toml"

MODEL=vit
# Datasets default to mickelin + shock_caps; override by passing names as args,
# e.g. `submit_vit_daint.sh cahn_hilliard`. Each must have a
# configs/data/<dataset>_hdf5.yaml and a batch_sizes.sh resolution entry.
if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=(mickelin shock_caps)
fi
LRS=(1e-4 5e-4 1e-3)

for dataset in "${DATASETS[@]}"; do
    bs=$(fots_batch_size "$MODEL" "$dataset")
    for lr in "${LRS[@]}"; do
        exports="MODEL_CONFIG=configs/models/${MODEL}.yaml,DATA_CONFIG=configs/data/${dataset}_hdf5.yaml,TRAIN_CONFIG=configs/train_4-to-1.yaml,LR=${lr},BATCH_SIZE=${bs},NAME=${dataset},TOMLPATH=${TOMLPATH_OVERRIDE}"

        TRAIN_JOB=$(sbatch --parsable --time=24:00:00 \
            --job-name="fots-${MODEL}-${dataset}-${lr}" \
            --export="${exports}" \
            scripts/train_daint.sbatch)
        echo "submitted train ${MODEL} ${dataset} lr=${lr} bs=${bs} -> ${TRAIN_JOB}"

        TEST_JOB=$(sbatch --parsable --time=02:00:00 \
            --job-name="fots-test-${MODEL}-${dataset}-${lr}" \
            --dependency=afterok:${TRAIN_JOB} \
            --export="${exports},TEST_MODE=true" \
            scripts/train_daint.sbatch)
        echo "submitted test  ${MODEL} ${dataset} lr=${lr}              -> ${TEST_JOB} (afterok ${TRAIN_JOB})"
    done
done
