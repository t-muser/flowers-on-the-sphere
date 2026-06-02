#!/bin/bash
#SBATCH --job-name=zarr2hdf5
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=scripts/logs/zarr2hdf5-%j.out
#SBATCH --error=scripts/logs/zarr2hdf5-%j.err
#
# Full zarr -> Well-style HDF5 conversion for one spherical dataset.
# CPU/IO + gzip only (no GPU). Peak mem ~= workers x run_size: the 2-D
# datasets are <2 GB, so the scicore partition / 16 GB default is plenty.
#
#   sbatch --export=ALL,DATASET=mickelin-gns scripts/zarr_to_hdf5.sbatch
#   sbatch --export=ALL,DATASET=galewsky-sw  scripts/zarr_to_hdf5.sbatch
#
# held-suarez loads ~5 GB/run eagerly per worker (~40 GB at 8 workers), so
# override onto bigmem for it:
#   sbatch --partition=bigmem --mem=192G \
#          --export=ALL,DATASET=held-suarez-clima scripts/zarr_to_hdf5.sbatch
#
# Output lands at ${ARCH}/${DATASET}-hdf5/data/{train,valid,test}/*.h5

set -euo pipefail

: "${DATASET:?set DATASET, e.g. --export=ALL,DATASET=held-suarez-clima}"
: "${ARCH:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs}"
mkdir -p scripts/logs

echo "=== converting ${DATASET} (full) -> ${ARCH}/${DATASET}-hdf5"
uv run --no-sync python -u scripts/zarr_to_hdf5.py \
    --dataset   "${DATASET}" \
    --src-root  "${ARCH}" \
    --dst-root  "${ARCH}" \
    --workers   "${SLURM_CPUS_PER_TASK}"

echo "=== done: ${ARCH}/${DATASET}-hdf5"
