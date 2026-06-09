#!/bin/bash
#SBATCH --job-name=hdf5stats
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=scripts/logs/hdf5stats-%j.out
#SBATCH --error=scripts/logs/hdf5stats-%j.err
#
# Parallel-Welford normalization stats over a dataset's train split, written to
# <dataset>-hdf5/stats.{yaml,json}. CPU/IO only.
#
#   sbatch --export=ALL,DATASET=galewsky-sw scripts/compute_hdf5_stats.sbatch
#
# held-suarez loads multi-GB fields per worker -> override onto bigmem:
#   sbatch --partition=bigmem --mem=128G \
#          --export=ALL,DATASET=held-suarez-clima scripts/compute_hdf5_stats.sbatch

set -euo pipefail

: "${DATASET:?set DATASET, e.g. --export=ALL,DATASET=galewsky-sw}"
: "${ARCH:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs}"
mkdir -p scripts/logs

echo "=== stats for ${DATASET}-hdf5"
uv run --no-sync python -u scripts/compute_hdf5_stats.py \
    --root    "${ARCH}/${DATASET}-hdf5" \
    --workers "${SLURM_CPUS_PER_TASK}"
echo "=== done"
