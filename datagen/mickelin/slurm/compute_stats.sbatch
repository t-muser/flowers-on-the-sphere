#!/bin/bash
#SBATCH --job-name=mickelin-stats
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=datagen/mickelin/slurm/logs/stats-%j.out
#SBATCH --error=datagen/mickelin/slurm/logs/stats-%j.err
#
# Compute per-field z-score normalization stats over the train split.
# Reads <root>/train/run_*.zarr and writes <root>/stats.json.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

UV="uv run --no-sync --project datagen"

: "${DATASET_ROOT:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/mickelin-gns}"

${UV} python datagen/scripts/compute_stats.py \
    --root "${DATASET_ROOT}" \
    --fields vorticity
