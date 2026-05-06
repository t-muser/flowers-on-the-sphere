#!/bin/bash
#SBATCH --job-name=mitgcm-go-stats-3d
#SBATCH --partition=scicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=datagen/mitgcm/global_ocean/slurm/logs/stats3d-%j.out
#SBATCH --error=datagen/mitgcm/global_ocean/slurm/logs/stats3d-%j.err

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

UV="uv run --no-sync --project datagen"

: "${DATA_ROOT:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/global-ocean-3D}"

mkdir -p datagen/mitgcm/global_ocean/slurm/logs

echo "=== compute 3-D stats over train split"
echo "    DATA_ROOT: ${DATA_ROOT}"

${UV} python -u datagen/scripts/compute_stats_3d.py \
    --root "${DATA_ROOT}" \
    --grid "${DATA_ROOT}/grid.zarr" \
    --workers 8
