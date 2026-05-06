#!/bin/bash
#SBATCH --job-name=mitgcm-go-reorg-3d
#SBATCH --partition=scicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=datagen/mitgcm/global_ocean/slurm/logs/reorg3d-%j.out
#SBATCH --error=datagen/mitgcm/global_ocean/slurm/logs/reorg3d-%j.err

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

UV="uv run --no-sync --project datagen"

: "${DATA_ROOT:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/global-ocean-3D}"

mkdir -p datagen/mitgcm/global_ocean/slurm/logs

echo "=== reorganize global-ocean 3D into train/val/test"
echo "    DATA_ROOT: ${DATA_ROOT}"

${UV} python datagen/scripts/reorganize_splits.py \
    --root "${DATA_ROOT}" \
    --src-dir sweep \
    --nested \
    --trim-first 12 \
    --workers 16
