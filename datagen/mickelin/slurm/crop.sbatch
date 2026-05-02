#!/bin/bash
#SBATCH --job-name=mickelin-crop
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=datagen/mickelin/slurm/logs/crop-%j.out
#SBATCH --error=datagen/mickelin/slurm/logs/crop-%j.err
#
# Crop Mickelin per-run Zarrs to t ∈ [0, 40 τ] (≈ 0–10 s of animation
# playback at 20 fps cadence). Reads from processed/ and writes to
# processed_cropped/, preserving originals.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

UV="uv run --no-sync --project datagen"

: "${DATASET_ROOT:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/mickelin-gns}"

${UV} python -m datagen.mickelin.scripts.crop_zarrs \
    --src "${DATASET_ROOT}/processed" \
    --dst "${DATASET_ROOT}/processed_cropped" \
    --splits datagen/mickelin/configs/splits.json \
    --t-start-tau 0 --t-stop-tau 40
