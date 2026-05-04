#!/bin/bash
#SBATCH --job-name=mitgcm-preflight
#SBATCH --partition=scicore
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-31

set -euo pipefail

# Environment Setup
module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

# Thread Limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Output Destination
export DATA_ROOT=/scicore/home/rothvo/abati0000/fots-data/mitgcm/held-suarez

# Map Slurm Array ID to zero-padded config files (e.g., 00, 01 ... 31)
TASK_ID=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
CONFIG="datagen/mitgcm/held_suarez/configs/preflight/corner_${TASK_ID}.json"
OUT_DIR="${DATA_ROOT}/preflight/corner_${TASK_ID}"

# Clean up any previous failed attempts
mkdir -p "${OUT_DIR%/*}"
rm -rf "${OUT_DIR}" "${OUT_DIR}.FAILED"

# Execute the preflight run
uv run --no-sync --project datagen python -m datagen.mitgcm.held_suarez.scripts.preflight run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --n-mpi "${SLURM_NTASKS}" \
    --executable "$(pwd)/datagen/mitgcm/held_suarez/build/mitgcmuv"
