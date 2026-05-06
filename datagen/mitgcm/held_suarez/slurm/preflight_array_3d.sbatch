#!/bin/bash
#SBATCH --job-name=mitgcm-preflight-3d
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-31
#SBATCH --output=datagen/mitgcm/held_suarez/slurm/logs/preflight3d-%A_%a.out
#SBATCH --error=datagen/mitgcm/held_suarez/slurm/logs/preflight3d-%A_%a.err
#
# 32-corner stability sweep for the Held-Suarez-3D dataset.
# Runs at Δt = 30 s with 8 ERA5-standard pressure levels, 30 d spin-up +
# 30 d data collection. Validates that the reduced timestep clears the
# CFL failures observed at the original Δt = 45 s.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

UV="uv run --no-sync --project datagen"

: "${DATA_ROOT:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/held-suarez-3d}"

TASK_ID=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
CONFIG="datagen/mitgcm/held_suarez/configs/preflight/corner_${TASK_ID}.json"
OUT_DIR="${DATA_ROOT}/preflight/corner_${TASK_ID}"

mkdir -p "${OUT_DIR%/*}" datagen/mitgcm/held_suarez/slurm/logs
rm -rf "${OUT_DIR}" "${OUT_DIR}.FAILED"

echo "=== held-suarez-3d preflight corner_${TASK_ID}"
echo "    config:           ${CONFIG}"
echo "    out_dir:          ${OUT_DIR}"
echo "    n_mpi:            ${SLURM_NTASKS}"
echo "    delta_t:          30.0"
echo "    pressure_levels:  50 100 250 500 700 850 925 1000 hPa"

${UV} python -m datagen.mitgcm.held_suarez.scripts.preflight run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --n-mpi "${SLURM_NTASKS}" \
    --delta-t 30.0 \
    --pressure-levels 50 100 250 500 700 850 925 1000
