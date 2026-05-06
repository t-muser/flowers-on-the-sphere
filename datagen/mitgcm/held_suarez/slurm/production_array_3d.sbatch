#!/bin/bash
#SBATCH --job-name=mitgcm-prod-3d
#SBATCH --partition=scicore
#SBATCH --qos=1day
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --array=0-71
#SBATCH --output=datagen/mitgcm/held_suarez/slurm/logs/prod3d-%A_%a.out
#SBATCH --error=datagen/mitgcm/held_suarez/slurm/logs/prod3d-%A_%a.err
#
# Held-Suarez-3D production array: 72 runs of the safe parameter grid
# (3×2×2×2×3) at Δt = 30 s with 8 ERA5-standard pressure levels.
# Run preflight_array_3d.sbatch first to verify stability at this Δt.

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

TASK_ID=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
CONFIG="datagen/mitgcm/held_suarez/configs/safe/run_${TASK_ID}.json"
OUT_DIR="${DATA_ROOT}/runs/run_${TASK_ID}"

mkdir -p "${OUT_DIR%/*}" datagen/mitgcm/held_suarez/slurm/logs
rm -rf "${OUT_DIR}" "${OUT_DIR}.FAILED"

echo "=== held-suarez-3d run_${TASK_ID}"
echo "    config:           ${CONFIG}"
echo "    out_dir:          ${OUT_DIR}"
echo "    n_mpi:            ${SLURM_NTASKS}"
echo "    delta_t:          30.0"
echo "    pressure_levels:  50 100 250 500 700 850 925 1000 hPa"

${UV} python -m datagen.mitgcm.held_suarez.scripts.run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --n-mpi "${SLURM_NTASKS}" \
    --delta-t 30.0 \
    --pressure-levels 50 100 250 500 700 850 925 1000
