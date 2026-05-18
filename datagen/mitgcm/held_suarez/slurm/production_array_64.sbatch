#!/bin/bash
#SBATCH --job-name=mitgcm-prod-64
#SBATCH --partition=scicore
#SBATCH --qos=1week
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=4-00:00:00
#SBATCH --array=0-71
#SBATCH --output=datagen/mitgcm/held_suarez/slurm/logs/prod64-%A_%a.out
#SBATCH --error=datagen/mitgcm/held_suarez/slurm/logs/prod64-%A_%a.err
#
# Held-Suarez "C64" (256×128×30) production array: 72 runs of the safe
# parameter grid at Δt = 15 s with 8 ERA5-standard pressure levels and a
# 2-day output cadence. Adjust Δt down to 12 s if the preflight failed.
# Runs ~ 4 days wall each at 8 MPI ranks; uses scicore's 1-week QoS.
#
# Auxiliary disk savings on by default:
#   --top-thickness 7500    (separates 50 vs 100 hPa cleanly)
#   --cleanup-run-dir       (rm mitgcm_run/ after the Zarr is written)
#   diagnostics levels restricted to the 8 requested k-levels
#
# Run preflight_array_64.sbatch first to verify stability at this Δt.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

UV="uv run --no-sync --project datagen"

: "${DATA_ROOT:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/held-suarez-64}"

TASK_ID=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
CONFIG="datagen/mitgcm/held_suarez/configs/safe/run_${TASK_ID}.json"
OUT_DIR="${DATA_ROOT}/runs/run_${TASK_ID}"

mkdir -p "${OUT_DIR%/*}" datagen/mitgcm/held_suarez/slurm/logs
rm -rf "${OUT_DIR}" "${OUT_DIR}.FAILED"

echo "=== held-suarez-64 run_${TASK_ID}"
echo "    config:           ${CONFIG}"
echo "    out_dir:          ${OUT_DIR}"
echo "    n_mpi:            ${SLURM_NTASKS}"
echo "    grid:             256x128x30, top_thickness=7500 Pa"
echo "    delta_t:          15.0"
echo "    snapshot cadence: 2 days"
echo "    pressure_levels:  50 100 250 500 700 850 925 1000 hPa"

${UV} python -m datagen.mitgcm.held_suarez.scripts.run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --n-mpi "${SLURM_NTASKS}" \
    --nlon 256 --nlat 128 --nr 30 --top-thickness 7500 \
    --delta-t 15.0 \
    --snapshot-interval-days 2.0 \
    --pressure-levels 50 100 250 500 700 850 925 1000 \
    --cleanup-run-dir
