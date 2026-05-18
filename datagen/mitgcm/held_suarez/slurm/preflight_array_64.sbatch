#!/bin/bash
#SBATCH --job-name=mitgcm-preflight-64
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --array=0-31
#SBATCH --output=datagen/mitgcm/held_suarez/slurm/logs/preflight64-%A_%a.out
#SBATCH --error=datagen/mitgcm/held_suarez/slurm/logs/preflight64-%A_%a.err
#
# 32-corner stability sweep for the high-resolution (256×128×30) Held-Suarez
# build at Δt = 15 s. Runs 30 d spin-up + 30 d data collection per corner,
# with the diagnostics restricted to the 8 ERA5-standard pressure levels
# and the mitgcm_run working dir kept around for forensics.
# Top model layer is 75 hPa (7500 Pa) so that 50 hPa and 100 hPa resolve to
# distinct k-levels.

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

TASK_ID=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
CONFIG="datagen/mitgcm/held_suarez/configs/preflight/corner_${TASK_ID}.json"
OUT_DIR="${DATA_ROOT}/preflight/corner_${TASK_ID}"

mkdir -p "${OUT_DIR%/*}" datagen/mitgcm/held_suarez/slurm/logs
rm -rf "${OUT_DIR}" "${OUT_DIR}.FAILED"

echo "=== held-suarez-64 preflight corner_${TASK_ID}"
echo "    config:           ${CONFIG}"
echo "    out_dir:          ${OUT_DIR}"
echo "    n_mpi:            ${SLURM_NTASKS}"
echo "    grid:             256x128x30, top_thickness=7500 Pa"
echo "    delta_t:          15.0"
echo "    pressure_levels:  50 100 250 500 700 850 925 1000 hPa"

${UV} python -m datagen.mitgcm.held_suarez.scripts.preflight run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --n-mpi "${SLURM_NTASKS}" \
    --nlon 256 --nlat 128 --nr 30 --top-thickness 7500 \
    --delta-t 15.0 \
    --pressure-levels 50 100 250 500 700 850 925 1000
