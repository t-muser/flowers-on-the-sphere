#!/bin/bash
#SBATCH --job-name=clima-hs-diag
#SBATCH --partition=bigmem
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --array=0-161
#SBATCH --output=datagen/clima_atmos/held_suarez/slurm/logs/diag-%A_%a.out
#SBATCH --error=datagen/clima_atmos/held_suarez/slurm/logs/diag-%A_%a.err
#
# Stage-1 physics-validation reduction: one task per corner streams its
# run.zarr, drops the 200-day spin-up, and writes
#   ${DATA_ROOT}/diagnostics/run_XXXX.npz
# CPU only (no GPU); the store is read in BLOCK-sized time slices so memory
# stays bounded. After the array lands, build the report on the login node:
#   uv run --no-sync --project datagen python \
#       -m datagen.clima_atmos.held_suarez.diagnostics report
#
# Submit:  sbatch datagen/clima_atmos/held_suarez/slurm/diagnostics.sbatch

set -euo pipefail

mkdir -p datagen/clima_atmos/held_suarez/slurm/logs

UV="uv run --no-sync --project datagen"

: "${DATA_ROOT:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/held-suarez-clima}"
export DATA_ROOT

TASK_ID=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")

echo "=== hs diagnostics reduce run_${TASK_ID}  (DATA_ROOT=${DATA_ROOT})"

${UV} python -u -m datagen.clima_atmos.held_suarez.diagnostics reduce \
    --run-id "${TASK_ID}"

echo "=== done run_${TASK_ID}"
