#!/bin/bash
#SBATCH --job-name=mickelin-preflight
#SBATCH --partition=scicore
#SBATCH --qos=30min
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=datagen/mickelin/slurm/preflight-%A_%a.out
#SBATCH --error=datagen/mickelin/slurm/preflight-%A_%a.err
#
# Mickelin GNS preflight: 8 extreme points of the
# (r_over_lambda × kappa_lambda) box at half resolution (Nphi=128,
# Ntheta=64) and 20 τ of simulated time, seed 0. Confirms the two-axis
# active-band parameter plane is numerically stable before the full
# 480-run sweep.
#
# Prereqs before sbatch:
#   uv run --project datagen python -m datagen.mickelin.scripts.generate_sweep
#   # (also emits datagen/mickelin/configs/preflight/corner_0N.json)
#
# DATA_ROOT defaults to $SCRATCH/flowers-data; override in the environment.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
UV="uv run --no-sync --project datagen"

: "${DATA_ROOT:=${SCRATCH:-/tmp}/flowers-data}"

CORNER_ID=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
CONFIG="datagen/mickelin/configs/preflight/corner_${CORNER_ID}.json"
OUT_DIR="${DATA_ROOT}/mickelin_preflight/raw/corner_${CORNER_ID}"

echo "=== Mickelin preflight corner ${CORNER_ID}"
echo "    config:  ${CONFIG}"
echo "    out_dir: ${OUT_DIR}"

mpirun -n "${SLURM_NTASKS}" ${UV} python -m datagen.mickelin.scripts.run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --nphi 128 --ntheta 64 \
    --stop-sim-time 20.0 \
    --snapshot-dt 1.0
