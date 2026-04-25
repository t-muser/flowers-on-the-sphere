#!/bin/bash
#SBATCH --job-name=galewsky-preflight
#SBATCH --partition=scicore
#SBATCH --qos=30min
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=datagen/galewsky/slurm/preflight-%A_%a.out
#SBATCH --error=datagen/galewsky/slurm/preflight-%A_%a.err
#
# Preflight: 32 corners of the 5-axis parameter box at half resolution
# (Nphi=256, Ntheta=128), 1 day of simulated time. Confirms the whole
# parameter box is numerically stable before the full 1920-run sweep.
#
# Prereqs before sbatch: `cd datagen && uv sync` has been run on a compute
# node, relevant toolchain modules are loaded (see README), and the corners
# configs have been generated:
#
#   uv run --project datagen python -m datagen.galewsky.scripts.preflight generate \
#       --out datagen/galewsky/configs/preflight/
#
# DATA_ROOT defaults to $SCRATCH/flowers-data; override in the environment.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

# Assume env was built on the login node (compute nodes can't reach PyPI).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
UV="uv run --no-sync --project datagen"

: "${DATA_ROOT:=${SCRATCH:-/tmp}/flowers-data}"

CORNER_ID=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
CONFIG="datagen/galewsky/configs/preflight/corner_${CORNER_ID}.json"
OUT_DIR="${DATA_ROOT}/preflight/raw/corner_${CORNER_ID}"

echo "=== Preflight corner ${CORNER_ID}"
echo "    config:  ${CONFIG}"
echo "    out_dir: ${OUT_DIR}"

mpirun -n "${SLURM_NTASKS}" ${UV} python -m datagen.galewsky.scripts.preflight run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --nphi 256 --ntheta 128 \
    --stop-sim-time 86400
