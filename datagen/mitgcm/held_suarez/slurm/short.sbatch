#!/bin/bash
#SBATCH --job-name=mitgcm-short
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=datagen/mitgcm/held_suarez/slurm/logs/short-%j.out
#SBATCH --error=datagen/mitgcm/held_suarez/slurm/logs/short-%j.err
#
# One short MITgcm Held-Suarez run: 5-day spin-up + 10 days of daily snapshots
# at 128x64x20. Uses the compiled mitgcmuv at
# datagen/mitgcm/held_suarez/build/mitgcmuv; build it once with
# datagen.mitgcm.held_suarez.scripts.build before submitting.
#
# DATA_ROOT defaults to $SCRATCH/flowers-data; override in the environment.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

# Compute nodes can't reach PyPI; rely on uv sync from the login node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
UV="uv run --no-sync --project datagen"

: "${DATA_ROOT:=${SCRATCH:-/tmp}/flowers-data}"

CONFIG="datagen/mitgcm/held_suarez/configs/short/short_run.json"
OUT_DIR="${DATA_ROOT}/mitgcm/short_run"

mkdir -p "${OUT_DIR%/*}"
rm -rf "${OUT_DIR}" "${OUT_DIR}.FAILED"

echo "=== MITgcm short run"
echo "    config:  ${CONFIG}"
echo "    out_dir: ${OUT_DIR}"
echo "    n_mpi:   ${SLURM_NTASKS}"

${UV} python -m datagen.mitgcm.held_suarez.scripts.run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --n-mpi "${SLURM_NTASKS}" \
    --spinup-days 5 \
    --data-days 10 \
    --snapshot-interval-days 0.5 \
    --delta-t 600
