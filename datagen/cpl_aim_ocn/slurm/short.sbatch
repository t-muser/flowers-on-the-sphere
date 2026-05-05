#!/bin/bash
#SBATCH --job-name=cpl_aim_ocn-short
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=datagen/cpl_aim_ocn/slurm/logs/short-%j.out
#SBATCH --error=datagen/cpl_aim_ocn/slurm/logs/short-%j.err
#
# One short coupled AIM+ocean run: 1-day spin-up + 5 days of daily snapshots
# on cs32. Useful as a validation step before launching the full 180-run
# sweep. Three MPI tasks → one each for the cpl, ocn, and atm binaries
# (must be exactly 3, matching the cs32 SIZE.h with nPx=nPy=1 each).
#
# Build pre-requisites (run once on a login node before submitting):
#
#   uv run --project datagen python -m datagen.cpl_aim_ocn.scripts.build \
#       --mitgcm-root $MITGCM_ROOT
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

CONFIG="datagen/cpl_aim_ocn/configs/run_0000.json"
OUT_DIR="${DATA_ROOT}/cpl_aim_ocn/short_run"

mkdir -p "${OUT_DIR%/*}"
rm -rf "${OUT_DIR}" "${OUT_DIR}.FAILED"

echo "=== cpl_aim+ocn short run"
echo "    config:  ${CONFIG}"
echo "    out_dir: ${OUT_DIR}"
echo "    n_tasks: ${SLURM_NTASKS}"

${UV} python -m datagen.cpl_aim_ocn.scripts.run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --spinup-days 1 \
    --data-days 5 \
    --snapshot-interval-days 0.5
