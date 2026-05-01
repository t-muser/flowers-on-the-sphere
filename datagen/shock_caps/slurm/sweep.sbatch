#!/bin/bash
#SBATCH --job-name=shock-caps-sweep
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --array=0-499
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=datagen/shock_caps/slurm/logs/sweep-%A_%a.out
#SBATCH --error=datagen/shock_caps/slurm/logs/sweep-%A_%a.err
#
# Full parameter sweep: 500 runs on the (K, delta, seed) grid
# (5 cap counts × 5 velocity scalings × 20 seeds), at production
# resolution (Nx=512, Ny=256 mapped grid; Nlat=256, Nlon=512 output
# grid), t_max = 1.5 non-dim time units, snapshot every 0.015. PyClaw
# is OMP-parallel — one task, 16 threads, no MPI. Walltime budget is
# generous to cover the K=16 stride, where extra shock fronts tighten
# the CFL early in each run.
#
# Before sbatch (on the login node):
#   uv run --project datagen python -m datagen.shock_caps.scripts.generate_sweep
#
# DATASET_ROOT defaults to the group-shared PDE datasets dir and can be
# overridden in the environment.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
UV="uv run --no-sync --project datagen"

: "${DATASET_ROOT:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/shock-caps}"

RUN_ID=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
CONFIG="datagen/shock_caps/configs/run_${RUN_ID}.json"
ZARR_OUT="${DATASET_ROOT}/processed/run_${RUN_ID}.zarr"
FAIL_MARKER="${DATASET_ROOT}/processed/run_${RUN_ID}.FAILED"

mkdir -p "${DATASET_ROOT}/processed"
# Idempotent: every task copies the manifest; identical content, last writer wins.
cp -f datagen/shock_caps/configs/manifest.json "${DATASET_ROOT}/manifest.json"

# Start clean: drop any stale FAILED marker from a prior attempt.
rm -f "${FAIL_MARKER}"

echo "=== Sweep run ${RUN_ID}"
echo "    config:       ${CONFIG}"
echo "    zarr (final): ${ZARR_OUT}"

set +e
${UV} python -m datagen.shock_caps.scripts.run \
    --config "${CONFIG}" \
    --out    "${ZARR_OUT}" \
    --nlat 256 --nlon 512 --nx 512 --ny 256 \
    --snapshot-dt 0.015 --stop-sim-time 1.5
SOLVER_RC=$?
set -e

if [ "${SOLVER_RC}" -ne 0 ]; then
    if [ ! -f "${FAIL_MARKER}" ]; then
        cp -f "${ZARR_OUT}.FAILED" "${FAIL_MARKER}"
    else
        printf '{"run_id": %s, "solver_rc": %d}\n' "${SLURM_ARRAY_TASK_ID}" "${SOLVER_RC}" \
            > "${FAIL_MARKER}"
    fi
    echo "=== Solver failed (rc=${SOLVER_RC}); wrote ${FAIL_MARKER}"
    exit "${SOLVER_RC}"
fi
