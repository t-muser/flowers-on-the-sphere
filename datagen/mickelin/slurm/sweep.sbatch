#!/bin/bash
#SBATCH --job-name=mickelin-sweep
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=datagen/mickelin/slurm/logs/sweep-%A_%a.out
#SBATCH --error=datagen/mickelin/slurm/logs/sweep-%A_%a.err
#
# Mickelin GNS parameter sweep: 480 runs at production resolution
# (Nphi=256, Ntheta=128), 130 τ simulated, snapshot cadence τ/5.
# Each array task: Dedalus solve in job-local scratch, then resample and
# write the final Zarr directly to the persistent group dataset path.
#
# Before sbatch (on the login node):
#   uv run --project datagen python -m datagen.mickelin.scripts.generate_sweep
#
# DATASET_ROOT defaults to the group-shared PDE datasets dir and can be
# overridden in the environment.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
UV="uv run --no-sync --project datagen"

: "${DATASET_ROOT:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/mickelin-gns}"
: "${JOB_SCRATCH:=${TMPDIR:-${SCRATCH:-/tmp}}}"

RUN_ID=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
CONFIG="datagen/mickelin/configs/run_${RUN_ID}.json"
RAW_DIR="${JOB_SCRATCH}/raw/run_${RUN_ID}"
ZARR_OUT="${DATASET_ROOT}/processed/run_${RUN_ID}.zarr"
FAIL_MARKER="${DATASET_ROOT}/processed/run_${RUN_ID}.FAILED"

mkdir -p "${DATASET_ROOT}/processed"
# Idempotent: every task copies the manifest; identical content, last writer wins.
cp -f datagen/mickelin/configs/manifest.json "${DATASET_ROOT}/manifest.json"

rm -f "${FAIL_MARKER}"

echo "=== Mickelin sweep run ${RUN_ID}"
echo "    config:       ${CONFIG}"
echo "    raw (tmp):    ${RAW_DIR}"
echo "    zarr (final): ${ZARR_OUT}"

set +e
mpirun -n "${SLURM_NTASKS}" ${UV} python -m datagen.mickelin.scripts.run \
    --config "${CONFIG}" \
    --out-dir "${RAW_DIR}"
SOLVER_RC=$?
set -e

if [ "${SOLVER_RC}" -ne 0 ]; then
    if [ -f "${RAW_DIR}.FAILED" ]; then
        cp -f "${RAW_DIR}.FAILED" "${FAIL_MARKER}"
    else
        printf '{"run_id": %s, "solver_rc": %d}\n' "${SLURM_ARRAY_TASK_ID}" "${SOLVER_RC}" \
            > "${FAIL_MARKER}"
    fi
    echo "=== Solver failed (rc=${SOLVER_RC}); wrote ${FAIL_MARKER}"
    exit "${SOLVER_RC}"
fi

${UV} python -m datagen.mickelin.scripts.postprocess \
    --raw "${RAW_DIR}" \
    --out "${ZARR_OUT}" \
    --config "${CONFIG}" \
    --nlat 128 --nlon 256

rm -rf "${RAW_DIR}"
