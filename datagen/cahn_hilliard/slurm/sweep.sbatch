#!/bin/bash
#SBATCH --job-name=cahn-hilliard-sweep
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --array=0-143
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=datagen/cahn_hilliard/slurm/logs/sweep-%A_%a.out
#SBATCH --error=datagen/cahn_hilliard/slurm/logs/sweep-%A_%a.err
#
# Cahn-Hilliard parameter sweep: 144 runs at production resolution
# (Nphi=256, Ntheta=128). Dedalus solver with Krekhov-style rotating Y_ell^m
# forcing. Each array task: Dedalus solve in job-local scratch, then resample
# and write the final Zarr directly to the persistent group dataset path.
#
# Before sbatch (on the login node):
#   uv run --project datagen python -m datagen.cahn_hilliard.scripts.generate_sweep
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

: "${DATASET_ROOT:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/cahn-hilliard-sphere}"
: "${JOB_SCRATCH:=${TMPDIR:-${SCRATCH:-/tmp}}}"

RUN_ID=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
CONFIG="datagen/cahn_hilliard/configs/run_${RUN_ID}.json"
RAW_DIR="${JOB_SCRATCH}/raw/run_${RUN_ID}"
ZARR_OUT="${DATASET_ROOT}/processed/run_${RUN_ID}.zarr"
FAIL_MARKER="${DATASET_ROOT}/processed/run_${RUN_ID}.FAILED"

mkdir -p "${DATASET_ROOT}/processed"
cp -f datagen/cahn_hilliard/configs/manifest.json "${DATASET_ROOT}/manifest.json"
if [ -f datagen/cahn_hilliard/configs/splits.json ]; then
    cp -f datagen/cahn_hilliard/configs/splits.json "${DATASET_ROOT}/splits.json"
fi

rm -f "${FAIL_MARKER}"

echo "=== CH sweep run ${RUN_ID}"
echo "    config:       ${CONFIG}"
echo "    raw (tmp):    ${RAW_DIR}"
echo "    zarr (final): ${ZARR_OUT}"

set +e
mpirun -n "${SLURM_NTASKS}" ${UV} python -m datagen.cahn_hilliard.scripts.run \
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

${UV} python -m datagen.cahn_hilliard.scripts.postprocess \
    --raw "${RAW_DIR}" \
    --out "${ZARR_OUT}" \
    --config "${CONFIG}" \
    --nlat 128 --nlon 256

rm -rf "${RAW_DIR}"
