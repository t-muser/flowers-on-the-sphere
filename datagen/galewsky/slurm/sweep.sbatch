#!/bin/bash
#SBATCH --job-name=galewsky-sweep
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --array=0-2559
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=datagen/galewsky/slurm/logs/sweep-%A_%a.out
#SBATCH --error=datagen/galewsky/slurm/logs/sweep-%A_%a.err
#
# Full parameter sweep: 2560 runs at production resolution
# (Nphi=512, Ntheta=256), 6 days simulated, hourly output. The grid is
# u_max × lat_center × h_hat × H × seed = 5·4·4·4·8.
# Each array task: Dedalus solve in job-local scratch, then resample and
# write the final Zarr directly to the persistent group dataset path.
#
# Before sbatch (on the login node):
#   uv run --project datagen python -m datagen.galewsky.scripts.generate_sweep
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

# Persistent dataset directory. Raw HDF5 snapshots stay in job-local scratch
# and are discarded when the job exits; only the resampled Zarr is persisted.
: "${DATASET_ROOT:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/galewsky-sw}"
: "${JOB_SCRATCH:=${TMPDIR:-${SCRATCH:-/tmp}}}"

RUN_ID=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
CONFIG="datagen/galewsky/configs/run_${RUN_ID}.json"
RAW_DIR="${JOB_SCRATCH}/raw/run_${RUN_ID}"
ZARR_OUT="${DATASET_ROOT}/processed/run_${RUN_ID}.zarr"
FAIL_MARKER="${DATASET_ROOT}/processed/run_${RUN_ID}.FAILED"

mkdir -p "${DATASET_ROOT}/processed"
# Idempotent: every task copies the manifest + splits file; identical content,
# last writer wins.
cp -f datagen/galewsky/configs/manifest.json "${DATASET_ROOT}/manifest.json"
if [ -f datagen/galewsky/configs/splits.json ]; then
    cp -f datagen/galewsky/configs/splits.json "${DATASET_ROOT}/splits.json"
fi

# Start clean: drop any stale FAILED marker from a prior attempt.
rm -f "${FAIL_MARKER}"

echo "=== Sweep run ${RUN_ID}"
echo "    config:       ${CONFIG}"
echo "    raw (tmp):    ${RAW_DIR}"
echo "    zarr (final): ${ZARR_OUT}"

# Dedalus solve (MPI) into job-local scratch.
set +e
mpirun -n "${SLURM_NTASKS}" ${UV} python -m datagen.galewsky.scripts.run \
    --config "${CONFIG}" \
    --out-dir "${RAW_DIR}"
SOLVER_RC=$?
set -e

if [ "${SOLVER_RC}" -ne 0 ]; then
    # Preserve the FAILED sentinel (written by galewsky/scripts/run.py) on the
    # persistent path so consolidate.py can report it.
    if [ -f "${RAW_DIR}.FAILED" ]; then
        cp -f "${RAW_DIR}.FAILED" "${FAIL_MARKER}"
    else
        printf '{"run_id": %s, "solver_rc": %d}\n' "${SLURM_ARRAY_TASK_ID}" "${SOLVER_RC}" \
            > "${FAIL_MARKER}"
    fi
    echo "=== Solver failed (rc=${SOLVER_RC}); wrote ${FAIL_MARKER}"
    exit "${SOLVER_RC}"
fi

# Resample to regular (lat, lon) Zarr directly on the persistent path.
${UV} python -m datagen.galewsky.scripts.postprocess \
    --raw "${RAW_DIR}" \
    --out "${ZARR_OUT}" \
    --config "${CONFIG}" \
    --nlat 256 --nlon 512

# Free local scratch promptly (SLURM usually cleans it up anyway).
rm -rf "${RAW_DIR}"
