#!/bin/bash
#SBATCH --job-name=galewsky-sweep
#SBATCH --account=uba03
#SBATCH --partition=normal
#SBATCH --array=0-959%32
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=datagen/galewsky/slurm/logs/sweep-%A_%a.out
#SBATCH --error=datagen/galewsky/slurm/logs/sweep-%A_%a.err
#
# Daint port of sweep.sbatch. Full 960-run parameter sweep at production
# resolution (Nphi=512, Ntheta=256), 32 days simulated, 4-hourly output
# (168 frames after a 4-day spinup is dropped in postprocess).
#
# Each array task runs the Dedalus solve in $TMPDIR (compute-node tmpfs),
# then resamples and writes the final Zarr directly to the persistent
# group dataset path.
#
# Before sbatch (on the login node, in the dedalus conda env):
#   conda activate dedalus
#   python -m datagen.galewsky.scripts.generate_sweep
#
# DATASET_ROOT defaults to the Daint scratch dataset dir and can be
# overridden in the environment.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p datagen/galewsky/slurm/logs

CONDA_BASE="${CONDA_BASE:-/capstor/scratch/cscs/tmuser/miniforge3}"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate dedalus

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}/datagen:${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

# Persistent dataset directory (must match configs/data/galewsky.yaml::root).
# Raw HDF5 snapshots stay in job-local scratch and are discarded when the
# job exits; only the resampled Zarr is persisted.
: "${DATASET_ROOT:=/iopsstor/scratch/cscs/${USER}/PDEDatasets/SphericalPDEs/galewsky-sw}"
: "${JOB_SCRATCH:=${TMPDIR:-/dev/shm/${USER}}}"
mkdir -p "${JOB_SCRATCH}"

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
srun --mpi=pmix --ntasks="${SLURM_NTASKS}" --cpus-per-task=1 \
    python -m datagen.galewsky.scripts.run \
    --config "${CONFIG}" \
    --out-dir "${RAW_DIR}" \
    --stop-sim-time 2764800 \
    --snapshot-dt 14400
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
    rm -rf "${RAW_DIR}"
    exit "${SOLVER_RC}"
fi

# Resample to regular (lat, lon) Zarr directly on the persistent path.
python -m datagen.galewsky.scripts.postprocess \
    --raw "${RAW_DIR}" \
    --out "${ZARR_OUT}" \
    --config "${CONFIG}" \
    --nlat 256 --nlon 512

# Free local scratch promptly.
rm -rf "${RAW_DIR}"
