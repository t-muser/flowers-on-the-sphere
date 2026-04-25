#!/bin/bash
#SBATCH --job-name=cahn-hilliard-single
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=datagen/cahn_hilliard/slurm/logs/single-%j.out
#SBATCH --error=datagen/cahn_hilliard/slurm/logs/single-%j.err
#
# Production-shape single-trajectory test. Writes the final Zarr to the
# same group dataset path as the full sweep so we can spot-check a
# representative trajectory in place; the sweep will overwrite it.
# Defaults to a representative pattern-forming config: xi=1, ell=m=6,
# amplitude=0.03, omega=0.005, seed=0 (run_0064 in the new grid).

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
: "${RUN_ID:=0124}"
RUN_DIR="${JOB_SCRATCH}/raw/run_${RUN_ID}"
ZARR_OUT="${DATASET_ROOT}/processed/run_${RUN_ID}.zarr"
CONFIG="datagen/cahn_hilliard/configs/run_${RUN_ID}.json"

mkdir -p "${DATASET_ROOT}/processed"
rm -rf "${RUN_DIR}" "${ZARR_OUT}" "${DATASET_ROOT}/processed/run_${RUN_ID}.FAILED"

echo "=== Single-trajectory production test"
echo "    config:    ${CONFIG}"
echo "    raw (dir): ${RUN_DIR}"
echo "    zarr:      ${ZARR_OUT}"

mpirun -n "${SLURM_NTASKS}" ${UV} python -m datagen.cahn_hilliard.scripts.run \
    --config "${CONFIG}" \
    --out-dir "${RUN_DIR}"

${UV} python -m datagen.cahn_hilliard.scripts.postprocess \
    --raw "${RUN_DIR}" \
    --out "${ZARR_OUT}" \
    --config "${CONFIG}" \
    --nlat 128 --nlon 256

SINGLE_ZARR="${ZARR_OUT}" ${UV} python - <<'PY'
import os
import xarray as xr
ds = xr.open_zarr(os.environ["SINGLE_ZARR"])
print(ds)
psi = ds.fields.sel(field="psi")
print("shape:", psi.shape)
print("psi range across run:", float(psi.min()), float(psi.max()))
print("final-frame psi range:",
      float(psi.isel(time=-1).min()), float(psi.isel(time=-1).max()))
print("first-frame psi range:",
      float(psi.isel(time=0).min()), float(psi.isel(time=0).max()))
PY
