#!/bin/bash
#SBATCH --job-name=cahn-hilliard-single
#SBATCH --partition=scicore
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=datagen/cahn_hilliard/slurm/single-%j.out
#SBATCH --error=datagen/cahn_hilliard/slurm/single-%j.err
#
# Production-shape single-trajectory test: run_0000 at production cell_size
# (0.3), stop_sim_time (1000), snapshot_dt (10), resampled to a 256x512 Zarr.
# Output goes under ${SINGLE_ROOT:=$HOME/scratch/ch_single} so it stays out of
# the group dataset path until we're happy with it.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
UV="uv run --no-sync --project datagen"

: "${SINGLE_ROOT:=${HOME}/scratch/ch_single}"
: "${RUN_ID:=0040}"
RUN_DIR="${SINGLE_ROOT}/run_${RUN_ID}"
RAW_FILE="${RUN_DIR}/cahn_s1.h5"
ZARR_OUT="${SINGLE_ROOT}/run_${RUN_ID}.zarr"
CONFIG="datagen/cahn_hilliard/configs/run_${RUN_ID}.json"

mkdir -p "${SINGLE_ROOT}"
rm -rf "${RUN_DIR}" "${ZARR_OUT}" "${SINGLE_ROOT}/run_${RUN_ID}.FAILED"

echo "=== Single-trajectory production test"
echo "    config:    ${CONFIG}"
echo "    raw:       ${RAW_FILE}"
echo "    zarr:      ${ZARR_OUT}"

${UV} python -m datagen.cahn_hilliard.scripts.run \
    --config "${CONFIG}" \
    --out-dir "${RUN_DIR}"

${UV} python -m datagen.cahn_hilliard.scripts.postprocess \
    --raw "${RAW_FILE}" \
    --out "${ZARR_OUT}" \
    --config "${CONFIG}" \
    --nlat 256 --nlon 512

SINGLE_ZARR="${ZARR_OUT}" ${UV} python - <<'PY'
import os
import xarray as xr
ds = xr.open_zarr(os.environ["SINGLE_ZARR"])
print(ds)
phi = ds.fields.sel(field="phi")
print("shape:", phi.shape)
print("phi range across run:", float(phi.min()), float(phi.max()))
print("final-frame phi range:", float(phi.isel(time=-1).min()), float(phi.isel(time=-1).max()))
print("first-frame phi range:", float(phi.isel(time=0).min()), float(phi.isel(time=0).max()))
PY
