#!/bin/bash
#SBATCH --job-name=cahn-hilliard-smoke
#SBATCH --partition=scicore
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=datagen/cahn_hilliard/slurm/smoke-%j.out
#SBATCH --error=datagen/cahn_hilliard/slurm/smoke-%j.err
#
# One-off smoke test: solve run_0000 on a coarse mesh and short window,
# resample to a small Zarr, then print a quick summary. Writes output under
# ${SMOKE_ROOT:=$HOME/scratch/ch_smoke} so it stays out of the production
# group dataset path.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
UV="uv run --no-sync --project datagen"

: "${SMOKE_ROOT:=${HOME}/scratch/ch_smoke}"
RUN_DIR="${SMOKE_ROOT}/run_0000"
RAW_FILE="${RUN_DIR}/cahn_s1.h5"
ZARR_OUT="${SMOKE_ROOT}/run_0000.zarr"
CONFIG="datagen/cahn_hilliard/configs/run_0000.json"

mkdir -p "${SMOKE_ROOT}"
rm -rf "${RUN_DIR}" "${ZARR_OUT}" "${SMOKE_ROOT}/run_0000.FAILED"

echo "=== Smoke run"
echo "    config:    ${CONFIG}"
echo "    raw:       ${RAW_FILE}"
echo "    zarr:      ${ZARR_OUT}"

${UV} python -m datagen.cahn_hilliard.scripts.run \
    --config "${CONFIG}" \
    --out-dir "${RUN_DIR}" \
    --cell-size 0.6 --stop-sim-time 50.0 --snapshot-dt 5.0

${UV} python -m datagen.cahn_hilliard.scripts.postprocess \
    --raw "${RAW_FILE}" \
    --out "${ZARR_OUT}" \
    --config "${CONFIG}" \
    --nlat 64 --nlon 128

SMOKE_ZARR="${ZARR_OUT}" ${UV} python - <<'PY'
import os
import xarray as xr
ds = xr.open_zarr(os.environ["SMOKE_ZARR"])
print(ds)
print("phi range:",
      float(ds.fields.sel(field="phi").min()),
      float(ds.fields.sel(field="phi").max()))
print("shape:", ds.fields.shape)
PY
