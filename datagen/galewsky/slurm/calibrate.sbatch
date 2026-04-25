#!/bin/bash
#SBATCH --job-name=galewsky-calib
#SBATCH --partition=scicore
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=datagen/galewsky/slurm/calibrate-%j.out
#SBATCH --error=datagen/galewsky/slurm/calibrate-%j.err
#
# Calibration: one preflight corner at full preflight resolution
# (Nphi=256, Ntheta=128, 1 day simulated) on 8 MPI ranks. Produces a
# real per-run wall-time data point for budgeting the full 1920-run sweep.
# Picks corner_31 (all-max axis values: u_max=100, lat_center=60, h_hat=240,
# H=14000, lon_c=300) as the worst-case timing proxy.
#
# Submit from the repo root:
#   sbatch datagen/galewsky/slurm/calibrate.sbatch

set -euo pipefail

# scicore: load a coherent toolchain (foss bundles GCC+OpenMPI+FFTW+OpenBLAS;
# HDF5 is separate but must match the gompi subset of the foss version).
module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

# Compute nodes can't reach PyPI; assume `uv sync` was done on the login node.
# Pin thread counts so each MPI rank stays on its own core.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

UV="uv run --no-sync --project datagen"

# Preflight corner configs (datagen/galewsky/configs/preflight/corner_XX.json).
${UV} python -m datagen.galewsky.scripts.preflight generate

CORNER_ID=31
CORNER_TAG=$(printf "%02d" "${CORNER_ID}")
: "${DATA_ROOT:=${SCRATCH:-/tmp}/flowers-data}"
OUT_DIR="${DATA_ROOT}/calibration/corner_${CORNER_TAG}"
ZARR_OUT="${DATA_ROOT}/calibration/corner_${CORNER_TAG}.zarr"

echo "=== Calibration corner ${CORNER_TAG}"
echo "    raw:  ${OUT_DIR}"
echo "    zarr: ${ZARR_OUT}"

T0=$(date +%s)
mpirun -n "${SLURM_NTASKS}" ${UV} python -m datagen.galewsky.scripts.preflight run \
    --config "datagen/galewsky/configs/preflight/corner_${CORNER_TAG}.json" \
    --out-dir "${OUT_DIR}" \
    --nphi 256 --ntheta 128 \
    --stop-sim-time 86400 \
    --snapshot-dt 3600
T1=$(date +%s)
echo "=== Solver wall time: $((T1 - T0)) s"

${UV} python -m datagen.galewsky.scripts.postprocess \
    --raw "${OUT_DIR}" \
    --out "${ZARR_OUT}" \
    --config "datagen/galewsky/configs/preflight/corner_${CORNER_TAG}.json" \
    --nlat 128 --nlon 256
T2=$(date +%s)
echo "=== Resample wall time: $((T2 - T1)) s"
echo "=== Total wall time:    $((T2 - T0)) s"

# Quick integrity check on the resampled Zarr.
${UV} python - <<'PY'
import os, numpy as np, xarray as xr
path = f"{os.environ.get('DATA_ROOT', os.environ.get('SCRATCH', '/tmp') + '/flowers-data')}/calibration/corner_31.zarr"
ds = xr.open_zarr(path)
print(ds)
print("\nfield sanity:")
for f in ds.field.values:
    arr = ds.fields.sel(field=f).values
    finite = np.isfinite(arr).all()
    print(f"  {f:9s}  shape={arr.shape}  min={arr.min():.3g}  max={arr.max():.3g}  all_finite={finite}")
PY
