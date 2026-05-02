#!/bin/bash
#SBATCH --job-name=galewsky-calib
#SBATCH --account=uba03
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=datagen/galewsky/slurm/logs/calibrate-%j.out
#SBATCH --error=datagen/galewsky/slurm/logs/calibrate-%j.err
#
# Daint port of calibrate.sbatch. Uses the conda env at
# /capstor/scratch/cscs/tmuser/miniforge3/envs/dedalus (dedalus 3.0.5 +
# parallel HDF5 + openmpi from conda-forge). Runs one preflight corner
# (corner_31) for an end-to-end timing data point before the full sweep.
#
# Submit from the repo root:
#   sbatch datagen/galewsky/slurm/calibrate_daint.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p datagen/galewsky/slurm/logs

# Conda env (dedalus + mpi4py + parallel h5py + xarray + zarr).
CONDA_BASE="${CONDA_BASE:-/capstor/scratch/cscs/tmuser/miniforge3}"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate dedalus

# Use the conda env's MPI launcher (openmpi from conda-forge) and pin
# threads so each rank stays on its own core.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Add the repo to PYTHONPATH so `datagen.*` imports resolve.
export PYTHONPATH="${SLURM_SUBMIT_DIR}/datagen:${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

# Generate the preflight corner configs (idempotent).
python -m datagen.galewsky.scripts.preflight generate

CORNER_ID=31
CORNER_TAG=$(printf "%02d" "${CORNER_ID}")
: "${DATA_ROOT:=/iopsstor/scratch/cscs/${USER}/flowers-data}"
OUT_DIR="${DATA_ROOT}/calibration/corner_${CORNER_TAG}"
ZARR_OUT="${DATA_ROOT}/calibration/corner_${CORNER_TAG}.zarr"

mkdir -p "${DATA_ROOT}/calibration"

echo "=== Calibration corner ${CORNER_TAG}"
echo "    raw:  ${OUT_DIR}"
echo "    zarr: ${ZARR_OUT}"

T0=$(date +%s)
srun --mpi=pmix --ntasks="${SLURM_NTASKS}" --cpus-per-task=1 \
    python -m datagen.galewsky.scripts.preflight run \
    --config "datagen/galewsky/configs/preflight/corner_${CORNER_TAG}.json" \
    --out-dir "${OUT_DIR}" \
    --nphi 256 --ntheta 128 \
    --stop-sim-time 86400 \
    --snapshot-dt 3600
T1=$(date +%s)
echo "=== Solver wall time: $((T1 - T0)) s"

python -m datagen.galewsky.scripts.postprocess \
    --raw "${OUT_DIR}" \
    --out "${ZARR_OUT}" \
    --config "datagen/galewsky/configs/preflight/corner_${CORNER_TAG}.json" \
    --nlat 128 --nlon 256 \
    --skip-seconds 0
T2=$(date +%s)
echo "=== Resample wall time: $((T2 - T1)) s"
echo "=== Total wall time:    $((T2 - T0)) s"

# Quick integrity check on the resampled Zarr.
DATA_ROOT="${DATA_ROOT}" python - <<'PY'
import os, numpy as np, xarray as xr
path = f"{os.environ['DATA_ROOT']}/calibration/corner_31.zarr"
ds = xr.open_zarr(path)
print(ds)
print("\nfield sanity:")
for f in ds.field.values:
    arr = ds.fields.sel(field=f).values
    finite = np.isfinite(arr).all()
    print(f"  {f:9s}  shape={arr.shape}  min={arr.min():.3g}  max={arr.max():.3g}  all_finite={finite}")
PY
