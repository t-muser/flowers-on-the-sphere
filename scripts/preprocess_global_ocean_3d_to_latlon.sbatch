#!/bin/bash
#SBATCH --job-name=fots-preprocess-global-ocean-3d-latlon
#SBATCH --account=uba03
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Preprocess the global-ocean-3D dataset onto a (nlat, nlon) grid by
# applying cs32→latlon IDW regrid + uv rotation + land imputation once.
# Runs the miniforge env directly (zarr 3.x + xarray); writes a v2
# zarr sibling tree the container can read.

set -euo pipefail
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

PY=/capstor/scratch/cscs/tmuser/miniforge3/bin/python
SRC=/iopsstor/scratch/cscs/tmuser/PDEDatasets/SphericalPDEs/global-ocean-3D-v2
DST=/iopsstor/scratch/cscs/tmuser/PDEDatasets/SphericalPDEs/global-ocean-3D-latlon
WORKERS=16

echo "PY:      $PY"
echo "SRC:     $SRC"
echo "DST:     $DST"
echo "WORKERS: $WORKERS"

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=72 --export=ALL -u -l \
    "$PY" scripts/preprocess_global_ocean_3d_to_latlon.py \
    --src "$SRC" --dst "$DST" --nlat 64 --nlon 128 \
    --chunk-time 1 --workers "$WORKERS"
