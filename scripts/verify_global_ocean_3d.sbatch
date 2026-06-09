#!/bin/bash
#SBATCH --job-name=fots-verify-global-ocean-3d
#SBATCH --account=uba03
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Verify the transferred global-ocean-3D zarr dataset by decompressing
# every chunk. Runs the miniforge env directly (zarr 3.x, Python 3.13);
# no container needed since the aurora_arm64_modulus2412 image ships
# zarr 2.x and cannot read v3 stores.

set -euo pipefail
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

PY=/capstor/scratch/cscs/tmuser/miniforge3/bin/python
ROOT="${ROOT:-/iopsstor/scratch/cscs/tmuser/PDEDatasets/SphericalPDEs/global-ocean-3D}"
WORKERS=16

echo "PY:      $PY"
echo "ROOT:    $ROOT"
echo "WORKERS: $WORKERS"

srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=72 --export=ALL -u -l \
    "$PY" scripts/verify_zarr_dataset.py "$ROOT" --workers "$WORKERS"
