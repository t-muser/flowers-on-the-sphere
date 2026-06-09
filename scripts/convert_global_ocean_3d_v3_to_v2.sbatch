#!/bin/bash
#SBATCH --job-name=fots-convert-global-ocean-3d-v3-to-v2
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
# Rewrite the global-ocean-3D zarr v3 dataset as v2 (sibling tree),
# so the aurora_arm64_modulus2412 container (zarr 2.x) can read it.
# Uses the miniforge env directly; no container needed.

set -euo pipefail
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

PY=/capstor/scratch/cscs/tmuser/miniforge3/bin/python
SRC=/iopsstor/scratch/cscs/tmuser/PDEDatasets/SphericalPDEs/global-ocean-3D
DST=/iopsstor/scratch/cscs/tmuser/PDEDatasets/SphericalPDEs/global-ocean-3D-v2
WORKERS=16

echo "PY:      $PY"
echo "SRC:     $SRC"
echo "DST:     $DST"
echo "WORKERS: $WORKERS"

srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=72 --export=ALL -u -l \
    "$PY" scripts/convert_global_ocean_zarr_v3_to_v2.py \
    --src "$SRC" --dst "$DST" --workers "$WORKERS"
