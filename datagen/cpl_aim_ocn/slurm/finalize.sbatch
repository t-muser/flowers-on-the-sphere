#!/bin/bash
#SBATCH --job-name=cpl-aim-ocn-finalize
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=datagen/cpl_aim_ocn/slurm/logs/finalize-%j.out
#SBATCH --error=datagen/cpl_aim_ocn/slurm/logs/finalize-%j.err
#
# Assemble the published dataset from the finished sweep:
#   1. Symlink each successful per-run zarr into <root>/{train,val,test}/
#      (random 80/10/10 split, fixed seed).
#   2. Compute per-channel z-score stats over the train split's
#      regridded lat/lon view → stats.json.
#
# Run this once, after `sbatch sweep.sbatch` has finished.
#
# DATA_ROOT defaults to $SCRATCH/flowers-data; override in the environment.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
UV="uv run --no-sync --project datagen"

: "${DATA_ROOT:=${SCRATCH:-/tmp}/flowers-data}"
: "${SPLIT_SEED:=42}"

RUNS_DIR="${DATA_ROOT}/cpl_aim_ocn/runs"
OUT_ROOT="${DATA_ROOT}/cpl_aim_ocn"

echo "=== cpl_aim+ocn finalize"
echo "    runs_dir: ${RUNS_DIR}"
echo "    out_root: ${OUT_ROOT}"
echo "    seed:     ${SPLIT_SEED}"

${UV} python -m datagen.cpl_aim_ocn.scripts.finalize_dataset \
    --runs-dir "${RUNS_DIR}" \
    --out-root "${OUT_ROOT}" \
    --seed "${SPLIT_SEED}"

echo "=== splits.json ==="
cat "${OUT_ROOT}/splits.json"
echo
echo "=== stats.json (head) ==="
head -40 "${OUT_ROOT}/stats.json"
