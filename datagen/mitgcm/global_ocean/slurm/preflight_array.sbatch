#!/bin/bash
#SBATCH --job-name=mitgcm-go-preflight
#SBATCH --partition=scicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-31
#SBATCH --output=datagen/mitgcm/global_ocean/slurm/logs/preflight-%A_%a.out
#SBATCH --error=datagen/mitgcm/global_ocean/slurm/logs/preflight-%A_%a.err

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

UV="uv run --no-sync --project datagen"

: "${DATA_ROOT:=/scicore/home/rothvo/abati0000/fots-data/mitgcm/global-ocean}"
: "${EXECUTABLE:=$(pwd)/datagen/mitgcm/global_ocean/build/mitgcmuv}"

TASK_ID=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
CONFIG="datagen/mitgcm/global_ocean/configs/preflight/corner_${TASK_ID}.json"
OUT_DIR="${DATA_ROOT}/preflight/corner_${TASK_ID}"

mkdir -p "${OUT_DIR%/*}" datagen/mitgcm/global_ocean/slurm/logs
rm -rf "${OUT_DIR}" "${OUT_DIR}.FAILED"

echo "=== global-ocean preflight corner_${TASK_ID}"
echo "    config:     ${CONFIG}"
echo "    out_dir:    ${OUT_DIR}"
echo "    executable: ${EXECUTABLE}"

${UV} python -m datagen.mitgcm.global_ocean.scripts.preflight run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --executable "${EXECUTABLE}"
