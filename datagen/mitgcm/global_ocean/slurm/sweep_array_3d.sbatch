#!/bin/bash
#SBATCH --job-name=mitgcm-go-sweep-3d
#SBATCH --partition=scicore
#SBATCH --qos=1day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-242
#SBATCH --output=datagen/mitgcm/global_ocean/slurm/logs/sweep3d-%A_%a.out
#SBATCH --error=datagen/mitgcm/global_ocean/slurm/logs/sweep3d-%A_%a.err

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

UV="uv run --no-sync --project datagen"

# 3-D dataset lives next to the legacy 2-D one but in a -3D suffixed root.
: "${DATA_ROOT:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/global-ocean-3D}"
: "${EXECUTABLE:=$(pwd)/datagen/mitgcm/global_ocean/build/mitgcmuv}"

TASK_ID=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
CONFIG="datagen/mitgcm/global_ocean/configs/run_${TASK_ID}.json"
OUT_DIR="${DATA_ROOT}/sweep/run_${TASK_ID}"

mkdir -p "${OUT_DIR%/*}" datagen/mitgcm/global_ocean/slurm/logs
rm -rf "${OUT_DIR}" "${OUT_DIR}.FAILED"

echo "=== global-ocean 3D sweep run_${TASK_ID}"
echo "    config:     ${CONFIG}"
echo "    out_dir:    ${OUT_DIR}"
echo "    executable: ${EXECUTABLE}"

${UV} python -m datagen.mitgcm.global_ocean.scripts.preflight run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --executable "${EXECUTABLE}" \
    --n-timesteps 36000 \
    --snapshot-interval-days 30 \
    --levels all
