#!/bin/bash
#SBATCH --job-name=mitgcm-go-preflight1
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=datagen/mitgcm/global_ocean/slurm/logs/preflight1-%j.out
#SBATCH --error=datagen/mitgcm/global_ocean/slurm/logs/preflight1-%j.err
#
# Single-corner global-ocean preflight.
# Override CORNER on the sbatch command line:
#   sbatch --export=ALL,CORNER=07 datagen/mitgcm/global_ocean/slurm/preflight_one.sbatch

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
: "${CORNER:=00}"

CONFIG="datagen/mitgcm/global_ocean/configs/preflight/corner_${CORNER}.json"
OUT_DIR="${DATA_ROOT}/preflight/corner_${CORNER}"

mkdir -p "${OUT_DIR%/*}" datagen/mitgcm/global_ocean/slurm/logs
rm -rf "${OUT_DIR}" "${OUT_DIR}.FAILED"

echo "=== global-ocean preflight corner_${CORNER}"
echo "    config:     ${CONFIG}"
echo "    out_dir:    ${OUT_DIR}"
echo "    executable: ${EXECUTABLE}"

${UV} python -m datagen.mitgcm.global_ocean.scripts.preflight run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --executable "${EXECUTABLE}"
