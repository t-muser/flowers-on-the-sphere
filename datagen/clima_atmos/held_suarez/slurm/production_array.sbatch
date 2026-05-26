#!/bin/bash
#SBATCH --job-name=clima-prod
#SBATCH --partition=l40s,a100,rtx4090
#SBATCH --qos=l40s-1day,a100-1day,rtx4090-1day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --array=0-161
#SBATCH --output=datagen/clima_atmos/held_suarez/slurm/logs/prod-%A_%a.out
#SBATCH --error=datagen/clima_atmos/held_suarez/slurm/logs/prod-%A_%a.err
#
# Production sweep at he24ze31 (~144x288 native output grid, ~0.75 deg
# equivalent). Full 162-corner sweep, dispatched across the L40s, A100,
# and RTX 4090 partitions so SLURM can pick whichever has capacity.
#
# Wall projection from the GPU smoke run: ~5 h per corner. With ~25
# concurrent GPUs the full sweep finishes in ~32 h wall.
#
# Generate per-run configs first if the configs/ dir is empty:
#   uv run --project datagen python \
#       -m datagen.clima_atmos.held_suarez.scripts.generate_sweep \
#       --out datagen/clima_atmos/held_suarez/configs/

set -euo pipefail

module purge
module load Julia/1.10.8-linux-x86_64
module load CUDA/12.4.0

export JULIA_DEPOT_PATH="${PWD}/datagen/clima_atmos/held_suarez/env/.julia_depot"

UV="uv run --no-sync --project datagen"

: "${DATA_ROOT:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/held-suarez-clima}"
: "${CLIMA_NUMERICS_CONFIG:=datagen/clima_atmos/held_suarez/configs/numerics/sphere_he24ze31.yml}"
: "${CLIMA_PROD_T_END:=565days}"

TASK_ID=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
CONFIG="datagen/clima_atmos/held_suarez/configs/run_${TASK_ID}.json"
OUT_DIR="${DATA_ROOT}/runs/run_${TASK_ID}"

mkdir -p "${OUT_DIR%/*}" datagen/clima_atmos/held_suarez/slurm/logs
rm -rf "${OUT_DIR}" "${OUT_DIR}.FAILED"

echo "=== clima production run_${TASK_ID}"
nvidia-smi -L 2>&1 | head -1 || true
echo "    partition:  ${SLURM_JOB_PARTITION}"
echo "    config:     ${CONFIG}"
echo "    out_dir:    ${OUT_DIR}"
echo "    numerics:   ${CLIMA_NUMERICS_CONFIG}"
echo "    t_end:      ${CLIMA_PROD_T_END}"

${UV} python -m datagen.clima_atmos.held_suarez.scripts.run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --base-config "${CLIMA_NUMERICS_CONFIG}" \
    --resolution-config "external/ClimaAtmos.jl/config/model_configs/held_suarez.yml" \
    --t-end "${CLIMA_PROD_T_END}" \
    --cleanup-clima
