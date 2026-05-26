#!/bin/bash
#SBATCH --job-name=clima-preflight
# Dispatch across L40s / A100 / RTX 4090 to maximize cluster throughput.
# SLURM picks the (partition, qos) pair that has capacity per array task.
#SBATCH --partition=l40s,a100,rtx4090
#SBATCH --qos=l40s-6hours,a100-6hours,rtx4090-6hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-7
#SBATCH --output=datagen/clima_atmos/held_suarez/slurm/logs/preflight-%A_%a.out
#SBATCH --error=datagen/clima_atmos/held_suarez/slurm/logs/preflight-%A_%a.err
#
# Phase 1.2: 8-corner stability preflight at production resolution
# (he24ze31) x 60 sim-days, on GPU. Each corner picks the extreme
# (low/high) values of the 3 physical sweep axes; seed is fixed.
#
# Wall projection from the GPU smoke: L40s ~5 h per 565 sim-d at he24
# -> ~30 min per 60 sim-d corner. Bumped QoS to 6h to keep margin.
#
# Generate corner configs first:
#   uv run --project datagen python \
#       -m datagen.clima_atmos.held_suarez.scripts.preflight generate

set -euo pipefail

module purge
module load Julia/1.10.8-linux-x86_64
# CUDA.jl is configured with local_toolkit=true in env/LocalPreferences.toml;
# load the system CUDA so libcudart is on LD_LIBRARY_PATH.
module load CUDA/12.4.0

export JULIA_DEPOT_PATH="${PWD}/datagen/clima_atmos/held_suarez/env/.julia_depot"

UV="uv run --no-sync --project datagen"

: "${DATA_ROOT:=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/held-suarez-clima}"
: "${CLIMA_NUMERICS_CONFIG:=datagen/clima_atmos/held_suarez/configs/numerics/sphere_he24ze31.yml}"
: "${CLIMA_PREFLIGHT_T_END:=60days}"

TASK_ID=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
echo "    preflight corner index: ${TASK_ID} (of 0-7)"
CONFIG="datagen/clima_atmos/held_suarez/configs/preflight/corner_${TASK_ID}.json"
OUT_DIR="${DATA_ROOT}/preflight/corner_${TASK_ID}"

mkdir -p "${OUT_DIR%/*}" datagen/clima_atmos/held_suarez/slurm/logs
rm -rf "${OUT_DIR}" "${OUT_DIR}.FAILED"

echo "=== clima preflight corner_${TASK_ID}"
nvidia-smi -L 2>&1 | head -1 || true
echo "    partition:  ${SLURM_JOB_PARTITION}"
echo "    config:     ${CONFIG}"
echo "    out_dir:    ${OUT_DIR}"
echo "    numerics:   ${CLIMA_NUMERICS_CONFIG}"
echo "    t_end:      ${CLIMA_PREFLIGHT_T_END}"

# `device: CUDADevice` is baked into the numerics YAML so no extra
# override layer is needed here. Layering order: numerics →
# held_suarez → run_override (automatic).
${UV} python -m datagen.clima_atmos.held_suarez.scripts.preflight run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --base-config "${CLIMA_NUMERICS_CONFIG}" \
    --resolution-config "external/ClimaAtmos.jl/config/model_configs/held_suarez.yml" \
    --t-end "${CLIMA_PREFLIGHT_T_END}"
