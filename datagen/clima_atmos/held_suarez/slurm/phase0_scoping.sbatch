#!/bin/bash
#SBATCH --job-name=clima-phase0-scoping
#SBATCH --partition=l40s,a100,rtx4090
#SBATCH --qos=l40s-6hours,a100-6hours,rtx4090-6hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=datagen/clima_atmos/held_suarez/slurm/logs/phase0_scoping-%j.out
#SBATCH --error=datagen/clima_atmos/held_suarez/slurm/logs/phase0_scoping-%j.err
#
# Phase 0.3: he24ze31 wall-time scoping on a single GPU. Runs 10 sim-days
# and projects to 565 sim-days. Decides whether the production array
# sbatch's 8-hour budget is enough headroom across L40s / A100 / RTX 4090.

set -euo pipefail

module purge
module load Julia/1.10.8-linux-x86_64
module load CUDA/12.4.0

export JULIA_DEPOT_PATH="${PWD}/datagen/clima_atmos/held_suarez/env/.julia_depot"

REPO_ROOT="${PWD}"
ENV_DIR="${REPO_ROOT}/datagen/clima_atmos/held_suarez/env"
CLIMA_DIR="${REPO_ROOT}/external/ClimaAtmos.jl"
DRIVER_JL="${REPO_ROOT}/datagen/clima_atmos/held_suarez/run_hs.jl"
NUMERICS="${REPO_ROOT}/datagen/clima_atmos/held_suarez/configs/numerics/sphere_he24ze31.yml"
OUT_ROOT="${REPO_ROOT}/datagen/clima_atmos/held_suarez/slurm/logs/phase0_scoping_out"
JOB_ID="phase0_he24ze31_10day_${SLURM_JOB_ID}"
OUT_DIR="${OUT_ROOT}/${JOB_ID}"

mkdir -p "${OUT_DIR}" datagen/clima_atmos/held_suarez/slurm/logs

# Stock-defaults corner — Phase 0.3 only measures wall time, not
# parameter response.
CORNER_JSON="${OUT_DIR}/stock_corner.json"
cat > "${CORNER_JSON}" <<'EOF'
{
  "run_id": 0,
  "run_name": "phase0_scoping",
  "params": {
    "omega_factor":  1.0,
    "delta_T_y":     60.0,
    "delta_theta_z": 10.0,
    "seed":          0
  }
}
EOF

echo "=== Phase 0.3 scoping  job=${JOB_ID}"
echo "    partition: ${SLURM_JOB_PARTITION}"
nvidia-smi -L 2>&1 | head -1 || true
echo "    numerics:  ${NUMERICS}"
echo "    out_dir:   ${OUT_DIR}"

START=$(date +%s)
julia --project="${ENV_DIR}" "${DRIVER_JL}" \
    --corner "${CORNER_JSON}" \
    --base-config "${NUMERICS}" \
    --resolution-config "${CLIMA_DIR}/config/model_configs/held_suarez.yml" \
    --t-end "10days" \
    --out-dir "${OUT_DIR}/run" \
    --job-id "${JOB_ID}"
END=$(date +%s)
WALL=$((END - START))
PROJ_565_H=$(python3 -c "print(round(${WALL} * 56.5 / 3600.0, 2))")

SUMMARY="${OUT_ROOT}/summary.csv"
if [ ! -f "${SUMMARY}" ]; then
    echo "job_id,partition,t_end_days,wall_s,projected_565d_hours" > "${SUMMARY}"
fi
echo "${JOB_ID},${SLURM_JOB_PARTITION},10,${WALL},${PROJ_565_H}" >> "${SUMMARY}"

echo "=== Phase 0.3 scoping done  wall=${WALL}s  projected_565d=${PROJ_565_H}h"
ls "${OUT_DIR}/run/" 2>&1
