#!/bin/bash
#SBATCH --job-name=cpl-aim-ocn-sweep
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --array=0-179
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --time=05:30:00
#SBATCH --output=datagen/cpl_aim_ocn/slurm/logs/sweep-%A_%a.out
#SBATCH --error=datagen/cpl_aim_ocn/slurm/logs/sweep-%A_%a.err
#
# Coupled AIM+ocean parameter sweep: 180 runs (CO2 × solar × κ_GM × seed).
# Each array task is one ensemble member: 30-day spin-up + 365-day data
# at cs32, three MPI ranks (cpl, ocn, atm) launched as MPMD by the
# solver via mpirun.
#
# Before sbatch (on the login node):
#   uv run --project datagen python -m datagen.cpl_aim_ocn.scripts.generate_sweep
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

RUN_ID=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
CONFIG="datagen/cpl_aim_ocn/configs/run_${RUN_ID}.json"
OUT_DIR="${DATA_ROOT}/cpl_aim_ocn/runs/run_${RUN_ID}"

mkdir -p "${DATA_ROOT}/cpl_aim_ocn/runs"
rm -rf "${OUT_DIR}" "${OUT_DIR}.FAILED"

echo "=== cpl_aim+ocn sweep run ${RUN_ID}"
echo "    config:  ${CONFIG}"
echo "    out_dir: ${OUT_DIR}"
echo "    n_mpi:   ${SLURM_NTASKS} (must be 3 for MPMD cpl/ocn/atm)"

${UV} python -m datagen.cpl_aim_ocn.scripts.run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}"
