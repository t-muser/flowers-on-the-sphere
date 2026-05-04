#!/bin/bash
#SBATCH --job-name=cpl-aim-ocn-smoke
#SBATCH --partition=scicore
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=datagen/cpl_aim_ocn/slurm/logs/smoke-%j.out
#SBATCH --error=datagen/cpl_aim_ocn/slurm/logs/smoke-%j.err
#
# 5-hour end-to-end smoke test of the coupled AIM+ocean wrapper.
# Uses `scripts.smoke_test` which builds the MPMD mpirun command
# itself (3 ranks: cpl, ocn, atm). SLURM allocates the 3 tasks; the
# smoke_test driver invokes mpirun internally.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
UV="uv run --no-sync --project datagen"

: "${RUN_DIR:=${SCRATCH:-/tmp}/cpl_aim_ocn_smoke}"

mkdir -p datagen/cpl_aim_ocn/slurm/logs

echo "=== cpl_aim+ocn smoke test"
echo "    run_dir: ${RUN_DIR}"

${UV} python -m datagen.cpl_aim_ocn.scripts.smoke_test \
    --source rendered \
    --run-dir "${RUN_DIR}" \
    --timeout-s 1500
