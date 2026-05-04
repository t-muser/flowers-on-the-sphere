#!/bin/bash
#SBATCH --job-name=mitgcm-preflight1
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=datagen/mitgcm/held_suarez/slurm/logs/preflight1-%j.out
#SBATCH --error=datagen/mitgcm/held_suarez/slurm/logs/preflight1-%j.err
#
# Single-corner preflight to validate the new apply_forcing.F.
# Override CORNER (default 00) on the sbatch command line:
#   sbatch --export=ALL,CORNER=07 datagen/mitgcm/held_suarez/slurm/preflight_one.sbatch

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

UV="uv run --no-sync --project datagen"

: "${DATA_ROOT:=/scicore/home/dokman0000/muser0001/fots-data/mitgcm/held-suarez}"
: "${CORNER:=00}"

CONFIG="datagen/mitgcm/held_suarez/configs/preflight/corner_${CORNER}.json"
OUT_DIR="${DATA_ROOT}/preflight/corner_${CORNER}"

mkdir -p "${OUT_DIR%/*}" datagen/mitgcm/held_suarez/slurm/logs
rm -rf "${OUT_DIR}" "${OUT_DIR}.FAILED"

echo "=== preflight corner_${CORNER}"
echo "    config:  ${CONFIG}"
echo "    out_dir: ${OUT_DIR}"
echo "    n_mpi:   ${SLURM_NTASKS}"

${UV} python -m datagen.mitgcm.held_suarez.scripts.preflight run \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --n-mpi "${SLURM_NTASKS}"
