#!/bin/bash
#SBATCH --job-name=cpl-aim-ocn-build
#SBATCH --partition=scicore
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=datagen/cpl_aim_ocn/slurm/logs/build-%j.out
#SBATCH --error=datagen/cpl_aim_ocn/slurm/logs/build-%j.err
#
# Compile the three coupled-AIM+ocean MITgcm binaries (cpl, ocn, atm)
# and stage their input bundle. Reads the MITgcm source tree from the
# `mitgcm/` git submodule.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

# genmake2 reads MPI_HOME to discover mpif77/mpi headers.
export MPI_HOME="${EBROOTOPENMPI:?EBROOTOPENMPI not set — is foss/2024a loaded?}"
echo "MPI_HOME=${MPI_HOME}"

UV="uv run --no-sync --project datagen"

mkdir -p datagen/cpl_aim_ocn/slurm/logs

# Wipe build dirs so a stale executable can't be picked up if compile fails.
for c in cpl ocn atm; do
    rm -f "datagen/cpl_aim_ocn/build_${c}/mitgcmuv"
done

${UV} python -m datagen.cpl_aim_ocn.scripts.build \
    --mitgcm-root mitgcm \
    --optfile mitgcm/tools/build_options/linux_amd64_gfortran \
    --jobs "${SLURM_CPUS_PER_TASK:-8}"

echo "=== build_info ==="
cat datagen/cpl_aim_ocn/build_info.json
echo "=== executables ==="
ls -l datagen/cpl_aim_ocn/build_cpl/mitgcmuv \
      datagen/cpl_aim_ocn/build_ocn/mitgcmuv \
      datagen/cpl_aim_ocn/build_atm/mitgcmuv
echo "=== inputs/ ==="
ls -l datagen/cpl_aim_ocn/inputs/
