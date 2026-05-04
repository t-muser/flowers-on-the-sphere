#!/bin/bash
#SBATCH --job-name=mitgcm-build
#SBATCH --partition=scicore
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=datagen/mitgcm/held_suarez/slurm/logs/build-%j.out
#SBATCH --error=datagen/mitgcm/held_suarez/slurm/logs/build-%j.err
#
# Compile MITgcm against datagen/mitgcm/held_suarez/code/ overrides.
# Reads the MITgcm source tree from the `mitgcm/` git submodule.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

# genmake2 reads MPI_HOME to discover mpif77/mpi headers when pkg-config
# isn't available on the build node. EasyBuild exposes the OpenMPI prefix.
export MPI_HOME="${EBROOTOPENMPI:?EBROOTOPENMPI not set — is foss/2024a loaded?}"
echo "MPI_HOME=${MPI_HOME}"

UV="uv run --no-sync --project datagen"

mkdir -p datagen/mitgcm/held_suarez/slurm/logs

# Wipe the build dir so a stale executable can't be picked up if compile fails.
rm -rf datagen/mitgcm/held_suarez/build

${UV} python -m datagen.mitgcm.held_suarez.scripts.build \
    --mitgcm-root mitgcm \
    --optfile mitgcm/tools/build_options/linux_amd64_gfortran

echo "=== build_info ==="
cat datagen/mitgcm/held_suarez/build/build_info.json
echo "=== executable ==="
ls -l datagen/mitgcm/held_suarez/build/mitgcmuv
echo "=== input/ ==="
ls -l datagen/mitgcm/held_suarez/input/
