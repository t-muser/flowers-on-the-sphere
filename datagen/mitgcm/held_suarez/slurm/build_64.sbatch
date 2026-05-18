#!/bin/bash
#SBATCH --job-name=mitgcm-build-64
#SBATCH --partition=scicore
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=datagen/mitgcm/held_suarez/slurm/logs/build64-%j.out
#SBATCH --error=datagen/mitgcm/held_suarez/slurm/logs/build64-%j.err
#
# Compile MITgcm for the 256×128×30 Held-Suarez build (a.k.a. "C64"). The
# generator writes a fresh SIZE.h for this resolution into a per-build mods/
# directory; the canonical 128×64×20 SIZE.h is left untouched.

set -euo pipefail

module purge
module load foss/2024a
module load HDF5/1.14.5-gompi-2024a

export MPI_HOME="${EBROOTOPENMPI:?EBROOTOPENMPI not set — is foss/2024a loaded?}"
echo "MPI_HOME=${MPI_HOME}"

UV="uv run --no-sync --project datagen"

mkdir -p datagen/mitgcm/held_suarez/slurm/logs

# Wipe the resolution-tagged build dir so a stale executable can't be picked
# up if compile fails. Leaves the legacy build/ (128×64×20) alone.
rm -rf datagen/mitgcm/held_suarez/build_256x128x30

${UV} python -m datagen.mitgcm.held_suarez.scripts.build \
    --mitgcm-root mitgcm \
    --optfile mitgcm/tools/build_options/linux_amd64_gfortran \
    --nlon 256 --nlat 128 --nr 30 --n-mpi 8

echo "=== build_info ==="
cat datagen/mitgcm/held_suarez/build_256x128x30/build_info.json
echo "=== executable ==="
ls -l datagen/mitgcm/held_suarez/build_256x128x30/mitgcmuv
echo "=== input/ ==="
ls -l datagen/mitgcm/held_suarez/input_256x128/
