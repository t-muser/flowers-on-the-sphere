#!/bin/bash
#SBATCH --job-name=clima-install-env
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=datagen/clima_atmos/held_suarez/slurm/logs/install_env-%j.out
#SBATCH --error=datagen/clima_atmos/held_suarez/slurm/logs/install_env-%j.err
#
# Precompile-only step of the Julia env install. The download/instantiate
# step MUST run on the login node first because scicore compute nodes
# are blocked from pkg.julialang.org and github.com (HTTP 403). See
# scripts/install_env.sh for the split — call it once on login with
# INSTALL_PHASE=instantiate, then submit this sbatch (which runs
# INSTALL_PHASE=precompile) to do the expensive compile work.

set -euo pipefail

module purge
module load Julia/1.10.8-linux-x86_64

# Keep the Julia package depot under the repo to avoid polluting $HOME
# and to make the install self-contained. Must match what
# install_env.sh used on the login node.
export JULIA_DEPOT_PATH="${PWD}/datagen/clima_atmos/held_suarez/env/.julia_depot"
mkdir -p "${JULIA_DEPOT_PATH}" datagen/clima_atmos/held_suarez/slurm/logs

echo "=== Precompiling Julia env"
echo "    JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH}"
echo "    pwd=$(pwd)"

INSTALL_PHASE=precompile bash datagen/clima_atmos/held_suarez/scripts/install_env.sh
