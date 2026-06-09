#!/bin/bash
#SBATCH --job-name=fots-smoke-global-ocean
#SBATCH --account=uba03
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# One-shot pytest smoke test for fots.data.global_ocean against the
# transferred dataset on /iopsstor. Mirrors the train_daint.sbatch
# container/env wiring, just runs pytest instead of fots.train.

set -euo pipefail
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
TOMLPATH="${TOMLPATH:-${WORKDIR}/scripts/torchcontainer_daint_iopsstor.toml}"

echo "WORKDIR:  $WORKDIR"
echo "TOMLPATH: $TOMLPATH"

srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=72 \
     --export=ALL \
     --environment="${TOMLPATH}" \
     -u -l \
  bash -c "
    set -euo pipefail
    cd '${WORKDIR}'
    export PYTHONPATH=\"${WORKDIR}:\${PYTHONPATH:-}\"
    # Point the test helper at the converted v2 store via a tmp dir
    # symlink (the helper resolves <DATA_ROOT>/global-ocean).
    export DATA_ROOT=\$(mktemp -d)
    ln -s /iopsstor/scratch/cscs/tmuser/PDEDatasets/SphericalPDEs/global-ocean-v2 \"\$DATA_ROOT/global-ocean\"

    python3 -c 'import torchinfo' 2>/dev/null || pip install --user --no-deps torchinfo 2>&1 | tail -3
    python3 -c 'import torch, torch_harmonics, zarr, xarray; print(\"torch:\", torch.__version__, \"th:\", torch_harmonics.__version__, \"zarr:\", zarr.__version__, \"xr:\", xarray.__version__)'

    python3 -m pytest fots/data/tests/test_global_ocean.py -x -q
  "
