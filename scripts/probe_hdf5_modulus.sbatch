#!/bin/bash
#SBATCH --job-name=fots-probe-hdf5-modulus
#SBATCH --account=uba03
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Deciding test for reverting the container: does the HDF5 dataset path load
# inside the OLD aurora_arm64_modulus2412 container (Python 3.10, torch 2.5)?
# If yes, the pt-25.04 migration (done only to read zarr v3) is unnecessary
# for HDF5 and we can go back to train_daint.sbatch's modulus container.
#
# Uses the capstor modulus toml (the iopsstor sqsh is currently missing).
# Userbase is the modulus default ~/.local (where torch_harmonics is already
# preinstalled); we only add the_well + h5py.

set -euo pipefail
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
TOMLPATH="${TOMLPATH:-${WORKDIR}/scripts/torchcontainer_daint.toml}"

srun --ntasks=1 --ntasks-per-node=1 --gpus=1 \
     --export=ALL --environment="${TOMLPATH}" \
     -u -l bash -c "
set -euo pipefail
cd '${WORKDIR}'
export PYTHONUNBUFFERED=1
export PYTHONPATH=\"${WORKDIR}:\${PYTHONPATH:-}\"

echo '=== modulus container baseline ==='
python3 --version
python3 -c 'import torch; print(\"torch:\", torch.__version__, \"cuda:\", torch.cuda.is_available())'
python3 -c 'import zarr; print(\"zarr:\", zarr.__version__)'

echo
echo '=== add the_well + h5py to the modulus ~/.local userbase (--no-deps) ==='
python3 -c 'import h5py, the_well' 2>/dev/null || \
    pip install --user --no-deps h5py 'the-well>=1.1.0' 2>&1 | tail -8
python3 -c 'import h5py, the_well; print(\"h5py:\", h5py.__version__, \"the_well:\", the_well.__version__)'
echo

python3 scripts/_probe_hdf5.py
"

echo "Modulus HDF5 probe finished at: $(date)"
