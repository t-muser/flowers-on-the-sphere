#!/bin/bash
#SBATCH --job-name=fots-smoke-vit-hdf5
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
# De-risk the real sweep: run the FULL training path the 6-job vit sweep
# will use — modulus container (capstor toml), the_well + h5py + timm from
# ~/.local, ViT model, mickelin HDF5 data — for one short epoch on a single
# GPU before committing 6x24h jobs. restrict_num_trajectories keeps the
# epoch tiny.

set -euo pipefail
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
TOMLPATH="${TOMLPATH:-${WORKDIR}/scripts/torchcontainer_daint.toml}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/iopsstor/scratch/cscs/${USER}/fots-runs}"

srun --ntasks=1 --ntasks-per-node=1 --gpus=1 \
     --export=ALL --environment="${TOMLPATH}" \
     -u -l bash -c "
set -euo pipefail
cd '${WORKDIR}'
export PYTHONUNBUFFERED=1
export PYTHONPATH=\"${WORKDIR}:\${PYTHONPATH:-}\"

# Same install block as train_daint.sbatch.
python3 -c 'import torchinfo' 2>/dev/null || pip install --user --no-deps torchinfo 2>&1 | tail -3
python3 -c 'import timm' 2>/dev/null || pip install --user timm 2>&1 | tail -5
python3 -c 'import h5py, the_well' 2>/dev/null || pip install --user --no-deps h5py 'the-well>=1.1.0' 2>&1 | tail -5

python3 -c 'import torch, timm, h5py, the_well, torch_harmonics; print(\"torch\", torch.__version__, \"timm\", timm.__version__, \"th\", torch_harmonics.__version__)'
nvidia-smi -L | head -1
echo

echo '=== single-GPU vit + mickelin HDF5 smoke (1 epoch, 8 trajectories) ==='
torchrun --standalone --nnodes=1 --nproc_per_node=1 \\
    -m fots.train \\
    --data configs/data/mickelin_hdf5.yaml \\
    --model configs/models/vit.yaml \\
    --train configs/train_4-to-1.yaml \\
    optimizer.lr=5e-4 \\
    data.batch_size=8 \\
    data.restrict_num_trajectories=8 \\
    name=smoke_vit_hdf5 \\
    experiment_dir='${EXPERIMENT_DIR}' \\
    trainer.epochs=1 \\
    trainer.ar_epochs=0 \\
    trainer.rollout_val_frequency=999 \\
    trainer.checkpoint_frequency=1 \\
    wandb.enabled=false
"

echo "vit+HDF5 smoke finished at: $(date)"
