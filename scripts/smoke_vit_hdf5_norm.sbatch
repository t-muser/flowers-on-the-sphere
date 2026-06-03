#!/bin/bash
#SBATCH --job-name=fots-smoke-vit-norm
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
# Confirm the inlined ZScore stats actually engage (the loader silently
# "proceeds without normalization" on any stats mismatch). Runs a short
# 1-epoch vit train on BOTH mickelin (scalar field) and shock_caps (scalar
# + vector field) HDF5 configs. Expect: NO "Proceeding without
# normalization" warning, and loss on the O(1) scale (raw vorticity gave
# ~1e-4 unnormalized).

set -euo pipefail
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
TOMLPATH="${TOMLPATH:-${WORKDIR}/scripts/torchcontainer_daint.toml}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/iopsstor/scratch/cscs/${USER}/fots-runs}"
# Space-separated dataset list; override to smoke a single dataset, e.g.
# SMOKE_DATASETS=cahn_hilliard sbatch scripts/smoke_vit_hdf5_norm.sbatch
SMOKE_DATASETS="${SMOKE_DATASETS:-mickelin shock_caps}"

srun --ntasks=1 --ntasks-per-node=1 --gpus=1 \
     --export=ALL --environment="${TOMLPATH}" \
     -u -l bash -c "
set -euo pipefail
cd '${WORKDIR}'
export PYTHONUNBUFFERED=1
export PYTHONPATH=\"${WORKDIR}:\${PYTHONPATH:-}\"

python3 -c 'import torchinfo' 2>/dev/null || pip install --user --no-deps torchinfo 2>&1 | tail -3
python3 -c 'import timm' 2>/dev/null || pip install --user timm 2>&1 | tail -5
python3 -c 'import h5py, the_well' 2>/dev/null || pip install --user --no-deps h5py 'the-well>=1.1.0' 2>&1 | tail -5

for ds in ${SMOKE_DATASETS}; do
  echo
  echo \"================ NORM SMOKE: \${ds} ================\"
  torchrun --standalone --nnodes=1 --nproc_per_node=1 \\
      -m fots.train \\
      --data configs/data/\${ds}_hdf5.yaml \\
      --model configs/models/vit.yaml \\
      --train configs/train_4-to-1.yaml \\
      optimizer.lr=5e-4 \\
      data.batch_size=8 \\
      data.restrict_num_trajectories=8 \\
      name=smoke_norm_\${ds} \\
      experiment_dir='${EXPERIMENT_DIR}' \\
      trainer.epochs=1 \\
      trainer.ar_epochs=0 \\
      trainer.rollout_val_frequency=999 \\
      trainer.checkpoint_frequency=999 \\
      wandb.enabled=false
done
"

echo "norm smoke finished at: $(date)"
