#!/bin/bash
#SBATCH --job-name=fots-train
#SBATCH --account=uba03
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Daint port of train.sbatch. Runs fots.train inside the aurora-arm64
# modulus container (torch 2.5 + CUDA + most fots deps preinstalled).
# A small set of missing python deps (torchinfo, etc.) is installed once
# into a persistent --user site under PYTHONUSERBASE, surviving across
# jobs.
#
# Submit via scripts/submit_train_and_test_daint.sh, which sets
#     MODEL_CONFIG, DATA_CONFIG, TRAIN_CONFIG, LR, BATCH_SIZE, NAME
# and TEST_MODE=true on the dependent test job.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
TOMLPATH="${TOMLPATH:-${WORKDIR}/scripts/torchcontainer_daint.toml}"

MODEL_CONFIG="${MODEL_CONFIG:-configs/models/sfno.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/galewsky.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_4-to-1.yaml}"
LR="${LR:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NAME="${NAME:-galewsky}"

# Pip --user installs done inside the container land in /users/$USER/.local
# (the default PYTHONUSERBASE). torch_harmonics 0.8 is already pre-installed
# there from a prior makani setup; we only need to add the few packages
# missing from the modulus container.

# WandB credentials, if any.
if [ -f "/users/${USER}/.wandb_env" ]; then
    # shellcheck source=/dev/null
    source "/users/${USER}/.wandb_env"
fi

# Single-node DDP across 4 GH200 GPUs via torchrun. No aws-ofi-nccl on
# Daint (Clariden-only hook), so disable the plugin lookup; for one node
# NCCL stays on NVLink + shared memory.
export NCCL_NET_PLUGIN=""
export NCCL_IB_DISABLE=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# /iopsstor is fast enough on Daint that the loader streams from the
# config'd `data.root` directly — no node-local staging needed (unlike
# scicore where the original sbatch had to cp into $TMPDIR).
DATA_OVERRIDE=""

echo "================= SLURM Info ================="
echo "Run started at:    $(date)"
echo "SLURM_JOB_ID:      $SLURM_JOB_ID"
echo "MODEL_CONFIG:      $MODEL_CONFIG"
echo "DATA_CONFIG:       $DATA_CONFIG"
echo "TRAIN_CONFIG:      $TRAIN_CONFIG"
echo "NAME:              $NAME"
echo "BATCH_SIZE:        $BATCH_SIZE   LR: $LR"
echo "TEST_MODE:         ${TEST_MODE:-false}"
echo "DATA_OVERRIDE:     ${DATA_OVERRIDE}"
echo "================================================"

srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=288 \
     --export=ALL \
     --environment="${TOMLPATH}" \
     -u -l \
  bash -c "
    set -euo pipefail
    cd '${WORKDIR}'
    export PYTHONPATH=\"${WORKDIR}:\${PYTHONPATH:-}\"

    # Install once-only extras the modulus container is missing.
    python3 -c 'import torchinfo' 2>/dev/null || pip install --user --no-deps torchinfo 2>&1 | tail -3

    python3 -c 'import torch; print(\"torch:\", torch.__version__, \"cuda:\", torch.cuda.is_available())'
    python3 -c 'import torch_harmonics; print(\"th:\", torch_harmonics.__version__)'
    nvidia-smi -L

    # 4 ranks per node (one per GH200 GPU). torchrun sets RANK / LOCAL_RANK /
    # WORLD_SIZE / MASTER_ADDR / MASTER_PORT for every rank — fots.dist
    # picks them up to init NCCL.
    if [ '${TEST_MODE:-false}' = 'true' ]; then
        torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
            -m fots.train \\
            --data '${DATA_CONFIG}' --model '${MODEL_CONFIG}' --train '${TRAIN_CONFIG}' \\
            optimizer.lr='${LR}' \\
            data.batch_size='${BATCH_SIZE}' \\
            name='${NAME}' \\
            test_mode=true \\
            auto_resume=true
    else
        torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
            -m fots.train \\
            --data '${DATA_CONFIG}' --model '${MODEL_CONFIG}' --train '${TRAIN_CONFIG}' \\
            optimizer.lr='${LR}' \\
            data.batch_size='${BATCH_SIZE}' \\
            name='${NAME}' \\
            ${DATA_OVERRIDE}
    fi
  "

echo "Run finished at: $(date)"
