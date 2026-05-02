#!/bin/bash
#SBATCH --job-name=fots-probe-mickelin
#SBATCH --partition=dokmanic-h200
#SBATCH --qos=dokmanic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Sweep peak GPU memory of the AR(x2) training step at the Mickelin
# shape: 128x256 equiangular, scalar field (in=4, out=1).
# Batch ranges are scaled ~4x from the PlanetSWE probe to bracket OOM.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

SWEEP=(
    "fno:128,144,160,176,192"
    "sfno:128,144,160,176,192"
    "flower:112,128,144,160,176"
    "zinnia:48,56,64,72,80,88"
    "local_r_transformer:40,48,56,64,72"
    "local_s2_transformer:20,24,28,32,36,40"
)

for entry in "${SWEEP[@]}"; do
    model="${entry%%:*}"
    batches="${entry#*:}"
    IFS=',' read -ra BS_LIST <<< "$batches"
    for bs in "${BS_LIST[@]}"; do
        echo "=================================================="
        echo "model=${model} batch=${bs}"
        echo "=================================================="
        uv run --no-sync python scripts/probe_memory.py \
            "configs/models/${model}.yaml" \
            --batch-size "$bs" \
            --inp-shape 128 256 --inp-chans 4 --out-chans 1 \
            --ar-steps 2 --iters 3 \
            || echo "[probe] ${model} bs=${bs} failed (likely OOM)"
    done
done
