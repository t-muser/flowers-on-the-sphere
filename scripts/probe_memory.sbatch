#!/bin/bash
#SBATCH --job-name=fots-probe
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
# Sweep peak GPU memory of the AR(x2) training step across {model, batch}
# on one H200 (141 GiB). Synthetic inputs match PlanetSWE shapes
# (12 -> 3 channels, 256x512). Prints a table; OOMs are non-fatal.
#
# Batch ranges chosen to bracket >90% VRAM (>=127 GiB peak_reserved) per
# model, scaling linearly from the previous probe (see BATCH_SIZE table
# in scripts/submit_train_and_test.sh).

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

# model:batch1,batch2,... — sweep ascending; OOMs are non-fatal.
SWEEP=(
    "fno:36,40,44,48"
    "sfno:36,40,44,48"
    "flower:34,36,38,40"
    "zinnia:16,20,24,28,32"
    "local_r_transformer:12,14,16,18"
    "local_s2_transformer:6,8,10,12"
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
            --inp-shape 256 512 --inp-chans 12 --out-chans 3 \
            --ar-steps 2 --iters 3 \
            || echo "[probe] ${model} bs=${bs} failed (likely OOM)"
    done
done
