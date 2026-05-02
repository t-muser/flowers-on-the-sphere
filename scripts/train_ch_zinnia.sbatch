#!/bin/bash
#SBATCH --job-name=fots-ch-zinnia
#SBATCH --partition=dokmanic-h200
#SBATCH --qos=dokmanic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

source scripts/batch_sizes.sh
BATCH_SIZE=$(fots_batch_size zinnia cahn_hilliard)

SRC=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/cahn-hilliard-sphere
SCRATCH="${TMPDIR}/cahn-hilliard-sphere"
cleanup() { rm -rf "$SCRATCH" 2>/dev/null; }
trap cleanup EXIT

echo "Copying dataset to ${SCRATCH} ..."
cp -r "$SRC" "$SCRATCH"
echo "Copy complete ($(du -sh "$SCRATCH" | cut -f1))."

uv run --no-sync python -m fots.train \
    --data  configs/data/cahn_hilliard.yaml \
    --model configs/models/zinnia.yaml \
    --train configs/train_4-to-1.yaml \
    data.root="$SCRATCH" \
    data.batch_size="$BATCH_SIZE" \
    name=ch-zinnia
