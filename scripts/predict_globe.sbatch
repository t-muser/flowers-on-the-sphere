#!/bin/bash
#SBATCH --job-name=predict-globe
#SBATCH --partition=dokmanic-h200,h200
#SBATCH --qos=dokmanic,h200-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Autoregressive rollout + rotating-globe MP4 (pred|truth) for a zarr
# spherical dataset, matching the dataset visualization notebooks.
#
# Submit from this worktree, e.g.:
#   DATASET=galewsky \
#   CKPT=/.../checkpoints/galewsky-galewsky-Flower2D-0.0005/best.pt \
#   sbatch scripts/predict_globe.sbatch
#   DATASET=shock_caps \
#   CKPT=/.../checkpoints/shock_caps-shock_caps-Flower2D-0.001/best.pt \
#   sbatch scripts/predict_globe.sbatch
#
set -euo pipefail
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

DATASET="${DATASET:?set DATASET=galewsky|shock_caps}"
CKPT="${CKPT:?set CKPT=path/to/best.pt}"
STEPS="${STEPS:-0}"          # 0 = full trajectory
DPI="${DPI:-100}"
FPS="${FPS:-12}"
OUT_DIR="${OUT_DIR:-notebooks/figures/clima_predict_renders}"

uv run --no-sync python scripts/predict_globe.py \
    --dataset "$DATASET" \
    --ckpt "$CKPT" \
    --steps "$STEPS" \
    --dpi "$DPI" \
    --fps "$FPS" \
    --device cuda \
    --out-dir "$OUT_DIR"
