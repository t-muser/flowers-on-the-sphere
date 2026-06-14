#!/bin/bash
#SBATCH --job-name=hs-predict-cutaway
#SBATCH --partition=dokmanic-h200
#SBATCH --qos=dokmanic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Autoregressive rollout + cutaway-sphere MP4 for a trained Held-Suarez
# (ClimaAtmos) model. Inference is seconds; the matplotlib 3-D rendering of the
# pred|truth frames is the bulk of the wall-clock.
#
# Submit from this worktree (origin/master + the fold feature):
#     CKPT=checkpoints/held_suarez_clima-held_suarez_hdf5-Flower2D-0.0005/best.pt \
#     sbatch scripts/predict_cutaway.sbatch
#
set -euo pipefail
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

CKPT="${CKPT:?set CKPT=path/to/best.pt}"
STEPS="${STEPS:-160}"
TRAJ="${TRAJ:-0}"
DPI="${DPI:-100}"
FPS="${FPS:-12}"
OUT_DIR="${OUT_DIR:-notebooks/figures/clima_predict_renders}"

uv run --no-sync python scripts/predict_cutaway.py \
    --ckpt "$CKPT" \
    --steps "$STEPS" \
    --traj-index "$TRAJ" \
    --dpi "$DPI" \
    --fps "$FPS" \
    --device cuda \
    --out-dir "$OUT_DIR"
