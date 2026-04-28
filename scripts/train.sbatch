#!/bin/bash
#SBATCH --job-name=fots-train
#SBATCH --partition=dokmanic-h200
#SBATCH --qos=dokmanic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Train (or test) one (model, lr) cell of the PlanetSWE baseline sweep.
# Submit via scripts/submit_train_and_test.sh, which sets:
#     MODEL_CONFIG, DATA_CONFIG, TRAIN_CONFIG, LR, BATCH_SIZE, NAME
# and TEST_MODE=true on the dependent test job. Walltime is set per-job
# from the submit script (--time=24:00:00 train, --time=02:00:00 test).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

MODEL_CONFIG="${MODEL_CONFIG:-configs/models/sfno.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/planetswe.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_4-to-1.yaml}"
LR="${LR:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NAME="${NAME:-planetswe}"

if [ "${TEST_MODE:-false}" = "true" ]; then
    uv run --no-sync python -m fots.train \
        --data "$DATA_CONFIG" --model "$MODEL_CONFIG" --train "$TRAIN_CONFIG" \
        optimizer.lr="$LR" \
        data.batch_size="$BATCH_SIZE" \
        name="$NAME" \
        test_mode=true \
        auto_resume=true
else
    uv run --no-sync python -m fots.train \
        --data "$DATA_CONFIG" --model "$MODEL_CONFIG" --train "$TRAIN_CONFIG" \
        optimizer.lr="$LR" \
        data.batch_size="$BATCH_SIZE" \
        name="$NAME"
fi
