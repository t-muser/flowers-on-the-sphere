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
    # Stage dataset on the node-local scratch ($TMPDIR) — reads from the
    # centralized GPFS path are too slow to keep the GPU fed.
    SRC_BASE=$(awk '/^\s*well_base_path:/ {print $2; exit}' "$DATA_CONFIG")
    DATASET_NAME=$(awk '/^\s*well_dataset_name:/ {print $2; exit}' "$DATA_CONFIG")
    SCRATCH_BASE="${TMPDIR}/the-well/datasets"
    cleanup() { rm -rf "${TMPDIR}/the-well" 2>/dev/null; }
    trap cleanup EXIT

    mkdir -p "$SCRATCH_BASE"
    echo "Copying ${SRC_BASE}/${DATASET_NAME} to ${SCRATCH_BASE}/ ..."
    cp -r "${SRC_BASE}/${DATASET_NAME}" "${SCRATCH_BASE}/"
    echo "Data copy complete ($(du -sh ${SCRATCH_BASE}/${DATASET_NAME} | cut -f1))."

    uv run --no-sync python -m fots.train \
        --data "$DATA_CONFIG" --model "$MODEL_CONFIG" --train "$TRAIN_CONFIG" \
        optimizer.lr="$LR" \
        data.batch_size="$BATCH_SIZE" \
        data.well_base_path="$SCRATCH_BASE" \
        name="$NAME"
fi
