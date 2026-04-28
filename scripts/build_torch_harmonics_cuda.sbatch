#!/bin/bash
#SBATCH --job-name=fots-build-th-cuda
#SBATCH --partition=dokmanic-h200
#SBATCH --qos=dokmanic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Build torch_harmonics from source with CUDA kernels enabled.
# PyPI's manylinux wheel ships only the CPU `_C.so`, so DISCO ops have no
# CUDA backend at runtime. We use the CUDA/13.0.0 module here to match the
# torch 2.11+cu130 runtime.
#
# Prereq: build deps must already be installed in the project venv (via
# `uv pip install setuptools wheel setuptools-scm` from the login node).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

source /scicore/soft/lmod/8.7.32/init/bash
module load CUDA/13.0.0

echo "nvcc: $(which nvcc)"
nvcc --version | tail -1
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"

FORCE_CUDA_EXTENSION=1 \
TORCH_CUDA_ARCH_LIST="9.0" \
uv pip install --no-deps --no-build-isolation --reinstall --offline \
    third_party/torch-harmonics

echo
echo "=== Verify CUDA symbols in rebuilt _C.so ==="
uv run --no-sync python - <<'PY'
import os, torch_harmonics
p = os.path.join(os.path.dirname(torch_harmonics.__file__), "disco", "_C.cpython-312-x86_64-linux-gnu.so")
print("disco _C:", p)
import subprocess
out = subprocess.run(["nm", "-D", p], capture_output=True, text=True).stdout
cuda_syms = [l for l in out.splitlines() if "cuda" in l.lower() or "cu_fwd" in l.lower() or "cu_bwd" in l.lower()]
print(f"CUDA-related symbols: {len(cuda_syms)} (first 5):")
for l in cuda_syms[:5]: print(" ", l)
PY
