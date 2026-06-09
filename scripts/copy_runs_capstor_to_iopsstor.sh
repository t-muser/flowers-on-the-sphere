#!/bin/bash
# Stage existing fots-runs directories from /capstor → /iopsstor so jobs
# launched with EXPERIMENT_DIR=/iopsstor/... can find their checkpoints.
#
# Covers two buckets identified during capstor outage planning:
#   B) crashed train+test cells (auto_resume from surviving checkpoints)
#   C) test-only cells (full train finished, only test job is missing)
# both filtered to galewsky / cahn_hilliard / mickelin and dropping
# zinnia / dahlia / dandelion (rerun fresh or out of scope).
#
# rsync makes this idempotent — safe to rerun, partial copies resume.
# Run while /capstor is at least readable; aborts on error per cell but
# keeps going.

set -uo pipefail
cd "$(dirname "$0")/.."

SRC=/capstor/scratch/cscs/${USER}/fots-runs
DST=/iopsstor/scratch/cscs/${USER}/fots-runs
mkdir -p "$DST"

# Shell model name -> CamelCase directory token.
declare -A MODEL_DIR=(
    [flower]=Flower2D
    [fno]=Fno
    [sfno]=Sfno
    [local_r_transformer]=LocalRTransformer
    [local_s2_transformer]=LocalS2Transformer
)

# LR shorthand -> decimal used in run dir names.
declare -A LR_DEC=(
    [1e-4]=0.0001
    [5e-4]=0.0005
    [1e-3]=0.001
)

# Bucket B — crashed train+test (need full run dir for auto_resume + test).
CELLS_B=(
    # cahn_hilliard
    "flower    cahn_hilliard 5e-4"
    "fno       cahn_hilliard 1e-3"
    # galewsky
    "local_r_transformer  galewsky 1e-3"
    "local_s2_transformer galewsky 1e-4"
    "local_s2_transformer galewsky 5e-4"
    "local_s2_transformer galewsky 1e-3"
    "sfno                 galewsky 5e-4"
    "sfno                 galewsky 1e-3"
    # mickelin
    "flower              mickelin 5e-4"
    "fno                 mickelin 1e-4"
    "local_r_transformer mickelin 1e-3"
)

# Bucket C — train finished, only test missing.
CELLS_C=(
    # cahn_hilliard (8)
    "flower              cahn_hilliard 1e-4"
    "flower              cahn_hilliard 1e-3"
    "fno                 cahn_hilliard 1e-4"
    "fno                 cahn_hilliard 5e-4"
    "local_r_transformer cahn_hilliard 5e-4"
    "sfno                cahn_hilliard 1e-4"
    "sfno                cahn_hilliard 5e-4"
    "sfno                cahn_hilliard 1e-3"
    # galewsky (7)
    "flower    galewsky 1e-4"
    "flower    galewsky 5e-4"
    "flower    galewsky 1e-3"
    "fno       galewsky 1e-4"
    "fno       galewsky 5e-4"
    "fno       galewsky 1e-3"
    "sfno      galewsky 1e-4"
    # mickelin (9)
    "flower              mickelin 1e-4"
    "flower              mickelin 1e-3"
    "fno                 mickelin 5e-4"
    "fno                 mickelin 1e-3"
    "local_r_transformer mickelin 1e-4"
    "local_r_transformer mickelin 5e-4"
    "sfno                mickelin 1e-4"
    "sfno                mickelin 5e-4"
    "sfno                mickelin 1e-3"
)

CELLS=("${CELLS_B[@]}" "${CELLS_C[@]}")

ok=0; missing=0; failed=0
for cell in "${CELLS[@]}"; do
    read -r model dataset lr <<< "$cell"
    mclass="${MODEL_DIR[$model]:-}"
    lrdec="${LR_DEC[$lr]:-}"
    if [ -z "$mclass" ] || [ -z "$lrdec" ]; then
        echo "SKIP unknown mapping: $cell" >&2
        failed=$((failed+1))
        continue
    fi
    name="${dataset}-${dataset}-${mclass}-${lrdec}"
    if [ ! -d "$SRC/$name" ]; then
        echo "MISSING on capstor: $name"
        missing=$((missing+1))
        continue
    fi
    echo "rsync  $name"
    if rsync -a --info=stats1 "$SRC/$name/" "$DST/$name/"; then
        ok=$((ok+1))
    else
        echo "FAILED rsync: $name" >&2
        failed=$((failed+1))
    fi
done

echo
echo "copied: $ok   missing: $missing   failed: $failed   total: ${#CELLS[@]}"
