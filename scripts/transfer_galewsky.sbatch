#!/bin/bash
#SBATCH --job-name=xfer-galewsky
#SBATCH --account=uba03
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Pull the galewsky-sw zarr ensemble from scicore over ssh into our
# Daint scratch. Uses ~/.ssh/config alias `transfer-scicore` for the
# endpoint. Resumable: --partial keeps half-transferred zarrs across
# retries, and rsync's per-file checks make a re-run a no-op once
# everything is local.
#
# If the xfer partition is unavailable, fall back to running this on
# the login node:
#   bash scripts/transfer_galewsky.sbatch
# (the SBATCH directives are inert in that case).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

REMOTE="${REMOTE:-transfer-scicore:/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/galewsky-sw}"
LOCAL="${LOCAL:-/iopsstor/scratch/cscs/${USER}/PDEDatasets/SphericalPDEs/galewsky-sw}"

mkdir -p "$(dirname "$LOCAL")"

echo "Transferring ${REMOTE} -> ${LOCAL}"
echo "Started at: $(date)"
T0=$(date +%s)

# -a: archive (recursion, perms, times). Compression off (-z) since zarr is
# already chunked binary. --info=progress2 for a single ETA line.
rsync -a --partial --info=progress2 \
    "${REMOTE}/" "${LOCAL}/"

T1=$(date +%s)
echo "Done at: $(date) (elapsed: $((T1 - T0))s)"
echo "Local size: $(du -sh "${LOCAL}" | cut -f1)"
echo "Top-level entries:"
ls -la "${LOCAL}" | head -20
