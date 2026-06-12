#!/bin/bash
#SBATCH --job-name=fots-smoke-held-suarez-hdf5
#SBATCH --account=uba04
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# In-container smoke for the HDF5 Held-Suarez (ClimaAtmos) path:
# SphericalHDF5Dataset with fold_levels_into_channels=true, driven through the
# shipped configs/data/held_suarez_hdf5.yaml. Verifies the (level,lat,lon)
# store folds to (T, 25, 144, 288) field-major, that metadata reports a 2-D
# 25-channel view, and that per-level normalization + folded denorm stats wire
# up. Counterpart to smoke_held_suarez.sbatch (which covers the zarr datamodule).

set -euo pipefail
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
TOMLPATH="${TOMLPATH:-${WORKDIR}/scripts/torchcontainer_daint_iopsstor.toml}"

echo "WORKDIR:  $WORKDIR"
echo "TOMLPATH: $TOMLPATH"

srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=72 \
     --export=ALL \
     --environment="${TOMLPATH}" \
     -u -l \
  bash -c "
    set -euo pipefail
    cd '${WORKDIR}'
    export PYTHONPATH=\"${WORKDIR}:\${PYTHONPATH:-}\"

    python3 -c 'import torch, h5py, hydra; print(\"torch:\", torch.__version__, \"h5py:\", h5py.__version__)'

    # Pure-logic fold tests + real-data fold smoke (the latter reads the
    # on-disk dataset; both live in the same module).
    python3 -m pytest -q fots/data/tests/test_spherical_hdf5_fold.py

    # Extra end-to-end check: a real batch through the train DataLoader, with
    # the channel-ordering assertion (ps must land at folded channel index 8).
    python3 - <<'PY'
from hydra.utils import instantiate
from omegaconf import OmegaConf

cfg = OmegaConf.load('configs/data/held_suarez_hdf5.yaml')
dm = instantiate(cfg.data)

md = dm.metadata
print('spatial_resolution:', md.spatial_resolution, 'dim_in:', md.dim_in,
      'dim_out:', md.dim_out, 'grid:', md.grid)
assert md.spatial_resolution == (144, 288), md.spatial_resolution
assert md.dim_out == 25, md.dim_out

names = md.field_names
print('folded channels:', names)
# Field-major: T_lev0..7 (0-7), ps (8), then velocity comps x 8 levels (9-24).
assert len(names) == 25, names
assert names[8] == 'ps', names[8]
assert all(n.startswith('T') for n in names[:8]), names[:8]

loader = dm.train_dataloader()
batch = next(iter(loader))
xi, xo = batch['input_fields'], batch['output_fields']
print('batch input :', tuple(xi.shape))
print('batch output:', tuple(xo.shape))
assert xi.shape[-3:] == (25, 144, 288), xi.shape
assert xo.shape[-3:] == (25, 144, 288), xo.shape

# Normalized data should be O(1) per channel; catch broken per-level stats.
per_ch_std = xi.float().std(dim=(0, 1, 3, 4)).numpy()
print('per-channel std min/max: %.3f / %.3f' % (per_ch_std.min(), per_ch_std.max()))
assert per_ch_std.min() > 0.05 and per_ch_std.max() < 20.0, per_ch_std

mean, std = dm.train_dataset.folded_denorm_stats
assert mean.numel() == 25 and std.numel() == 25
print('SMOKE OK')
PY
  "
