#!/bin/bash
#SBATCH --job-name=fots-smoke-held-suarez
#SBATCH --account=uba03
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# In-container smoke for fots.data.held_suarez.HeldSuarezDataModule against
# the v2 store. Mirrors smoke_global_ocean.sbatch — verifies the per-variable
# zarr schema (u/v/T over levels + ps) is correctly stacked into a (T, 25,
# 64, 128) `fields` array and that stats.json normalization is wired up.

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

    python3 -c 'import torchinfo' 2>/dev/null || pip install --user --no-deps torchinfo 2>&1 | tail -3
    python3 -c 'import torch, zarr, xarray; print(\"torch:\", torch.__version__, \"zarr:\", zarr.__version__, \"xr:\", xarray.__version__)'

    python3 - <<'PY'
import numpy as np
from fots.data.held_suarez import HeldSuarezDataModule

field_names = (
    'u_50hpa','u_100hpa','u_250hpa','u_500hpa','u_700hpa','u_850hpa','u_925hpa','u_1000hpa',
    'v_50hpa','v_100hpa','v_250hpa','v_500hpa','v_700hpa','v_850hpa','v_925hpa','v_1000hpa',
    'T_50hpa','T_100hpa','T_250hpa','T_500hpa','T_700hpa','T_850hpa','T_925hpa','T_1000hpa',
    'ps',
)
dm = HeldSuarezDataModule(
    root='/iopsstor/scratch/cscs/tmuser/PDEDatasets/SphericalPDEs/held-suarez-3d-v2',
    time_steps_per_run=365,
    dim_in=25, dim_out=25,
    spatial_resolution=(64, 128),
    field_names=field_names,
    n_steps_input=4, n_steps_output=1,
    batch_size=2, num_workers=0,
)
print('splits:', len(dm.train_dataset), len(dm.val_dataset), len(dm.test_dataset))

x = dm.train_dataset[0]['input_fields']
y = dm.train_dataset[0]['output_fields']
print('input :', tuple(x.shape), x.dtype, 'mean', float(x.mean()), 'std', float(x.std()))
print('output:', tuple(y.shape), y.dtype)
assert tuple(x.shape) == (4, 25, 64, 128), x.shape
assert tuple(y.shape) == (1, 25, 64, 128), y.shape

# Per-channel std should be near 1 across the dataset; on a single window
# it varies but should be O(1). Catch grossly-wrong stats wiring.
per_ch_std = x.std(dim=(0,2,3)).numpy()
print('per-channel std (first sample):', np.array2string(per_ch_std, precision=2))
assert per_ch_std.min() > 0.05, f'channel near-zero std: {per_ch_std}'
assert per_ch_std.max() < 20.0, f'channel huge std (norm broken?): {per_ch_std}'

# Pull one full batch via the train DataLoader to verify the collator path.
loader = dm.train_dataloader()
batch = next(iter(loader))
print('batch input :', tuple(batch['input_fields'].shape))
print('batch output:', tuple(batch['output_fields'].shape))
assert tuple(batch['input_fields'].shape) == (2, 4, 25, 64, 128)
assert tuple(batch['output_fields'].shape) == (2, 1, 25, 64, 128)
print('SMOKE OK')
PY
  "
