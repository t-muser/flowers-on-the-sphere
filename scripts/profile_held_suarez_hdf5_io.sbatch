#!/bin/bash
#SBATCH --job-name=fots-profile-hs-hdf5-io
#SBATCH --account=uba04
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Attribute held_suarez_hdf5 per-sample data-loading cost so we stop guessing.
# Single process (no workers): measures one worker's throughput and splits it
# into (A) full __getitem__, (B) file open+close, (C) window read+decompress,
# plus a raw-vs-decompressed read to isolate gzip. Aggregate throughput with W
# workers ~= W / (A) until a ceiling (compute / FS bandwidth).

set -euo pipefail
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
# /iopsstor container copy is gone; use the /capstor image.
TOMLPATH="${TOMLPATH:-${WORKDIR}/scripts/torchcontainer_daint.toml}"

srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=72 \
     --export=ALL --environment="${TOMLPATH}" -u -l \
  bash -c "
    set -euo pipefail
    cd '${WORKDIR}'
    export PYTHONPATH=\"${WORKDIR}:\${PYTHONPATH:-}\"
    python3 -c 'import hdf5plugin' 2>/dev/null || pip install --user --no-deps hdf5plugin 2>&1 | tail -2
    python3 - <<'PY'
import hdf5plugin  # noqa: F401  (registers LZ4 read filter)
import glob, time
import numpy as np
import h5py
from omegaconf import OmegaConf
from hydra.utils import instantiate

ROOT = '/iopsstor/scratch/cscs/tmuser/PDEDatasets/SphericalPDEs/held-suarez-clima-hdf5'
files = sorted(glob.glob(f'{ROOT}/data/train/*.h5'))
f0 = files[0]
N = 50
rng = np.random.RandomState(0)

# --- on-disk layout: chunking + codec (does it even chunk/compress?) -------
print('=== dataset layout ===')
with h5py.File(f0, 'r') as f:
    nt = None
    for grp in ('t0_fields', 't1_fields'):
        for name in f[grp].attrs.get('field_names', []):
            d = f[grp][name]
            nt = nt or d.shape[1]
            print(f'  {grp}/{name}: shape={d.shape} chunks={d.chunks} '
                  f'compression={d.compression} opts={d.compression_opts}')
print(f'  timesteps/run={nt}')

# --- A) full pipeline: real per-sample cost (open+read+norm+pad+fold) ------
cfg = OmegaConf.load('configs/data/held_suarez_hdf5.yaml')
dm = instantiate(cfg.data)
ds = dm.train_dataset
idx = rng.randint(0, len(ds), size=N)
ds[int(idx[0])]  # warm imports/JIT
t = time.time()
for i in idx:
    _ = ds[int(i)]
A = (time.time() - t) / N
print(f'\\nA) full __getitem__         : {A*1e3:7.1f} ms/sample')

# --- B) file open+close only (the per-sample re-open overhead) -------------
t = time.time()
for _ in range(N):
    with h5py.File(f0, 'r') as f:
        pass
B = (time.time() - t) / N
print(f'B) h5py open+close          : {B*1e3:7.1f} ms/open')

# --- C) window read+decompress, open amortized -----------------------------
W = 6  # n_steps_input(4)+n_steps_output(2)
with h5py.File(f0, 'r') as f:
    T, ps, vel = f['t0_fields']['T'], f['t0_fields']['ps'], f['t1_fields']['velocity']
    nt = T.shape[1]
    t = time.time()
    for _ in range(N):
        s = rng.randint(0, nt - W)
        _ = T[0, s:s+W]; _ = ps[0, s:s+W]; _ = vel[0, s:s+W]
    C = (time.time() - t) / N
print(f'C) window read+decompress   : {C*1e3:7.1f} ms/window (open amortized)')

# --- decompress isolation: raw compressed bytes vs decompressed read -------
# read_direct_chunk returns the stored (compressed) bytes WITHOUT decoding;
# normal [] read decodes. Delta ~= gzip decompression cost.
print('\\n=== decompress isolation (velocity) ===')
with h5py.File(f0, 'r') as f:
    d = f['t1_fields']['velocity']
    ch = d.chunks
    if ch is None:
        print('  not chunked -> not compressed; gzip is not the cost')
    else:
        # iterate chunks covering one timestep slab
        import itertools
        offs = []
        for c0 in range(0, min(ch[1]*4, d.shape[1]), ch[1]):
            offs.append((0, c0) + tuple(0 for _ in d.shape[2:]))
        raw = dec = 0.0; nb = 0
        for off in offs:
            t = time.time(); _, blob = d.id.read_direct_chunk(off); raw += time.time()-t
            nb += len(blob)
            sl = tuple(slice(o, o+cs) for o, cs in zip(off, ch))
            t = time.time(); arr = d[sl]; dec += time.time()-t
        mb = nb/1e6
        print(f'  {len(offs)} chunks, {mb:.0f} MB compressed')
        print(f'  raw chunk read (IO only)    : {raw*1e3:7.1f} ms -> {mb/raw:6.0f} MB/s')
        print(f'  full read (IO+decompress)   : {dec*1e3:7.1f} ms')
        print(f'  decompress share            : {100*(dec-raw)/dec:5.1f}% of read time')

print('\\n=== verdict ===')
print(f'  re-open overhead (B/A)      : {100*B/A:5.1f}% of per-sample time')
print(f'  read+decompress (C/A)       : {100*C/A:5.1f}% of per-sample time')
print('  (B large -> port ocean handle-cache; decompress share large -> recodec)')
PY
  "
