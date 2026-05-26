# Held-Suarez producer (ClimaAtmos.jl)

Replaces the cubed-sphere MITgcm path (`datagen/mitgcm/held_suarez_cs/`).
ClimaAtmos uses a spectral-element method on the cubed sphere; the
finite-volume cube-corner instability that killed the MITgcm pipeline at
cs64 does not exist there.

The Python orchestration mirrors the MITgcm producer; only the inner
solver call is replaced by a small Julia driver (`run_hs.jl`).

## Layout

```
.
├── run_hs.jl                   Julia driver (corner JSON → AtmosSimulation → NetCDF)
├── postprocess.py              ClimaAtmos NetCDF → (time, level, lat, lon) Zarr
├── env/Project.toml            Pinned Julia env
├── configs/
│   ├── resolution/he12ze31.yml he4-h16ze31 resolution overrides
│   └── resolution/he16ze31.yml
├── scripts/
│   ├── install_env.sh          One-time Pkg.instantiate + precompile
│   ├── run.py                  Single-run driver (mirrors mitgcm/.../run.py)
│   ├── preflight.py            32-corner stability sweep
│   └── finalize_dataset.py     Splits + stats (defers to held_suarez/finalize_3d)
├── slurm/
│   ├── install_env.sbatch
│   ├── phase0_smoke.sbatch
│   ├── phase0_scoping.sbatch
│   ├── preflight_array.sbatch
│   └── production_array.sbatch
└── tests/
    ├── test_param_override.py  Override-keys land on forcing tendency
    ├── test_postprocess.py     Zarr matches consumer schema
    └── test_e2e_smoke.py       Full one-corner end-to-end
```

## One-time setup

1. Clone the pinned ClimaAtmos source and apply our IC-seed patch:
   ```
   git clone --depth 1 --branch v0.39.0 \
       https://github.com/CliMA/ClimaAtmos.jl external/ClimaAtmos.jl
   cd external/ClimaAtmos.jl && git apply \
       ../../datagen/clima_atmos/held_suarez/env/source_patch.diff
   ```
   The patch at `env/source_patch.diff` replaces `DecayingProfile`'s
   deterministic `0.1·sind(long)·mask(z<5km)` temperature bump with
   per-cell Gaussian noise seeded by `hash(seed, lat, long, z)` when
   the `HS_IC_SEED` env var is set; behavior is bit-identical to stock
   when it's unset. `run_hs.jl` sets this var from each corner's
   `seed` value so the 6-seed-per-regime design has independent IC
   realizations. (The HS forcing-timescale patch from earlier revisions
   is still in the diff but inactive — the current sweep does not vary
   k_a/k_s/k_f.)

2. Install the Julia environment in two phases (scicore compute nodes
   are blocked from `pkg.julialang.org` / `github.com`, see memory):
   ```
   # 2a. Download + resolve on the LOGIN node (network OK):
   PATH=/scicore/soft/easybuild/apps/Julia/1.10.8-linux-x86_64/bin:$PATH \
   JULIA_DEPOT_PATH=$PWD/datagen/clima_atmos/held_suarez/env/.julia_depot \
   INSTALL_PHASE=instantiate \
       bash datagen/clima_atmos/held_suarez/scripts/install_env.sh

   # 2b. Precompile on a compute node (slow; no network):
   sbatch datagen/clima_atmos/held_suarez/slurm/install_env.sbatch
   ```
   The committed `env/Manifest.toml` pins resolved package versions so
   subsequent installs are reproducible. The depot under
   `env/.julia_depot/` is gitignored.

## Phase 0 — Skeptic phase

```
# 0.2 smoke run at the stock he6ze31 × 2 days config
sbatch datagen/clima_atmos/held_suarez/slurm/phase0_smoke.sbatch

# 0.3 resolution + duration scoping at he24ze31 on GPU
sbatch datagen/clima_atmos/held_suarez/slurm/phase0_scoping.sbatch

# 0.4-0.5 validated by pytest
uv run pytest datagen/clima_atmos/held_suarez/tests/test_postprocess.py -v
# Integration tests need julia + env installed + RUN_CLIMA_INTEGRATION=1
```

The go/no-go gate is laid out in `docs/held_suarez_clima.md`.

## Phase 1 — Preflight

```
# Generate 8 corner configs (axis-extreme physical regimes, seed=0)
uv run --project datagen python \
    -m datagen.clima_atmos.held_suarez.scripts.preflight generate

# Submit
sbatch datagen/clima_atmos/held_suarez/slurm/preflight_array.sbatch
```

## Phase 2 — Production

```
# Generate the 162 production configs (3 physical x 6 seed sweep)
uv run --project datagen python \
    -m datagen.clima_atmos.held_suarez.scripts.generate_sweep \
    --out datagen/clima_atmos/held_suarez/configs

# Submit
sbatch datagen/clima_atmos/held_suarez/slurm/production_array.sbatch

# Finalize when done
uv run --project datagen python \
    -m datagen.clima_atmos.held_suarez.scripts.finalize_dataset \
    --root "$DATA_ROOT/held-suarez-clima"
```

## Output schema

Identical to the MITgcm HS-3D dataset (`datagen/mitgcm/held_suarez/`):

  - Per-run store at `<split>/run_XXXX.zarr` with vars
    `u, v, T (time, level, lat, lon)` + `ps (time, lat, lon)`
  - 8 ERA5-standard pressure levels: 50, 100, 250, 500, 700, 850, 925, 1000 hPa
  - `time` in seconds since simulation start, `lat`/`lon` in degrees
  - `param_*` attrs from the corner JSON
  - `splits.json` + `stats.json` at the root

This means `fots.data.held_suarez.HeldSuarezDataModule` consumes
ClimaAtmos and MITgcm runs with **zero code changes**. The contract is
checked by `tests/test_postprocess.py::TestDataModuleCompat`.
