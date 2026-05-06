# Held-Suarez Atmospheric GCM on the Sphere — 3-D variant

**One-line description.** A 72-member ensemble of dry Held-Suarez
atmospheric simulations on the sphere, integrated with MITgcm and
exported on **8 ERA5-standard isobaric levels** plus surface pressure.

**Relation to the 2-D dataset.** This is the volumetric companion to
[`docs/held_suarez.md`](held_suarez.md). The physics, IC perturbation
recipe, and two-phase spin-up/data-collection design are identical.
The differences are:

1. **Output schema.** The 2-D variant keeps a single 500 hPa slice;
   the 3-D variant keeps `u`, `v`, `T` at 8 isobaric levels.
2. **Parameter sweep is conservative.** The original 162-run sweep at
   `Δt = 45 s` had a 26 % CFL-failure rate concentrated in the
   weak-drag / strong-meridional-forcing corner. The 3-D variant drops
   the riskiest extremes (`delta_T_y = 80 K`, `delta_theta_z = 5 K`)
   and integrates at `Δt = 30 s` for ~50 % CFL margin. All 32 corner
   preflights and all 72 production runs completed without failures.
3. **Naming.** Dataset slug `held-suarez-3d`, separate from
   `held-suarez`.

Refer to the 2-D doc for the physical framework, mathematical
formulation, and IC details — those are unchanged.

## Associated resources

- **Paper.** Held, I. M. & Suarez, M. J. (1994). *A proposal for the
  intercomparison of the dynamical cores of atmospheric general
  circulation models*. Bulletin of the American Meteorological Society.
- **Data generator.** Till Muser (University of Basel), 2026.
- **Generation software.** [MITgcm](../mitgcm) with the Held-Suarez
  customizations in
  [`datagen/mitgcm/held_suarez/`](../datagen/mitgcm/held_suarez).
- **Solver entry point.** `solver.py` (3-D extract via
  `RunConfig.pressure_levels`); `scripts/run.py` and
  `scripts/preflight.py` (`--pressure-levels` flag).
- **Sweep generator.**
  [`scripts/generate_sweep_safe.py`](../datagen/mitgcm/held_suarez/scripts/generate_sweep_safe.py).
- **Finalisation.**
  [`scripts/finalize_3d.py`](../datagen/mitgcm/held_suarez/scripts/finalize_3d.py)
  (manifest copy, stratified train/val/test split, parallel-Welford
  per-(var, level) stats).
- **Zarr writer.** `datagen.resample.write_latlon_zarr_3d`
  (per-variable schema with a `level` axis; sibling of the 2-D
  `write_latlon_zarr`).
- **Tests.**
  [`datagen/mitgcm/held_suarez/tests/test_extract_3d.py`](../datagen/mitgcm/held_suarez/tests/test_extract_3d.py)
  (single-level regression, 1-level and 8-level extraction shape and
  near-pressure matching, duplicate-request handling).

## Data specifications

### Grid

- **Discretization.** Regular `(lat, lon)` grid with `Nlat = 64` and
  `Nlon = 128`, equispaced in both directions and excluding the
  poles. Same as the 2-D dataset.
- **Vertical coordinate.** **8 isobaric levels** chosen to mirror
  ERA5/WeatherBench conventions:

  | Requested (hPa) | Nearest model centre (hPa) |
  | ---             | ---                        |
  | 50              | 75 (top fat layer)         |
  | 100             | 75 (same model k)          |
  | 250             | 262                        |
  | 500             | 486                        |
  | 700             | 709                        |
  | 850             | 843                        |
  | 925             | 933                        |
  | 1000            | 978                        |

  Note that `50 hPa` and `100 hPa` both resolve to MITgcm's top
  layer (centre at ≈ 75 hPa, thickness 150 hPa) — the model's vertical
  grid is too coarse there to separate them. The actual matched
  pressures are stored as `pressure_actual_hpa(level)` in every
  `run.zarr` for audit.

### Temporal layout

- **Spin-up.** `200` simulated days, no diagnostics written.
- **Data collection.** `365` simulated days, daily snapshots written.
- **Total run.** `565` simulated days per production trajectory.
- **Timestep.** `Δt = 30 s` (vs `45 s` in the 2-D variant; see CFL
  notes above).

### Available fields

Per-variable layout (unlike the 2-D `fields(time, field, lat, lon)`
stack):

| Field | Shape                        | Units | Description                  |
| ---   | ---                          | ---   | ---                          |
| `u`   | `(time, level, lat, lon)`    | m/s   | Zonal wind                   |
| `v`   | `(time, level, lat, lon)`    | m/s   | Meridional wind              |
| `T`   | `(time, level, lat, lon)`    | K     | Potential temperature        |
| `ps`  | `(time, lat, lon)`           | Pa    | Surface pressure             |

Coords: `time` (seconds since data-collection start), `level`
(requested hPa values), `pressure_actual_hpa(level)` (actual model
centre), `lat`, `lon` (degrees).

All fields are `float32`.

### Dataset size

- **Number of trajectories.** 72.
- **Per-trajectory size.** ≈ 240 MB on disk (compressed Zarr).
- **Total dataset size.** ≈ 17 GB across all 72 runs (excluding the
  per-run `mitgcm_run/` working dirs, which are kept on the
  generation cluster as forensics).

### Storage format and on-disk layout

```
<DATA_ROOT>/held-suarez-3d/
    manifest.json     72 runs × {run_id, params, param_hash}
    splits.json       stratified on tau_drag_days, seed=42 (57/6/9)
    stats.json        per-(var, level) z-score (25 entries)
    train/  run_NNNN.zarr   57 trajectories
    val/    run_NNNN.zarr    6
    test/   run_NNNN.zarr    9
    runs/   run_NNNN/mitgcm_run/   forensic working dirs (not shipped)
    preflight/  corner_NN/run.zarr    32 stability-validation runs
```

The default `DATA_ROOT/held-suarez-3d/` lives in the group archive at
`/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/`.

## Parameter space (safe grid)

Runs are laid out on a 5-axis tensor product, indexed
`run_0000 … run_0071` in row-major order over
`(tau_drag_days, delta_T_y, delta_theta_z, tau_surf_days, seed)`.

| Parameter | Symbol | Values | Count |
| --- | --- | --- | --- |
| Surface drag timescale | `tau_drag_days` | `0.5, 1.0, 2.0` | 3 |
| Meridional temperature contrast | `delta_T_y` | `40, 60` K | 2 |
| Vertical temperature contrast | `delta_theta_z` | `10, 20` K | 2 |
| Surface cooling timescale | `tau_surf_days` | `4, 8` days | 2 |
| IC seed | `seed` | `0, 1, 2` | 3 |
| **Total runs** |  |  | **72** |

The free-atmosphere cooling timescale is held fixed at
`tau_atm_days = 40` in every run. The dropped values
(`delta_T_y = 80 K`, `delta_theta_z = 5 K`) are the regimes where the
2-D 162-run sweep at `Δt = 45 s` saw the most CFL failures.

## End-to-end data flow

```bash
# Login node: install Python deps
uv sync --project datagen

# Build the MITgcm executable (shared with the 2-D variant)
sbatch datagen/mitgcm/held_suarez/slurm/build.sbatch

# Emit the 72-run safe sweep
uv run --project datagen python -m datagen.mitgcm.held_suarez.scripts.generate_sweep_safe \
  --out datagen/mitgcm/held_suarez/configs/safe

# Optional: run the 32-corner preflight at Δt = 30 s with 8 plev levels
sbatch datagen/mitgcm/held_suarez/slurm/preflight_array_3d.sbatch

# Production array (72 runs × 565 sim-days × 8 plev levels)
sbatch datagen/mitgcm/held_suarez/slurm/production_array_3d.sbatch

# After production finishes: assemble manifest, splits, symlinks→data, stats
uv run --project datagen python -m datagen.mitgcm.held_suarez.scripts.finalize_3d \
  --root /scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/held-suarez-3d
```

The companion notebook
[`notebooks/mitgcm_visualize.ipynb`](../notebooks/mitgcm_visualize.ipynb)
opens the 2-D Zarr stores; for the 3-D variant, open any
`<DATA_ROOT>/held-suarez-3d/train/run_NNNN.zarr` directly with
`xarray.open_zarr` and select levels via `ds.u.sel(level=500)`.
