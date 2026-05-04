# MITgcm Global Ocean Tutorial

**One-line description.** A MITgcm implementation of the
`tutorial_global_oce_latlon` ocean benchmark, integrated on a
4-degree spherical-polar grid and exported to regular lat-lon Zarr
stores.

**Extended description.** Each trajectory runs the standard MITgcm
global ocean tutorial with realistic bathymetry, Levitus hydrography
and restoring, Trenberth wind stress, and NCEP heat and freshwater
fluxes. The solver stages the tutorial code from the vendored MITgcm
tree, patches only the runtime settings needed for data generation
(`nTimeSteps`, output cadence, checkpoint cadence, timestep values,
and a small GM/Redi sweep hook), and then extracts a small set of
diagnostic fields into a regular `(lat, lon)` grid. The production
workflow is split into a corner-based preflight and a longer regular
run. Preflight is designed to test the corners of the parameter box
under reduced duration before committing to the full 32-corner array.

## Associated resources

- **Tutorial.** MITgcm `verification/tutorial_global_oce_latlon`.
- **Data generator.** Till Muser (University of Basel), 2026.
- **Generation software.** [MITgcm](../mitgcm) with the global-ocean
  customizations in
  [`datagen/mitgcm/global_ocean/`](../datagen/mitgcm/global_ocean).
- **Code.** Build, run, preflight, and sweep scripts live in the
  global-ocean package of this repository.

## Physical framework

The benchmark is the MITgcm global ocean tutorial, a primitive-equation
ocean simulation on a spherical-polar grid. The tutorial setup includes:

- realistic bathymetry
- Levitus hydrographic initial conditions and restoring
- Trenberth wind stress forcing
- NCEP heat and freshwater fluxes
- GM/Redi mixing

The solver is configured through the tutorial namelist files in the
vendored MITgcm tree, with a small set of runtime overrides applied by
the Python driver.

## Data specifications

### Grid

- **Discretization.** Regular `(lat, lon)` output grid with `Nlat = 40`
  and `Nlon = 90`, equispaced and pole-excluding. The output latitudes
  are cell centers spanning `80S` to `80N`.
- **Native solver grid.** MITgcm spherical-polar grid with 90 longitude
  cells, 40 latitude cells, and 15 vertical z-levels. The native
  tutorial domain is restricted to `80S..80N`.
- **Vertical coordinate.** 15 depth levels with layer thicknesses from
  `50 m` to `690 m`.

The exported benchmark is a horizontal slice through the 3-D ocean
state rather than the full 3-D volume. The selected levels are the same
ones used by the driver defaults.

### Temporal layout

- **Production run.** `36000` time steps on a 1-day model clock, i.e.
  `100` model years.
- **Snapshot cadence.** `30` simulated days by default for production
  runs.
- **Preflight.** `30` time steps, with `30` days of output by default
  unless the SLURM wrapper overrides the cadence.
- **Time coordinate.** Stored as seconds since the start of the kept
  window; the driver re-bases the MDS iterations so the first saved
  snapshot starts at `t = 0`.

### Available fields

The exported Zarr stores the following 2-D slices:

| Field name | Units | Description |
| --- | --- | --- |
| `theta_k1` | `degC` | Potential temperature at tracer level 1 |
| `salt_k1` | `psu` | Salinity at tracer level 1 |
| `u_k2` | `m/s` | Zonal velocity at velocity level 2 |
| `v_k2` | `m/s` | Meridional velocity at velocity level 2 |
| `eta` | `m` | Sea surface height |

All fields are stored as `float32` on the regular lat-lon grid.

### Dataset size

- **Number of trajectories.** 243 for the current sweep grid
- **Shape per trajectory.** `(time≈1200, field=5, lat=40, lon=90)` for
  the default 100-year, monthly-output production run
- **Per-trajectory size.** on the order of tens of MB uncompressed
- **Total ensemble size.** on the order of a few GB for the default
  sweep

The exact file size depends on the chosen snapshot cadence. Monthly
output keeps the 100-year integration manageable; daily output is
useful for preflight and short test runs.

### Storage format

Per-run Zarr stores are written at
`…/mitgcm/global-ocean/run.zarr`. The standard layout is
`(time, field, lat, lon)` with chunking optimized for time-slice
inspection. See
[`datagen/mitgcm/global_ocean/solver.py`](../datagen/mitgcm/global_ocean/solver.py)
for the writer and
[`notebooks/mitgcm_global_ocean_visualize.ipynb`](../notebooks/mitgcm_global_ocean_visualize.ipynb)
for the visualization workflow.

## Initial and boundary conditions

The ocean tutorial uses the MITgcm-provided input files from
`verification/tutorial_global_oce_latlon/input`. These include the
initial hydrography, bathymetry, forcing fields, and the fixed
namelists needed for the verification experiment. The Python driver
only adjusts the runtime settings and the GM/Redi diffusion parameter.

The default output levels are:

- tracer level `1` for `theta_k1` and `salt_k1`
- velocity level `2` for `u_k2` and `v_k2`

Those defaults are chosen to match the tutorial diagnostics and to give
near-surface fields with clear large-scale structure.

## Physical setup

- **Domain.** Global ocean on a spherical-polar grid, restricted to
  `80S..80N`.
- **Boundary conditions.** Physical lower boundary from the bathymetry;
  the driver symlinks the tutorial binary inputs into each run
  directory.
- **Dynamics.** MITgcm primitive-equation ocean model with GM/Redi
  mixing and the tutorial restoring / surface forcing fields.
- **Timestepper.** MITgcm native time integration, with `deltaTmom`,
  `deltaTtracer`, `deltaTClock`, and `deltaTfreesurf` patched by the
  Python driver.
- **Restart strategy.** Preflight and production both use the same run
  driver; the preflight phase simply shortens the integration window.

## Parameter space

The current sweep varies five physically meaningful ocean controls
around the tutorial defaults:

| Parameter | Symbol | Values | Count |
| --- | --- | --- | --- |
| GM/Redi background diffusivity | `gm_background_k` | `250, 1000, 2500` | 3 |
| Horizontal viscosity | `visc_ah` | `2e5, 5e5, 1e6` | 3 |
| Vertical diffusivity | `diff_kr` | `1e-5, 3e-5, 1e-4` | 3 |
| Surface temperature restoring | `tau_theta_relax_days` | `30, 60, 120` | 3 |
| Surface salinity restoring | `tau_salt_relax_days` | `90, 180, 360` | 3 |

Held fixed:

- `n_timesteps = 36000` for the regular run
- `delta_t_clock = 86400 s`
- `tracer_level = 1`
- `velocity_level = 2`

The current sweep is a `3^5 = 243` run tensor product and is designed
to span weak, standard, and strong mixing / restoring regimes rather
than only tiny perturbations around the tutorial settings.

## Preflight strategy

The preflight configuration uses the same parameter grid but runs each
corner at reduced duration. The current defaults are:

- `30` timesteps
- `30` days of data collection
- daily snapshots if you override the cadence to `1` day

For visual inspection and failure detection, the recommended setup is:

- 30-day spin-up corner run
- 1-day snapshot cadence
- one SLURM array element per corner

## Numerical-stability strategy

1. **Corner preflight.** The 32-corner preflight run checks extreme
   parameter combinations before launching the full sweep.
2. **Runtime-only overrides.** The Python driver stages the tutorial
   namelists and changes only the fields needed for reproducible data
   generation.
3. **Failure markers.** SLURM wrappers write `.FAILED` files when a
   run crashes, so one failure does not cancel the rest of the array.
4. **Shared-storage outputs.** All outputs are written under a user
   controlled `DATA_ROOT`, which avoids coupling the docs to any
   personal path.

## Computational details

- **Per-run wall.** The 100-year production run is substantially more
  expensive than the preflight corner run; submit it as a SLURM array
  only after the corner run is healthy.
- **Total compute.** The `243`-run production sweep is intended for
  batch execution.
- **Solver precision.** MITgcm runs in double precision internally;
  exported Zarr stores are `float32`.
- **Cluster setup.** The provided scripts assume the `foss/2024a` and
  `HDF5/1.14.5-gompi-2024a` module stack on sciCORE, but the commands
  below are written so they can be launched from any folder if you set
  `--chdir` and `DATA_ROOT` appropriately.

## End-to-end data flow

```bash
# Login node: install Python deps
uv sync --project datagen

# Build the MITgcm executable for the global-ocean tutorial
uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.build \
  --mitgcm-root mitgcm \
  --optfile mitgcm/tools/build_options/linux_amd64_gfortran

# Emit the full sweep configs
uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.generate_sweep \
  --out datagen/mitgcm/global_ocean/configs

# Emit the preflight corner configs
uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.preflight generate \
  --out datagen/mitgcm/global_ocean/configs/preflight

# Run one corner interactively
uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.preflight run \
  --config datagen/mitgcm/global_ocean/configs/preflight/corner_00.json \
  --out-dir /scratch/$USER/fots-data/mitgcm/global-ocean/preflight/corner_00 \
  --executable datagen/mitgcm/global_ocean/build/mitgcmuv

# Submit the 32-corner preflight array from any directory
sbatch --chdir /path/to/flowers-on-the-sphere \
  --export=ALL,DATA_ROOT=/scratch/$USER/fots-data/mitgcm/global-ocean \
  /path/to/flowers-on-the-sphere/datagen/mitgcm/global_ocean/slurm/preflight_array.sbatch

# Submit the single-corner preflight wrapper from any directory
sbatch --chdir /path/to/flowers-on-the-sphere \
  --export=ALL,DATA_ROOT=/scratch/$USER/fots-data/mitgcm/global-ocean,CORNER=00 \
  /path/to/flowers-on-the-sphere/datagen/mitgcm/global_ocean/slurm/preflight_one.sbatch
```

The important portability rule is the same one used for the
Held-Suarez documentation: keep the repository root as the job working
directory with `--chdir`, and point `DATA_ROOT` at a writable shared
location. That makes the SLURM scripts usable from any launch folder.

For interactive inspection, the notebook
[`notebooks/mitgcm_global_ocean_visualize.ipynb`](../notebooks/mitgcm_global_ocean_visualize.ipynb)
is the companion view on the exported global-ocean Zarr stores.
