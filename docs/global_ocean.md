# MITgcm Global Ocean (cs32x15)

**One-line description.** A MITgcm implementation of the
`verification/global_ocean.cs32x15` ocean benchmark, integrated on a
cubed-sphere grid (six 32×32 faces, 15 vertical levels) and exported
to a native cubed-sphere Zarr layout.

**Extended description.** Each trajectory runs the cs32x15 verification
case with realistic bathymetry, Levitus hydrography (3-D initial
condition and surface restoring), Trenberth wind stress, and
shi*/ncep heat and freshwater fluxes. Runs warm-start from the
`pickup.0000072000` file shipped with the tutorial, which contains
roughly 200 model years of free spin-up. The Python driver stages the
tutorial code from the vendored MITgcm tree, patches only the runtime
settings needed for data generation (`nIter0`, `nTimeSteps`, output
cadence, checkpoint cadence, timestep values, restoring timescales, and
a small GM/Redi sweep hook), and reads the resulting MDS state dumps
into a face-major Zarr store. The production workflow is split into a
corner-based preflight and a longer regular run.

## Associated resources

- **Tutorial.** MITgcm `verification/global_ocean.cs32x15`.
- **Data generator.** Till Muser (University of Basel), 2026.
- **Generation software.** [MITgcm](../mitgcm) with the global-ocean
  customizations in
  [`datagen/mitgcm/global_ocean/`](../datagen/mitgcm/global_ocean).
- **Code.** Build, run, preflight, and sweep scripts live in the
  global-ocean package of this repository.

## Physical framework

The benchmark is a primitive-equation ocean simulation on the cubed
sphere with non-linear free surface (`z*` coordinate, real fresh-water
flux) and vector-invariant momentum. The configuration includes:

- realistic cs32 bathymetry (`bathy_Hmin50.bin`)
- Levitus hydrographic initial conditions (`lev_T_cs_15k.bin`,
  `lev_S_cs_15k.bin`) and monthly surface restoring (`lev_surfT_cs_12m.bin`,
  `lev_surfS_cs_12m.bin`)
- Trenberth wind stress (`trenberth_taux.bin`, `trenberth_tauy.bin`)
- shi*/ncep heat and freshwater fluxes (`shiQnet_cs32.bin`,
  `shiEmPR_cs32.bin`)
- GM/Redi mesoscale eddy parameterization
- GGL90 vertical mixing

The solver is configured through the tutorial namelist files in the
vendored MITgcm tree, with a small set of runtime overrides applied by
the Python driver.

## Data specifications

### Grid

- **Discretization.** Cubed-sphere grid with six 32×32 faces, supplied
  via `grid_cs32.face00{1..6}.bin` (symlinked from the
  `tutorial_held_suarez_cs` input directory at run time).
- **Compile-time domain decomposition.** `sNx=32, sNy=32, nSx=6, nSy=1,
  nPx=1, nPy=1` — single rank with one tile per cube face. Global
  output shape is `(Nr, 32, 192)` and reshapes trivially to
  `(Nr, 6, 32, 32)`.
- **Vertical coordinate.** 15 depth levels with layer thicknesses from
  `50 m` to `690 m`.

The exported benchmark is a horizontal slice through the 3-D ocean
state rather than the full 3-D volume. The selected levels are the same
ones used by the driver defaults.

### Temporal layout

- **Pickup.** Runs warm-start from `pickup.0000072000` (`nIter0=72000`),
  which provides roughly 200 model years of free spin-up.
- **Production run.** `36000` time steps on a 1-day model clock, i.e.
  `100` model years of post-pickup integration.
- **Snapshot cadence.** `30` simulated days by default for production
  runs.
- **Preflight.** `30` time steps, with `10`-day snapshots by default.
- **Time coordinate.** Stored as seconds since the start of the kept
  window; the driver re-bases the MDS iterations so the first saved
  snapshot starts at `t = 0`.

### Available fields

The exported Zarr stores the following 2-D (per-face) slices:

| Field name | Units | Description |
| --- | --- | --- |
| `theta_k1` | `degC` | Potential temperature at tracer level 1 |
| `salt_k1` | `psu` | Salinity at tracer level 1 |
| `u_k2` | `m/s` | Zonal velocity at velocity level 2 |
| `v_k2` | `m/s` | Meridional velocity at velocity level 2 |
| `eta` | `m` | Sea surface height |

All fields are stored as `float32`. Per-cell longitudes (`xc`) and
latitudes (`yc`) are stored as `(face, y, x)` coordinate arrays
alongside the data.

### Storage format

Per-run Zarr stores are written at `…/mitgcm/global-ocean/run.zarr`.
The native layout is `(time, field, face, y, x)` with `face=6` and
`y=x=32`. See
[`datagen/mitgcm/global_ocean/solver.py`](../datagen/mitgcm/global_ocean/solver.py)
for the writer and
[`notebooks/mitgcm_global_ocean_visualize.ipynb`](../notebooks/mitgcm_global_ocean_visualize.ipynb)
for the visualization workflow.

## Initial and boundary conditions

The cs32x15 verification case ships its own input files in
`mitgcm/verification/global_ocean.cs32x15/input/`. The Python driver
symlinks every binary file from that directory plus the six
`grid_cs32.face00?.bin` files into the run directory, then writes
patched `data`, `data.gmredi`, `data.pkg`, and the static `eedata`
namelist on top.

The default output levels are:

- tracer level `1` for `theta_k1` and `salt_k1`
- velocity level `2` for `u_k2` and `v_k2`

Those defaults are chosen to match the tutorial diagnostics and to give
near-surface fields with clear large-scale structure.

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

- `n_iter0 = 72000` (warm-start from the shipped pickup)
- `n_timesteps = 36000` for the regular run
- `delta_t_clock = 86400 s`
- `tracer_level = 1`
- `velocity_level = 2`

The current sweep is a `3^5 = 243` run tensor product and is designed
to span weak, standard, and strong mixing / restoring regimes rather
than only tiny perturbations around the tutorial settings.

## Preflight strategy

The preflight configuration uses the same parameter grid but runs each
corner at reduced duration:

- `30` timesteps after the pickup
- `10`-day snapshots

For visual inspection and failure detection, run the 32-corner array,
then open the visualization notebook against any successful corner.

## Numerical-stability strategy

1. **Corner preflight.** The 32-corner preflight run checks extreme
   parameter combinations before launching the full sweep.
2. **Runtime-only overrides.** The Python driver stages the tutorial
   namelists and changes only the fields needed for reproducible data
   generation.
3. **Failure markers.** SLURM wrappers write `.FAILED` files when a
   run crashes, so one failure does not cancel the rest of the array.
4. **Shared-storage outputs.** All outputs are written under a user
   controlled `DATA_ROOT`.

## Computational details

- **Per-run wall.** The 100-year production run is substantially more
  expensive than the preflight corner run; submit it as a SLURM array
  only after the corner run is healthy.
- **Total compute.** The `243`-run production sweep is intended for
  batch execution.
- **Solver precision.** MITgcm runs in double precision internally;
  exported Zarr stores are `float32`.
- **Cluster setup.** The provided scripts assume the `foss/2024a` and
  `HDF5/1.14.5-gompi-2024a` module stack on sciCORE.

## End-to-end data flow

```bash
# Login node: install Python deps
uv sync --project datagen

# Build the MITgcm executable for the cs32x15 case
sbatch datagen/mitgcm/global_ocean/slurm/build.sbatch

# Emit the full sweep configs
uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.generate_sweep \
  --out datagen/mitgcm/global_ocean/configs

# Emit the preflight corner configs
uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.preflight generate \
  --out datagen/mitgcm/global_ocean/configs/preflight

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
location.

For interactive inspection, the notebook
[`notebooks/mitgcm_global_ocean_visualize.ipynb`](../notebooks/mitgcm_global_ocean_visualize.ipynb)
is the companion view on the exported cubed-sphere Zarr stores.
