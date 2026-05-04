# Held-Suarez Atmospheric GCM on the Sphere

**One-line description.** A 162-member ensemble of Held-Suarez dry
atmospheric simulations on the sphere, integrated with MITgcm and
written to regular lat-lon Zarr stores for analysis and machine-learning
workflows.

**Extended description.** Each trajectory solves the standard
Held-Suarez idealized general circulation model on a spherical-polar
grid with an aqua-planet lower boundary, Newtonian cooling toward a
prescribed equilibrium temperature profile, and Rayleigh friction in
the lower atmosphere. The solver runs in two phases. A spin-up phase
evolves the atmosphere from a seeded initial perturbation without
writing diagnostics, allowing the flow to settle into the familiar jet
and eddy structure. The data-collection phase restarts from the
spin-up pickup and writes daily output that is resampled to a regular
`(lat, lon)` grid. Ensemble diversity comes from a five-axis parameter
sweep over drag timescale, Newtonian-cooling timescales, meridional
and vertical temperature contrasts, and an initial-condition seed.

## Associated resources

- **Paper.** Held, I. M. & Suarez, M. J. (1994). *A proposal for the
  intercomparison of the dynamical cores of atmospheric general
  circulation models*. Bulletin of the American Meteorological Society.
- **Data generator.** Till Muser (University of Basel), 2026.
- **Generation software.** [MITgcm](../mitgcm) with the Held-Suarez
  customizations in
  [`datagen/mitgcm/held_suarez/`](../datagen/mitgcm/held_suarez).
- **Code.** Solver, IC generation, namelists, SLURM scripts, and sweep
  logic live in the Held-Suarez package of this repository.

## Mathematical framework

The benchmark is the dry Held-Suarez atmospheric GCM. In pressure
coordinates, the model evolves horizontal winds and potential
temperature under adiabatic dynamics, Newtonian cooling, and Rayleigh
drag. The forcing is parameterized by the standard Held-Suarez
profiles:

- equilibrium potential temperature `θ_eq(φ, p)`
- free-atmosphere Newtonian cooling rate `k_a`
- stronger near-surface Newtonian cooling rate `k_s`
- lower-level Rayleigh friction rate `k_f`

The implementation follows the canonical Held-Suarez formulas and
stores the forcing parameters in SI units internally. See
[`datagen/mitgcm/held_suarez/_physics.py`](../datagen/mitgcm/held_suarez/_physics.py)
and [`datagen/mitgcm/held_suarez/solver.py`](../datagen/mitgcm/held_suarez/solver.py)
for the exact parameterization.

### Fixed scales

| Symbol | Value | Meaning |
| --- | --- | --- |
| `R_EARTH` | `6.37122e6 m` | Planetary radius |
| `P0` | `1.0e5 Pa` | Reference pressure |
| `CP` | physical constant | Dry-air specific heat |
| `R_DRY` | physical constant | Dry-air gas constant |

The atmosphere is dry and hydrostatic, with a flat aqua-planet lower
boundary.

## Data specifications

### Grid

- **Discretization.** Regular `(lat, lon)` grid with `Nlat = 64` and
  `Nlon = 128`, equispaced in both directions and excluding the poles.
  Latitude centers are `lat_k = -π/2 + (k + 0.5)·π/Nlat` and longitude
  centers are `lon_k = k·2π/Nlon`.
- **Native solver grid.** MITgcm spherical-polar grid with `Nr = 20`
  pressure levels, `Nlon = 128`, `Nlat = 64`, and `ygOrigin = -90`.
  The native model covers the full sphere; the exported regular grid is
  the pole-excluding remap used throughout this repository.
- **Vertical coordinate.** The benchmark exports a single pressure
  slice at 500 hPa, plus surface pressure.

### Temporal layout

- **Spin-up.** `200` simulated days, no diagnostics written.
- **Data collection.** `365` simulated days, daily snapshots written.
- **Total run.** `565` simulated days per production trajectory.
- **Preflight.** `30` days spin-up plus `30` days data collection for
  corner testing.
- **Time coordinate.** Stored as seconds since the start of the saved
  data-collection window, not since the spin-up start.

### Available fields

The exported Zarr stores a single pressure level, chosen at
`pressure_hpa = 500` by default, plus surface pressure:

| Field name | Units | Description |
| --- | --- | --- |
| `u_500hpa` | `m/s` | Zonal wind at 500 hPa |
| `v_500hpa` | `m/s` | Meridional wind at 500 hPa |
| `T_500hpa` | `K` | Potential temperature at 500 hPa |
| `ps` | `Pa` | Surface pressure |

All fields are stored as `float32` on the regular lat-lon grid.

### Dataset size

- **Number of trajectories.** 162
- **Shape per trajectory.** `(time≈365, field=4, lat=64, lon=128)`
- **Per-trajectory size.** roughly 45 MB uncompressed
- **Total ensemble size.** roughly 7.5 GB uncompressed

The exact number of time samples depends on the daily diagnostic
cadence and the restart/output convention of the data-collection
phase, but the intended window is one year of daily output after the
spin-up has been discarded.

### Storage format

Per-run Zarr stores are written at `…/mitgcm/held-suarez/run.zarr`.
The standard layout is `(time, field, lat, lon)` with chunking
optimized for snapshot access and time-series inspection. See
[`datagen/mitgcm/held_suarez/solver.py`](../datagen/mitgcm/held_suarez/solver.py)
for the writer and
[`notebooks/mitgcm_visualize.ipynb`](../notebooks/mitgcm_visualize.ipynb)
for the corresponding visualization workflow.

## Initial conditions

Each run is initialized from the Held-Suarez equilibrium potential
temperature profile at every pressure level, plus a small seeded random
perturbation. The perturbation is:

- drawn from `np.random.default_rng(seed)`
- Gaussian in amplitude with RMS scale `0.1 K`
- smoothed with a Gaussian kernel of width `5` grid points
- wrapped periodically in longitude and reflected in latitude

The smoothing keeps the IC perturbation band-limited and avoids
injecting unnecessary grid-scale noise into the spin-up.

The lower boundary is an aqua-planet: bathymetry is identically zero,
which is the correct flat-surface choice for this dry atmospheric setup.

## Physical setup

- **Domain.** Full sphere `S²`, represented by a spherical-polar
  atmospheric grid.
- **Boundary conditions.** No solid boundaries; the lower boundary is a
  flat aqua-planet surface.
- **Dynamics.** Dry hydrostatic atmosphere with Coriolis forces,
  Newtonian cooling, and Rayleigh drag in the lower boundary layer.
- **Timestepper.** MITgcm native time integration with the standard
  Held-Suarez namelist settings.
- **Two-phase execution.** The spin-up phase writes no diagnostics and
  ends with a pickup file; the data-collection phase restarts from that
  pickup and writes output.

The two-phase design is intentional. It discards the initial
stochastic adjustment transient and preserves only the physically
meaningful post-spin-up evolution.

## Parameter space

Runs are laid out on a five-axis tensor product, indexed
`run_0000 … run_0161` in row-major order over
`(tau_drag_days, delta_T_y, delta_theta_z, tau_surf_days, seed)`.

| Parameter | Symbol | Values | Count |
| --- | --- | --- | --- |
| Surface drag timescale | `tau_drag_days` | `0.5, 1.0, 2.0` | 3 |
| Meridional temperature contrast | `delta_T_y` | `40, 60, 80` K | 3 |
| Vertical temperature contrast | `delta_theta_z` | `5, 10, 20` K | 3 |
| Surface cooling timescale | `tau_surf_days` | `4, 8` days | 2 |
| IC seed | `seed` | `0, 1, 2` | 3 |
| **Total runs** |  |  | **162** |

The free-atmosphere cooling timescale is held fixed at
`tau_atm_days = 40` in every run.

The sweep deliberately spans a range of dynamical regimes:

- stronger vs weaker lower-level drag
- stronger vs weaker meridional equilibrium temperature gradients
- shallow vs steep vertical stratification
- stronger vs weaker surface relaxation

The seed axis provides distinct initial perturbations on top of the
same large-scale forcing.

## Numerical-stability strategy

1. **Spin-up before diagnostics.** The production run is split into a
   long spin-up and a diagnostic phase so the seeded IC transient is not
   saved in the final dataset.
2. **Smoothed stochastic ICs.** Initial perturbations are low-pass
   filtered to avoid seeding grid-scale noise that would contaminate the
   spin-up.
3. **Held-Suarez physics choices.** The forcing is the standard dry,
   idealized benchmark, which is well-conditioned compared with a fully
   moist atmospheric model.
4. **Failure markers.** SLURM wrappers write `.FAILED` markers on
   exceptions so array jobs continue running and failures can be
   summarized later.

## Computational details

- **Per-run wall.** Roughly a few hours or less on the current scicore
  setup, depending on node load and MPI placement.
- **Total compute.** The 162-run production sweep is intended to be
  launched as a SLURM array.
- **Solver precision.** MITgcm runs in double precision internally;
  the exported Zarr stores are `float32`.
- **Cluster setup.** The provided scripts assume the `foss/2024a` and
  `HDF5/1.14.5-gompi-2024a` module stack on sciCORE.

## End-to-end data flow

```bash
# Login node: install Python deps
uv sync --project datagen

# Build the MITgcm executable for Held-Suarez
sbatch datagen/mitgcm/held_suarez/slurm/build.sbatch

# Emit the 162-run sweep configs
uv run --project datagen python -m datagen.mitgcm.held_suarez.scripts.generate_sweep \
  --out datagen/mitgcm/held_suarez/configs

# Optional: emit the 32-corner preflight configs
uv run --project datagen python -m datagen.mitgcm.held_suarez.scripts.preflight generate \
  --out datagen/mitgcm/held_suarez/configs/preflight

# Run the 32-corner preflight array
sbatch datagen/mitgcm/held_suarez/slurm/preflight_array.sbatch

# Launch one production run with the same driver used by the array job
uv run --project datagen python -m datagen.mitgcm.held_suarez.scripts.run \
  --config datagen/mitgcm/held_suarez/configs/run_0000.json \
  --out-dir /scratch/$USER/fots-data/mitgcm/held-suarez/run_0000
```

To submit from another directory, keep the repo root as the job working
directory with `--chdir` and point `DATA_ROOT` at any writable shared
location:

```bash
sbatch --chdir /path/to/flowers-on-the-sphere \
  --export=ALL,DATA_ROOT=/scratch/$USER/fots-data/mitgcm/held-suarez \
  /path/to/flowers-on-the-sphere/datagen/mitgcm/held_suarez/slurm/preflight_one.sbatch
```

The SLURM wrappers and the Python CLI both assume they are launched
from the repository root, but the `--chdir` pattern makes that true
even when the `sbatch` command itself is issued from another folder.

For interactive inspection, the notebook
[`notebooks/mitgcm_visualize.ipynb`](../notebooks/mitgcm_visualize.ipynb)
is the companion view on the exported Held-Suarez Zarr stores.
