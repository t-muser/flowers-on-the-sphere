# Galewsky Shallow-Water Barotropic Instability on the Sphere

**One-line description.** A parametric ensemble of rotating shallow-water
simulations on the 2-sphere, initialized with a zonally-symmetric balanced
jet plus a localized height perturbation that seeds barotropic instability
and a rollup of cyclones along the jet's poleward flank.

**Extended description.** Each trajectory solves the global shallow-water
equations on a rotating sphere of Earth-like radius for sixteen simulated
days, of which the first four (linear spinup) are discarded, leaving a
twelve-day window in the post-saturation / turbulent regime. The initial
condition is the Galewsky et al. (2004) test case: a
compact-support zonal jet in latitude, geostrophically + cyclostrophically
balanced by an elevation step across the jet, with a bi-Gaussian height
bump breaking zonal symmetry. The perturbation grows into a wave train
whose breakdown into filamentary cyclones by day 4–6 is a standard
benchmark for atmospheric-flow solvers on the sphere. The ensemble
explores a 5-axis parameter box over the jet strength, jet latitude,
perturbation amplitude, mean fluid depth, and perturbation longitude.

## Associated resources

- **Papers.** Galewsky, J., Scott, R. K., and Polvani, L. M. (2004).
  *An initial-value problem for testing numerical models of the global
  shallow-water equations*. Tellus A, 56(5), 429–440.
- **Data generator.** Till Muser (University of Basel), 2026.
- **Generation software.** [Dedalus v3](https://dedalus-project.org/) —
  MPI-parallel spectral PDE solver with `SphereBasis` spin-weighted
  spherical-harmonic transforms.
- **Code.** This repository, subsystem
  [`datagen/`](../datagen). Data generation, initial conditions,
  resampling, and SLURM scripts are self-contained in that subproject.

## Mathematical framework

The governing equations are the rotating shallow-water system on the
sphere, with biharmonic hyperviscosity `ν∇⁴·` added to both equations
for grid-scale dissipation:

$$
\begin{aligned}
\partial_t \mathbf{u} + \mathbf{u}\!\cdot\!\nabla\mathbf{u}
    + g\,\nabla h + 2\Omega\,\sin(\phi)\,\hat{\mathbf{k}}\!\times\!\mathbf{u}
    + \nu\,\nabla^4\mathbf{u} &= 0,\\
\partial_t h + \nabla\!\cdot\!\bigl((H + h)\,\mathbf{u}\bigr)
    + \nu\,\nabla^4 h &= 0,
\end{aligned}
$$

where `u(λ, φ, t)` is the horizontal velocity tangent to the sphere,
`h(λ, φ, t)` is the surface-height perturbation about the mean depth
`H`, `φ` is latitude, and `k̂×u = zcross(u)` is a 90° rotation of the
horizontal vector on `S²`. Total depth is `H + h`.

### Constants

| Symbol | Value | Meaning |
| --- | --- | --- |
| `R`   | `6.371 22 × 10⁶ m` | Earth radius |
| `Ω`   | `7.292 × 10⁻⁵ rad/s` | Rotation rate |
| `g`   | `9.806 16 m/s²` | Gravitational acceleration |
| `ν`   | `1.0 × 10⁵ m²/s` at `N_θ = 256` | Hyperviscosity, scaled ∝ `1/N_θ²` |

### Hyperviscosity scaling

`ν(N_θ) = ν₀ · (N_θ,ref / N_θ)²` with `ν₀ = 10⁵ m²/s`, `N_θ,ref = 256`.
This keeps the grid-scale damping fixed in non-dimensional spectral
space as the resolution changes. All trajectories in this ensemble use
`ν = 10⁵ m²/s`.

## Data specifications

### Grid

- **Discretization.** Regular `(lat, lon)` grid with `N_lat = 256`,
  `N_lon = 512`, equispaced and pole-excluding. Latitude centers
  are `lat_k = -π/2 + (k + 0.5)·π/N_lat` in radians, stored as
  `degrees_north`. Longitude centers are `lon_k = k·2π/N_lon` in
  `[0, 2π)`, stored as `degrees_east`.
- **Native solver grid.** Gauss-Legendre in colatitude with
  `N_θ = 256` and equispaced longitude with `N_φ = 512`, dealias
  factor `3/2`. Snapshots are resampled from the native grid to the
  regular `(lat, lon)` grid via per-longitude cubic-spline interpolation
  in colatitude — effectively exact below Nyquist.

### Temporal layout

- **Snapshot cadence.** Every 3 600 s (1 h of simulated time).
- **Simulation length.** 16 simulated days = 1 382 400 s.
- **Output window.** Simulation days 4 – 16 only; the first 4 days
  (linear spinup) are discarded at resample time. The remaining 289
  snapshots span 0 – 12 d on the exported `time` axis (rebased so the
  first kept snapshot is `t = 0`).
- **Trajectory shape.** 289 snapshots per trajectory, each 1 h apart.
- **Time coordinate.** Stored as seconds since the start of the kept
  window (`float64`).

### Available fields

| Field name | Units | Description |
| --- | --- | --- |
| `u_phi`     | `m/s` | Zonal (eastward) component of the horizontal velocity |
| `u_theta`   | `m/s` | Meridional component in the native colatitude direction (southward) |
| `h`         | `m`   | Surface-height perturbation about mean depth `H` |
| `vorticity` | `1/s` | Relative vorticity `ζ = -∇·skew(u)` = vertical component of `∇×u` |

All fields stored as `float32`.

### Dataset size

- **Number of trajectories.** 960.
- **Shape per trajectory.** `(time=289, field=4, lat=256, lon=512)`.
- **Per-trajectory size.** ≈ 503 MB (float32, uncompressed).
- **Total ensemble size.** ≈ 483 GB.
- **Consolidated shape.** `(run=960, time=289, field=4, lat=256, lon=512)`.

### Storage format

Per-run Zarr v3 stores at
`…/galewsky-sw/processed/run_XXXX.zarr`, chunked as
`(time=1, field=4, lat=256, lon=512)` for fast per-snapshot random access.
A consolidated `dataset.zarr` stacks all runs along a leading `run`
dimension with parameter arrays carried as `param_*` coords. The
`manifest.json` at the dataset root carries the full list of
`(run_id, param_hash, params)` tuples.

### Train / val / test split

A fixed 80 / 10 / 10 split of the 960 run IDs is distributed with the
dataset as `splits.json` alongside `manifest.json`. The split is
**stratified on `u_max`** with a deterministic RNG seed so every split
sees all five jet-strength values in the target proportions. Counts are
`{train: 770, val: 95, test: 95}`. Typical use:

```python
import json, xarray as xr

root   = "/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/galewsky-sw"
splits = json.load(open(f"{root}/splits.json"))
train_runs = [xr.open_zarr(f"{root}/processed/run_{i:04d}.zarr") for i in splits["train"]]
```

## Initial conditions

All trajectories share the same three-stage IC construction, parameterized
by the 5-axis grid below.

### 1. Zonal jet profile

A compact-support analytic jet in latitude, nonzero only inside
`[lat_center − 20°, lat_center + 20°]`, reaching peak amplitude
`u_max` at the jet center:

$$
u_\phi(\varphi) = \frac{u_\mathrm{max}}{e_n}\,
  \exp\!\Bigl(\tfrac{1}{(\varphi - \varphi_0)(\varphi - \varphi_1)}\Bigr),
  \quad \varphi_0 < \varphi < \varphi_1,
$$

with `e_n = exp(−4/(φ₁ − φ₀)²)` chosen so the maximum equals `u_max`.
Outside the band `u_φ ≡ 0`. Meridional velocity `u_θ ≡ 0` at initial
time.

### 2. Geostrophic + cyclostrophic balance for `h`

The balanced surface-height field is obtained by 1-D numerical
integration of the zonal-flow balance ODE in latitude,

$$
\frac{\partial h}{\partial \varphi} = -\frac{1}{g}\,u_\phi\,
  \Bigl(2\Omega R\sin\varphi + u_\phi \tan\varphi\Bigr),
$$

on a fine uniform grid in `φ`, followed by cos-weighted area-mean
subtraction so that `⟨h⟩ = 0`. The resulting 1-D profile `h(φ)` is
evaluated at the native solver colatitude nodes by cubic interpolation.
Because `u_φ = 0` near the poles, `u_φ · tan(φ)` is well-behaved at
the singular endpoints.

### 3. Bi-Gaussian height perturbation

A localized Galewsky bump is added on top of the balanced height:

$$
h'(\lambda, \varphi) = \hat{h}\,\cos(\varphi)\,
  \exp\!\Bigl(-\bigl(\tfrac{\lambda - \lambda_c}{\beta}\bigr)^{\!2}\Bigr)\,
  \exp\!\Bigl(-\bigl(\tfrac{\varphi_c - \varphi}{\alpha}\bigr)^{\!2}\Bigr),
$$

with fixed widths `α = 1/3` (meridional) and `β = 1/15` (zonal),
perturbation amplitude `ĥ = h_hat`, and perturbation center
`(λ_c, φ_c) = (lon_c, lat_center)` — i.e. the bump sits at the jet
center in latitude and at the prescribed longitude.

## Physical setup

- **Domain.** Full sphere `S²`.
- **Boundary conditions.** None in the usual sense: `SphereBasis`
  enforces periodicity in longitude and smoothness at the poles
  intrinsically via the spin-weighted spherical-harmonic expansion.
- **Timestepper.** `RK222` with CFL-adaptive `dt`
  (safety 0.3, `max_dt = 600 s`, `initial_dt = 120 s`, max change 1.5
  per step, min change 0.5 per step).
- **Integration window.** `0 ≤ t_sim ≤ 16 · 86 400 s`. Snapshots are
  stored hourly only for `t_sim ≥ 4 · 86 400 s`; the first four days
  of each run (linear instability growth) are discarded so every
  trajectory begins in the post-saturation / turbulent regime.

## Parameter space

Runs are laid out on an explicit tensor grid over five axes
(5 · 4 · 4 · 4 · 3 = **960** runs). Runs are indexed `run_0000 …
run_0959` in row-major order over the tuple
`(u_max, lat_center, h_hat, H, lon_c)`.

| Parameter       | Units  | Values                                  | Count |
| --- | --- | --- | --- |
| `u_max`         | m/s    | 60, 70, 80, 90, 100                     | 5 |
| `lat_center`    | °N     | 30, 40, 50, 60                          | 4 |
| `h_hat`         | m      | 60, 120, 180, 240                       | 4 |
| `H`             | m      | 8 000, 10 000, 12 000, 14 000           | 4 |
| `lon_c`         | °      | 0, 120, 240                             | 3 |

Jet half-width (`40°` full width), perturbation widths
(`α = 1/3`, `β = 1/15`), grid resolution, and physics constants are
identical across the ensemble.

The `lon_c` axis is a cheap rotational-augmentation factor: the
physics is zonally symmetric, so `lon_c` shifts the instability trigger
by a pure rotation of the initial condition. The other four axes are
physically meaningful.

## Numerical-stability strategy

1. **Preflight sweep** over the 32 box corners at half resolution
   (`N_φ = 256, N_θ = 128`) for 1 simulated day. All 32 corners
   completed without NaNs before the full sweep launched; this confirms
   the 5-axis box is numerically stable.
2. **CFL-adaptive `dt`** inside every run, with a hard ceiling
   `max_dt = 600 s`.
3. **Per-run try/except** around the solver loop: on failure the
   driver writes a `run_XXXX.FAILED` sentinel JSON with the exception
   and parameters, then exits non-zero. The SLURM array continues; a
   final `consolidate.py` step emits a report of any missing runs.
4. **Hyperviscosity** scaled so the hardest corner (`u_max = 100`,
   `H = 14 000`) is still sufficiently dissipative at the grid scale.

## Computational details

- **Per-run wall.** ≈ 20 – 25 min on one scicore compute node with 16
  MPI ranks (AMD Epyc / Intel Xeon class nodes, 128-core `scicore`
  partition, one MPI rank per core with `OMP_NUM_THREADS = 1`).
- **Total compute.** ≈ 5 000 – 6 500 core-hours for the full
  ensemble (16 sim-days × 960 runs × 16 ranks × ~22 min).
- **Solver precision.** `float64` throughout the simulation; all
  output fields are downcast to `float32` at resample time.
- **Cluster.** sciCORE @ Universität Basel, `scicore` partition,
  `6hours` QoS, Easybuild `foss/2024a` +
  `HDF5/1.14.5-gompi-2024a` toolchain.

## Using the dataset

```python
import xarray as xr

root = "/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/galewsky-sw"

# One trajectory — lightweight random access.
run = xr.open_zarr(f"{root}/processed/run_0000.zarr")
print(run)

# Full ensemble — run, time, field, lat, lon.
ds = xr.open_zarr(f"{root}/dataset.zarr")
print(ds)

# Parameter lookup via the carried coords.
ds.sel(run=ds.param_u_max == 80.0)
```

See the repo notebook
[`notebooks/galewsky_visualize.ipynb`](../notebooks/galewsky_visualize.ipynb)
for plots of the initial condition, vorticity rollup over time, zonal-mean
Hovmöller diagrams, and an optional MP4 animation.

## Citation

If you use this dataset, please cite the original Galewsky test case:

```bibtex
@article{galewsky2004,
  title   = {An initial-value problem for testing numerical models
             of the global shallow-water equations},
  author  = {Galewsky, Joseph and Scott, Richard K. and Polvani, Lorenzo M.},
  journal = {Tellus A: Dynamic Meteorology and Oceanography},
  volume  = {56},
  number  = {5},
  pages   = {429--440},
  year    = {2004},
  doi     = {10.3402/tellusa.v56i5.14436},
}
```

along with the solver used to generate this ensemble:

```bibtex
@article{burns2020dedalus,
  title   = {Dedalus: A flexible framework for numerical simulations
             with spectral methods},
  author  = {Burns, Keaton J. and Vasil, Geoffrey M. and Oishi, Jeffrey S.
             and Lecoanet, Daniel and Brown, Benjamin P.},
  journal = {Physical Review Research},
  volume  = {2},
  number  = {2},
  pages   = {023068},
  year    = {2020},
  doi     = {10.1103/PhysRevResearch.2.023068},
}
```
