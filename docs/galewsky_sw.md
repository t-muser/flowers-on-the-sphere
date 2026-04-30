# Galewsky Shallow-Water Barotropic Instability on the Sphere

**One-line description.** A parametric ensemble of rotating shallow-water
simulations on the 2-sphere, initialized with a zonally-symmetric balanced
jet plus a localized height perturbation that seeds barotropic instability
and a rollup of cyclones along the jet's poleward flank.

**Extended description.** Each trajectory solves the global shallow-water
equations on a rotating sphere of Earth-like radius for sixteen simulated
days, of which the first four (linear spinup) are discarded, leaving a
twelve-day window in the post-saturation / turbulent regime. The initial
condition is a bi-hemispheric extension of the Galewsky et al. (2004) test
case: a pair of compact-support zonal jets — the canonical northern jet
plus a mirrored southern jet at the opposite latitude — geostrophically
+ cyclostrophically balanced by the corresponding height profile, with a
bi-Gaussian height bump applied **only on the northern jet** to break
hemispheric symmetry. The asymmetric trigger forces cross-equatorial
Rossby propagation as the dominant signal distinguishing trajectories.
The perturbed jet rolls up into filamentary cyclones by day 4–6 (the
standard barotropic-instability benchmark) while the unperturbed jet
slowly receives the Rossby wave train across the equator. The solver
itself runs in the canonical (polar-aligned) frame; a per-trajectory
random `SO(3)` tilt is applied at postprocess time so the physical
rotation axis is **not** aligned with the computational pole, removing
the grid-memorisation shortcut a CNN/ViT would otherwise exploit. The
ensemble explores a 5-axis parameter box over the jet strength, jet
latitude, perturbation amplitude, mean fluid depth, and SO(3)-tilt seed.

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
| `ν₀`  | `1.0 × 10⁵ m²/s`     | Reference Laplacian-equivalent viscosity (matched at `ℓ = 32`) |

### Hyperviscosity scaling

The biharmonic coefficient `ν` is set so its damping rate at spherical
harmonic degree `ℓ_match = 32` matches a Laplacian viscosity
`ν₀ = 10⁵ m²/s`, i.e. `ν · [ℓ(ℓ+1)/R²]² = ν₀ · ℓ(ℓ+1)/R²` at `ℓ = ℓ_match`,
giving `ν = ν₀ · R² / ℓ_match²`. The resolution scaling is
`ν(N_θ) = ν_base · (N_θ,ref / N_θ)⁴` with `N_θ,ref = 256`, so the
grid-scale damping time is invariant under resolution changes (the
biharmonic eigenvalue grows as `ℓ⁴`). All trajectories in this ensemble
run at `N_θ = 256`.

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
| `u_phi`     | `m/s` | Zonal (eastward) component of the horizontal velocity, in the **post-tilt** local east basis |
| `u_theta`   | `m/s` | Meridional component in the native colatitude direction (southward), in the **post-tilt** local south basis |
| `h`         | `m`   | Surface-height perturbation about mean depth `H` |
| `vorticity` | `1/s` | Relative vorticity `ζ = -∇·skew(u)` = vertical component of `∇×u` |

All fields stored as `float32`. After the SO(3) tilt the physical rotation
axis no longer coincides with the computational pole, so the Coriolis
parameter `f(λ, φ)` must be inferred from the velocity field itself
rather than read off the grid.

Two scalar metadata arrays are also written into each per-run Zarr to
make the tilt recoverable:

| Variable name    | Shape  | Dtype     | Description |
| ---              | ---    | ---       | --- |
| `so3_axis_xyz`   | `(3,)` | `float64` | Unit rotation axis in 3-D, drawn uniformly on `S²` from `seed` |
| `so3_angle_rad`  | `()`   | `float64` | Rotation angle in radians, drawn uniformly on `[0, 2π)` from `seed` |

### Dataset size

- **Number of trajectories.** 2 560.
- **Shape per trajectory.** `(time=289, field=4, lat=256, lon=512)`.
- **Per-trajectory size.** ≈ 577 MB (float32, uncompressed).
- **Total ensemble size.** ≈ 1.4 TB.
- **Consolidated shape.** `(run=2560, time=289, field=4, lat=256, lon=512)`.

### Storage format

Per-run Zarr v3 stores at
`…/galewsky-sw/processed/run_XXXX.zarr`, chunked as
`(time=1, field=4, lat=256, lon=512)` for fast per-snapshot random access.
A consolidated `dataset.zarr` stacks all runs along a leading `run`
dimension with parameter arrays carried as `param_*` coords. The
`manifest.json` at the dataset root carries the full list of
`(run_id, param_hash, params)` tuples.

### Train / val / test split

A fixed 80 / 10 / 10 split of the 2 560 run IDs is distributed with the
dataset as `splits.json` alongside `manifest.json`. The split is
**stratified on `u_max`** with a deterministic RNG seed so every split
sees all five jet-strength values in the target proportions. Counts are
`{train: 2048, val: 256, test: 256}`. Typical use:

```python
import json, xarray as xr

root   = "/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/galewsky-sw"
splits = json.load(open(f"{root}/splits.json"))
train_runs = [xr.open_zarr(f"{root}/processed/run_{i:04d}.zarr") for i in splits["train"]]
```

## Initial conditions

All trajectories share the same three-stage IC construction, parameterized
by the 5-axis grid below. The IC is built in the canonical (polar-aligned)
frame; the per-trajectory SO(3) tilt is applied at postprocess time (see
[Per-trajectory SO(3) tilt](#per-trajectory-so3-tilt)).

### 1. Bi-hemispheric zonal jet profile

A pair of compact-support analytic jets in latitude, summing the
canonical northern jet at `+lat_center` and a mirrored southern jet
at `−lat_center`. Each jet is nonzero only inside
`[lat_c − 20°, lat_c + 20°]` and reaches peak amplitude `u_max` at its
own jet centre:

$$
u_\phi(\varphi) \;=\; J_+(\varphi)\;+\;J_-(\varphi),\qquad
J_{\pm}(\varphi) \;=\; \frac{u_\mathrm{max}}{e_n}\,
  \exp\!\Bigl(\tfrac{1}{(\varphi - \varphi_0^{\pm})(\varphi - \varphi_1^{\pm})}\Bigr)
  \text{ inside its support, } 0 \text{ elsewhere},
$$

with `e_n = exp(−4/(φ₁ − φ₀)²)` chosen so the per-jet maximum equals
`u_max`. The jet supports never overlap because every grid value of
`lat_center` (≥ 30°) exceeds the half-width (`20°`). Meridional
velocity `u_θ ≡ 0` at initial time.

### 2. Geostrophic + cyclostrophic balance for `h`

The balanced surface-height field is obtained by 1-D numerical
integration of the zonal-flow balance ODE in latitude,

$$
\frac{\partial h}{\partial \varphi} = -\frac{1}{g}\,u_\phi\,
  \Bigl(2\Omega R\sin\varphi + u_\phi \tan\varphi\Bigr),
$$

driven by the **two-jet** `u_φ` profile, on a fine uniform grid in `φ`,
followed by cos-weighted area-mean subtraction so that `⟨h⟩ = 0`. The
resulting 1-D profile `h(φ)` is evaluated at the native solver
colatitude nodes by cubic interpolation. Because `u_φ = 0` near the
poles, `u_φ · tan(φ)` is well-behaved at the singular endpoints.

### 3. Asymmetric bi-Gaussian height perturbation

A localized Galewsky bump is added on top of the balanced height **on
the northern jet only**:

$$
h'(\lambda, \varphi) = \hat{h}\,\cos(\varphi)\,
  \exp\!\Bigl(-\bigl(\tfrac{\lambda - \lambda_c}{\beta}\bigr)^{\!2}\Bigr)\,
  \exp\!\Bigl(-\bigl(\tfrac{\varphi_c - \varphi}{\alpha}\bigr)^{\!2}\Bigr),
$$

with fixed widths `α = 1/3` (meridional) and `β = 1/15` (zonal),
perturbation amplitude `ĥ = h_hat`, and perturbation centre
`(λ_c, φ_c) = (0, +lat_center)`. The longitude of the bump is
hard-coded to `0` in the canonical frame; rotational diversity across
the ensemble is supplied by the per-trajectory SO(3) tilt rather than
by the previous `lon_c` axis. The southern jet is left unperturbed, so
delayed Rossby waves propagating across the equator are the only signal
that disturbs it within the simulation window.

## Per-trajectory SO(3) tilt

The Dedalus shallow-water solver uses the standard Coriolis term
`2·Ω·sin(φ)·k̂×u`, which is only valid in the frame where the rotation
axis is the polar axis. Running with a tilted rotation axis would break
that closed form. Instead, every trajectory runs in the canonical frame
and a per-trajectory rotation `R(ê, α) ∈ SO(3)` is applied **once**, at
postprocess time, to the resampled `(lat, lon)` snapshots. The pair
`(ê, α)` is drawn deterministically from the run's `seed`:

- `ê` uniform on `S²` (z uniform in `[−1, 1]`, longitude uniform in
  `[0, 2π)`), stored as `so3_axis_xyz`.
- `α` uniform in `[0, 2π)`, stored as `so3_angle_rad`.

Sampling the polar-frame field at the back-rotated grid points is
equivalent to running the simulation on a tilted-axis sphere. Scalar
fields (`h`, `vorticity`) are bicubic-interpolated at the back-rotated
query grid (with periodic-wrap padding in longitude). The velocity pair
`(u_phi, u_theta)` additionally gets the local-frame Jacobian applied
per gridpoint, so its components are expressed in the **output** local
east/south basis at every rotated location. The PCG64 generator is
keyed on `run_id` rather than on the `seed` axis value, so every one of
the 960 trajectories receives a distinct rotation — the `seed` axis is
purely the rotation-multiplicity index (see
[Parameter space](#parameter-space)). Implementation in
[`datagen/galewsky/so3.py`](../datagen/galewsky/so3.py); the
postprocess step that consumes it is in
[`datagen/galewsky/scripts/postprocess.py`](../datagen/galewsky/scripts/postprocess.py).

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
(5 · 4 · 2 · 4 · 6 = **960** runs). Runs are indexed `run_0000 …
run_0959` in row-major order over the tuple
`(u_max, lat_center, h_hat, H, seed)`.

| Parameter       | Units  | Values                                  | Count |
| --- | --- | --- | --- |
| `u_max`         | m/s    | 60, 70, 80, 90, 100                     | 5 |
| `lat_center`    | °N     | 30, 40, 50, 60                          | 4 |
| `h_hat`         | m      | 60, 240                                 | 2 |
| `H`             | m      | 8 000, 10 000, 12 000, 14 000           | 4 |
| `seed`          | —      | 0, 1, 2, 3, 4, 5                        | 6 |

Jet half-width (`40°` full width), perturbation widths
(`α = 1/3`, `β = 1/15`), grid resolution, and physics constants are
identical across the ensemble. The hard-coded perturbation longitude
`lon_c = 0` (in the canonical frame) is also identical across the
ensemble.

The `seed` axis is a rotation-multiplicity index: every
`(u_max, lat_center, h_hat, H)` combination is replicated 6 times
along it, and each replicate gets a distinct SO(3) tilt because the
tilt is keyed on `run_id` (not on the `seed` value itself). All 960
trajectories therefore receive 960 unique rotations. The other four
axes are physically meaningful; `seed` exists purely to multiply the
ensemble along the rotational direction so the model sees several
rotations of every physics combination. The tilt also breaks the
alignment between the physical rotation axis and the computational
pole, so a model can no longer infer the Coriolis parameter from the
grid latitude alone.

## Numerical-stability strategy

1. **Preflight sweep** over the 32 corners of the four physical axes
   (`u_max × lat_center × h_hat × H`, `seed = 0`) at half resolution
   (`N_φ = 256, N_θ = 128`) for 1 simulated day. All corners must
   complete without NaNs before the full sweep launches; this confirms
   the box is numerically stable.
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
- **Total compute.** ≈ 13 000 – 18 000 core-hours for the full
  ensemble (16 sim-days × 2 560 runs × 16 ranks × ~22 min).
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
