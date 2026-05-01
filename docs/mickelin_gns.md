# Mickelin Generalized Navier–Stokes Active Turbulence on the Sphere

**One-line description.** A parametric ensemble of chaotic,
active-matter vorticity fields on the 2-sphere, driven by a band-limited
linear instability that injects energy at a characteristic vortex size
`Λ` and relaxes into the anomalous A-phase "chained turbulence" regime
of Mickelin et al. (2018).

**Extended description.** Each trajectory solves the
vorticity–stream-function form of the Mickelin generalized
Navier–Stokes (GNS) equation on a unit sphere. The linear operator
`f(Δ+4K)(Δ+2K)` has a narrow band of unstable spherical-harmonic modes
centred on `ℓ ≈ πR/Λ` with half-width `≈ κR/2`. Inside the band,
vortices of size `Λ` are continuously injected; outside, dissipation
dominates. In the target A-phase regime `R⁻¹ < κ < Λ⁻¹` these vortices
self-organise into percolating anti-ferromagnetic chains. The
constant-mode and rigid-rotation directions of the configuration space
are removed by gauge constraints (`⟨ψ⟩ = 0`, `⟨ω⟩ = 0`) and by
excluding `ℓ = 0, 1` from the initial condition. The initial condition
seeds modes **directly inside** the unstable band (rather than far
below it) so the saturation is reached on an `O(τ)` timescale instead
of after tens of thousands of solver steps spent waiting for round-off
to bootstrap the instability. Unlike Galewsky (deterministic rollup), the Mickelin flow
is chaotic, and ensemble diversity comes from **random initial
conditions** rather than IC-parameter variation. Each trajectory is
nevertheless a fully deterministic function of the triple
`(r_over_lambda, kappa_lambda, seed)`, so the mapping from initial
state to trajectory is a well-defined learnable target.

## Associated resources

- **Paper.** Mickelin, O., Słomka, J., Burns, K. J., Lecoanet, D.,
  Le Bars, M., Fauve, S., & Dunkel, J. (2018). *Anomalous chained
  turbulence in actively driven flows on spheres*. Physical Review
  Letters, 120, 164503. [arXiv:1710.05525](https://arxiv.org/abs/1710.05525).
- **Data generator.** Till Muser (University of Basel), 2026.
- **Generation software.** [Dedalus v3](https://dedalus-project.org/) —
  MPI-parallel spectral PDE solver with `SphereBasis` spin-weighted
  spherical-harmonic transforms.
- **Code.** This repository, subsystem
  [`datagen/`](../datagen). Coefficient map, initial conditions, solver,
  resampling, and SLURM scripts are self-contained in that subproject.

## Mathematical framework

The governing equations are the vorticity–stream-function form of the
GNS equation on a sphere of radius `R` with Gaussian curvature
`K = 1/R²`:

$$
\begin{aligned}
\Delta\psi &= -\omega,\\
\partial_t \omega + J(\psi,\omega) &= f(\Delta + 4K)(\Delta + 2K)\,\omega,
\end{aligned}
$$

where `ω(θ, φ, t)` is the vertical component of vorticity, `ψ(θ, φ, t)`
the stream function, the divergence-free velocity is `v = skew(∇ψ)`,
and the Jacobian is

$$
J(\psi,\omega)\;=\;\frac{1}{R^2 \sin\theta}\bigl(\partial_\theta\omega\,\partial_\varphi\psi
  -\partial_\varphi\omega\,\partial_\theta\psi\bigr)
  \;=\; v\cdot\nabla\omega.
$$

The driving polynomial has the Mickelin form

$$
f(x) \;=\; \Gamma_0 \;-\; \Gamma_2\,x \;+\; \Gamma_4\,x^2,
\qquad \Gamma_0,\,\Gamma_4>0,\; \Gamma_2<0.
$$

### Spectral growth rate and band-limited instability

On a spherical-harmonic mode `Y_ℓ^m` the linear operator has eigenvalue

$$
\sigma(\ell) \;=\; f\!\bigl(-k^2 + 4K\bigr)\cdot(2K - k^2),
\qquad k^2 \equiv \ell(\ell+1)/R^2 .
$$

Because `Γ₄ > 0`, `f(x)` is an upward-opening parabola with two real
roots `X_lo < X_hi`, negative between them. Mapping back through
`x = −k² + 4K`, the unstable band in wavenumber is

$$
k_-^2 \;=\; (\pi/\Lambda)^2\bigl(1 - \kappa\Lambda/2\bigr)^2,\qquad
k_+^2 \;=\; (\pi/\Lambda)^2\bigl(1 + \kappa\Lambda/2\bigr)^2,
$$

with peak growth rate `σ(ℓ_peak) = 1/τ` by construction of the
amplitude rescaling. Peak ℓ and band edges in spherical-harmonic
units:

$$
\ell_\mathrm{peak} \approx \pi R/\Lambda,\qquad
\text{band width}\approx \pi\,\kappa R.
$$

The closed-form map `(R, Λ, κ, τ) → (Γ₀, Γ₂, Γ₄)` is implemented in
[`datagen/mickelin/coeffs.py`](../datagen/mickelin/coeffs.py) as
`coefficients_from_RLkT` and is unit-tested against the spectral
predictions in
[`datagen/mickelin/tests/test_coeffs.py`](../datagen/mickelin/tests/test_coeffs.py).

### Fixed non-dimensional scales

| Symbol | Value      | Meaning |
| ---    | ---        | --- |
| `R`    | `1`        | Sphere radius (non-dimensional) |
| `τ`    | `1`        | Inverse peak driving rate (time unit) |
| `K`    | `1/R² = 1` | Gaussian curvature |

All times are reported in units of `τ`; all lengths in units of `R`.

## Data specifications

### Grid

- **Discretization.** Regular `(lat, lon)` grid with `N_lat = 128`,
  `N_lon = 256`, equispaced and pole-excluding. Latitude centres
  `lat_k = -π/2 + (k + 0.5)·π/N_lat` in radians, stored as
  `degrees_north`. Longitude centres `lon_k = k·2π/N_lon` in `[0, 2π)`,
  stored as `degrees_east`.
- **Native solver grid.** Gauss–Legendre in colatitude with
  `N_θ = 128` and equispaced longitude with `N_φ = 256`, dealias factor
  `3/2`. Snapshots are resampled from the native grid to the regular
  `(lat, lon)` grid via per-longitude cubic-spline interpolation in
  colatitude — effectively exact below Nyquist.

### Temporal layout

- **Snapshot cadence.** Every `τ/5` = `0.2 τ` of simulated time.
- **Simulation length.** `130 τ`.
- **Output window.** `30 τ ≤ t ≤ 130 τ`; the first `30 τ` (activity /
  dissipation balance settling in) is discarded at resample time. The
  remaining 500 snapshots span `0 – 100 τ` on the exported `time` axis
  (rebased so the first kept snapshot is `t = 0`).
- **Trajectory shape.** 500 snapshots per trajectory, `τ/5` apart.
- **Time coordinate.** Stored as `τ` (`float64`).

### Available fields

| Field name   | Units   | Description                                   |
| ---          | ---     | ---                                           |
| `vorticity`  | `1/τ`   | Vorticity `ω` (full state for this system)    |

Stored as `float32`.

The velocity is *derivable* from vorticity via a single sphere-Poisson
solve `Δψ = −ω` followed by `v = skew(∇ψ)`; models that need velocity
should add a learned stream-function head or perform the elliptic
solve at load time.

### Dataset size

- **Number of trajectories.** 480.
- **Shape per trajectory.** `(time=500, field=1, lat=128, lon=256)`.
- **Per-trajectory size.** ≈ 65 MB (float32, uncompressed).
- **Total ensemble size.** ≈ 31 GB.
- **Consolidated shape.** `(run=480, time=500, field=1, lat=128, lon=256)`.

### Storage format

Per-run Zarr v3 stores at
`…/mickelin-gns/processed/run_XXXX.zarr`, chunked as
`(time=1, field=1, lat=128, lon=256)` for fast per-snapshot random
access. A consolidated `dataset.zarr` stacks all runs along a leading
`run` dimension with parameter arrays carried as `param_*` coords. The
`manifest.json` at the dataset root carries the full list of
`(run_id, param_hash, params)` tuples.

## Initial conditions

Each run is seeded with a small-amplitude random vorticity field

$$
\omega_0(\theta,\varphi) \;=\; \varepsilon\sum_{\ell=2}^{\ell_\mathrm{init}}
  \sum_{m=-\ell}^{\ell} a_{\ell m}\,Y_{\ell m}(\theta,\varphi),
\qquad a_{\ell m} \sim \mathcal{N}(0,1)\ \text{i.i.d.},
$$

where `Y_{ℓm}` is a real orthonormal basis of spherical harmonics
(cosine combinations for `m > 0`, sine combinations for `m < 0`,
implemented in [`datagen/_ylm.py`](../datagen/_ylm.py)), and the
coefficients `a_{ℓm}` are drawn from
`np.random.Generator(np.random.PCG64(seed))`. Fixed values:

- `ℓ_init` is chosen **inside the unstable band**, derived per-run as
  `⌈R·k₊⌉ + 4 = ⌈πR/Λ · (1 + κΛ/2)⌉ + 4` (a few degrees above the
  upper band edge). For the production grid this is
  `ℓ_init ∈ {11, 18, 28, 38}` for `r_over_lambda ∈ {2, 4, 7, 10}` at
  `kappa_lambda = 0.4`. Seeding directly inside the actively unstable
  band saturates the nonlinear balance on an `O(τ)` timescale; the
  previous choice of `ℓ_init = 6` (well below the band) wasted tens of
  thousands of solver steps waiting for round-off to bootstrap the
  instability.
- `ε = 10⁻³` — well inside the linear regime; saturation amplitude is
  then set by the nonlinear balance of drive, advection, and Rayleigh
  drag, not by the seed scale.
- `ℓ = 0, 1` are excluded: the constant and rigid-rotation modes are
  not dynamically relevant and can be dropped without loss.

This keeps each trajectory a deterministic function of the triple
`(r_over_lambda, kappa_lambda, seed)`.

## Physical setup

- **Domain.** Full sphere `S²`, unit radius, no boundary.
- **Boundary conditions.** None in the usual sense: `SphereBasis`
  enforces periodicity in longitude and smoothness at the poles
  intrinsically via the spin-weighted spherical-harmonic expansion.
- **Stream-function gauge.** `⟨ψ⟩ = 0` enforced via a scalar-tau
  constraint on the Poisson solve (the sphere Laplacian has a 1-D
  constant null space). `⟨ω⟩ = 0` is similarly enforced as a gauge.
- **Timestepper.** `RK222` IMEX; the 6th-order linear operator
  `f(Δ+4K)(Δ+2K)` goes on the implicit side, the advective Jacobian on
  the explicit side.
- **CFL-adaptive `dt`** (safety `0.3`, `max_dt = 0.05 τ`,
  `initial_dt = 0.005 τ`, max change 1.5, min change 0.5).
- **Integration window.** `0 ≤ t ≤ 130 τ`; snapshots stored at cadence
  `τ/5` only for `t ≥ 30 τ`.

## Parameter space

Runs are laid out on an explicit tensor grid over three axes
(`4 · 6 · 20 = 480` runs). Runs are indexed `run_0000 … run_0479` in
row-major order over the tuple
`(r_over_lambda, kappa_lambda, seed)`.

| Parameter        | Symbol          | Values                              | Count |
| ---              | ---             | ---                                 | ---   |
| Sphere/vortex    | `r_over_lambda` | 2, 4, 7, 10                         | 4     |
| Active bandwidth | `kappa_lambda`  | 0.2, 0.4, 0.7, 1.0, 1.4, 1.8        | 6     |
| IC seed          | `seed`          | 0, 1, …, 19                         | 20    |

Physics constants `R = 1` and `τ = 1` are fixed across the ensemble;
the per-run derived parameters are `Λ = R / r_over_lambda`
and `κ = kappa_lambda / Λ`, and `(Γ₀, Γ₂, Γ₄)` from the closed-form
coefficient map. `ℓ_init` is derived from `(R, Λ, κ)` so the IC always
seeds inside the unstable band. Resolution and time horizon are
identical across the ensemble.

The grid intentionally spans all three Mickelin regimes rather than
restricting to A-phase. The A-phase band `R⁻¹ < κ < Λ⁻¹` (equivalently
`κR > 1` and `κΛ < 1`) covers the subset with `kappa_lambda ∈ {0.2,
0.4, 0.7}` at sufficiently large `r_over_lambda`; the remaining
combinations probe the laminar (`κR < 1`) and broad-band (`κΛ ≥ 1`)
regimes. The solver constraint `κΛ < 2` (so that `k₋² > 0`) is satisfied
at every grid point, with `κΛ ≤ 1.8` at the upper edge.

## Numerical-stability strategy

1. **Preflight sweep** over the 8 extremes of the `r_over_lambda ×
   kappa_lambda` plane (all `r_over_lambda` × min/max `kappa_lambda`)
   at half resolution (`N_φ = 128, N_θ = 64`) for `20 τ` at `seed = 0`.
   Confirms every corner of the parameter plane is numerically stable
   before launching the full 480-run array.
2. **CFL-adaptive `dt`** inside every run, with a hard ceiling
   `max_dt = 0.05 τ`.
3. **Per-run try/except** around the solver loop: on failure the
   driver writes a `run_XXXX.FAILED` sentinel JSON with the exception
   and parameters, then exits non-zero. The SLURM array continues; a
   final `consolidate.py` step emits a report of any missing runs.
4. **IMEX split.** The 6th-order driving operator is treated
   implicitly so grid-scale damping does not impose a prohibitive
   timestep constraint; the advective Jacobian (at most quadratic in
   `∇ψ, ∇ω`) goes on the explicit side.

## Computational details

- **Per-run wall.** ≈ 5 – 15 min on one scicore compute node with 16
  MPI ranks (128-core `scicore` partition, one MPI rank per core with
  `OMP_NUM_THREADS = 1`).
- **Total compute.** ≈ 50 – 150 core-hours for the full ensemble.
- **Solver precision.** `float64` throughout the simulation; all
  output fields are downcast to `float32` at resample time.
- **Cluster.** sciCORE @ Universität Basel, `scicore` partition,
  `6hours` QoS, Easybuild `foss/2024a` +
  `HDF5/1.14.5-gompi-2024a` toolchain.

## End-to-end data flow

```
# Login-node (need PyPI access).
uv sync --project datagen

# Emit per-run JSON configs + manifest.
uv run --project datagen python -m datagen.mickelin.scripts.generate_sweep

# Optional preflight array (8 corners, low resolution, seed 0).
sbatch datagen/mickelin/slurm/preflight.sbatch

# Full 480-run sweep (Nphi=256, Ntheta=128, 130 τ).
sbatch datagen/mickelin/slurm/sweep.sbatch

# Consolidate per-run Zarrs into the final dataset.zarr.
DATASET_ROOT=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/mickelin-gns
uv run --project datagen python -m datagen.scripts.consolidate \
    --processed $DATASET_ROOT/processed/ \
    --manifest  $DATASET_ROOT/manifest.json \
    --out       $DATASET_ROOT/dataset.zarr
```

## Using the dataset

```python
import xarray as xr

root = "/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/mickelin-gns"

# One trajectory — lightweight random access.
run = xr.open_zarr(f"{root}/processed/run_0000.zarr")
print(run)

# Full ensemble — run, time, field, lat, lon.
ds = xr.open_zarr(f"{root}/dataset.zarr")
print(ds)

# Parameter lookup via the carried coords.
a_phase_small = ds.sel(run=(ds.param_r_over_lambda == 4.0)
                           & (ds.param_kappa_lambda == 0.4))
```

## Citation

If you use this dataset, please cite the original Mickelin et al. paper:

```bibtex
@article{mickelin2018anomalous,
  title   = {Anomalous chained turbulence in actively driven flows on spheres},
  author  = {Mickelin, Oscar and S{\l}omka, Jonasz and Burns, Keaton J.
             and Lecoanet, Daniel and Le Bars, Michael and Fauve, Stephan
             and Dunkel, J{\"o}rn},
  journal = {Physical Review Letters},
  volume  = {120},
  number  = {16},
  pages   = {164503},
  year    = {2018},
  doi     = {10.1103/PhysRevLett.120.164503},
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
