# Cahn-Hilliard Phase Separation on the Sphere

**One-line description.** A parametric ensemble of forced Cahn-Hilliard
trajectories on the 2-sphere, driven by a single rotating spherical
harmonic `Y_ℓ^m` (Krekhov-style), evolved through spinodal decomposition
in a silent burn-in window and then captured in the mature, statistically
steady state.

**Extended description.** Each trajectory solves the
Krekhov / Weith–Krekhov–Zimmermann (2009) variant of the Cahn-Hilliard
equation on a sphere of non-dimensional radius `R = 5`. The free energy
is the symmetric `ψ⁴` double well (`ψ ∈ [-1, 1]`) and the linear
control parameter `ε > 0` puts the system in the ordered phase. A
spatially-periodic external forcing `a · F(x̂, t)` enters through the
chemical potential; `F = Re[Y_ℓ^m(R_ê(ω·t)·x̂)]` is a single real
spherical harmonic whose pattern rotates rigidly about a per-trajectory
tilted axis `ê` drawn from `seed`. Setting the rotation rate `ω = 0`
recovers the Krekhov locked pattern; non-zero `ω` breaks both spatial
and temporal symmetry simultaneously. Each run consists of two phases:
a **silent burn-in** during which the spinodal-decomposition transient
plays out without writing any snapshots, followed by a **production
window** during which `ψ` and the instantaneous forcing field are
written at fixed cadence on the regular `(lat, lon)` grid. Storing the
forcing field alongside `ψ` lets a dataloader concatenate the local
forcing context with the `ψ` history at training time, so the network
does not need to deduce the external heating schedule from `ψ`
snapshots alone.

## Associated resources

- **Papers.** Cahn, J. W. & Hilliard, J. E. (1958), *Free Energy of a
  Nonuniform System*. Cahn, J. W. (1961), *On Spinodal Decomposition*.
  Weith, V., Krekhov, A., & Zimmermann, W. (2009),
  *Periodic structures in a model with spatial extension and external
  forcing* ([arXiv:0809.0211](https://arxiv.org/abs/0809.0211)).
- **Data generator.** Till Muser (University of Basel), 2026.
- **Generation software.** [Dedalus v3](https://dedalus-project.org/) —
  MPI-parallel spectral PDE solver with `SphereBasis` spin-weighted
  spherical-harmonic transforms.
- **Code.** This repository, subsystem
  [`datagen/`](../datagen). Solver, initial conditions, rotating
  forcing, resampling, and SLURM scripts are in
  [`datagen/cahn_hilliard/`](../datagen/cahn_hilliard); the shared
  real-spherical-harmonic basis is in
  [`datagen/_ylm.py`](../datagen/_ylm.py).

## Mathematical framework

The governing equation is the Krekhov-form Cahn-Hilliard equation on
the sphere `S²`,

$$
\partial_t \psi
\;=\; \nabla^2\!\Bigl[
  -\,\varepsilon\,\psi
  \;+\; \psi^3
  \;-\; \xi^2\,\nabla^2 \psi
  \;+\; a\,F(\hat{\mathbf{x}},\,t)
\Bigr]
\;=\;
-\,\varepsilon\,\nabla^2\psi
\;-\; \xi^2\,\nabla^4\psi
\;+\; \nabla^2(\psi^3)
\;+\; a\,\nabla^2 F .
$$

Here `ψ ∈ [-1, 1]` is the order parameter, `ε > 0` the bulk control
parameter (positive `⇒` ordered phase), `ξ` the interface width, and
`a` the forcing amplitude. The external field

$$
F(\hat{\mathbf{x}},\,t) \;=\;
  \mathrm{Re}\!\left[\,Y_{\ell}^{m}\!\bigl(R_{\hat{\mathbf{e}}}(\omega\,t)\cdot\hat{\mathbf{x}}\bigr)\,\right]
$$

is a single real spherical harmonic of degree `ℓ` and order `m` whose
pattern rotates rigidly at rate `ω` about a per-trajectory unit axis
`ê` (drawn uniformly on `S²` from `seed`). Because `∇²Y_ℓ^m =
−ℓ(ℓ+1)/R² · Y_ℓ^m` is exact in the spectral basis, `∇²F` is computed
spectrally with no aliasing leakage. The PDE conserves the spatial mean
`⟨ψ⟩` (only `∇²` of the chemical potential acts on `ψ`); the gauge
condition `⟨ψ⟩ = ψ_mean` is enforced via a scalar `tau`.

### Linear instability and pattern wavelength

Linearising about `ψ = 0`, modes of wavenumber `k` grow at rate

$$
\sigma(k) \;=\; \varepsilon\,k^2 \;-\; \xi^2\,k^4,
$$

so the homogeneous state is unstable for `0 < k² < ε/ξ²`, with
fastest-growing wavenumber `k_max = √(ε/(2ξ²))`. On a sphere of radius
`R` patterns require `ε/ξ² > 2/R²` (so at least one `ℓ ≥ 2` mode is
unstable). The Krekhov-locked solution under spatially-periodic forcing
matches the forcing wavelength `λ ≈ 2π/k_max`; setting the forcing
harmonic at `ℓ ≈ R · k_max` keeps the locking regime accessible.

### Fixed non-dimensional scales

| Symbol      | Value | Meaning |
| ---         | ---   | --- |
| `R`         | `5`   | Sphere radius (non-dimensional) |
| `ε`         | `1.0` | Bulk control parameter (ordered phase) |
| `mean_init` | `0.0` | Initial spatial mean of `ψ` |
| `psi_mean`  | `0.0` | Conserved gauge mean of `ψ` |
| `variance`  | `0.01`| IC noise variance |
| `ell_init`  | `6`   | Highest `ℓ` in the bandlimited Gaussian IC |

Time is dimensionless. With `ε = 1` the linear instability rate scales
with `1/ξ²` and the wavelength with `ξ`, so the three `ξ` values in the
sweep span a factor-of-four range in pattern scale.

## Data specifications

### Grid

- **Discretization.** Regular `(lat, lon)` grid with `N_lat = 128`,
  `N_lon = 256`, equispaced and pole-excluding. Latitude centres
  `lat_k = -π/2 + (k + 0.5)·π/N_lat` in radians, stored as
  `degrees_north`. Longitude centres `lon_k = k·2π/N_lon` in `[0, 2π)`,
  stored as `degrees_east`.
- **Native solver grid.** Gauss-Legendre in colatitude with
  `N_θ = 128` and equispaced longitude with `N_φ = 256`, dealias factor
  `3/2`. Snapshots are resampled from the native grid to the regular
  `(lat, lon)` grid via per-longitude cubic-spline interpolation in
  colatitude — effectively exact below Nyquist.

### Temporal layout

- **Snapshot cadence.** Every `10` solver-time units.
- **Burn-in window.** `200` solver-time units, run silently before any
  snapshots are written. The first ~3% of an unforced CH simulation
  consists of an unphysically violent spinodal-decomposition transient
  from pure Gaussian noise; that transient is unrelated to the mature,
  forcing-driven dynamics we want to learn, so it is discarded by
  evolving the PDE without an attached `FileHandler`. The forcing field
  is updated every step during burn-in so the system already feels the
  correct rotating pattern before the first snapshot.
- **Production length.** `2 000` solver-time units after burn-in.
- **Output window.** Production only. Snapshot times on disk start at
  `t = 0` and are not offset by the burn-in.
- **Trajectory shape.** 201 snapshots per trajectory, `10` solver units
  apart (`t = 0, 10, …, 2 000`).
- **Time coordinate.** Stored as solver units (`float64`).

### Available fields

| Field name | Units            | Description |
| ---        | ---              | --- |
| `psi`      | dimensionless    | Order parameter `ψ ∈ [-1, 1]`; coexisting phases at `ψ ≈ ±1` |
| `forcing`  | dimensionless    | Instantaneous forcing field `F(x̂, t) = Re[Y_ℓ^m(R_ê(ω·t)·x̂)]` |

Both fields stored as `float32`. The `forcing` field is written at the
**same** snapshot cadence as `psi`, so a dataloader can concatenate the
local thermal/forcing context with the `ψ` history at training time —
the trick used by *The Well* to give the network direct access to the
external driving environment instead of forcing it to deduce the
heating schedule from `ψ` alone. Small transient overshoots of `ψ`
outside `[-1, 1]` are expected during interface formation; they relax
back as interfaces sharpen.

### Dataset size

- **Number of trajectories.** 144.
- **Shape per trajectory.** `(time=201, field=2, lat=128, lon=256)`.
- **Per-trajectory size.** ≈ 53 MB (float32, uncompressed).
- **Total ensemble size.** ≈ 7.6 GB.
- **Consolidated shape.** `(run=144, time=201, field=2, lat=128, lon=256)`.

### Storage format

Per-run Zarr v3 stores at
`…/cahn-hilliard-sphere/processed/run_XXXX.zarr`, chunked as
`(time=1, field=2, lat=128, lon=256)` for fast per-snapshot random
access. A consolidated `dataset.zarr` stacks all runs along a leading
`run` dimension with parameter arrays carried as `param_*` coords. The
`manifest.json` at the dataset root carries the full list of
`(run_id, param_hash, params)` tuples; per-run configs additionally
store the held-fixed scalars under a `fixed` key.

## Initial conditions

Each run is seeded with a small-amplitude bandlimited Gaussian noise
field

$$
\psi_0(\theta,\varphi) \;=\;
  m_0 \;+\; \sqrt{\mathrm{var}_0}\,
  \sum_{\ell=2}^{\ell_\mathrm{init}}\sum_{m=-\ell}^{\ell}
   a_{\ell m}\,Y_{\ell m}(\theta,\varphi),
\qquad a_{\ell m}\sim\mathcal{N}(0,1)\ \text{i.i.d.},
$$

with `m₀ = mean_init = 0`, `var₀ = 0.01`, and `ℓ_init = 6`. The
coefficients are drawn from `np.random.Generator(np.random.PCG64(seed))`
with the same `seed` that picks the per-trajectory rotation axis `ê`,
so each trajectory is bit-reproducible from the parameter tuple.
Modes `ℓ = 0` (the mean is set explicitly via `mean_init`) and `ℓ = 1`
(no preferred axis — the rotational symmetry would be broken only by
numerics) are both excluded.

The forcing axis `ê` is drawn separately from the same `seed`: `ê.z`
uniform on `[-1, 1]`, longitude uniform on `[0, 2π)`, giving a unit
vector uniform on `S²`. See
[`datagen/cahn_hilliard/forcing.py`](../datagen/cahn_hilliard/forcing.py).

## Physical setup

- **Domain.** Full sphere `S²`, radius `R = 5`, no boundary.
- **Boundary conditions.** None: `SphereBasis` enforces periodicity in
  longitude and smoothness at the poles intrinsically via the
  spin-weighted spherical-harmonic expansion.
- **Mass conservation.** The whole RHS is `∇²(...)`, so `⟨ψ⟩` is
  exactly conserved by the continuous problem; the Dedalus
  discretisation conserves it to within solver tolerance, with the
  gauge `⟨ψ⟩ = psi_mean` enforced via a scalar `tau`.
- **Timestepper.** `RK443` IMEX (4-stage, 3rd-order) — the explicit
  region is generous enough to handle the `∇²(ψ³)` term, whose local
  eigenvalue at saturation can spike well above `3ε·k²`.
- **IMEX split.** The stiff linear terms (`+ε∇²ψ`, `+ξ²∇⁴ψ`, plus the
  scalar gauge tau) are implicit; the cubic `∇²(ψ³)` and the forcing
  `a·∇²F` are explicit.
- **Time-step selection.** Explicit CFL `dt ≲ 0.2 · ξ² / (3ε)`, capped
  by `max_dt = 0.5`. When `ω ≠ 0` the step is additionally bounded so
  there are at least 12 substeps per visible forcing period
  `2π/(|ω|·max(m, 1))`, ensuring the rotating pattern is resolved
  smoothly.
- **Burn-in.** `burn_in_time = 200` solver-time units of silent
  evolution (no `FileHandler` attached) so the spinodal transient
  finishes before any snapshot is written. After burn-in the solver's
  `sim_time` and `iteration` are reset to `0` and the production
  `FileHandler` is attached, so on-disk snapshot times read
  `0, 10, …, 2 000` rather than `200, 210, …, 2 200`.
- **Integration window.** Total wall-time per run covers
  `[0, burn_in_time + stop_sim_time] = [0, 2 200]` solver units;
  snapshots cover only the production window `[0, 2 000]`.

## Parameter space

Runs are laid out on an explicit tensor grid over five axes
(`3 · 2 · 2 · 3 · 4 = 144` runs). Runs are indexed `run_0000 …
run_0143` in row-major order over the tuple
`(xi, ell, amplitude, omega, seed)`.

| Parameter             | Symbol      | Values                | Count |
| ---                   | ---         | ---                   | ---   |
| Interface width       | `xi`        | 0.5, 1.0, 2.0         | 3     |
| Forcing degree        | `ell`       | 6, 12                 | 2     |
| Forcing amplitude     | `amplitude` | 0.01, 0.03            | 2     |
| Forcing rotation rate | `omega`     | 0.0, 0.005, 0.015     | 3     |
| IC + axis seed        | `seed`      | 0, 1, 2, 3            | 4     |

The sectoral order `m = ell` is locked to `ell` (sectoral mode, e.g.
`Y_6^6`). Held fixed across the ensemble: `epsilon = 1`,
`mean_init = 0`, `variance = 0.01`, `psi_mean = 0`, `ell_init = 6`,
`R = 5`, `burn_in_time = 200`, `stop_sim_time = 2 000`,
`snapshot_dt = 10`. Held-fixed scalars are embedded in every per-run
config so the solver never reads from a default at run time.

The three `omega` values span three regimes: `0.0` (locked Krekhov
pattern), `0.005` (slow rotation, near the locking-to-irregular
boundary), `0.015` (fast rotation, irregular dynamics). The two `ell`
values span an order-of-magnitude range in pattern wavenumber; each
combines with the three `xi` values to produce both
`ξ²·ℓ²/R² ≪ ε` (intrinsic-pattern-dominated) and `ξ²·ℓ²/R² ≳ ε`
(forcing-pattern-dominated) regimes.

The `seed` axis randomises both the IC noise field and the forcing
rotation axis `ê`, so two runs with the same physical parameters but
different seeds disagree about *where* the rotating pattern points, in
addition to disagreeing about the IC.

## Numerical-stability strategy

1. **Burn-in.** The 200-unit silent burn-in window absorbs the violent
   spinodal-decomposition transient before any snapshot is captured;
   the network is therefore trained on the mature, statistically steady
   regime rather than on a one-off start-up shock.
2. **CFL-bounded `dt`.** Explicit step bounded by
   `0.2 · ξ²/(3ε)` and by `forcing_period / 12`, in addition to a hard
   ceiling `max_dt = 0.5`. The implicit `RK443` split keeps the stiff
   linear terms unconditionally stable.
3. **Per-step finite-check.** Every 200 iterations the driver does an
   `MPI.allreduce` of `np.isfinite(ψ).all()` and the global min/max of
   `ψ` so non-finite values (or runaway saturation) are caught early
   rather than silently corrupting the trajectory.
4. **Per-run try/except.** On exception the driver writes a
   `run_XXXX.FAILED` sentinel JSON with the parameters and traceback,
   then exits non-zero. The SLURM array continues; a final
   `consolidate.py` step emits a report of any missing runs.
5. **Spectral forcing.** Because the forcing pattern is a single
   `Y_ℓ^m`, `∇²F` is exact in the spectral basis; the forcing
   contributes no spectral leakage that would need extra hyperviscosity
   to control.

## Computational details

- **Per-run wall.** ≈ 5 – 15 min on one scicore compute node with 16
  MPI ranks (128-core `scicore` partition, one MPI rank per core with
  `OMP_NUM_THREADS = 1`).
- **Total compute.** ≈ 15 – 50 core-hours for the full 144-run
  ensemble.
- **Solver precision.** `float64` throughout the simulation; output
  fields are downcast to `float32` at resample time.
- **Cluster.** sciCORE @ Universität Basel, `scicore` partition,
  `6hours` QoS, Easybuild `foss/2024a` +
  `HDF5/1.14.5-gompi-2024a` toolchain.

## End-to-end data flow

```
# Login-node (need PyPI access).
uv sync --project datagen

# Emit per-run JSON configs + manifest.
uv run --project datagen python -m datagen.cahn_hilliard.scripts.generate_sweep

# Full 144-run sweep (one trajectory per array task, 16 MPI ranks each).
sbatch datagen/cahn_hilliard/slurm/sweep.sbatch

# Consolidate per-run Zarrs into the final dataset.zarr.
DATASET_ROOT=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/cahn-hilliard-sphere
uv run --project datagen python -m datagen.scripts.consolidate \
    --processed $DATASET_ROOT/processed/ \
    --manifest  $DATASET_ROOT/manifest.json \
    --out       $DATASET_ROOT/dataset.zarr
```

## Using the dataset

```python
import xarray as xr

root = "/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/cahn-hilliard-sphere"

# One trajectory — lightweight random access.
run = xr.open_zarr(f"{root}/processed/run_0000.zarr")
print(run)

# Full ensemble — run, time, field, lat, lon.
ds = xr.open_zarr(f"{root}/dataset.zarr")
print(ds)

# Concatenate forcing context with psi history at training time:
psi     = ds["fields"].sel(field="psi")
forcing = ds["fields"].sel(field="forcing")
inputs  = xr.concat([psi, forcing], dim="channel")  # (run, time, channel, lat, lon)

# Parameter lookup via the carried coords (e.g. all locked-pattern runs).
locked = ds.sel(run=ds.param_omega == 0.0)
```

See the repo notebook
[`notebooks/cahn_hilliard_visualize.ipynb`](../notebooks/cahn_hilliard_visualize.ipynb)
for snapshots of the IC, panels through the post-burn-in production
window, the rotating forcing field, and an optional MP4 animation on a
rotating globe.

## Citation

If you use this dataset, please cite the foundational Cahn-Hilliard
papers and the Krekhov-style forcing model:

```bibtex
@article{cahn1958free,
  title   = {Free Energy of a Nonuniform System.\
             I. Interfacial Free Energy},
  author  = {Cahn, John W. and Hilliard, John E.},
  journal = {The Journal of Chemical Physics},
  volume  = {28},
  number  = {2},
  pages   = {258--267},
  year    = {1958},
  doi     = {10.1063/1.1744102},
}

@article{cahn1961spinodal,
  title   = {On Spinodal Decomposition},
  author  = {Cahn, John W.},
  journal = {Acta Metallurgica},
  volume  = {9},
  number  = {9},
  pages   = {795--801},
  year    = {1961},
  doi     = {10.1016/0001-6160(61)90182-1},
}

@article{weith2009krekhov,
  title   = {Periodic structures in a model with spatial extension
             and external forcing},
  author  = {Weith, V. and Krekhov, A. and Zimmermann, W.},
  journal = {arXiv:0809.0211},
  year    = {2009},
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
