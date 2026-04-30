# Cahn-Hilliard Phase Separation on the Sphere

**One-line description.** A parametric ensemble of Cahn-Hilliard
phase-separation trajectories on the 2-sphere, evolved through spinodal
decomposition and coarsening from random Gaussian-noise initial
conditions on an unstructured spherical mesh.

**Extended description.** Each trajectory solves the conservative,
mobility-form Cahn-Hilliard equation on a thin spherical shell using
the NIST FiPy `examples/cahnHilliard/sphere.py` recipe: a Gmsh
`Gmsh2DIn3DSpace` mesh of the sphere extruded radially to a one-cell
shell so that face-based fluxes have a well-defined orientation. The
order parameter `φ ∈ [0, 1]` interpolates between two coexisting phases
at `φ ≈ 0` and `φ ≈ 1`, separated by diffuse interfaces of width set by
`ε`. From a small-amplitude per-cell Gaussian noise field around a
prescribed mean `mean_init`, the system rapidly undergoes spinodal
decomposition into the two phases, then coarsens at a slower power-law
rate. To resolve both regimes within a single trajectory, the
time-stepper uses an exponentially growing `dt`: tiny steps capture the
violent decomposition transient near `t = 0`, then `dt` ramps up to the
ceiling `max_dt` once the system enters the slow coarsening phase. Each
trajectory is bit-reproducible from the parameter tuple; ensemble
diversity comes from a 5-axis grid over the bulk free-energy scale, the
mean composition, the IC noise variance, the sphere radius, and the IC
seed.

## Associated resources

- **Papers.** Cahn, J. W. & Hilliard, J. E. (1958), *Free Energy of a
  Nonuniform System*. Cahn, J. W. (1961), *On Spinodal Decomposition*.
- **Data generator.** Till Muser (University of Basel), 2026.
- **Generation software.** [FiPy](https://www.ctcms.nist.gov/fipy/) —
  NIST finite-volume PDE solver, with the
  `examples/cahnHilliard/sphere.py` recipe used as the reference
  implementation.
- **Code.** This repository, subsystem
  [`datagen/`](../datagen). Solver, initial conditions, resampling, and
  SLURM scripts are in
  [`datagen/cahn_hilliard/`](../datagen/cahn_hilliard).

## Mathematical framework

The governing equation is the conservative, mobility-form Cahn-Hilliard
equation on the sphere `S²`,

$$
\partial_t \varphi
\;=\; \nabla\!\cdot\!\Bigl[D\,a^2\,\bigl(1 - 6\varphi(1-\varphi)\bigr)\,\nabla\varphi\Bigr]
\;-\; \nabla\!\cdot\!\Bigl[D\,\varepsilon^2\,\nabla(\nabla^2\varphi)\Bigr],
$$

which is equivalent to the chemical-potential form
`∂_tφ = D·∇²μ` with

$$
\mu \;=\; a^2\,f'(\varphi) \;-\; \varepsilon^2\,\nabla^2\varphi,
\qquad
f(\varphi) \;=\; \tfrac{1}{2}\,\varphi^2 (1-\varphi)^2,
$$

since `f'(φ) = φ − 3φ² + 2φ³` has derivative `1 − 6φ(1−φ)`. Here
`φ ∈ [0, 1]` is the order parameter, `D` the (constant) Cahn-Hilliard
mobility, `a` the bulk free-energy scale, and `ε` the interface-width
parameter. The two phases coexist at `φ ≈ 0` and `φ ≈ 1`. Because the
RHS is the divergence of a flux, the spatial integral `∫φ dA` is
exactly conserved by the continuous problem.

### Linear instability and pattern wavelength

Linearising about the spatial mean `φ = m₀`, the homogeneous state is
unstable when `f''(m₀) = 6m₀² − 6m₀ + 1 < 0`, i.e. for
`m₀ ∈ ((3−√3)/6, (3+√3)/6) ≈ (0.211, 0.789)`. Inside that window the
fastest-growing wavenumber is `k_max = √(a²·|f''(m₀)|/(2ε²))` and the
characteristic pattern wavelength is `λ ≈ 2π/k_max`. With `a = D = 1`
and small `m₀ − 0.5`, the wavelength scales linearly with `ε` and the
domain footprint scales linearly with `R`, so the dimensionless count
of patterns on the sphere is `O(R/ε)`.

### Fixed non-dimensional scales

| Symbol | Value | Meaning |
| ---    | ---   | --- |
| `D`    | `1.0` | Cahn-Hilliard mobility |
| `a`    | `1.0` | Bulk free-energy amplitude |

All quantities are dimensionless; the saved `time` axis is in solver
time units.

## Data specifications

### Grid

- **Discretization (output).** Regular `(lat, lon)` grid with
  `N_lat = 256`, `N_lon = 512`, equispaced and pole-excluding. Latitude
  centres `lat_k = -π/2 + (k + 0.5)·π/N_lat` in radians, stored as
  `degrees_north`. Longitude centres `lon_k = k·2π/N_lon` in `[0, 2π)`,
  stored as `degrees_east`.
- **Native solver grid.** Unstructured Gmsh `Gmsh2DIn3DSpace` mesh of
  the sphere surface (`cell_size = 0.3`, six surface patches stitched
  together for well-conditioned meshing) radially extruded by a factor
  `1.1` to a one-cell-thick shell. Cell count grows quadratically with
  `radius`; cell-centre Cartesian coordinates `(x, y, z)` are stored
  alongside the snapshots.
- **Resampling.** Snapshots are mapped from the unstructured mesh to
  the regular `(lat, lon)` grid by inverse-distance weighting on the
  `k = 4` nearest cell centres on the unit sphere (single `cKDTree`
  query, reused across all snapshots).

### Temporal layout

- **Snapshot cadence.** Every `10` solver-time units.
- **Simulation length.** `500` solver-time units.
- **Output window.** `0 ≤ t ≤ 500`. No burn-in is discarded — the
  `t = 0` Gaussian-noise IC and the violent spinodal-decomposition
  transient are part of what the dataset is designed to capture.
- **Trajectory shape.** 51 snapshots per trajectory (`t = 0, 10, …,
  500`).
- **Time coordinate.** Stored as solver time units (`float64`).

### Available fields

| Field name | Units         | Description |
| ---        | ---           | --- |
| `phi`      | dimensionless | Order parameter `φ ∈ [0, 1]`; coexisting phases at `φ ≈ 0` and `φ ≈ 1` |

Stored as `float32`. Small transient overshoots of `φ` outside `[0, 1]`
are expected during interface formation and relax back as interfaces
sharpen.

### Dataset size

- **Number of trajectories.** 576.
- **Shape per trajectory.** `(time=51, field=1, lat=256, lon=512)`.
- **Per-trajectory size.** ≈ 27 MB (float32, uncompressed).
- **Total ensemble size.** ≈ 15 GB.
- **Consolidated shape.** `(run=576, time=51, field=1, lat=256, lon=512)`.

### Storage format

Per-run Zarr v3 stores at
`…/cahn-hilliard-sphere/processed/run_XXXX.zarr`, chunked as
`(time=1, field=1, lat=256, lon=512)` for fast per-snapshot random
access. A consolidated `dataset.zarr` stacks all runs along a leading
`run` dimension with parameter arrays carried as `param_*` coords. The
`manifest.json` at the dataset root carries the full list of
`(run_id, param_hash, params)` tuples.

## Initial conditions

Each run is seeded with a small-amplitude per-cell Gaussian field

$$
\varphi_0(c) \;\sim\; \mathcal{N}\!\bigl(\text{mean\_init},\ \text{variance}\bigr),
\qquad c \in \text{mesh cells},
$$

drawn from `np.random.default_rng(seed)` with one i.i.d. sample per
mesh cell. The IC is therefore unstructured; no spectral truncation is
applied. The mean composition `mean_init` and the noise scale
`variance` are both swept axes (see [Parameter space](#parameter-space)).
Implementation in
[`datagen/cahn_hilliard/ic.py`](../datagen/cahn_hilliard/ic.py).

## Physical setup

- **Domain.** Full sphere `S²`, radius `radius`, no boundary. The thin
  radial extrusion gives the FV stencil unambiguous face normals; the
  dynamics are tangential because the IC and the equation are.
- **Boundary conditions.** None: the extruded shell is closed on
  itself.
- **Mass conservation.** The whole RHS is the divergence of a flux, so
  `∫φ dA` is exactly conserved by the continuous problem and to within
  solver tolerance by the FV discretisation.
- **Time stepper.** First-order implicit (FiPy default for
  `TransientTerm() == DiffusionTerm(...) - DiffusionTerm(...)`), one
  linear solve per step.
- **Time-step schedule.** `dt = min(max_dt, exp(dexp))` with
  `dexp` initialised to `−5.0` and incremented by `0.01` per step,
  capped by `max_dt = 100`. The early steps are sub-`O(10⁻²)` (resolving
  the spinodal-decomposition transient) and `dt` saturates at the
  ceiling once coarsening is the dominant dynamic. The final step is
  shrunk to land exactly on `stop_sim_time`.
- **Snapshot logic.** `t = 0` is always written; subsequent snapshots
  are appended whenever the running `elapsed` first crosses an integer
  multiple of `snapshot_dt`.

## Parameter space

Runs are laid out on an explicit tensor grid over five axes
(`4 · 3 · 4 · 3 · 4 = 576` runs). Runs are indexed `run_0000 …
run_0575` in row-major order over the tuple
`(epsilon, mean_init, variance, radius, seed)`.

| Parameter           | Symbol      | Values                          | Count |
| ---                 | ---         | ---                             | ---   |
| Interface width     | `epsilon`   | 0.5, 1.0, 1.5, 2.0              | 4     |
| Mean composition    | `mean_init` | 0.35, 0.50, 0.65                | 3     |
| IC noise variance   | `variance`  | 0.001, 0.005, 0.01, 0.05        | 4     |
| Sphere radius       | `radius`    | 5.0, 7.5, 10.0                  | 3     |
| IC seed             | `seed`      | 0, 1, 2, 3                      | 4     |

Held fixed across the ensemble: `D = 1`, `a = 1`. The held-fixed
scalars are embedded in every per-run config so the solver never reads
from a default at run time.

The `mean_init` axis spans symmetric (`0.50`) and asymmetric
(`0.35, 0.65`) compositions, all inside the spinodally-unstable window
`((3−√3)/6, (3+√3)/6) ≈ (0.211, 0.789)`. The `epsilon` and `radius`
axes together control the dimensionless pattern count on the sphere
(`O(radius/epsilon)`); the four `epsilon` and three `radius` values
combine to span an order-of-magnitude range. The `variance` axis
ranges from a barely-perturbed near-uniform IC (`variance = 0.001`) to
a strongly-perturbed IC (`variance = 0.05`) that already contains
domain-scale structure at `t = 0`.

## Numerical-stability strategy

1. **Exponential `dt` schedule.** The `dt = exp(dexp)` ramp keeps the
   step small during the early spinodal transient (when interfaces are
   still forming and the cubic nonlinearity is stiff), then grows it
   into the slow coarsening regime where the RHS is small. The
   `max_dt = 100` ceiling keeps the implicit linear solver
   well-conditioned.
2. **Per-step finite check.** After every step the driver verifies
   `np.all(np.isfinite(φ))` and raises `RuntimeError` on the first
   non-finite value, so a runaway step is caught immediately rather
   than silently corrupting the trajectory.
3. **Per-run try/except.** On exception the driver writes a
   `run_XXXX.FAILED` sentinel JSON with the parameters and traceback,
   then exits non-zero. The SLURM array continues; a final
   `consolidate.py` step emits a report of any missing runs.
4. **Thin radial extrusion.** Extruding the 2-D-in-3-D surface mesh
   into a one-cell-thick shell (extrude factor `1.1`) gives every face
   a well-defined inward/outward normal; this is the workaround the
   NIST FiPy CH-on-sphere example uses to make face-based fluxes
   well-posed on a curved manifold.

## Computational details

- **Per-run wall.** ≈ 5 – 30 min on one scicore compute node, single
  process (FiPy is not MPI-parallelised on this mesh). Wall scales
  with mesh cell count, which grows quadratically with `radius`.
- **Total compute.** ≈ 50 – 250 core-hours for the full 576-run
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

# Full 576-run sweep (one trajectory per array task, single-process).
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

# Parameter lookup via the carried coords (e.g. all symmetric-mixture runs).
symmetric = ds.sel(run=ds.param_mean_init == 0.5)
```

See the repo notebook
[`notebooks/cahn_hilliard_visualize.ipynb`](../notebooks/cahn_hilliard_visualize.ipynb)
for snapshots of the IC, panels through the spinodal-decomposition
transient and the coarsening regime, and an optional MP4 animation on
a rotating globe.

## Citation

If you use this dataset, please cite the foundational Cahn-Hilliard
papers:

```bibtex
@article{cahn1958free,
  title   = {Free Energy of a Nonuniform System.
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
```

along with the solver used to generate this ensemble:

```bibtex
@article{guyer2009fipy,
  title   = {{FiPy}: Partial Differential Equations with {Python}},
  author  = {Guyer, Jonathan E. and Wheeler, Daniel and Warren, James A.},
  journal = {Computing in Science \& Engineering},
  volume  = {11},
  number  = {3},
  pages   = {6--15},
  year    = {2009},
  doi     = {10.1109/MCSE.2009.52},
}
```
