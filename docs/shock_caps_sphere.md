# Shock Caps: Random-Cap Riemann Shallow Water Dynamics on the Sphere

**One-line description.** A 500-trajectory ensemble of hyperbolic
shallow-water dynamics on the 2-sphere, initialized with $K$ random
geodesic caps — each carrying an independent piecewise-constant fluid
state — that produce multi-shock collisions, expanding bores, and
antipodal focus without any grid-aligned symmetry. The ensemble is
laid out on a $(K, \delta, \text{seed})$ grid that exposes cap count
and velocity-flow strength as explicit axes.

**Extended description.** Each trajectory solves the rotation-free
shallow water equations on a unit sphere. The initial condition places
$K$ geodesic disks (spherical caps) at uniformly random locations on
the sphere; each cap has a randomly drawn angular radius and a
piecewise-constant primitive state $(h, u, v)$. The flow-strength
parameter $\delta \in [0, 1]$ scales the velocity envelope only:
$u, v \in [-0.5\delta, 0.5\delta]$. Depth jumps are independent of
$\delta$, so even at $\delta=0$ the initial pressure imbalance drives
full-amplitude Riemann fans; at $\delta=1$ caps additionally carry
nontrivial advective momentum. Where caps overlap, a painter's
algorithm gives precedence to higher-index caps. A background state
covers any region not claimed by a cap. Because cap centres are drawn
uniformly on $\mathbb{S}^2$ from the outset, no separate $SO(3)$ tilt
step is required — the shock interfaces are already non-aligned with
any computational axis. The resulting dynamics include expanding
bores, multi-shock collisions at cap boundaries, and antipodal
convergence as fronts wrap around the closed manifold. Compared to the
shock-quadrants dataset, the cap geometry avoids the fourfold symmetry
of equator/meridian partitioning and naturally generates a wider
variety of partial-sphere Riemann configurations, including isolated
island-like states and multi-region interactions.

## Associated resources

- **Papers.**
  - LeVeque, R. J., & Calhoun, D. A. (2002). *Shallow water flow on the sphere*.
  - Calhoun, D. A., Helzel, C., & LeVeque, R. J. (2008). *Logically Rectangular Grids and Finite Volume Methods for PDEs in Circular and Spherical Domains*. SIAM Review 50, 723–752.
  - Lax, P. D., & Liu, X.-D. (1998). *Solution of two-dimensional Riemann problems of gas dynamics by positive schemes*. SIAM Journal on Scientific Computing, 19(2), 319–340.
- **Data generator.** Till Muser (University of Basel), 2026.
- **Generation software.** [Clawpack / PyClaw](https://www.clawpack.org/) — high-resolution finite-volume solver with mapped spherical-grid support and wave-propagation algorithms for hyperbolic systems; `riemann.shallow_sphere_2D` Riemann kernel paired with the `classic2_sw_sphere` Fortran time-step module.
- **Code.** This repository, subsystem [`datagen/`](../datagen). Cap sampling, IC fill, solver bindings, resampling, and SLURM scripts are self-contained in `datagen/shock_caps/`.

## Mathematical framework

The system is the rotation-free shallow water equations on the unit
sphere $\mathbb{S}^2$, written in conservation form for fluid depth $h$
and depth-integrated momentum $h\mathbf{u}$:

$$
\begin{aligned}
\frac{\partial h}{\partial t} + \nabla_{\mathbb{S}^2} \cdot (h\mathbf{u}) &= 0 \\
\frac{\partial (h\mathbf{u})}{\partial t} + \nabla_{\mathbb{S}^2} \cdot \left( h\mathbf{u} \otimes \mathbf{u} + \tfrac{1}{2} g h^2 \mathbf{I} \right) &= 0
\end{aligned}
$$

where $h(\lambda, \phi, t)$ is the fluid depth and
$\mathbf{u}(\lambda, \phi, t)$ is the velocity vector field tangent to
the sphere. The gravity-wave celerity $c = \sqrt{g h}$ closes the
characteristic structure; with the chosen non-dimensional
$g \approx 9.806$ and $h \in [0.5, 2.0]$, $c \in [2.21, 4.43]$ — fast
enough that shock fronts traverse a large fraction of the sphere within
the simulated window. No Coriolis term is included; the dataset is
intentionally rotation-free so the shock physics dominates.

### Fixed non-dimensional scales

| Symbol | Value | Meaning |
| --- | --- | --- |
| $R$ | $1$ | Sphere radius (non-dimensional) |
| $g$ | $9.80616$ | Gravitational acceleration (non-dimensional) |
| $t_{max}$ | $1.5$ | Maximum simulation time |

Unlike spectral solvers — which require hyperviscosity to stabilize and
smear shocks — the numerical dissipation needed to enforce the entropy
condition is provided organically by the finite-volume Riemann solver
plus TVD limiters.

## Data specifications

### Grid

- **Discretization.** Regular `(lat, lon)` grid with `N_lat = 256`, `N_lon = 512`, equispaced and pole-excluding. Latitude centers are `lat_k = -π/2 + (k + 0.5)·π/N_lat` in radians. Longitude centers are `lon_k = k·2π/N_lon` in `[0, 2π)`.
- **Native solver grid.** Calhoun-Helzel single-patch mapped sphere with `(N_x, N_y) = (512, 256)` finite-volume cells over the computational rectangle `[-3, 1] × [-1, 1]`. Snapshots are nearest-neighbour-interpolated from the FV cells to the regular `(lat, lon)` grid at output time (a single `cKDTree` query, reused across all snapshots).

### Temporal layout

- **Snapshot cadence.** Every $\Delta t = 0.015$ of simulated time.
- **Simulation length.** $1.5$ simulated time units.
- **Output window.** $0 \le t \le 1.5$. No burn-in is discarded — the primary physics of interest includes the immediate resolution of the $t=0$ discontinuities.
- **Trajectory shape.** 101 snapshots per trajectory, each $0.015$ apart.
- **Time coordinate.** Stored as non-dimensional time (`float64`).

### Available fields

| Field name | Units | Description |
| --- | --- | --- |
| `height` | dimensionless | Fluid depth $h$ |
| `momentum_u` | dimensionless | Zonal (eastward) depth-integrated momentum $hu$ in the local east basis |
| `momentum_v` | dimensionless | Meridional (northward) depth-integrated momentum $hv$ in the local north basis |

All fields stored as `float32`. The Fortran solver carries 3-D Cartesian
momentum internally (`h u_x, h u_y, h u_z`); the output projection back
to local east/north is exact up to floating-point round-off because the
flow is constrained to the tangent plane by the source-split projection
step in `sw_sphere_problem.src2`.

### Dataset size

- **Number of trajectories.** 500 on a $(K, \delta, \text{seed})$ grid (5 × 5 × 20).
- **Shape per trajectory.** `(time=101, field=3, lat=256, lon=512)`.
- **Per-trajectory size.** $\approx$ 159 MB (float32, uncompressed).
- **Total ensemble size.** $\approx$ 79 GB.

### Storage format

Per-run Zarr v3 stores chunked as `(time=1, field=3, lat=256, lon=512)`
for fast per-snapshot random access. After splitting (see
"Train/val/test split" below) the layout under `…/shock-caps/` is

```
shock-caps/
├── manifest.json
├── splits.json
├── train/run_XXXX.zarr   (400 runs)
├── val/run_XXXX.zarr     (50 runs)
└── test/run_XXXX.zarr    (50 runs)
```

`splits.json` records the seed, the stratification keys, and the
per-split run-ID lists.

## Initial conditions

### 1. Random Cap Placement

At $t=0$, $K$ geodesic disks (spherical caps) are placed on the unit
sphere; $K$ is a grid axis (see "Parameter space" below). Each cap
$k \in \{0, \ldots, K-1\}$ is defined by:

- **Centre** $\hat{\mathbf{c}}_k \in \mathbb{S}^2$, drawn uniformly via the $(z, \varphi)$ parameterisation.
- **Angular radius** $r_k \in [0.3, 1.0]$ rad ($\approx 17°$–$57°$), drawn uniformly. At the mean radius of $\approx 37°$, a single cap covers $\approx 17\%$ of the sphere.
- **Primitive state** $(h_k, u_k, v_k)$ with $h_k \in [0.5, 2.0]$ and $u_k, v_k \in [-0.5\delta, 0.5\delta]$, all drawn uniformly. The depth range is fixed; $\delta$ scales velocities only. At $\delta=0$ the cap is at rest with a pure depth perturbation; at $\delta=1$ the cap carries the full velocity envelope.

A separate **background state** $(h_\text{bg}, u_\text{bg}, v_\text{bg})$, drawn from the same $\delta$-scaled ranges, fills all points not claimed by any cap.

For a given $(K, \delta, \text{seed})$ tuple all stochastic
quantities — centres, radii, per-cap states, and the background — are
determined deterministically from `seed`.

### 2. Painter's Algorithm

Caps are assigned in index order $0 \to K-1$: a cell belongs to cap $k$
if $\hat{\mathbf{p}} \cdot \hat{\mathbf{c}}_k \ge \cos r_k$, with
higher-index caps overwriting lower-index ones in overlap regions.
Expected total cap coverage at $K=4$ is roughly 50–70%; at $K=1$ a
single cap covers $\sim 17\%$ on average, and at $K=16$ the background
becomes a thin connecting tissue between near-fully-tiled caps.

### 3. Sub-cell Antialiasing

At IC time every solver cell is sub-sampled $S \times S = 4 \times 4$
times in computational coordinates. Each sub-point is classified into a
cap (or background) by the dot-product test; the cell IC is the mean of
the primitives $(h, u_\text{east}, u_\text{north})$ across the 16
sub-points, with momentum projected to 3-D Cartesian at the cell centre.
This eliminates staircase artefacts at cap boundaries and gives the
solver a well-resolved initial shock width of $\sim 1$ FV cell.

### Regime

The $\delta$-scaled velocity bounds keep the initial flow strictly
subcritical at all $\delta \in [0, 1]$:
$\max\,\text{Fr} = 0.5\delta / \sqrt{g \cdot h_{\min}} \approx 0.23\delta$,
peaking at $\approx 0.23$ when $\delta=1$ and reaching $0$ at
$\delta=0$. The depth lower bound ($h \ge 0.5$) is independent of
$\delta$, so positivity is preserved throughout the parameter space
and shock fronts cannot drive $h$ to zero.

## Physical setup

- **Domain.** Full sphere $\mathbb{S}^2$, unit radius.
- **Boundary conditions.** Periodic mapping intrinsic to the spherical domain (the Calhoun-Helzel single patch wraps onto the sphere with custom $y$-fold BCs that identify the top/bottom strips with their reflected counterparts).
- **Timestepper.** Explicit finite-volume updates with Clawpack's high-resolution wave-propagation method, MC limiter, and 2-D transverse-wave correction. A source-split step (`source_split=2`) projects momentum back to the tangent plane every macro-step to keep the velocity tangent to the sphere.
- **CFL-adaptive $dt$.** Desired CFL $= 0.45$, hard ceiling $\le 0.9$; handled internally by Clawpack.

## Parameter space

Runs are laid out on a three-axis grid, indexed `run_0000 … run_0499`
in `K → delta → seed` order, so each $(K, \delta)$ block of 20 seeds is
contiguous in the SLURM array.

| Parameter | Symbol | Values | Count |
| --- | --- | --- | --- |
| Number of caps | $K$ | $1, 2, 4, 8, 16$ | 5 |
| Velocity scaling | $\delta$ | $0.0, 0.25, 0.5, 0.75, 1.0$ | 5 |
| IC seed | `seed` | $0, 1, \ldots, 19$ | 20 |
| **Total runs** | | | **500** |

For a given $(K, \delta)$ block, `seed` dictates all stochastic
quantities: cap centres, radii, per-cap states, and the background
state. Different $(K, \delta)$ blocks at the same `seed` draw their
caps independently — there is no subset relationship between cap counts.
Because cap centres are uniformly distributed on $\mathbb{S}^2$ by
construction, no additional $SO(3)$ tilt parameter is needed —
geometric variety comes from `seed` (placement) and $K$ (count); flow
strength comes from $\delta$.

### Train/val/test split

A fixed 80 / 10 / 10 split of the 500 run IDs is distributed with the
dataset as `splits.json` alongside `manifest.json`. The split is
stratified jointly on $(K, \delta)$ so each split sees every cell of
the parameter grid in the target proportions: **16 / 2 / 2** seeds per
$(K, \delta)$ cell, **400 / 50 / 50** runs total. Generated by
[`datagen/shock_caps/scripts/generate_split.py`](../datagen/shock_caps/scripts/generate_split.py)
with `seed=42`; deterministic and reproducible.

## Numerical-stability strategy

1. **Entropy-Stable Riemann Solvers.** Hyperbolic discontinuities trigger severe Gibbs phenomena in spectral codes. Clawpack pairs an approximate Riemann solver (`shallow_sphere_2D`) with TVD limiters (`MC`) to maintain sharp shock fronts without unphysical oscillations.
2. **Pole Singularity Mitigation.** Standard lat-lon grids force time-steps to near zero at the poles. The Calhoun-Helzel single-patch mapped sphere wraps the entire 2-sphere with a logically rectangular grid that has *no* coordinate singularity at the poles.
3. **Tangent Projection.** A Fortran source-split step (`sw_sphere_problem.src2`) projects momentum onto the local tangent plane every macro-step, so the 3-D Cartesian momentum stored in `state.q[1:4]` does not drift off the sphere.
4. **Per-run try/except.** On failure, the driver writes a `run_XXXX.FAILED` sentinel JSON with the exception, allowing the SLURM array to proceed.

## Computational details

- **Per-run wall.** Estimated $\approx$ 10 – 30 min on one scicore compute node (OMP-parallel Clawpack, 16 threads). Wall time scales roughly with $K$ — the $K=16$ stride is the budget driver because extra shock fronts tighten the CFL early in each run.
- **Total compute.** $\approx$ 100 – 250 core-hours for the full ensemble.
- **Solver precision.** `float64` throughout the simulation; downcast to `float32` at resample time.
- **Cluster.** sciCORE @ Universität Basel, `scicore` partition, `6hours` QoS, Easybuild `foss/2024a` toolchain.

## End-to-end data flow

```bash
# Login-node (need PyPI access).
uv sync --project datagen

# Emit per-run JSON configs + manifest.
uv run --project datagen python -m datagen.shock_caps.scripts.generate_sweep

# Full 500-run sweep.
sbatch datagen/shock_caps/slurm/sweep.sbatch

# Emit train/val/test splits (stratified on (K, delta), seed=42).
uv run --project datagen python -m datagen.shock_caps.scripts.generate_split

# Move per-run Zarrs into train/val/test/ subdirectories.
DATASET_ROOT=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/shock-caps
cp -f datagen/shock_caps/configs/splits.json $DATASET_ROOT/splits.json
uv run --project datagen python datagen/scripts/reorganize_splits.py \
    --root $DATASET_ROOT
```

## Using the dataset

```python
import json
import xarray as xr

root = "/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/shock-caps"

# One trajectory — lightweight random access.
run = xr.open_zarr(f"{root}/train/run_0000.zarr")
print(run)

# Iterate a split.
splits = json.load(open(f"{root}/splits.json"))
val_runs = [
    xr.open_zarr(f"{root}/val/run_{i:04d}.zarr") for i in splits["val"]
]
```

## Citation

If you use this dataset, please cite the foundational solver and the
shallow-water-on-sphere literature:

```bibtex
@article{mandli2016clawpack,
  title     = {Clawpack: building an open source ecosystem for solving hyperbolic PDEs},
  author    = {Mandli, Kyle T and Ahmadia, Aron J and Berger, Marsha and Calhoun, Donna and George, David L and Hadjimichael, Yiannis and Ketcheson, David I and Lemoine, Grady I and LeVeque, Randall J},
  journal   = {PeerJ Computer Science},
  volume    = {2},
  pages     = {e68},
  year      = {2016},
  publisher = {PeerJ Inc.}
}

@article{calhoun2008logically,
  title   = {Logically rectangular grids and finite volume methods for {PDE}s in circular and spherical domains},
  author  = {Calhoun, Donna A and Helzel, Christiane and LeVeque, Randall J},
  journal = {SIAM Review},
  volume  = {50},
  number  = {4},
  pages   = {723--752},
  year    = {2008},
  publisher = {SIAM}
}
```
