# Shock Quadrants: Multi-Shock Shallow Water Dynamics on the Sphere

**One-line description.** A 500-trajectory ensemble of hyperbolic
shallow-water dynamics on the 2-sphere, initialized with randomized
4-quadrant Riemann problems that produce strong discontinuous shock fronts,
cross-quadrant wave interactions, and antipodal shock convergence.

**Extended description.** Each trajectory solves the rotation-free shallow
water equations on a unit sphere. The initial condition maps the classic
2-D multi-quadrant Riemann problem onto the spherical topology: the
sphere is partitioned into four sectors, each initialized with a
piecewise-constant state of fluid depth and tangent velocity. To prevent
deep-learning models from exploiting grid-aligned boundaries, the entire
four-quadrant configuration is subjected to a per-trajectory random
`SO(3)` tilt so the shock interfaces do not trivially align with the
computational equator or prime meridian. The dataset captures the wave
interactions — propagating bores, expansion fans, and shock-shock
collisions — that emerge as four piecewise-constant states fight for
sphere area. Because the manifold is closed, expanding shocks eventually
cross the globe and converge at antipodal points, providing a sharp
benchmark for geometric and entropy-conserving neural PDE modelling.
Compared to the (abandoned) compressible-Euler quadrant variant, the
shallow-water system has only three conserved variables and no
adiabatic-index axis, so the parameter space collapses onto a single
`seed` axis while keeping the brutal-shock motivation.

## Associated resources

- **Papers.**
  - LeVeque, R. J., & Calhoun, D. A. (2002). *Shallow water flow on the sphere*.
  - Calhoun, D. A., Helzel, C., & LeVeque, R. J. (2008). *Logically Rectangular Grids and Finite Volume Methods for PDEs in Circular and Spherical Domains*. SIAM Review 50, 723-752.
  - Lax, P. D., & Liu, X.-D. (1998). *Solution of two-dimensional Riemann problems of gas dynamics by positive schemes*. SIAM Journal on Scientific Computing, 19(2), 319-340.
- **Data generator.** Till Muser (University of Basel), 2026.
- **Generation software.** [Clawpack / PyClaw](https://www.clawpack.org/) — high-resolution finite-volume solver with mapped spherical-grid support and wave-propagation algorithms for hyperbolic systems; `riemann.shallow_sphere_2D` Riemann kernel paired with the `classic2_sw_sphere` Fortran time-step module.
- **Code.** This repository, subsystem [`datagen/`](../datagen). Coefficient map, initial conditions, SO(3) rotations, solver bindings, resampling, and SLURM scripts are self-contained in that subproject.

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
| `momentum_u` | dimensionless | Zonal (eastward) momentum component $hu$, in the **post-tilt** local east basis |
| `momentum_v` | dimensionless | Meridional (northward) momentum component $hv$, in the **post-tilt** local north basis |

All fields stored as `float32`. The Fortran solver carries 3-D Cartesian
momentum internally (`h u_x, h u_y, h u_z`); the output projection back
to local east/north is exact up to floating-point round-off because the
flow is constrained to the tangent plane by the source-split projection
step in `sw_sphere_problem.src2`.

### Dataset size

- **Number of trajectories.** 500 (single `seed` axis).
- **Shape per trajectory.** `(time=101, field=3, lat=256, lon=512)`.
- **Per-trajectory size.** $\approx$ 159 MB (float32, uncompressed).
- **Total ensemble size.** $\approx$ 79 GB.
- **Consolidated shape.** `(run=500, time=101, field=3, lat=256, lon=512)`.

### Storage format

Per-run Zarr v3 stores at `…/shock-quadrants/processed/run_XXXX.zarr`,
chunked as `(time=1, field=3, lat=256, lon=512)` for fast per-snapshot
random access. A consolidated `dataset.zarr` stacks all runs along a
leading `run` dimension with parameter arrays carried as `param_*` coords.

## Initial conditions

### 1. The Canonical 4-Quadrant Setup
At $t=0$, in a canonical frame, the sphere is divided into four equal
quadrants separated by the equator ($\phi = 0$) and the prime meridian
($\lambda = 0$ / $\lambda = \pi$). Each quadrant $\Omega_i$
($i \in \{1,2,3,4\}$) is initialized with a uniform, independently drawn
primitive state:
$$\mathbf{U}_i = [h_i, u_i, v_i]^T$$
The values are drawn uniformly from $h \in [0.5, 2.0]$,
$u, v \in [-0.5, 0.5]$, parameterized completely by the run's `seed`.
The bounds keep $h$ strictly positive (well-posedness) and the velocities
subcritical relative to the gravity-wave speed
$c = \sqrt{g h_{\min}} \approx 2.21$, so no individual quadrant is in a
single-state supersonic regime.

### 2. Per-trajectory SO(3) Tilt

To prevent networks from memorising grid-aligned features or exploiting
the native lat-lon singularities, the entire initial condition is
subjected to a per-trajectory rotation
$R(\hat{\mathbf{e}}, \alpha) \in SO(3)$ drawn deterministically from the
run's `seed`.

* $\hat{\mathbf{e}}$ uniform on $\mathbb{S}^2$, stored as `so3_axis_xyz`.
* $\alpha$ uniform in $[0, 2\pi)$, stored as `so3_angle_rad`.

The shock interfaces, which would natively lie on the equator and
meridians, are thus randomly oriented across the globe. The per-quadrant
$(u, v)$ are interpreted as scalars in the **local east/north basis at
each (post-tilt) cell** — the natural spherical analogue of the planar
Riemann problem's Cartesian velocities. At IC time these are projected
into the ambient 3-D Cartesian frame consumed by the SWE Riemann kernel
($h\mathbf{u}_{xyz}$); at output time they are projected back to local
east/north for the saved $h u, h v$ channels.

## Physical setup

- **Domain.** Full sphere $\mathbb{S}^2$, unit radius.
- **Boundary conditions.** Periodic mapping intrinsic to the spherical domain (the Calhoun-Helzel single patch wraps onto the sphere with custom $y$-fold BCs that identify the top/bottom strips with their reflected counterparts).
- **Timestepper.** Explicit finite-volume updates with Clawpack's high-resolution wave-propagation method, MC limiter, and 2-D transverse-wave correction. A source-split step (`source_split=2`) projects momentum back to the tangent plane every macro-step to keep the velocity tangent to the sphere.
- **CFL-adaptive $dt$.** Strict CFL limit $\le 0.9$ required for hyperbolic stability, handled internally by Clawpack.

## Parameter space

Runs are laid out on a single axis (500 runs). Runs are indexed
`run_0000 … run_0499` by `seed`.

| Parameter | Symbol | Values | Count |
| --- | --- | --- | --- |
| IC + SO(3) tilt seed | `seed` | $0, 1, \ldots, 499$ | 500 |

The `seed` dictates both the four piecewise-constant states $(h, u, v)$
and the orientation of the initial quadrants on the globe. There is no
adiabatic-index analogue in shallow water, so the dataset uses the full
500-trajectory budget on seed alone.

## Numerical-stability strategy

1. **Entropy-Stable Riemann Solvers.** Hyperbolic discontinuities trigger severe Gibbs phenomena in spectral codes. Clawpack pairs an approximate Riemann solver (`shallow_sphere_2D`) with TVD limiters (`MC`) to maintain sharp shock fronts without unphysical oscillations.
2. **Pole Singularity Mitigation.** Standard lat-lon grids force time-steps to near zero at the poles. The Calhoun-Helzel single-patch mapped sphere wraps the entire 2-sphere with a logically rectangular grid that has *no* coordinate singularity at the poles.
3. **Tangent Projection.** A Fortran source-split step (`sw_sphere_problem.src2`) projects momentum onto the local tangent plane every macro-step, so the 3-D Cartesian momentum stored in `state.q[1:4]` does not drift off the sphere.
4. **Per-run try/except.** On failure, the driver writes a `run_XXXX.FAILED` sentinel JSON with the exception, allowing the SLURM array to proceed.

## Computational details

- **Per-run wall.** Estimated $\approx$ 10 – 30 min on one scicore compute node (OMP-parallel Clawpack). Final figure to be confirmed by the smoke-test run.
- **Total compute.** $\approx$ 100 – 250 core-hours for the full ensemble.
- **Solver precision.** `float64` throughout the simulation; downcast to `float32` at resample time.
- **Cluster.** sciCORE @ Universität Basel, `scicore` partition, `6hours` QoS, Easybuild `foss/2024a` toolchain.

## End-to-end data flow

```bash
# Login-node (need PyPI access).
uv sync --project datagen

# Emit per-run JSON configs + manifest.
uv run --project datagen python -m datagen.shock_quadrants.scripts.generate_sweep

# Full 500-run sweep.
sbatch datagen/shock_quadrants/slurm/sweep.sbatch

# Consolidate per-run Zarrs into the final dataset.zarr.
DATASET_ROOT=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/shock-quadrants
uv run --project datagen python -m datagen.scripts.consolidate \
    --processed $DATASET_ROOT/processed/ \
    --manifest  $DATASET_ROOT/manifest.json \
    --out       $DATASET_ROOT/dataset.zarr
```

## Using the dataset

```python
import xarray as xr

root = "/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/shock-quadrants"

# One trajectory — lightweight random access.
run = xr.open_zarr(f"{root}/processed/run_0000.zarr")
print(run)

# Full ensemble — run, time, field, lat, lon.
ds = xr.open_zarr(f"{root}/dataset.zarr")
print(ds)
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
