"""PyClaw shallow-water solver on the single-patch Calhoun–Helzel sphere.

Each simulation:

1. Builds the mapped single-patch sphere domain over ``[-3, 1] × [-1, 1]``
   using the Fortran ``sw_sphere_problem.setaux`` aux array.
2. Draws a per-trajectory ``SO(3)`` tilt from the run's ``seed`` and fills
   the 4-quadrant Riemann IC (piecewise-constant ``h``, east/north ``u, v``
   per quadrant, rotated to random orientation on the sphere).
3. Advances the solution with ``ClawSolver2D(riemann.shallow_sphere_2D)``
   driven by the ``classic2_sw_sphere`` Fortran module, collecting snapshots
   at fixed ``snapshot_dt`` cadence.
4. At each snapshot, projects the 3-D Cartesian momenta ``(h·ux, h·uy, h·uz)``
   back to local ``(momentum_east, momentum_north)`` and gathers everything
   onto a regular ``(lat, lon)`` grid via nearest-neighbour lookup.

Non-dimensional parameters
--------------------------
- ``RSPHERE = 1.0``  sphere radius (non-dimensional).
- ``G = 9.80616``    non-dim gravity (same numerical value as g_phys in SI).
  With h ∈ [0.5, 2.0], wave speed c = √(G·h) ∈ [2.21, 4.43] sphere-radii
  per unit time. A shock crosses the globe in ≈ π/2.21 ≈ 1.4 time units,
  so ``stop_sim_time = 1.5`` captures roughly one hemisphere crossing.

Boundary conditions
-------------------
- x-direction: periodic (the patch wraps).
- y-direction: custom pole-fold BC (``qbc_lower_y`` / ``qbc_upper_y``),
  identical to the upstream Rossby-wave example.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree

from clawpack import pyclaw, riemann
from clawpack.pyclaw.classic import classic2_sw_sphere as classic2
from clawpack.pyclaw.examples.shallow_sphere import sw_sphere_problem

from datagen.galewsky.so3 import rotation_from_seed
from datagen.shock_quadrants import geometry as geo
from datagen.shock_quadrants.ic import fill_ic


logger = logging.getLogger(__name__)

RSPHERE: float = geo.RSPHERE
G: float = 9.80616   # non-dim gravity (g ≈ 9.806 m/s², R=1, h=O(1))
FIELD_NAMES = ("height", "momentum_u", "momentum_v")


# ---------------------------------------------------------------------------
# y-direction BCs  (pole-fold, identical to Rossby_wave.py)
# ---------------------------------------------------------------------------

def _qbc_lower_y(state, dim, t, qbc, auxbc, num_ghost):
    for j in range(num_ghost):
        qbc1d = np.copy(qbc[:, :, 2 * num_ghost - 1 - j])
        qbc[:, :, j] = qbc1d[:, ::-1]


def _qbc_upper_y(state, dim, t, qbc, auxbc, num_ghost):
    my = state.grid.num_cells[1]
    for j in range(num_ghost):
        qbc1d = np.copy(qbc[:, :, my + num_ghost - 1 - j])
        qbc[:, :, my + num_ghost + j] = qbc1d[:, ::-1]


# ---------------------------------------------------------------------------
# Geometric source term  (wraps Fortran src2 — handles spherical geometry)
# ---------------------------------------------------------------------------

def _source_term(solver, state, dt):
    grid = state.grid
    mx, my = grid.num_cells
    num_ghost = solver.num_ghost
    xlower, ylower = grid.lower
    dx, dy = grid.delta
    state.q = sw_sphere_problem.src2(
        mx, my, num_ghost, xlower, ylower, dx, dy,
        state.q, state.aux, state.t, dt, RSPHERE,
    )


# ---------------------------------------------------------------------------
# Solver / domain / state construction
# ---------------------------------------------------------------------------

def _build_solver(
    Nx: int, Ny: int, full_aux: np.ndarray,
    cfl_desired: float, cfl_max: float,
):
    """Build the ``ClawSolver2D`` for the shallow-sphere Riemann system.

    ``full_aux`` is the precomputed extended aux array used by the aux-BC
    callables to fill y-ghost rows of ``auxbc`` from cache.
    """
    xlower, ylower = geo.COMPUTATIONAL_LOWER
    dx = (geo.COMPUTATIONAL_UPPER[0] - xlower) / Nx
    dy = (geo.COMPUTATIONAL_UPPER[1] - ylower) / Ny

    solver = pyclaw.ClawSolver2D(riemann.shallow_sphere_2D)
    solver.fmod = classic2

    # dx/dy and gravity must be set in both the fmod and the rp common blocks.
    solver.fmod.comxyt.dxcom = dx
    solver.fmod.comxyt.dycom = dy
    solver.fmod.sw.g = G
    solver.rp.comxyt.dxcom = dx
    solver.rp.comxyt.dycom = dy
    solver.rp.sw.g = G

    solver.cfl_desired = float(cfl_desired)
    solver.cfl_max = float(cfl_max)
    solver.order = 2
    solver.limiters = pyclaw.limiters.tvd.MC
    solver.dimensional_split = 0
    solver.transverse_waves = 2
    solver.source_split = 2
    solver.step_source = _source_term

    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic
    solver.bc_lower[1] = pyclaw.BC.custom
    solver.bc_upper[1] = pyclaw.BC.custom
    solver.user_bc_lower = _qbc_lower_y
    solver.user_bc_upper = _qbc_upper_y

    solver.aux_bc_lower[0] = pyclaw.BC.periodic
    solver.aux_bc_upper[0] = pyclaw.BC.periodic
    solver.aux_bc_lower[1] = pyclaw.BC.custom
    solver.aux_bc_upper[1] = pyclaw.BC.custom
    solver.user_aux_bc_lower = geo.make_aux_bc("lower", full_aux)
    solver.user_aux_bc_upper = geo.make_aux_bc("upper", full_aux)

    solver.dt_initial = 1e-3
    solver.dt_max = 0.1
    solver.verbosity = 4
    return solver


def _build_state(Nx: int, Ny: int, num_ghost: int = 2):
    """Build the PyClaw ``Domain``/``State``/``Solution`` triple.

    Returns the precomputed full (ghost-extended) aux array alongside the
    usual triple, so the solver builder can wire it into the aux-BC fills.
    """
    if Nx % 2 != 0 or Ny % 2 != 0:
        raise ValueError(f"Nx and Ny must be even for the pole-fold BC: {Nx}, {Ny}")

    xlower, ylower = geo.COMPUTATIONAL_LOWER
    xupper, yupper = geo.COMPUTATIONAL_UPPER
    dx = (xupper - xlower) / Nx
    dy = (yupper - ylower) / Ny

    x = pyclaw.Dimension(xlower, xupper, Nx, name="x")
    y = pyclaw.Dimension(ylower, yupper, Ny, name="y")
    domain = pyclaw.Domain([x, y])

    state = pyclaw.State(domain, num_eqn=4, num_aux=geo.NUM_AUX)
    state.index_capa = 0   # aux[0] = kappa (cell-area capacity)

    full_aux = geo.compute_full_aux(
        Nx, Ny, xlower, ylower, dx, dy, num_ghost=num_ghost,
    )
    state.aux[:, :, :] = full_aux[:, num_ghost:-num_ghost, num_ghost:-num_ghost]

    state.grid.mapc2p = geo.mapc2p_sphere
    solution = pyclaw.Solution(state, domain)
    return domain, state, solution, dx, dy, full_aux


# ---------------------------------------------------------------------------
# Cell-centre → regular (lat, lon) nearest-neighbour gather
# ---------------------------------------------------------------------------

def _build_target_xyz(Nlat: int, Nlon: int) -> np.ndarray:
    lat = -np.pi / 2.0 + (np.arange(Nlat) + 0.5) * np.pi / Nlat
    lon = np.arange(Nlon) * 2.0 * np.pi / Nlon
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    cos_lat = np.cos(lat_g)
    return np.stack(
        [cos_lat * np.cos(lon_g), cos_lat * np.sin(lon_g), np.sin(lat_g)],
        axis=-1,
    ).reshape(-1, 3)


def _regular_lat_grid(Nlat: int) -> np.ndarray:
    return -np.pi / 2.0 + (np.arange(Nlat) + 0.5) * np.pi / Nlat


def _regular_lon_grid(Nlon: int) -> np.ndarray:
    return np.arange(Nlon) * 2.0 * np.pi / Nlon


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_simulation(
    params: Dict[str, float],
    *,
    Nlat: int = 256,
    Nlon: int = 512,
    Nx: int = 512,
    Ny: int = 256,
    snapshot_dt: float = 0.015,
    stop_sim_time: float = 1.5,
    cfl_desired: float = 0.45,
    cfl_max: float = 0.9,
) -> Dict:
    """Run one shallow-water shock-quadrant simulation.

    Returns
    -------
    dict
        ``time``          : ``(Nt,)`` float64 snapshot times.
        ``fields``        : ``(Nt, 3, Nlat, Nlon)`` float32 array,
                            channel order ``FIELD_NAMES``.
        ``field_names``   : list of channel names.
        ``so3_axis_xyz``  : ``(3,)`` float64 unit vector.
        ``so3_angle_rad`` : float — rotation angle in radians.
    """
    seed = int(params["seed"])
    axis, angle = rotation_from_seed(seed)
    logger.info(
        "Starting run: seed=%d axis=%s angle=%.6f rad "
        "Nx=%d Ny=%d Nlat=%d Nlon=%d stop_sim_time=%g snapshot_dt=%g",
        seed, axis.tolist(), angle, Nx, Ny, Nlat, Nlon,
        stop_sim_time, snapshot_dt,
    )

    t0 = time.time()
    domain, state, solution, dx, dy, full_aux = _build_state(Nx, Ny)
    solver = _build_solver(
        Nx, Ny, full_aux, cfl_desired=cfl_desired, cfl_max=cfl_max,
    )
    logger.info("Built solver + state in %.2fs", time.time() - t0)

    xlower, ylower = geo.COMPUTATIONAL_LOWER

    t0 = time.time()
    lat_cell, lon_cell = geo.cell_centers_latlon(Nx, Ny, xlower, ylower, dx, dy)
    pts_xyz_src = np.stack(
        [
            np.cos(lat_cell) * np.cos(lon_cell),
            np.cos(lat_cell) * np.sin(lon_cell),
            np.sin(lat_cell),
        ],
        axis=-1,
    ).reshape(-1, 3)
    logger.info("Built cell-centre lat/lon arrays in %.2fs", time.time() - t0)

    t0 = time.time()
    h0 = np.empty((Nx, Ny), dtype=np.float64)
    momx0 = np.empty((Nx, Ny), dtype=np.float64)
    momy0 = np.empty((Nx, Ny), dtype=np.float64)
    momz0 = np.empty((Nx, Ny), dtype=np.float64)
    fill_ic(
        h0, momx0, momy0, momz0,
        seed=seed,
        lat_centers=lat_cell,
        lon_centers=lon_cell,
        axis=axis,
        angle=angle,
    )
    state.q[0, ...] = h0
    state.q[1, ...] = momx0
    state.q[2, ...] = momy0
    state.q[3, ...] = momz0
    logger.info("Filled IC in %.2fs", time.time() - t0)

    t0 = time.time()
    pts_xyz_target = _build_target_xyz(Nlat, Nlon)
    tree = cKDTree(pts_xyz_src)
    _, nn_indices = tree.query(pts_xyz_target, k=1)
    logger.info(
        "Built cKDTree NN map (%d src, %d tgt) in %.2fs",
        pts_xyz_src.shape[0], pts_xyz_target.shape[0], time.time() - t0,
    )

    Nt = int(round(stop_sim_time / snapshot_dt)) + 1
    snapshot_times = np.linspace(0.0, stop_sim_time, Nt)
    fields = np.empty((Nt, 3, Nlat, Nlon), dtype=np.float32)

    t0 = time.time()
    _record_snapshot(fields, 0, state, lat_cell, lon_cell, Nx, Ny, Nlat, Nlon, nn_indices)
    logger.info("Recorded t=0 snapshot in %.2fs; entering time loop.", time.time() - t0)

    wallclock_start = time.time()
    for k, t_target in enumerate(snapshot_times[1:], start=1):
        solver.evolve_to_time(solution, t_target)
        _record_snapshot(fields, k, state, lat_cell, lon_cell, Nx, Ny, Nlat, Nlon, nn_indices)
        logger.info(
            "snapshot %3d / %d (t=%.4f) wallclock=%.1fs",
            k, Nt - 1, t_target, time.time() - wallclock_start,
        )

    logger.info("Simulation finished in %.1f s wallclock.", time.time() - wallclock_start)
    return {
        "time": snapshot_times.astype(np.float64),
        "fields": fields,
        "field_names": list(FIELD_NAMES),
        "so3_axis_xyz": np.asarray(axis, dtype=np.float64),
        "so3_angle_rad": float(angle),
    }


def _record_snapshot(
    fields_out: np.ndarray,
    k: int,
    state,
    lat_cell: np.ndarray,
    lon_cell: np.ndarray,
    Nx: int,
    Ny: int,
    Nlat: int,
    Nlon: int,
    nn_indices: np.ndarray,
) -> None:
    """Project the current PyClaw state onto the regular lat/lon grid.

    ``state.q`` carries ``(h, h·ux, h·uy, h·uz)`` in 3-D Cartesian.
    The momentum components are projected to local east/north for output.
    """
    h = np.asarray(state.q[0])
    momx = np.asarray(state.q[1])
    momy = np.asarray(state.q[2])
    momz = np.asarray(state.q[3])

    # Divide by h to get velocity, project to east/north, multiply back.
    h_safe = np.where(h > 0.0, h, 1.0)
    ux = momx / h_safe
    uy = momy / h_safe
    uz = momz / h_safe
    u_east, u_north = geo.xyz_to_en(lat_cell, lon_cell, ux, uy, uz)
    mom_east = h * u_east
    mom_north = h * u_north

    _F32_MAX = np.finfo(np.float32).max
    target_shape = (Nlat, Nlon)
    for c, field_cells in enumerate((h, mom_east, mom_north)):
        gathered = field_cells.reshape(-1)[nn_indices].reshape(target_shape)
        fields_out[k, c] = np.clip(np.nan_to_num(gathered), -_F32_MAX, _F32_MAX).astype(np.float32)
