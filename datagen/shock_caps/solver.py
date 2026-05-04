"""PyClaw shallow-water solver for the shock-caps dataset.

Each simulation:

1. Builds the mapped single-patch Calhoun-Helzel sphere domain over
   ``[-3, 1] × [-1, 1]`` using the Fortran ``sw_sphere_problem.setaux`` aux
   array (no coordinate singularity at the poles).
2. Fills the random-cap Riemann IC defined in ``datagen.shock_caps.ic``
   from the run's ``(K, delta, seed)`` triple. Cap centres are uniform on
   ``S²`` already, so no extra ``SO(3)`` rotation is required.
3. Advances the solution with ``ClawSolver2D(riemann.shallow_sphere_2D)``
   driven by the ``classic2_sw_sphere`` Fortran module, collecting snapshots
   at fixed ``snapshot_dt`` cadence.
4. At each snapshot, projects the 3-D Cartesian momenta ``(h·ux, h·uy, h·uz)``
   back to local ``(momentum_east, momentum_north)`` and remaps onto a
   regular ``(lat, lon)`` grid via the conservative spherical-polygon
   area-overlap operator from ``datagen.shock_caps.remap``.

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
from typing import Dict

import numpy as np

from clawpack import pyclaw, riemann
from clawpack.pyclaw.classic import classic2_sw_sphere as classic2
from clawpack.pyclaw.examples.shallow_sphere import sw_sphere_problem

from datagen.shock_caps import geometry as geo
from datagen.shock_caps.ic import fill_ic
from datagen.shock_caps.remap import apply_remap, build_remap_matrix


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
    sub_samples: int = 4,
) -> Dict:
    """Run one shallow-water shock-caps simulation.

    Returns
    -------
    dict
        ``time``        : ``(Nt,)`` float64 snapshot times.
        ``fields``      : ``(Nt, 3, Nlat, Nlon)`` float32 array,
                          channel order ``FIELD_NAMES``.
        ``field_names`` : list of channel names.
    """
    seed = int(params["seed"])
    K = int(params["K"])
    delta = float(params["delta"])
    logger.info(
        "Starting run: seed=%d K=%d delta=%g Nx=%d Ny=%d Nlat=%d Nlon=%d "
        "stop_sim_time=%g snapshot_dt=%g sub_samples=%d",
        seed, K, delta, Nx, Ny, Nlat, Nlon, stop_sim_time, snapshot_dt, sub_samples,
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
    logger.info("Built cell-centre lat/lon arrays in %.2fs", time.time() - t0)

    t0 = time.time()
    h0 = np.empty((Nx, Ny), dtype=np.float64)
    momx0 = np.empty((Nx, Ny), dtype=np.float64)
    momy0 = np.empty((Nx, Ny), dtype=np.float64)
    momz0 = np.empty((Nx, Ny), dtype=np.float64)
    fill_ic(
        h0, momx0, momy0, momz0,
        seed=seed,
        K=K,
        delta=delta,
        lat_centers=lat_cell,
        lon_centers=lon_cell,
        xlower=xlower, ylower=ylower, dx=dx, dy=dy,
        sub_samples=sub_samples,
    )
    state.q[0, ...] = h0
    state.q[1, ...] = momx0
    state.q[2, ...] = momy0
    state.q[3, ...] = momz0
    logger.info("Filled IC in %.2fs", time.time() - t0)

    t0 = time.time()
    W = build_remap_matrix(Nx, Ny, Nlat, Nlon)
    logger.info(
        "Built conservative area-overlap remap %d src → %d tgt (%d nnz) in %.2fs",
        Nx * Ny, Nlat * Nlon, W.nnz, time.time() - t0,
    )

    Nt = int(round(stop_sim_time / snapshot_dt)) + 1
    snapshot_times = np.linspace(0.0, stop_sim_time, Nt)
    fields = np.empty((Nt, 3, Nlat, Nlon), dtype=np.float32)

    t0 = time.time()
    _record_snapshot(fields, 0, state, lat_cell, lon_cell, W, Nlat, Nlon)
    logger.info("Recorded t=0 snapshot in %.2fs; entering time loop.", time.time() - t0)

    wallclock_start = time.time()
    for k, t_target in enumerate(snapshot_times[1:], start=1):
        solver.evolve_to_time(solution, t_target)
        _record_snapshot(fields, k, state, lat_cell, lon_cell, W, Nlat, Nlon)
        logger.info(
            "snapshot %3d / %d (t=%.4f) wallclock=%.1fs",
            k, Nt - 1, t_target, time.time() - wallclock_start,
        )

    logger.info("Simulation finished in %.1f s wallclock.", time.time() - wallclock_start)
    return {
        "time": snapshot_times.astype(np.float64),
        "fields": fields,
        "field_names": list(FIELD_NAMES),
    }


def _record_snapshot(
    fields_out: np.ndarray,
    k: int,
    state,
    lat_cell: np.ndarray,
    lon_cell: np.ndarray,
    W,
    Nlat: int,
    Nlon: int,
) -> None:
    """Project the current PyClaw state onto the regular lat/lon grid.

    The 3-D Cartesian momentum components ``(h·ux, h·uy, h·uz)`` are
    remapped to the target grid first, *then* projected to local
    east/north using the *target* cell coordinates. This avoids the
    near-pole cancellation that arises if you project on the source grid
    first: the local east/north basis rotates rapidly with longitude
    around the pole, and area-averaging local-frame scalars across
    source cells pointing in different physical directions silently
    cancels real momentum. Cartesian components are well-defined
    globally and area-average correctly.
    """
    h_src = np.asarray(state.q[0])
    momx_src = np.asarray(state.q[1])
    momy_src = np.asarray(state.q[2])
    momz_src = np.asarray(state.q[3])

    h_tgt = apply_remap(W, h_src, Nlat, Nlon)
    momx_tgt = apply_remap(W, momx_src, Nlat, Nlon)
    momy_tgt = apply_remap(W, momy_src, Nlat, Nlon)
    momz_tgt = apply_remap(W, momz_src, Nlat, Nlon)

    dlat = np.pi / Nlat
    dlon = (2.0 * np.pi) / Nlon
    lat_tgt_1d = -np.pi / 2.0 + (np.arange(Nlat) + 0.5) * dlat
    lon_tgt_1d = (np.arange(Nlon) + 0.5) * dlon
    lat_tgt, lon_tgt = np.meshgrid(lat_tgt_1d, lon_tgt_1d, indexing="ij")

    h_safe = np.where(h_tgt > 0.0, h_tgt, 1.0)
    ux_tgt = momx_tgt / h_safe
    uy_tgt = momy_tgt / h_safe
    uz_tgt = momz_tgt / h_safe

    u_east_tgt, u_north_tgt = geo.xyz_to_en(
        lat_tgt, lon_tgt, ux_tgt, uy_tgt, uz_tgt,
    )
    mom_east_tgt = h_tgt * u_east_tgt
    mom_north_tgt = h_tgt * u_north_tgt

    _F32_MAX = np.finfo(np.float32).max
    for c, field in enumerate((h_tgt, mom_east_tgt, mom_north_tgt)):
        fields_out[k, c] = np.clip(
            np.nan_to_num(field), -_F32_MAX, _F32_MAX,
        ).astype(np.float32)
