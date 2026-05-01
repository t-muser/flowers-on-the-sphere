"""PyClaw shallow-water solver for the shock-caps dataset.

Thin wrapper around the ``shock_quadrants`` Calhoun-Helzel SWE machinery:
same FV solver glue, same Calhoun-Helzel mapped sphere, same antialiased
output gather, but swaps the 4-quadrant Riemann IC for the random-cap
Riemann IC defined in ``datagen.shock_caps.ic``. No SO(3) tilt — cap
centres are already uniform on ``S²`` so no extra randomisation step is
needed.

Boundary conditions, source term, time-stepper, output projection, and
field channel order all match shock_quadrants exactly.
"""

from __future__ import annotations

import logging
import time
from typing import Dict

import numpy as np
from scipy.spatial import cKDTree

from datagen.shock_quadrants import geometry as geo
from datagen.shock_quadrants.solver import (
    FIELD_NAMES,
    _build_solver,
    _build_state,
    _build_target_xyz,
    _record_snapshot,
)
from datagen.shock_caps.ic import fill_ic


logger = logging.getLogger(__name__)


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
    pts_xyz_target = _build_target_xyz(Nlat, Nlon, oversample=sub_samples)
    tree = cKDTree(pts_xyz_src)
    _, nn_indices = tree.query(pts_xyz_target, k=1)
    logger.info(
        "Built cKDTree NN map (%d src, %d tgt @ oversample=%d) in %.2fs",
        pts_xyz_src.shape[0], pts_xyz_target.shape[0], sub_samples,
        time.time() - t0,
    )

    Nt = int(round(stop_sim_time / snapshot_dt)) + 1
    snapshot_times = np.linspace(0.0, stop_sim_time, Nt)
    fields = np.empty((Nt, 3, Nlat, Nlon), dtype=np.float32)

    t0 = time.time()
    _record_snapshot(
        fields, 0, state, lat_cell, lon_cell, Nx, Ny, Nlat, Nlon,
        nn_indices, oversample=sub_samples,
    )
    logger.info("Recorded t=0 snapshot in %.2fs; entering time loop.", time.time() - t0)

    wallclock_start = time.time()
    for k, t_target in enumerate(snapshot_times[1:], start=1):
        solver.evolve_to_time(solution, t_target)
        _record_snapshot(
            fields, k, state, lat_cell, lon_cell, Nx, Ny, Nlat, Nlon,
            nn_indices, oversample=sub_samples,
        )
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
