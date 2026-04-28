"""4-quadrant Riemann initial conditions for shallow water on the sphere.

The canonical partition divides the sphere by the equator (lat = 0) and
the prime/anti-meridian (lon = 0, lon = π) into four quadrants:

    Q1: lat > 0, 0   ≤ lon < π     (north-east)
    Q2: lat > 0, π   ≤ lon < 2π    (north-west)
    Q3: lat < 0, π   ≤ lon < 2π    (south-west)
    Q4: lat < 0, 0   ≤ lon < π     (south-east)

Each quadrant carries a uniform piecewise-constant state ``{h, u, v}``
drawn deterministically from the run's ``seed``. A per-trajectory ``SO(3)``
tilt is applied to the partition: every grid cell is back-rotated into the
canonical frame and classified there, so the shock interfaces are randomly
oriented across the globe rather than coinciding with the equator/meridians.

Per-quadrant ``(u, v)`` are local east/north velocity scalars. At IC time
they are projected into the solver's 3-D Cartesian momentum basis via
``geometry.en_to_xyz``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation

from datagen.galewsky.so3 import rotation_from_seed
from datagen.shock_quadrants.geometry import en_to_xyz, subcell_centers_latlon


# h strictly positive (keeps h far from 0 to prevent blow-up at shock fronts).
# Velocities subcritical: max Fr = 0.5 / sqrt(9.806 * 0.5) ≈ 0.23.
QUADRANT_RANGES: Dict[str, tuple[float, float]] = {
    "h": (0.5, 2.0),    # non-dimensional fluid depth
    "u": (-0.5, 0.5),   # non-dimensional east velocity
    "v": (-0.5, 0.5),   # non-dimensional north velocity
}


def sample_quadrant_states(seed: int) -> List[Dict[str, float]]:
    """Draw 4 ``{h, u, v}`` dicts deterministically from ``seed``.

    Draw order is fixed: Q1 → Q4, variables ``h, u, v`` within each quadrant.
    """
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    states: List[Dict[str, float]] = []
    for _ in range(4):
        s: Dict[str, float] = {}
        for name in ("h", "u", "v"):
            lo, hi = QUADRANT_RANGES[name]
            s[name] = float(rng.uniform(lo, hi))
        states.append(s)
    return states


def quadrant_index_canonical(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Classify each ``(lat, lon)`` point into 0–3 (Q1–Q4), canonical frame."""
    lon_w = np.mod(lon, 2.0 * np.pi)
    north = lat > 0.0
    east = lon_w < np.pi
    out = np.empty(np.broadcast(lat, lon).shape, dtype=np.int8)
    out[north & east] = 0   # Q1
    out[north & ~east] = 1  # Q2
    out[~north & ~east] = 2 # Q3
    out[~north & east] = 3  # Q4
    return out


def fill_ic(
    h_out: np.ndarray,
    momx_out: np.ndarray,
    momy_out: np.ndarray,
    momz_out: np.ndarray,
    *,
    seed: int,
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
    xlower: float,
    ylower: float,
    dx: float,
    dy: float,
    axis: np.ndarray,
    angle: float,
    sub_samples: int = 4,
) -> None:
    """Fill the four conserved-state arrays in place at every cell.

    Sub-cell antialiased classification: every cell is sub-sampled
    ``sub_samples × sub_samples`` times in computational ``(X, Y)``,
    every sub-point is back-rotated to the canonical frame and classified
    into one of four quadrants, and the cell IC is the mean of primitives
    ``(h, u_east, u_north)`` across sub-points. The momentum is converted
    to 3-D Cartesian at the cell centre, which keeps each cell's momentum
    exactly tangent to the sphere. Boundary cells get a clean one-cell
    soft transition instead of the cell-centre staircase the rotated
    quadrant interfaces produced before.
    """
    states = sample_quadrant_states(seed)
    Nx, Ny = lat_centers.shape

    # Sub-point lat/lon, shape (Nx, Ny, S, S).
    lat_s, lon_s = subcell_centers_latlon(
        Nx, Ny, xlower, ylower, dx, dy, sub_samples,
    )

    # Back-rotate every sub-point into the canonical frame.
    cos_lat_s = np.cos(lat_s)
    r_out = np.stack(
        [cos_lat_s * np.cos(lon_s),
         cos_lat_s * np.sin(lon_s),
         np.sin(lat_s)],
        axis=-1,
    )
    R_inv = Rotation.from_rotvec(angle * np.asarray(axis)).inv().as_matrix()
    r_in = r_out @ R_inv.T
    r_in[..., 2] = np.clip(r_in[..., 2], -1.0, 1.0)
    lat_in = np.arcsin(r_in[..., 2])
    lon_in = np.mod(np.arctan2(r_in[..., 1], r_in[..., 0]), 2.0 * np.pi)
    qidx = quadrant_index_canonical(lat_in, lon_in)  # (Nx, Ny, S, S)

    h_sub = np.empty(qidx.shape, dtype=np.float64)
    u_sub = np.empty(qidx.shape, dtype=np.float64)
    v_sub = np.empty(qidx.shape, dtype=np.float64)
    for q, s in enumerate(states):
        mask = qidx == q
        h_sub[mask] = s["h"]
        u_sub[mask] = s["u"]
        v_sub[mask] = s["v"]

    # Cell-mean primitives over sub-points.
    h_cell = h_sub.mean(axis=(-2, -1))
    u_cell = u_sub.mean(axis=(-2, -1))
    v_cell = v_sub.mean(axis=(-2, -1))

    # Convert (u_east, u_north) → 3-D Cartesian at cell centre, then ×h.
    ux, uy, uz = en_to_xyz(lat_centers, lon_centers, u_cell, v_cell)

    h_out[...] = h_cell
    momx_out[...] = h_cell * ux
    momy_out[...] = h_cell * uy
    momz_out[...] = h_cell * uz
