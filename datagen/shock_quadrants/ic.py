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
from datagen.shock_quadrants.geometry import en_to_xyz


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
    axis: np.ndarray,
    angle: float,
) -> None:
    """Fill the four conserved-state arrays in place at every cell.

    ``lat_centers`` and ``lon_centers`` are 2-D ``(Nx, Ny)`` arrays of
    cell-centre coordinates in radians.

    The per-quadrant ``(u, v)`` east/north velocities are projected into
    the solver's 3-D Cartesian momentum frame via ``geometry.en_to_xyz``,
    consistent with ``shallow_sphere_2D``'s ``(h, h·ux, h·uy, h·uz)``
    conserved-variable layout.
    """
    states = sample_quadrant_states(seed)

    # Back-rotate every cell centre into the canonical frame.
    sin_t = np.cos(lat_centers)   # sin(colatitude) = cos(lat)
    cos_t = np.sin(lat_centers)   # cos(colatitude) = sin(lat)
    r_out = np.stack(
        [sin_t * np.cos(lon_centers),
         sin_t * np.sin(lon_centers),
         cos_t],
        axis=-1,
    )
    R_inv = Rotation.from_rotvec(angle * np.asarray(axis)).inv().as_matrix()
    r_in = r_out @ R_inv.T
    r_in[..., 2] = np.clip(r_in[..., 2], -1.0, 1.0)
    lat_in = np.arcsin(r_in[..., 2])
    lon_in = np.mod(np.arctan2(r_in[..., 1], r_in[..., 0]), 2.0 * np.pi)
    qidx = quadrant_index_canonical(lat_in, lon_in)

    h_field = np.empty_like(lat_centers)
    u_field = np.empty_like(lat_centers)
    v_field = np.empty_like(lat_centers)
    for q, s in enumerate(states):
        mask = qidx == q
        h_field[mask] = s["h"]
        u_field[mask] = s["u"]
        v_field[mask] = s["v"]

    # Convert (u_east, u_north) → 3-D Cartesian, then multiply by h.
    ux, uy, uz = en_to_xyz(lat_centers, lon_centers, u_field, v_field)

    h_out[...] = h_field
    momx_out[...] = h_field * ux
    momy_out[...] = h_field * uy
    momz_out[...] = h_field * uz
